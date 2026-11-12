#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-spacemit.h"
#include "ocr-ai-complete.h"

#include "spacemit-session.h"
#include "spacemit-kernels.h"
#include "spacemit-opnode.h"
#include "spacemit-context.h"
#include "ocr-norm-affine.h"

#include "ime.h"
#include "repack.h"
#include "spacemit-env.h"
#include "spine_mem_pool.h"
#include "rvv_kernels.h"

#include "ggml-cpu.h"
#include "ggml-profile.h"
#include "ocr-op-timing.h"
#include "ocr-op-progress.h"
#include "ocr-pc8-options.h"
#include "ocr-fa-phases-v50.h"
#include "ocr-im2col-pack.h"
#include "ocr-cast-kv-pack.h"
#include "spacemit-host-memory.h"
#include "ocr-pv-pipeline.h"
#include "ocr-quant-phases.h"
#include "ocr-fa-schedule.h"

// spine-runtime C++ API (hard dependency)
#include <spert.hpp>
#define SPACEMIT_HAS_SPERT 1

using namespace ggml::cpu::riscv64_spacemit;

// Host-only boundaries: tracing never allocates storage or logs on AI workers.
static void spacemit_host_trace(const char * stage, size_t bytes = 0) {
    const char * enabled = std::getenv("GGML_SPACEMIT_HOST_TRACE");
    if (!enabled || std::strcmp(enabled, "1") != 0) return;
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    std::fprintf(stderr, "spacemit-host: %lld.%09ld stage=%s bytes=%zu\n",
                 static_cast<long long>(now.tv_sec), now.tv_nsec, stage, bytes);
    std::fflush(stderr);
}

static void spacemit_host_prefault(void * base, size_t bytes) {
    if (!ggml::spacemit::host_memory::prefault_enabled()) return;
    const long page_size = sysconf(_SC_PAGESIZE);
    spacemit_host_trace("prefault-begin", bytes);
    ggml::spacemit::host_memory::prefault(base, bytes, page_size > 0 ? size_t(page_size) : 4096);
    spacemit_host_trace("prefault-end", bytes);
}

// global_spine_env_info is initialized from spert::backend_info(),
// GGML_SPACEMIT_WORKERS and SPACEMIT_PERFER_CORE_ID instead of parsing
// /proc/cpuinfo.

namespace ggml::cpu::riscv64_spacemit {

spine_env_info::spine_env_info() {
    // Query spine-runtime for hardware info
    spacemit_host_trace("backend-info-begin");
    spert::BackendInfo info = spert::backend_info();
    spacemit_host_trace("backend-info-end");

    // Determine IME support from the CC core architecture id
    uint16_t arch = (uint16_t) info.core_arch_id;

    // A100 = 0xA064, A200 = 0xA0C8, X100 = 0x5064
    perfer_core_arch_id = spine_core_arch_id{ arch };

    use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
               perfer_core_arch_id == spine_core_arch_id::core_arch_x100;
    use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

    // num_cores from env, fallback to spert backend_info
    const char * workers_str = getenv("GGML_SPACEMIT_WORKERS");
    if (workers_str) {
        num_cores = atoi(workers_str);
        if (num_cores <= 0) num_cores = 1;
    } else {
        num_cores = (int) info.num_cores;
        if (num_cores <= 0) num_cores = 1;
    }

    // Optional core pinning: SPACEMIT_PERFER_CORE_ID lists preferred physical
    // CC-core ids (comma-separated, same semantics as the old ggml-cpu/spacemit
    // backend). The graph-compute stream is then bound to these cores and the
    // worker count is capped by the pool size.
    if (const char * perfer_str = getenv("SPACEMIT_PERFER_CORE_ID"); perfer_str && *perfer_str) {
        const int n_cpus = (int) sysconf(_SC_NPROCESSORS_CONF);
        std::vector<int> ids;
        std::string       list(perfer_str);
        size_t            start = 0;
        while (start <= list.size()) {
            size_t end = list.find(',', start);
            if (end == std::string::npos) {
                end = list.size();
            }
            std::string tok = list.substr(start, end - start);
            if (!tok.empty()) {
                int core_id = atoi(tok.c_str());
                if (core_id < 0 || (n_cpus > 0 && core_id >= n_cpus)) {
                    GGML_ABORT("invalid core id %d in SPACEMIT_PERFER_CORE_ID, should be between 0 and %d\n",
                               core_id, n_cpus - 1);
                }
                ids.push_back(core_id);
            }
            start = end + 1;
        }
        if (!ids.empty()) {
            perfer_core_ids = std::move(ids);
            if (num_cores > (int) perfer_core_ids.size()) {
                num_cores = (int) perfer_core_ids.size();
            }
        }
    }

    vlen = info.vlen;
    mem_backend = ggml::spacemit::host_memory::parse_backend(getenv("SPACEMIT_MEM_BACKEND"));
    GGML_LOG_INFO("ggml-spacemit: memory_pool=%s host_prefault=%d\n",
                  ggml::spacemit::host_memory::effective_backend_name(mem_backend),
                  int(ggml::spacemit::host_memory::prefault_enabled()));

    // Allocate init_barrier (needed by ime.cpp kernel barriers)
    init_barrier = new spine_barrier_t[spine_init_barrier_count];
    spine_barrier_init(init_barrier, spine_init_barrier_count, 2);

    std::string core_ids_str = "auto";
    if (!perfer_core_ids.empty()) {
        core_ids_str.clear();
        for (int core_id : perfer_core_ids) {
            core_ids_str += std::to_string(core_id) + ",";
        }
        core_ids_str.pop_back();
    }
    GGML_LOG_INFO("ggml-spacemit: num_cores=%d, core_ids=%s, arch_id=0x%x, vlen=%zu, shared_mem=%zu, use_ime1=%d, use_ime2=%d\n",
                  num_cores, core_ids_str.c_str(), (unsigned) arch, info.vlen, info.shared_mem_size, use_ime1, use_ime2);
}

spine_env_info::~spine_env_info() {
    delete[] init_barrier;
    init_barrier = nullptr;
}

spine_env_info global_spine_env_info;

}  // namespace ggml::cpu::riscv64_spacemit

//** static config

//** helpers

static inline bool op_is_compute(ggml_tensor * node) {
    return !ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags & GGML_TENSOR_FLAG_COMPUTE);
}

static spacemit_op_code op_remap_to_spacemit(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:    return SPACEMIT_OP_MUL_MAT;
        case GGML_OP_MUL_MAT_ID: return SPACEMIT_OP_MUL_MAT_ID;
        case GGML_OP_ADD:        return SPACEMIT_OP_ADD;
        case GGML_OP_RMS_NORM:   return SPACEMIT_OP_RMS_NORM;
        case GGML_OP_ROPE:       return SPACEMIT_OP_ROPE;
        case GGML_OP_SOFT_MAX:   return SPACEMIT_OP_SOFTMAX;
        case GGML_OP_RESHAPE:    return SPACEMIT_OP_RESHAPE;
        case GGML_OP_VIEW:       return SPACEMIT_OP_VIEW;
        case GGML_OP_PERMUTE:    return SPACEMIT_OP_PERMUTE;
        case GGML_OP_TRANSPOSE:  return SPACEMIT_OP_TRANSPOSE;
        case GGML_OP_NONE:       return SPACEMIT_OP_NONE;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_SILU: return SPACEMIT_OP_UNARY_SILU;
                case GGML_UNARY_OP_GELU: return SPACEMIT_OP_UNARY_GELU;
                default:                 break;
            }
            break;
        default:
            break;
    }
    return SPACEMIT_OP_INVALID;
}

//** op fusion helpers
//
// Fusion patterns, matched on the graph when the opnode list is built:
//   1. RMS_NORM + MUL    -> SPACEMIT_OP_RMS_NORM_MUL (one opnode at the rms position)
//   2. ADD + GELU        -> SPACEMIT_OP_ADD_GELU (write directly to the GELU output)
//   3. CONCAT + SSM_CONV -> SPACEMIT_OP_CONCAT_STATE at the concat position (partial write)
//                           + SPACEMIT_OP_SSM_CONV_CONCAT at the ssm_conv position (reads the
//                           concat inputs directly; the concat only materialized its trailing
//                           columns). Both halves must keep their original node positions: the
//                           graph allocator plans buffer reuse on the node order.
//
// Fused opnodes are dispatched by ggml_spacemit_compute_opnode through the
// tensor traits of their primary node (compute_forward_fused).
//
// try_fuse_node contains ALL fusion matching and emission: Pattern 3 consumes
// two nodes at different positions (the concat now, the ssm_conv when the loop
// reaches it â€?opnode list order is execution order, so the halves keep their
// original node positions); `emitted` carries the pairing across the gap.

// RMS_NORM followed by MUL (rms, weight) that can run as one fused RVV kernel.
static bool rms_norm_mul_fusable(const ggml_tensor * rms, const ggml_tensor * mul) {
    if (mul->op != GGML_OP_MUL || (mul->src[0] != rms && mul->src[1] != rms)) {
        return false;
    }
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    return x && w && x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
           mul->type == GGML_TYPE_F32 && mul->nb[0] == sizeof(float) &&
           w->type == GGML_TYPE_F32 && w->nb[0] == sizeof(float) &&
           w->ne[0] == rms->ne[0] && ggml_are_same_shape(x, mul);
}

static bool add_gelu_fusable(const ggml_tensor * add, const ggml_tensor * gelu) {
    if (add->op != GGML_OP_ADD || gelu->op != GGML_OP_UNARY ||
        ggml_get_unary_op(gelu) != GGML_UNARY_OP_GELU || gelu->src[0] != add ||
        !ggml_are_same_shape(add, gelu)) {
        return false;
    }
    const ggml_tensor * x = add->src[0];
    const ggml_tensor * b = add->src[1];
    const bool scalar_bias = b && ggml_nelements(b) == 1;
    const bool row_bias = x && b && b->ne[0] == x->ne[0] && b->ne[1] == 1 &&
                          b->ne[2] == 1 && b->ne[3] == 1;
    return x && b && x->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
           add->type == GGML_TYPE_F32 && gelu->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(x) && ggml_is_contiguous(b) && ggml_is_contiguous(add) &&
           ggml_is_contiguous(gelu) && ggml_are_same_shape(x, add) &&
           ggml_can_repeat(b, x) && (scalar_bias || row_bias);
}

// Q4_0 MatMul followed by its constant F32 row-bias Add. Keep this narrowly
// gated: the no-zero-point IME2 HP kernel adds bias before storing full N32
// tiles into the Add output. All other shapes retain the unfused graph.
static bool matmul_add_bias_fusable(const ggml_tensor * matmul, const ggml_tensor * add) {
    if (!matmul || !add || matmul->op != GGML_OP_MUL_MAT || add->op != GGML_OP_ADD ||
        add->src[0] != matmul || matmul->src[0] == nullptr || matmul->src[1] == nullptr || add->src[1] == nullptr ||
        matmul->src[0]->type != GGML_TYPE_Q4_0 || matmul->src[1]->type != GGML_TYPE_F32 ||
        matmul->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(matmul, add) || !ggml_is_contiguous(matmul) || !ggml_is_contiguous(add) ||
        !global_spine_env_info.use_ime2 || matmul->src[0]->ne[0] % 256 != 0 ||
        matmul->src[0]->ne[1] % 32 != 0 || !ggml_is_contiguous(matmul->src[0]) ||
        !ggml_is_contiguous(matmul->src[1])) {
        return false;
    }
    const ggml_tensor * bias = add->src[1];
    return bias->op == GGML_OP_NONE && bias->src[0] == nullptr && bias->type == GGML_TYPE_F32 &&
           bias->ne[0] == matmul->ne[0] && bias->ne[1] == 1 && bias->ne[2] == 1 && bias->ne[3] == 1 &&
           ggml_is_contiguous(bias) && ggml_can_repeat(bias, matmul);
}

// Experimental until exact-output and model-level checks pass. Read before the
// op list is cached; use a separate process when changing this switch.
static bool add_gelu_fusion_enabled() {
    const char * value = getenv("GGML_SPACEMIT_ADD_GELU");
    return value && strcmp(value, "1") == 0;
}

static bool matmul_add_fusion_enabled() {
    const char * value = getenv("GGML_SPACEMIT_MUL_MAT_ADD");
    return value && strcmp(value, "1") == 0;
}

// GGML_SPACEMIT_CONV_FUSE: 0/unset off, 1 concat + ssm_conv fusion with the concat inputs kept alive
// through graph_optimize, 2 additionally the scalar few-token kernels. Host thread only.
static int conv_fuse_mode() {
    const char * value = getenv("GGML_SPACEMIT_CONV_FUSE");
    if (value == nullptr) {
        return 0;
    }
    if (strcmp(value, "1") == 0) {
        return 1;
    }
    if (strcmp(value, "2") == 0) {
        return 2;
    }
    return 0;
}

static bool gdn_inplace_enabled() {
    const char * value = getenv("GGML_SPACEMIT_GDN_INPLACE");
    return value && strcmp(value, "1") == 0;
}

static bool byte_ranges_overlap(const void * a, size_t an, const void * b, size_t bn) {
    const char * a0 = (const char *) a;
    const char * b0 = (const char *) b;
    return a0 < b0 + bn && b0 < a0 + an;
}

// In-place recurrent state update. Graph shape (build_rs + build_delta_net_fused + build_recurrent_attn):
//   R = get_rows(states {state_elems, n_rs}, ids {1})         copy of the previous state out of the cache
//   G = gated_delta_net(q, k, v, g, beta, view(R), K = 1)     output = [attention | new state]
//   C = cpy(view(G, new state), view(states, row kv_head))    new state back into the cache
// When the gathered row is the row C writes (single sequence), R and C only move 1 MiB per layer out of
// the cache and back. G can then read and write the cache row directly and R / C have nothing to do.
// This matcher proves at list-build time that nobody else reads R's copy or G's new-state region and
// that the cache row is not written between R and G nor observed between G and C; the row identity
// itself is checked at run time (ggml_spacemit_gdn_inplace_now), the row index being a graph input.
static bool gdn_inplace_match(const ggml_cgraph * graph, int gi, ggml_tensor ** gather_out, ggml_tensor ** store_out) {
    ggml_tensor * g = graph->nodes[gi];
    if (g->op != GGML_OP_GATED_DELTA_NET || ggml_get_op_params_i32(g, 0) != 1 || g->type != GGML_TYPE_F32 ||
        g->view_src != nullptr || g->data == nullptr || (g->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    for (int j = 0; j < 6; ++j) {
        if (g->src[j] == nullptr) {
            return false;
        }
    }
    const ggml_tensor * v        = g->src[2];
    const int64_t       S_v      = v->ne[0];
    const int64_t       H        = v->ne[1];
    const int64_t       n_tokens = v->ne[2];
    const int64_t       n_seqs   = v->ne[3];
    if (n_seqs != 1 || g->src[3]->ne[0] == S_v) {
        return false;  // one sequence, scalar gate only
    }
    const int64_t state_elems = S_v * S_v * H;
    const int64_t attn_elems  = S_v * H * n_tokens * n_seqs;
    const size_t  state_bytes = (size_t) state_elems * sizeof(float);
    const size_t  attn_bytes  = (size_t) attn_elems * sizeof(float);
    if (ggml_nelements(g) != attn_elems + state_elems || !ggml_is_contiguous(g)) {
        return false;
    }

    // the state input: a plain window on the whole gathered copy
    const ggml_tensor * s = g->src[5];
    ggml_tensor *       r = s->view_src;
    if (r == nullptr || r->op != GGML_OP_GET_ROWS || s->view_offs != 0 || s->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(s) || ggml_nelements(s) != state_elems) {
        return false;
    }
    if (r->type != GGML_TYPE_F32 || r->view_src != nullptr || r->data == nullptr ||
        (r->flags & GGML_TENSOR_FLAG_OUTPUT) || r->ne[0] != state_elems || ggml_nelements(r) != state_elems) {
        return false;
    }
    const ggml_tensor * states = r->src[0];
    const ggml_tensor * ids    = r->src[1];
    if (states == nullptr || ids == nullptr || states->type != GGML_TYPE_F32 || states->data == nullptr ||
        states->ne[0] != state_elems || states->nb[0] != sizeof(float) || states->nb[1] != state_bytes ||
        states->ne[2] != 1 || states->ne[3] != 1 || ids->type != GGML_TYPE_I32 || ggml_nelements(ids) != 1 ||
        ids->data == nullptr) {
        return false;
    }

    // readers of R and G
    ggml_tensor * store     = nullptr;
    int           store_idx = -1;
    int           r_idx     = -1;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * t = graph->nodes[i];
        if (t == r) {
            r_idx = i;
        }
        if ((t->view_src == r || t->view_src == g) && (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
        if (!op_is_compute(t)) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * x = t->src[j];
            if (x == nullptr) {
                continue;
            }
            if (x == r || x->view_src == r) {
                if (t != g || j != 5) {
                    return false;  // the gathered copy has another reader (or a writer)
                }
            } else if (x == g) {
                return false;  // reads the whole output including the new-state region
            } else if (x->view_src == g) {
                if (x->view_offs + ggml_nbytes(x) <= attn_bytes) {
                    continue;  // attention output
                }
                // the only reader of the new state must be the store into the cache
                if (store != nullptr || t->op != GGML_OP_CPY || j != 0 || x->view_offs != attn_bytes ||
                    x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ggml_nelements(x) != state_elems) {
                    return false;
                }
                store     = t;
                store_idx = i;
            }
        }
    }
    if (store == nullptr || r_idx < 0 || !(r_idx < gi && gi < store_idx)) {
        return false;
    }
    if (store->type != GGML_TYPE_F32 || store->data == nullptr || !ggml_is_contiguous(store) ||
        ggml_nelements(store) != state_elems) {
        return false;
    }

    const void * row = store->data;  // the cache row written by C
    // nothing between R and G may write the cache row: G now reads it where R used to
    for (int i = r_idx + 1; i < gi; ++i) {
        ggml_tensor * t = graph->nodes[i];
        if (!op_is_compute(t)) {
            continue;
        }
        if (t->data == nullptr || byte_ranges_overlap(t->data, ggml_nbytes(t), row, state_bytes)) {
            return false;
        }
    }
    // C must be the next compute node after G: nobody may observe the cache row in between
    for (int i = gi + 1; i < store_idx; ++i) {
        if (op_is_compute(graph->nodes[i])) {
            return false;
        }
    }
    // G's own output and its other inputs must not live in the cache row
    if (byte_ranges_overlap(g->data, ggml_nbytes(g), row, state_bytes)) {
        return false;
    }
    for (int j = 0; j < 5; ++j) {
        const ggml_tensor * in   = g->src[j];
        const ggml_tensor * root = in->view_src ? in->view_src : in;
        if (root->data == nullptr || byte_ranges_overlap(root->data, ggml_nbytes(root), row, state_bytes)) {
            return false;
        }
    }

    *gather_out = r;
    *store_out  = store;
    return true;
}

// Marks the {R, G, C} opnodes of every matching gated_delta_net. All three must be plain single-op
// opnodes. Returns the number of layers marked.
static int gdn_inplace_mark(const ggml_cgraph * graph, std::vector<spacemit_opnode> & nodes) {
    int n_marked = 0;
    for (int gi = 0; gi < graph->n_nodes; ++gi) {
        ggml_tensor * g = graph->nodes[gi];
        if (g->op != GGML_OP_GATED_DELTA_NET || !op_is_compute(g)) {
            continue;
        }
        ggml_tensor * gather = nullptr;
        ggml_tensor * store  = nullptr;
        if (!gdn_inplace_match(graph, gi, &gather, &store)) {
            continue;
        }
        ggml_tensor * const want[3] = {gather, g, store};
        size_t              pos[3]  = {0, 0, 0};
        int                 seen[3] = {0, 0, 0};
        bool                ok      = true;
        for (size_t k = 0; k < nodes.size() && ok; ++k) {
            const spacemit_opnode & on = nodes[k];
            for (int w = 0; w < 3; ++w) {
                if (on.node == want[w]) {
                    if (on.opcode != SPACEMIT_OP_INVALID || !on.fused.empty() || !on.extra_dsts.empty() ||
                        on.packed_im2col != nullptr || on.elided_pad != nullptr ||
                        on.prepared_attention != nullptr) {
                        ok = false;
                    }
                    pos[w] = k;
                    seen[w]++;
                }
                if (on.packed_im2col == want[w] || on.elided_pad == want[w] || on.prepared_attention == want[w]) {
                    ok = false;
                }
                for (const ggml_tensor * f : on.fused) {
                    if (f == want[w]) {
                        ok = false;
                    }
                }
            }
        }
        if (!ok || seen[0] != 1 || seen[1] != 1 || seen[2] != 1 || !(pos[0] < pos[1] && pos[1] + 1 == pos[2])) {
            continue;
        }
        static const spacemit_op_code codes[3] = {SPACEMIT_OP_GDN_STATE_GATHER, SPACEMIT_OP_GDN_INPLACE,
                                                  SPACEMIT_OP_GDN_STATE_STORE};
        for (int w = 0; w < 3; ++w) {
            spacemit_opnode & on = nodes[pos[w]];
            on.opcode            = codes[w];
            on.gdn_state[0]      = gather;
            on.gdn_state[1]      = g;
            on.gdn_state[2]      = store;
        }
        n_marked++;
    }
    return n_marked;
}

// concat(conv_states, transpose(qkv), dim 0) as built by build_conv_state: contiguous F32 rows, the
// second input laid out transposed (column-major) so the fused kernel reads it column by column.
static bool is_transposed_concat_f32(const ggml_tensor * t) {
    if (t->op != GGML_OP_CONCAT || ggml_get_op_params_i32(t, 0) != 0) {
        return false;
    }
    const ggml_tensor * s0 = t->src[0];
    const ggml_tensor * s1 = t->src[1];
    return t->type == GGML_TYPE_F32 && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 &&
           t->nb[0] == sizeof(float) && s0->nb[0] == sizeof(float) && s1->nb[1] == sizeof(float) &&
           s1->nb[0] != sizeof(float) && s0->ne[1] == t->ne[1] && s1->ne[1] == t->ne[1] && s0->ne[2] == t->ne[2] &&
           s1->ne[2] == t->ne[2] && t->ne[3] == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
}

// First column of concat c covered by view v, or -1 if v is not a plain window of whole rows' trailing
// columns (same row/plane strides, starting in row 0).
static int64_t concat_view_first_col(const ggml_tensor * c, const ggml_tensor * v) {
    if (v->type != c->type || v->nb[0] != c->nb[0] || (v->ne[1] > 1 && v->nb[1] != c->nb[1]) ||
        (v->ne[2] > 1 && v->nb[2] != c->nb[2]) || v->ne[3] != 1 || v->view_offs / c->nb[1] != 0) {
        return -1;
    }
    const int64_t col = (int64_t) ((v->view_offs % c->nb[1]) / c->nb[0]);
    if (col + v->ne[0] > c->ne[0] || v->ne[1] > c->ne[1] || v->ne[2] > c->ne[2]) {
        return -1;
    }
    return col;
}

// concat -> ssm_conv fusion analysis: SSM_CONV reads the two concat inputs directly and CONCAT only
// writes the trailing columns that are read back through views (the conv state update). Returns the
// single ssm_conv consumer (with its graph index) and the first column any view reads back. Fails if
// the concat has any other consumer (or a second ssm_conv) that needs the full concat output.
static bool ssm_conv_concat_fusable(const ggml_cgraph * graph, int ci,
                                    ggml_tensor ** ssm_out, int * ssm_idx_out, int64_t * col0_out) {
    ggml_tensor * c = graph->nodes[ci];
    if (!is_transposed_concat_f32(c) || c->src[0]->ne[0] != 3) {
        return false;
    }

    ggml_tensor * ssm    = nullptr;
    int           ssm_idx = -1;
    int64_t       col0    = c->ne[0];  // min column read back through views

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * t = graph->nodes[i];

        if (t->view_src == c) {
            const int64_t col = concat_view_first_col(c, t);
            if (col < 0) {
                return false;
            }
            col0 = std::min(col0, col);
        }

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * s = t->src[j];
            if (s == nullptr) {
                continue;
            }
            if (s == c) {
                if (t->op == GGML_OP_SSM_CONV && j == 0 && t->src[1] && t->src[1]->ne[0] == 4 &&
                    ((t->src[2] == c->src[1] && t->src[3] == c->src[0]) ||
                     (conv_fuse_mode() > 0 && t->src[2] == nullptr && t->src[3] == nullptr))) {
                    // the graph keeps the concat inputs alive until this node (build_ssm_conv)
                    if (ssm != nullptr) {
                        return false;  // more than one ssm_conv consumer
                    }
                    ssm     = t;
                    ssm_idx = i;
                } else if (t->view_src != c) {
                    return false;  // some other consumer needs the full concat output
                }
            } else if (s->view_src == c) {
                const int64_t col = concat_view_first_col(c, s);
                if (col < 0) {
                    return false;
                }
                col0 = std::min(col0, col);
            }
        }
    }

    if (ssm == nullptr || ssm_idx <= ci) {
        return false;
    }

    // The fused kernels re-check their layouts at compute time and abort on a mismatch (there is no
    // fallback once the concat half has materialized only its trailing columns), so the gate must be
    // a superset of forward_concat_dim0_transposed_f32 + forward_ssm_conv_concat_f32: types, element
    // strides and the n_t / d_inner shape agreement between the concat and the ssm_conv output.
    const ggml_tensor * w = ssm->src[1];
    if (ssm->type != GGML_TYPE_F32 || ssm->nb[0] != sizeof(float) ||
        w->type != GGML_TYPE_F32 || w->nb[0] != sizeof(float) ||
        c->src[1]->ne[0] != ssm->ne[1] || c->ne[1] != ssm->ne[0]) {
        return false;
    }

    *ssm_out     = ssm;
    *ssm_idx_out = ssm_idx;
    *col0_out    = col0;
    return true;
}

// The fused ssm_conv reads the concat inputs at its own (later) graph position. With the alloc deps
// from graph_optimize their buffers stay allocated; verify on the final addresses that no compute
// node in [concat, ssm_conv] writes into them. On failure both nodes simply stay unfused.
static bool ssm_conv_concat_inputs_intact(const ggml_cgraph * graph, int ci, int ssm_idx) {
    const ggml_tensor * c = graph->nodes[ci];
    for (int k = 0; k < 2; ++k) {
        const ggml_tensor * in   = c->src[k];
        const ggml_tensor * root = in->view_src ? in->view_src : in;
        if (root->data == nullptr) {
            return false;
        }
        const char * r0 = (const char *) root->data;
        const char * r1 = r0 + ggml_nbytes(root);
        for (int i = ci; i <= ssm_idx; ++i) {
            ggml_tensor * t = graph->nodes[i];
            if (!op_is_compute(t)) {
                continue;
            }
            if (t->data == nullptr) {
                return false;
            }
            const char * t0 = (const char *) t->data;
            const char * t1 = t0 + ggml_nbytes(t);
            if (t0 < r1 && r0 < t1) {
                return false;
            }
        }
    }
    return true;
}

static bool try_fuse_node(const ggml_cgraph * graph, int & i, std::vector<spacemit_opnode> & nodes,
                          std::unordered_set<ggml_tensor *> & emitted) {
    ggml_tensor * n = graph->nodes[i];

    if (const auto plan = ocr_cast_kv_pack::match(graph,i)) {
        spacemit_opnode fused(n,{},SPACEMIT_OP_OCR_CAST_KV_PACK);
        fused.add_fused(graph->nodes[plan.value_index],true);
        fused.prepared_attention=plan.attention;
        nodes.push_back(std::move(fused));
        i=plan.value_index;
        return true;
    }

    const char * norm_option = getenv("GGML_SPACEMIT_NORM_AFFINE");
    if (norm_option && strcmp(norm_option, "1") == 0) {
        const auto plan = ocr_norm_affine::match(graph, i);
        if (plan && ocr_norm_affine::storage_safe(plan)) {
            spacemit_opnode fused(n, {}, plan.residual ? SPACEMIT_OP_RESIDUAL_NORM_AFFINE : SPACEMIT_OP_NORM_AFFINE);
            if (plan.residual) fused.add_fused(plan.norm);
            fused.add_fused(plan.scale);
            fused.add_fused(plan.bias);
            if (plan.residual) fused.extra_dsts.push_back(plan.bias);
            nodes.push_back(std::move(fused));
            i = plan.last;
            return true;
        }
    }

    const char * pack_option=getenv("GGML_SPACEMIT_IM2COL_PACK");
    const char * quant_capture=getenv("GGML_SPACEMIT_OCR_QUANT_CAPTURE");
    if(pack_option&&strcmp(pack_option,"1")==0&&(!quant_capture||!*quant_capture)) {
        const int matrix_index=ocr_im2col_pack::match(graph,i);
        if(matrix_index>=0) {
            auto * matrix=graph->nodes[matrix_index];
            // A shape match alone does not prove a physical hardware layout.
            if(matrix->src[0]->extra&&ggml_spacemit_get_tensor_traits(matrix)==matrix->src[0]->extra) {
                spacemit_opnode fused(matrix,{},SPACEMIT_OP_IM2COL_MUL_MAT);
                fused.packed_im2col=n;nodes.push_back(std::move(fused));i=matrix_index;return true;
            }
        }
    }

    // Second half of Pattern 3, arriving here: an ssm_conv paired with an earlier concat.
    // Emitted at its own position â€?opnode list order is execution order, and the graph
    // allocator planned the ssm_conv output buffer on the original node order (see Pattern 3).
    if (emitted.count(n)) {
        nodes.emplace_back(n, std::vector<ggml_tensor *>{}, SPACEMIT_OP_SSM_CONV_CONCAT);
        nodes.back().kernel_params[1] = conv_fuse_mode() >= 2;
        return true;
    }

    // Find the next compute node, skipping empty (NONE) nodes. test-backend-ops
    // inserts NONE sentinels between ops for out-of-bounds detection, which must
    // not break fusion adjacency. Production graphs have no such sentinels.
    int next_idx = -1;
    for (int j = i + 1; j < graph->n_nodes; ++j) {
        if (op_is_compute(graph->nodes[j])) {
            next_idx = j;
            break;
        }
    }
    ggml_tensor * next_node = (next_idx >= 0) ? graph->nodes[next_idx] : nullptr;

    // PC8 INT32 epilogue applies the separate F32 bias add before writeback.
    // Keep the ordinary ggml use/output/view guards and constant row bias only.
    const char *pc8_bias=getenv("GGML_SPACEMIT_OCR_PC8_BIAS_FUSE");
    if(pc8_bias&&!strcmp(pc8_bias,"1")&&n->op==GGML_OP_MUL_MAT&&next_node&&
       next_node->op==GGML_OP_ADD&&next_node->src[0]==n&&
       ggml_spacemit_can_fuse_pc8_bias(n)&&next_node->type==GGML_TYPE_F32&&
       ggml_are_same_shape(n,next_node)&&ggml_is_contiguous(next_node)) {
        const auto*b=next_node->src[1];
        if(b&&b->op==GGML_OP_NONE&&b->src[0]==nullptr&&b->type==GGML_TYPE_F32&&
           b->ne[0]==n->ne[0]&&b->ne[1]==1&&b->ne[2]==1&&b->ne[3]==1&&ggml_is_contiguous(b)) {
            const int idxs[2]={i,next_idx};const enum ggml_op ops[2]={GGML_OP_MUL_MAT,GGML_OP_ADD};
            if(ggml_can_fuse_subgraph_ext(graph,idxs,2,ops,&next_idx,1)) {
                const char *gelu_option=getenv("GGML_SPACEMIT_OCR_PC8_GELU_FUSE");
                int gelu_index=-1;
                for(int j=next_idx+1;j<graph->n_nodes;++j)if(op_is_compute(graph->nodes[j])){gelu_index=j;break;}
                if(gelu_option&&!strcmp(gelu_option,"1")&&gelu_index>=0) {
                    auto*gelu=graph->nodes[gelu_index];
                    const int all_idx[3]={i,next_idx,gelu_index};
                    const enum ggml_op all_ops[3]={GGML_OP_MUL_MAT,GGML_OP_ADD,GGML_OP_UNARY};
                    if(gelu->op==GGML_OP_UNARY&&ggml_get_unary_op(gelu)==GGML_UNARY_OP_GELU&&
                       gelu->src[0]==next_node&&gelu->type==GGML_TYPE_F32&&
                       ggml_are_same_shape(n,gelu)&&ggml_is_contiguous(gelu)&&
                       ggml_can_fuse_subgraph_ext(graph,all_idx,3,all_ops,&gelu_index,1)) {
                        spacemit_opnode fused(n,{},SPACEMIT_OP_PC8_MUL_MAT_BIAS);
                        fused.add_fused(next_node);fused.add_fused(gelu);
                        nodes.push_back(std::move(fused));i=gelu_index;return true;
                    }
                }
                spacemit_opnode fused(n,{},SPACEMIT_OP_PC8_MUL_MAT_BIAS);
                fused.add_fused(next_node);nodes.push_back(std::move(fused));i=next_idx;return true;
            }
        }
    }

    // Pack the unpadded activation directly into the PC8 INT8 workspace.
    // Retain graph allocations; all input reads and the packing barrier finish
    // before the matmul writes output, even if gallocr reuses the input buffer.
    const char * pc8_pad=getenv("GGML_SPACEMIT_OCR_PC8_PAD_FUSE");
    if(pc8_pad&&!strcmp(pc8_pad,"1")&&n->op==GGML_OP_PAD&&next_node&&
       next_node->src[1]==n&&ggml_spacemit_can_fuse_pc8_pad(next_node)) {
        const int idxs[2]={i,next_idx};
        const enum ggml_op ops[2]={GGML_OP_PAD,GGML_OP_MUL_MAT};
        if(ggml_can_fuse_subgraph_ext(graph,idxs,2,ops,&next_idx,1)) {
            spacemit_opnode fused(next_node,{},SPACEMIT_OP_PC8_PAD_MUL_MAT);
            fused.elided_pad=n;
            const char*option=getenv("GGML_SPACEMIT_OCR_PC8_DOWN_EPILOGUE");
            int last=next_idx;
            if(option&&!strcmp(option,"1")){
                const auto following=[&](int x){for(int j=x+1;j<graph->n_nodes;++j)if(op_is_compute(graph->nodes[j]))return j;return -1;};
                const int bi=following(next_idx);
                if(bi>=0){
                    auto*b=graph->nodes[bi];
                    const int ids[3]={i,next_idx,bi};const ggml_op ops[3]={GGML_OP_PAD,GGML_OP_MUL_MAT,GGML_OP_ADD};
                    if(b->op==GGML_OP_ADD&&b->src[0]==next_node&&b->type==GGML_TYPE_F32&&
                       ggml_are_same_shape(b,next_node)&&ggml_is_contiguous(b)&&
                       ocr_norm_affine::row_weight(b->src[1],b->ne[0])&&b->src[1]->op==GGML_OP_NONE&&
                       b->src[1]->src[0]==nullptr&&ggml_can_fuse_subgraph_ext(graph,ids,3,ops,&bi,1)){
                        fused.add_fused(b);last=bi;
                        const int ri=following(bi);
                        if(ri>=0){
                            auto*res=graph->nodes[ri];
                            auto*skip=res->src[0]==b?res->src[1]:res->src[1]==b?res->src[0]:nullptr;
                            const int all[4]={i,next_idx,bi,ri};const ggml_op allops[4]={GGML_OP_PAD,GGML_OP_MUL_MAT,GGML_OP_ADD,GGML_OP_ADD};
                            if(res->op==GGML_OP_ADD&&skip&&res->type==GGML_TYPE_F32&&skip->type==GGML_TYPE_F32&&
                               ggml_are_same_shape(res,b)&&ggml_are_same_shape(skip,b)&&ggml_is_contiguous(res)&&ggml_is_contiguous(skip)&&
                               (!ocr_norm_affine::overlaps(res,skip)||res->data==skip->data)&&
                               ggml_can_fuse_subgraph_ext(graph,all,4,allops,&ri,1)){
                                fused.add_fused(res);last=ri;
                            }
                        }
                    }
                }
            }
            nodes.push_back(std::move(fused));i=last;return true;
        }
    }

    // Pattern 1: RMS_NORM + MUL
    if (n->op == GGML_OP_RMS_NORM && next_node && next_node->op == GGML_OP_MUL) {
        const int          idxs[2] = { i, next_idx };
        const enum ggml_op ops[2]  = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse_ext(graph, idxs, ops, 2) && rms_norm_mul_fusable(n, next_node)) {
            spacemit_opnode node(n, {}, SPACEMIT_OP_RMS_NORM_MUL);
            node.add_fused(next_node);
            nodes.push_back(std::move(node));
            i = next_idx;  // skip past the fused MUL node
            return true;
        }
    }

    // Pattern 2: ADD + GELU. The fused kernel reads ADD's two inputs and writes the final activation
    // directly, avoiding a full-size pre-activation materialization and an inter-node barrier.
    if (add_gelu_fusion_enabled() && n->op == GGML_OP_ADD && next_node && next_node->op == GGML_OP_UNARY &&
        ggml_get_unary_op(next_node) == GGML_UNARY_OP_GELU) {
        const int          idxs[2] = { i, next_idx };
        const enum ggml_op ops[2]  = { GGML_OP_ADD, GGML_OP_UNARY };
        if (ggml_can_fuse_ext(graph, idxs, ops, 2) && add_gelu_fusable(n, next_node)) {
            spacemit_opnode node(n, {}, SPACEMIT_OP_ADD_GELU);
            node.add_fused(next_node);
            nodes.push_back(std::move(node));
            i = next_idx;
            return true;
        }
    }

    // Pattern 4: Q4_0 MUL_MAT + constant row-bias ADD. The fused op retains
    // the matmul's graph position but writes its tile to the final Add output.
    if (matmul_add_fusion_enabled() && n->op == GGML_OP_MUL_MAT && next_node && next_node->op == GGML_OP_ADD &&
        matmul_add_bias_fusable(n, next_node)) {
        const int          idxs[2] = { i, next_idx };
        const enum ggml_op ops[2]  = { GGML_OP_MUL_MAT, GGML_OP_ADD };
        if (ggml_can_fuse_ext(graph, idxs, ops, 2)) {
            spacemit_opnode node(n, {}, SPACEMIT_OP_MUL_MAT_ADD);
            node.add_fused(next_node);
            nodes.push_back(std::move(node));
            i = next_idx;
            return true;
        }
    }

    // Pattern 3: CONCAT + SSM_CONV. Keep both opnodes at the original positions:
    // the allocator may reuse buffers between the concat and ssm_conv nodes.
    if (n->op == GGML_OP_CONCAT) {
        ggml_tensor * ssm     = nullptr;
        int           ssm_idx = -1;
        int64_t       col0    = 0;
        if (ssm_conv_concat_fusable(graph, i, &ssm, &ssm_idx, &col0) &&
            (ssm->src[2] != nullptr || ssm_conv_concat_inputs_intact(graph, i, ssm_idx))) {
            spacemit_opnode cat(n, {}, SPACEMIT_OP_CONCAT_STATE);
            cat.kernel_params[0] = (int32_t) col0;
            cat.kernel_params[1] = conv_fuse_mode() >= 2;
            nodes.push_back(std::move(cat));
            // the ssm_conv half is emitted by try_fuse_node when the loop reaches its own
            // node position (see the top of this function) â€?never here at the concat position
            emitted.insert(ssm);
            return true;
        }
    }

    return false;
}

//** buffer interface
//
// Reuses spine_mem_pool_alloc/free for allocation and repack for init_tensor/set_tensor.
// Pattern adapted from ggml-cpu/spacemit/ime.cpp buffer management.

static void ggml_backend_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    if (base == nullptr) {
        return;
    }

    spine_mem_pool_free(base);
}

static void * ggml_backend_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    return base;
}

static enum ggml_status ggml_backend_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                   ggml_tensor *         tensor) {
    // Embedding lookup needs the original row-addressable quantized layout.
    if (tensor->flags & GGML_TENSOR_FLAG_NO_REPACK) { tensor->extra = nullptr; return GGML_STATUS_SUCCESS; }
    tensor->extra =
        (void *) const_cast<ggml::spacemit::tensor_traits_base *>(ggml_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                         ggml_tensor *         tensor,
                                                         uint8_t               value,
                                                         size_t                offset,
                                                         size_t                size) {
    GGML_ASSERT(tensor);
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

static void ggml_backend_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                      ggml_tensor *         tensor,
                                                      const void *          data,
                                                      size_t                offset,
                                                      size_t                size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    // Quantized tensors with tensor traits are physically repacked, so they
    // must be uploaded atomically. Plain tensors preserve the ggml layout and
    // support the partial writes used by test-backend-ops and graph inputs.
    if (tensor->extra) {
        GGML_ASSERT(offset == 0);
        GGML_ASSERT(size == ggml_nbytes(tensor));
        auto ok = ggml_riscv64_spacemit_repack_tensor(tensor, data, size);
        GGML_ASSERT(ok == 0);
    } else {
        memcpy((char *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

static void ggml_backend_spacemit_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                     const ggml_tensor *   tensor,
                                                     void *                data,
                                                     size_t                offset,
                                                     size_t                size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static const ggml_backend_buffer_i ggml_backend_spacemit_buffer_i = {
    /* .free_buffer     = */ ggml_backend_spacemit_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_spacemit_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_spacemit_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_spacemit_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_spacemit_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_spacemit_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_spacemit_buffer_clear,
    /* .reset           = */ nullptr,
};

//** buffer type interface

static const char * ggml_backend_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "SPACEMIT";
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                              size_t size) {
    spacemit_host_trace("buffer-alloc-begin", size);
    void * base = spine_mem_pool_alloc(size, 64);
    if (base == nullptr) {
        return nullptr;
    }

    spacemit_host_prefault(base, size);
    spacemit_host_trace("buffer-alloc-end", size);
    return ggml_backend_buffer_init(buft, ggml_backend_spacemit_buffer_i, base, size);
}

static size_t ggml_backend_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_spacemit_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX;
    GGML_UNUSED(buft);
}

static size_t ggml_backend_spacemit_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_spacemit_nbytes(buft, tensor);
}

static bool ggml_backend_spacemit_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_spacemit_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_spacemit_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_spacemit_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr, // default: alloc_buffer + linear allocator
    /* .get_alignment    = */ ggml_backend_spacemit_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_spacemit_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_spacemit_buffer_type_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr, // default: per-tensor get_alloc_size
    /* .is_host          = */ ggml_backend_spacemit_buffer_type_is_host,
};

static ggml_backend_buffer_type_t ggml_backend_spacemit_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type buft_s = {
        /* .iface   = */ ggml_backend_spacemit_buffer_type_interface,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    if (buft_s.device == nullptr) {
        buft_s.device = dev;
    }
    return &buft_s;
}

static bool ggml_backend_buffer_is_spacemit(const struct ggml_backend_buffer * b) {
    return b && b->buft->iface.get_name == ggml_backend_spacemit_buffer_type_get_name;
}

//** backend interface

spacemit_session::~spacemit_session() {
    free(workspace);
}

static const char * ggml_backend_spacemit_name(ggml_backend_t backend) {
    auto sess = static_cast<spacemit_session *>(backend->context);
    return sess->c_name();
}

static void ggml_backend_spacemit_free(ggml_backend_t backend) {
    // The device registry owns the shared session.
    delete backend;
}

// Determinism probe for transformer layer outputs. Only these checkpoints gain
// one extra barrier, after the normal node barrier publishes all writes and
// before the next node can reuse an alias. This is diagnostic only, not a
// benchmark mode.
static bool ggml_spacemit_is_trace_checkpoint(const ggml_tensor * tensor, int graph_nodes) {
    if (tensor == nullptr) {
        return false;
    }
    if (graph_nodes <= 2 && tensor->type == GGML_TYPE_F32 && ggml_nelements(tensor) >= 1024) {
        return true;
    }
    static constexpr const char * prefixes[] = { "layer_out" };
    for (const char * prefix : prefixes) {
        const size_t length = strlen(prefix);
        if (strncmp(tensor->name, prefix, length) != 0 || tensor->name[length] != '-') {
            continue;
        }
        const char * suffix = tensor->name + length + 1;
        if (*suffix < '0' || *suffix > '9') {
            continue;
        }
        while (*suffix >= '0' && *suffix <= '9') {
            ++suffix;
        }
        if (*suffix == '\0') {
            return true;
        }
    }
    return false;
}

static void ggml_spacemit_trace_op_output(FILE * file, uint64_t sequence, uint64_t graph_uid,
                                         size_t node_index, int opcode, int graph_nodes, const ggml_tensor * tensor) {
    if (file == nullptr || !ggml_spacemit_is_trace_checkpoint(tensor, graph_nodes) || tensor->data == nullptr ||
        tensor->type != GGML_TYPE_F32) {
        return;
    }

    const size_t elements = static_cast<size_t>(ggml_nelements(tensor));
    if (elements < 1024) {
        return;
    }

    const size_t samples = std::min<size_t>(elements, 256);
    const auto * data = static_cast<const uint8_t *>(tensor->data);
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < samples; ++i) {
        uint32_t bits = 0;
        const size_t element = (i * elements) / samples;
        memcpy(&bits, data + element * sizeof(bits), sizeof(bits));
        for (unsigned int byte = 0; byte < sizeof(bits); ++byte) {
            hash ^= (bits >> (byte * 8)) & 0xff;
            hash *= UINT64_C(1099511628211);
        }
    }

    std::fprintf(file, "T,%llu,%llu,%zu,%d,%s,%zu,%lld,%lld,%lld,%lld,%zu,%016llx\n",
                 static_cast<unsigned long long>(sequence), static_cast<unsigned long long>(graph_uid),
                 node_index, opcode, tensor->name, samples,
                 static_cast<long long>(tensor->ne[0]), static_cast<long long>(tensor->ne[1]),
                 static_cast<long long>(tensor->ne[2]), static_cast<long long>(tensor->ne[3]),
                 ggml_nbytes(tensor), static_cast<unsigned long long>(hash));
}

static std::mutex g_spacemit_execution_mutex;

void ggml_backend_spacemit_acquire(void) {
    g_spacemit_execution_mutex.lock();
}

void ggml_backend_spacemit_release(void) {
    g_spacemit_execution_mutex.unlock();
}

static ggml_status ggml_backend_spacemit_graph_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    std::lock_guard<std::mutex> execution_lock(g_spacemit_execution_mutex);
    auto sess = static_cast<spacemit_session *>(backend->context);
    // Serialize hardware use across backend handles. Streams are graph-scoped
    // by default; the opt-in persistent mode retains its CC grant per session.
    std::lock_guard<std::mutex> stream_lock(sess->stream_mutex);

    GGML_LOG_DEBUG("ggml-spacemit: %s graph-compute n_nodes %d\n", sess->c_name(), graph->n_nodes);

    // Build (or reuse) the opnode list for this graph. The graph uid identifies a
    // stable shape-class, so the list is only rebuilt when the topology changes.
    // try_fuse_node folds matching patterns into fused opnodes and consumes their
    // nodes; non-matching nodes are appended as single ops.
    if (!(graph->uid != 0 && sess->cached_graph.uid == graph->uid)) {
        sess->cached_graph.uid = graph->uid;
        sess->cached_graph.nodes.clear();
        sess->cached_graph.nodes.reserve(graph->n_nodes);
        std::unordered_set<ggml_tensor *> emitted;  // nodes already emitted by try_fuse_node
        for (int i = 0; i < graph->n_nodes; ++i) {
            ggml_tensor * n = graph->nodes[i];
            if (!op_is_compute(n)) {
                continue;
            }
            // On a match, try_fuse_node pushes the fused opnodes and advances i past
            // the nodes it consumed; otherwise the node is appended as a single op.
            if (try_fuse_node(graph, i, sess->cached_graph.nodes, emitted)) {
                continue;
            }
            sess->cached_graph.nodes.emplace_back(n);
        }

        const int n_gdn_inplace = gdn_inplace_enabled() ? gdn_inplace_mark(graph, sess->cached_graph.nodes) : 0;

        int n_fused = 0;
        int n_add_gelu = 0;
        int n_matmul_add = 0;
        int n_im2col_pack = 0;
        int n_norm_affine = 0;
        int n_residual_norm_affine = 0;
        int n_conv_seen = 0;
        int n_conv_fused = 0;
        int n_conv_scalar = 0;
        int n_gdn_seen = 0;
        for (const auto & node : sess->cached_graph.nodes) {
            n_add_gelu += node.opcode == SPACEMIT_OP_ADD_GELU;
            n_conv_seen += node.node->op == GGML_OP_CONCAT;
            n_conv_fused += node.opcode == SPACEMIT_OP_CONCAT_STATE;
            n_conv_scalar += node.opcode == SPACEMIT_OP_CONCAT_STATE && node.kernel_params[1] != 0;
            n_gdn_seen += node.node->op == GGML_OP_GATED_DELTA_NET;
            n_matmul_add += node.opcode == SPACEMIT_OP_MUL_MAT_ADD;
            n_im2col_pack += node.opcode == SPACEMIT_OP_IM2COL_MUL_MAT;
            n_norm_affine += node.opcode == SPACEMIT_OP_NORM_AFFINE;
            n_residual_norm_affine += node.opcode == SPACEMIT_OP_RESIDUAL_NORM_AFFINE;
            if (!node.fused.empty() || node.opcode != SPACEMIT_OP_INVALID) {
                n_fused++;
            }
        }
        GGML_LOG_DEBUG("ggml-spacemit: %s graph %llu: %zu opnodes, %d fused\n", sess->c_name(),
                       (unsigned long long) graph->uid, sess->cached_graph.nodes.size(), n_fused);
        const char * recurrent_trace = getenv("GGML_SPACEMIT_RECURRENT_TRACE");
        if (recurrent_trace && strcmp(recurrent_trace, "1") == 0 && n_gdn_seen > 0) {
            GGML_LOG_INFO("ggml-spacemit: recurrent conv fused=%d/%d scalar=%d gdn in-place=%d/%d graph=%llu\n",
                          n_conv_fused, n_conv_seen, n_conv_scalar, n_gdn_inplace, n_gdn_seen,
                          (unsigned long long) graph->uid);
        }
        const char * trace = getenv("GGML_SPACEMIT_ADD_GELU_TRACE");
        if (n_add_gelu > 0 && trace && strcmp(trace, "1") == 0) {
            GGML_LOG_INFO("ggml-spacemit: ADD+GELU fused=%d graph=%llu\n", n_add_gelu,
                          (unsigned long long) graph->uid);
        }
        const char * matmul_add_trace = getenv("GGML_SPACEMIT_MUL_MAT_ADD_TRACE");
        if (n_matmul_add > 0 && matmul_add_trace && strcmp(matmul_add_trace, "1") == 0) {
            GGML_LOG_INFO("ggml-spacemit: Q4_0 MUL_MAT+BIAS fused=%d graph=%llu\n", n_matmul_add,
                          (unsigned long long) graph->uid);
        }
        const char * pack_trace=getenv("GGML_SPACEMIT_IM2COL_PACK_TRACE");
        const char * norm_trace = getenv("GGML_SPACEMIT_NORM_AFFINE_TRACE");
        if (norm_trace && strcmp(norm_trace, "1") == 0)
            GGML_LOG_INFO("ggml-spacemit: NORM_AFFINE fused=%d residual=%d graph=%llu\n",
                          n_norm_affine, n_residual_norm_affine, (unsigned long long) graph->uid);
        if(n_im2col_pack>0&&pack_trace&&strcmp(pack_trace,"1")==0)
            GGML_LOG_INFO("ggml-spacemit: IM2COL+PACK+MUL_MAT fused=%d graph=%llu\n",n_im2col_pack,
                          (unsigned long long)graph->uid);
    }
    const auto & opnodes = sess->cached_graph.nodes;

    int n_threads = sess->num_cores > 0 ? sess->num_cores : 1;
    size_t workspace_size = 0;
    for (const auto & node : opnodes) {
        size_t node_workspace_size = 0;
        if (ggml_spacemit_get_work_size(n_threads, node.node, &node_workspace_size)) {
            if(node.packed_im2col)
                node_workspace_size=GGML_PAD(node_workspace_size,64)+ocr_im2col_pack::scratch_bytes(node.node,n_threads);
            workspace_size = std::max(workspace_size, node_workspace_size);
        }
    }

    if (sess->workspace_size < workspace_size) {
        spacemit_host_trace("workspace-alloc-begin", workspace_size);
        void * new_workspace = realloc(sess->workspace, workspace_size);
        if (new_workspace == nullptr) {
            GGML_LOG_ERROR("ggml-spacemit: failed to allocate work buffer (%zu bytes)\n", workspace_size);
            return GGML_STATUS_ALLOC_FAILED;
        }
        sess->workspace      = new_workspace;
        sess->workspace_size = workspace_size;
        spacemit_host_prefault(new_workspace, workspace_size);
        spacemit_host_trace("workspace-alloc-end", workspace_size);
    }
    auto * workspace = static_cast<uint8_t *>(sess->workspace);

    spert::StreamConfig stream_cfg;
    stream_cfg.n_cores = (uint32_t) sess->num_cores;
    {
        const auto & pool = global_spine_env_info.perfer_core_ids;
        stream_cfg.core_ids.assign(pool.begin(),
                                   pool.begin() + std::min<size_t>(pool.size(), stream_cfg.n_cores));
    }
    // CPU fallback also requests AI cores. Release this graph's grant when its
    // stream leaves scope, before the scheduler launches another backend.
    spacemit_host_trace("stream-open-begin");
    spert::Stream stream(stream_cfg);
    spacemit_host_trace("stream-open-end");
    if (!stream.valid()) {
        GGML_LOG_ERROR("ggml-spacemit: failed to create spert stream\n");
        return GGML_STATUS_FAILED;
    }

    FILE * op_hash_trace = nullptr;
    uint64_t op_hash_trace_sequence = 0;
    const char * op_hash_trace_path = getenv("GGML_SPACEMIT_OP_HASH_TRACE_PATH");
    if (op_hash_trace_path != nullptr && *op_hash_trace_path != '\0') {
        op_hash_trace = std::fopen(op_hash_trace_path, "ab");
        if (op_hash_trace == nullptr) {
            GGML_LOG_ERROR("ggml-spacemit: failed to open op hash trace '%s'\n", op_hash_trace_path);
            return GGML_STATUS_FAILED;
        }
        std::setvbuf(op_hash_trace, nullptr, _IOFBF, 64 * 1024);
        op_hash_trace_sequence = ++sess->op_hash_trace_sequence;
        std::fprintf(op_hash_trace, "G,%llu,%llu,%d\n",
                     static_cast<unsigned long long>(op_hash_trace_sequence),
                     static_cast<unsigned long long>(graph->uid), graph->n_nodes);
    }

    const bool op_timing=ocr_op_timing::path()!=nullptr;
    const uint64_t op_timing_sequence=op_timing?ocr_op_timing::storage::get().next.fetch_add(1):0;
    // Phase storage registers atexit and allocates several MiB. Initialize and
    // prefault it on the host, before any worker can enter record::begin().
    if(ocr_fa_phases_v50::enabled())(void)ocr_fa_phases_v50::storage::get();
    ocr_op_progress::storage * op_progress = nullptr;
    ggml::spacemit::scratch_mode scratch_mode;
    const uint64_t progress_sequence = ocr_op_progress::next_sequence();
    const size_t shared_mem_size = spert::backend_info().shared_mem_size;
    try {
        scratch_mode = ggml::spacemit::configured_scratch_mode();
        if (scratch_mode == ggml::spacemit::scratch_mode::locked_ddr && shared_mem_size)
            sess->ddr_scratch.prepare(n_threads, shared_mem_size);
        op_progress = ocr_op_progress::get();
        ocr_op_progress::catalog(op_progress, progress_sequence, graph->uid, opnodes);
    } catch (const std::exception & error) {
        if (op_hash_trace) std::fclose(op_hash_trace);
        GGML_LOG_ERROR("ggml-spacemit: host scratch/progress setup failed: %s\n", error.what());
        return GGML_STATUS_FAILED;
    }
    auto * contexts = sess->contexts.data();
    // Keep first-use option parsing and diagnostic allocation off AI workers.
    const char * pc8_host_options = std::getenv("GGML_SPACEMIT_PC8_HOST_OPTIONS");
    if (!pc8_host_options || std::strcmp(pc8_host_options, "0") != 0)
        ocr_q4_pc8::prepare_options_on_host();
    spacemit_kernels::rvv::prepare_ocr_worker_state();
    (void)ocr_fa_schedule::configured_rows(ggml_fa_tile_config::Q);
    (void)ocr_fa_schedule::separate_barrier_enabled();
    ocr_quant_phases::prepare_on_host();
    spacemit_host_trace("graph-launch-begin");
    auto fut = stream.launch(
        spert::Grid{(uint32_t)n_threads},
        [&opnodes, workspace_size = sess->workspace_size, workspace, contexts,
         shared_mem_size, scratch_mode, ddr_scratch = &sess->ddr_scratch, graph_nodes = graph->n_nodes,
         op_hash_trace, op_hash_trace_sequence, graph_uid = graph->uid,
         attention_work_queue = &sess->attention_work_queue,op_timing,op_timing_sequence,
         op_progress, progress_sequence](spert::Context * runtime) {
            const uint32_t ith = runtime->program_id(0);
            if (getenv("GGML_SPACEMIT_STRICT_AI_CORES") && (sched_getcpu() < 8 || sched_getcpu() > 15))
                throw std::runtime_error("inference worker is outside AI cores 8-15");
            ocr_op_progress::record(op_progress,ith,progress_sequence,UINT32_MAX,10,nullptr);
            const bool use_ddr = scratch_mode == ggml::spacemit::scratch_mode::locked_ddr;
            ggml::spacemit::tcm_lease lease(*runtime, shared_mem_size,
                scratch_mode == ggml::spacemit::scratch_mode::pinned_tcm);
            ocr_op_progress::record(op_progress,ith,progress_sequence,UINT32_MAX,11,nullptr);
            auto shared = use_ddr ? ddr_scratch->lane(ith) :
                (shared_mem_size > 0 ? runtime->shared_buffer() : spert::SharedBufferView{});
            if (shared_mem_size > 0 && (!shared || shared.size < shared_mem_size)) {
                throw std::runtime_error("ggml-spacemit: worker shared buffer unavailable");
            }

            auto & ctx = contexts[ith];
            ctx.reset(*runtime, ith, runtime->grid_dim(0), workspace, workspace_size, shared,
                      attention_work_queue, !use_ddr && shared_mem_size > 0 &&
                          scratch_mode == ggml::spacemit::scratch_mode::pinned_tcm);

            ctx.diagnostic_subphase = op_progress ? &op_progress->lanes[ith].subphase : nullptr;

            size_t node_index = 0;
            for (const auto & node : opnodes) {
                ggml_tensor * op = node.node;
                ocr_op_progress::record(op_progress,ith,progress_sequence,unsigned(node_index),1,op);
                const uint64_t op_begin=op_timing?ocr_op_timing::now():0;

                ggml_profile_log_spacemit_op_begin(op, (int) ith, (int) runtime->grid_dim(0));

                // Every non-PC8 node can change workspace or activations.
                // Only directly adjacent PC8 matrix nodes may share a pack.
                if(!ggml_spacemit_can_fuse_pc8_bias(op))ctx.pc8_packed_source=nullptr;
                if (!ggml_spacemit_compute_opnode(ctx, node)) {
                    throw std::runtime_error(std::string("ggml-spacemit: failed to dispatch op ") +
                                             ggml_op_desc(op) + " (" + ggml_type_name(op->type) + ")");
                }
                ggml_profile_log_spacemit_op_end(op, (int) ith, (int) runtime->grid_dim(0));
                const uint64_t op_end=op_timing?ocr_op_timing::now():0;
                ocr_op_progress::record(op_progress,ith,progress_sequence,unsigned(node_index),2,op);

                ++node_index;

                // Every core must finish the current node before any core starts
                // the next one. A bounded pointer look-ahead is not sufficient:
                // dependencies can be farther away or hidden behind aliases/views.
                ctx.sync();
                ocr_op_progress::record(op_progress,ith,progress_sequence,unsigned(node_index-1),3,op);
                if(op_timing)ocr_op_timing::record(ith,op_timing_sequence,unsigned(node_index-1),op,
                    int(node.opcode),op_begin,op_end,ocr_op_timing::now());
                ocr_op_progress::record(op_progress,ith,progress_sequence,unsigned(node_index-1),4,op);

                if (op_hash_trace != nullptr &&
                    (node.extra_dsts.empty() ? ggml_spacemit_is_trace_checkpoint(node.dst(), graph_nodes)
                                             : ggml_spacemit_is_trace_checkpoint(node.node, graph_nodes))) {
                    if (ith == 0) {
                        if (node.extra_dsts.empty()) {
                            ggml_spacemit_trace_op_output(op_hash_trace, op_hash_trace_sequence, graph_uid,
                                                          node_index - 1, static_cast<int>(node.opcode), graph_nodes, node.dst());
                        } else {
                            ggml_spacemit_trace_op_output(op_hash_trace, op_hash_trace_sequence, graph_uid,
                                                          node_index - 1, static_cast<int>(node.opcode), graph_nodes, node.node);
                            for (const ggml_tensor * extra_dst : node.extra_dsts) {
                                ggml_spacemit_trace_op_output(op_hash_trace, op_hash_trace_sequence, graph_uid,
                                                              node_index - 1, static_cast<int>(node.opcode), graph_nodes, extra_dst);
                            }
                        }
                    }
                    ctx.sync();
                }
            }

            ctx.clear();
#ifdef GGML_BUILD_PROFILE
            ggml_profile_flush_tls();
#endif
        }
    );

    spacemit_host_trace("graph-launch-end");
    const spert::Status status = fut.sync();
    spacemit_host_trace("graph-sync-end");
    spacemit_kernels::rvv::drain_ocr_worker_events();

    if (op_hash_trace != nullptr) {
        std::fprintf(op_hash_trace, "E,%llu,%d\n",
                     static_cast<unsigned long long>(op_hash_trace_sequence), static_cast<int>(status));
        std::fflush(op_hash_trace);
        std::fclose(op_hash_trace);
    }

    if (status != spert::Status::Ok) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_spacemit_synchronize(ggml_backend_t backend) {
    // graph_compute waits on its launch future before returning, so there is
    // no outstanding asynchronous work to synchronize here.
    GGML_UNUSED(backend);
}

static void ggml_backend_spacemit_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * graph,
                                                 struct ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(backend);
    if (params == nullptr || params->add_alloc_dep == nullptr || conv_fuse_mode() == 0) {
        return;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * n = graph->nodes[i];
        if (n->op != GGML_OP_CONCAT) {
            continue;
        }
        ggml_tensor * ssm     = nullptr;
        int           ssm_idx = -1;
        int64_t       col0    = 0;
        if (ssm_conv_concat_fusable(graph, i, &ssm, &ssm_idx, &col0) && ssm->src[2] == nullptr) {
            params->add_alloc_dep(params->user_data, n->src[0], ssm);
            params->add_alloc_dep(params->user_data, n->src[1], ssm);
        }
    }
}

static struct ggml_backend_i spacemit_backend_i = {
    /* .get_name                = */ ggml_backend_spacemit_name,
    /* .free                    = */ ggml_backend_spacemit_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_spacemit_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_spacemit_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ ggml_backend_spacemit_graph_optimize,
};

static ggml_guid_t ggml_backend_spacemit_guid() {
    static ggml_guid guid = { 0x8b, 0x68, 0xed, 0xbf, 0xef, 0x23, 0x2e, 0x4a,
                              0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22 };
    return &guid;
}

bool ggml_backend_is_spacemit(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_spacemit_name;
}

//** device interface

static ggml_backend_t ggml_backend_spacemit_device_init(ggml_backend_dev_t dev, const char * params) {
    auto sess = static_cast<spacemit_session *>(dev->context);

    return new ggml_backend{
        /* .guid      = */ ggml_backend_spacemit_guid(),
        /* .interface = */ spacemit_backend_i,
        /* .device    = */ dev,
        /* .context   = */ sess,
    };

    GGML_UNUSED(params);
}

static const char * ggml_backend_spacemit_device_get_name(ggml_backend_dev_t dev) {
    auto sess = static_cast<spacemit_session *>(dev->context);
    return sess->c_name();
}

static const char * ggml_backend_spacemit_device_get_description(ggml_backend_dev_t dev) {
    return "Spacemit K3/X200 AI Engine";
    GGML_UNUSED(dev);
}

static void ggml_backend_spacemit_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // SPACEMIT backend shares host memory (weights are in DRAM accessible by CC cores).
    // Report available system memory so the scheduler assigns tensors to us.
    size_t vfree = 0, vtotal = 0;
    ggml_backend_dev_t cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    if (cpu_dev && cpu_dev->iface.get_memory) {
        cpu_dev->iface.get_memory(cpu_dev, &vfree, &vtotal);
    }
    *free  = vfree;
    *total = vtotal;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_spacemit_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

static void ggml_backend_spacemit_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_spacemit_device_get_name(dev);
    props->description = ggml_backend_spacemit_device_get_description(dev);
    props->type        = ggml_backend_spacemit_device_get_type(dev);
    ggml_backend_spacemit_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        // the model is mmapped in host memory, then copied into spine pool buffers
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_buffer_type_t ggml_backend_spacemit_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_spacemit_buffer_type(dev);
}

static bool ggml_backend_spacemit_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // all srcs and dst must be on our buffer type
    auto check_buf = [&](const ggml_tensor * t) -> bool {
        if (!t || !t->buffer) return true;  // unallocated is OK
        return ggml_backend_buffer_is_spacemit(t->buffer);
    };

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && !check_buf(op->src[i])) {
            return false;
        }
    }
    if (!check_buf(op)) {
        return false;
    }

    if (ocr_ai_complete::supported(op)) return true;
    bool supp = false;
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            supp = true;
            break;

        case GGML_OP_SCALE:
        case GGML_OP_IM2COL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ROPE:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
        case GGML_OP_GET_ROWS:
        case GGML_OP_CONCAT:
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
        case GGML_OP_CONT:
        case GGML_OP_REPEAT:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_L2_NORM:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_PAD:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SET:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_SSM_CONV:
        case GGML_OP_ARGSORT:
        case GGML_OP_CLAMP:
            supp = ggml_spacemit_get_tensor_traits(op) != nullptr;
            break;

        default:
            break;
    }

    return supp;

    GGML_UNUSED(dev);
}

static bool ggml_backend_spacemit_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_spacemit_buffer_type_get_name;
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_spacemit_device_i = {
    /* .get_name             = */ ggml_backend_spacemit_device_get_name,
    /* .get_description      = */ ggml_backend_spacemit_device_get_description,
    /* .get_memory           = */ ggml_backend_spacemit_device_get_memory,
    /* .get_type             = */ ggml_backend_spacemit_device_get_type,
    /* .get_props            = */ ggml_backend_spacemit_device_get_props,
    /* .init_backend         = */ ggml_backend_spacemit_device_init,
    /* .get_buffer_type      = */ ggml_backend_spacemit_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_spacemit_device_supports_op,
    /* .supports_buft        = */ ggml_backend_spacemit_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

//** backend registry

#define GGML_SPACEMIT_MAX_DEVICES 1

struct ggml_spacemit_registry {
    ggml_spacemit_registry(ggml_backend_reg_t reg);
    ~ggml_spacemit_registry();

    ggml_backend_device devices[GGML_SPACEMIT_MAX_DEVICES];
};

ggml_spacemit_registry::ggml_spacemit_registry(ggml_backend_reg_t reg) {
    GGML_LOG_INFO("ggml-spacemit: Spacemit backend (experimental) : allocating new registry\n");

    for (size_t i = 0; i < GGML_SPACEMIT_MAX_DEVICES; i++) {
        devices[i].iface = ggml_backend_spacemit_device_i;
        devices[i].reg   = reg;

        auto * sess = new spacemit_session();

        // populate session from spert backend_info and env
        spert::BackendInfo info = spert::backend_info();
        sess->num_cores = global_spine_env_info.num_cores;
        sess->arch_id   = static_cast<int64_t>(global_spine_env_info.perfer_core_arch_id);
        sess->vlen      = (int64_t) info.vlen;
        sess->use_ime1  = global_spine_env_info.use_ime1;
        sess->use_ime2  = global_spine_env_info.use_ime2;
        sess->contexts.resize(sess->num_cores);

        sess->name = "SPACEMIT" + std::to_string(i);

        devices[i].context = sess;
    }
}

ggml_spacemit_registry::~ggml_spacemit_registry() {
    for (size_t i = 0; i < GGML_SPACEMIT_MAX_DEVICES; i++) {
        auto sess = static_cast<spacemit_session *>(devices[i].context);
        delete sess;
    }
}

static const char * ggml_backend_spacemit_reg_get_name(ggml_backend_reg_t reg) {
    return "SPACEMIT";
    GGML_UNUSED(reg);
}

static bool ggml_spacemit_disabled(void) {
    const char * str = getenv("GGML_SPACEMIT_BACKEND");
    return str == nullptr || strcmp(str, "1") != 0;
}

static size_t ggml_backend_spacemit_reg_get_device_count(ggml_backend_reg_t reg) {
    if (ggml_spacemit_disabled()) {
        return 0;
    }
    return GGML_SPACEMIT_MAX_DEVICES;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_spacemit_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto hreg = static_cast<ggml_spacemit_registry *>(reg->context);

    if (hreg == nullptr) {
        return nullptr;
    }

    if (index >= GGML_SPACEMIT_MAX_DEVICES || !hreg->devices[index].context) {
        return nullptr;
    }

    return &hreg->devices[index];
}

static void * ggml_backend_spacemit_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return NULL;
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static void ggml_spacemit_init(ggml_backend_reg * reg) {
    if (ggml_spacemit_disabled()) {
        return;
    }

    reg->context = new ggml_spacemit_registry(reg);
}

static const struct ggml_backend_reg_i ggml_backend_spacemit_reg_i = {
    /* .get_name         = */ ggml_backend_spacemit_reg_get_name,
    /* .get_device_count = */ ggml_backend_spacemit_reg_get_device_count,
    /* .get_device       = */ ggml_backend_spacemit_reg_get_device,
    /* .get_proc_address = */ ggml_backend_spacemit_get_proc_address,
};

ggml_backend_reg_t ggml_backend_spacemit_reg(void) {
    static bool initialized = false;

    static ggml_backend_reg reg = { /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_spacemit_reg_i,
                                    /* .context     = */ NULL };

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_spacemit_init(&reg);
        }
        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_spacemit_init(void) {
    ggml_backend_reg_t reg = ggml_backend_spacemit_reg();
    if (!reg) {
        return nullptr;
    }

    if (ggml_spacemit_disabled()) {
        return nullptr;
    }

    ggml_backend_dev_t dev = ggml_backend_spacemit_reg_get_device(reg, 0);
    if (!dev) {
        return nullptr;
    }

    return ggml_backend_spacemit_device_init(dev, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_spacemit_reg)
