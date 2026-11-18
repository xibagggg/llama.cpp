#ifndef OCR_D2T_SCATTER_H
#define OCR_D2T_SCATTER_H

#include "ocr-d2t-scatter-core.h"
#include "ocr-dspark-custom-contract.h"
#include "ggml-backend.h"
#include "spacemit-opnode.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace ocr_d2t_scatter {

struct plan {
    const ggml_tensor * op = nullptr;
    const uint32_t * offsets = nullptr;
    size_t rows = 0;
    size_t full_rows = 0;
    size_t columns = 0;
    unsigned mode = 0;
    mutable size_t worker_hits = 0; // Only lane zero writes; host reads after fut.sync().
};

struct mapping {
    const ggml_tensor * ids = nullptr;
    std::vector<uint32_t> offsets;
    validation result = validation::invalid;
};

struct launch_catalog {
    std::vector<mapping> mappings;
    std::vector<plan> plans;
    std::vector<uint8_t> seen;
    bool trace_worker_hits = false;

    static bool disjoint(const void * a, size_t an, const void * b, size_t bn) {
        const uintptr_t av = reinterpret_cast<uintptr_t>(a);
        const uintptr_t bv = reinterpret_cast<uintptr_t>(b);
        if (an > UINTPTR_MAX - av || bn > UINTPTR_MAX - bv) return false;
        return av + an <= bv || bv + bn <= av;
    }

    // A graph reshape/view has buffer==nullptr and is not GGML_OP_NONE.
    // Prove a bounded contiguous I64 alias chain ending in immutable WEIGHTS.
    // view_offs is relative to view_src (normally already flattened by ggml),
    // not relative to src[0]; never accumulate it through the operation graph.
    // The scheduler duplicates a cross-backend input with NONE + COMPUTE,
    // dropping its WEIGHTS flag and original view/source provenance. Admit it
    // only after proving that no allocated compute output in this complete
    // backend graph can write any byte of the entire I64 root. Scanning the
    // original graph also covers outputs hidden by opnode fusion/elision.
    static bool graph_preserves_ids(const ggml_tensor * root, const ggml_cgraph * graph,
                                    const char ** failure = nullptr) {
        const auto reject = [&](const char * reason) { if (failure) *failure = reason; return false; };
        if (!graph || graph->n_nodes <= 0 || !graph->nodes) return reject("root_usage_no_graph");
        const size_t bytes = ggml_nbytes(root);
        if (!bytes || bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(root->data))
            return reject("root_range");
        for (int index = 0; index < graph->n_nodes; ++index) {
            const auto * output = graph->nodes[index];
            if (!output) return reject("null_graph_node");
            switch (output->op) {
                // Only this registered callback has an audited writes-only-dst
                // contract. Its exact identity and full current shape are checked
                // on the host; names never participate in admission.
                case GGML_OP_CUSTOM:
                    if (!ocr_dspark_custom_contract::writes_only_output(output, root))
                        return reject("graph_opaque_callback");
                    break;
                case GGML_OP_MAP_CUSTOM1: case GGML_OP_MAP_CUSTOM2: case GGML_OP_MAP_CUSTOM3:
                    return reject("graph_opaque_callback");
                // These operations only describe aliases; no output is written.
                case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE:
                case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
                    continue;
                default: break;
            }
            // Elided fusion intermediates may have no allocated output. Every
            // allocated output is checked, even when fusion does not write it.
            if (output->data && !disjoint(root->data, bytes, output->data, ggml_nbytes(output)))
                return reject("graph_write_alias");
        }
        return true;
    }

    static bool immutable_ids(const ggml_tensor * ids, const ggml_tensor ** root = nullptr,
                              const char ** failure = nullptr, const ggml_cgraph * graph = nullptr,
                              bool * graph_readonly_copy = nullptr) {
        const auto reject = [&](const char * reason) { if (failure) *failure = reason; return false; };
        if (failure) *failure = "none";
        if (graph_readonly_copy) *graph_readonly_copy = false;
        const ggml_tensor * visited[64]{};
        const ggml_tensor * current = ids;
        for (size_t depth = 0; depth < 64; ++depth) {
            if (!current) return reject("null_tensor");
            if (!current->data) return reject("null_data");
            if (reinterpret_cast<uintptr_t>(current->data) % alignof(int64_t)) return reject("alignment");
            if (current->type != GGML_TYPE_I64) return reject("type");
            if (!ggml_is_contiguous(current)) return reject("contiguity");
            for (size_t i = 0; i < depth; ++i)
                if (visited[i] == current) return reject("cycle");
            visited[depth] = current;
            if (!current->view_src) {
                if (current->op != GGML_OP_NONE) return reject("root_op");
                if (!current->buffer) return reject("root_buffer");
                const auto usage = ggml_backend_buffer_get_usage(current->buffer);
                if (usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    if (usage != GGML_BACKEND_BUFFER_USAGE_ANY && usage != GGML_BACKEND_BUFFER_USAGE_COMPUTE)
                        return reject("root_usage_unknown");
                    if (!graph_preserves_ids(current, graph, failure)) return false;
                    if (graph_readonly_copy) *graph_readonly_copy = true;
                }
                if (root) *root = current;
                return true;
            }
            if (current->op != GGML_OP_VIEW && current->op != GGML_OP_RESHAPE) return reject("view_op");
            const auto * parent = current->view_src;
            if (!parent->data) return reject("parent_null_data");
            if (parent->type != GGML_TYPE_I64) return reject("parent_type");
            if (!ggml_is_contiguous(parent)) return reject("parent_contiguity");
            const size_t bytes = ggml_nbytes(current), parent_bytes = ggml_nbytes(parent);
            if (bytes > parent_bytes) return reject("parent_size");
            if (current->view_offs > parent_bytes - bytes) return reject("view_range");
            const uintptr_t base = reinterpret_cast<uintptr_t>(parent->data);
            if (current->view_offs > UINTPTR_MAX - base) return reject("pointer_overflow");
            if (reinterpret_cast<uintptr_t>(current->data) != base + current->view_offs)
                return reject("data_offset");
            current = parent;
        }
        return reject("depth_limit");
    }

    // Host-only, opt-in, bounded metadata trace; never reads ID contents or
    // changes admission. The reason comes from the actual immutable predicate.
    static void diagnose_immutable_ids(const ggml_tensor * op, const ggml_cgraph * graph) {
        const char * option = std::getenv("GGML_SPACEMIT_OCR_D2T_SCATTER_VERBOSE");
        if (!option || std::strcmp(option, "1") != 0) return;
        static std::atomic<unsigned> rejected_chains{0};
        const unsigned sequence = rejected_chains.fetch_add(1, std::memory_order_relaxed);
        if (sequence >= 4) return;
        const auto * ids = op->src[1];
        const char * reason = "unset";
        (void)immutable_ids(ids, nullptr, &reason, graph);
        std::fprintf(stderr, "OCR_D2T_IDS_REJECT sequence=%u reason=%s op=%p name=%.63s ids=%p\n",
                     sequence, reason, static_cast<const void *>(op), op->name,
                     static_cast<const void *>(ids));
        if (graph && graph->nodes && std::strcmp(reason, "graph_write_alias") == 0) {
            const ggml_tensor * root = ids;
            for (unsigned depth = 0; root && root->view_src && depth < 64; ++depth) root = root->view_src;
            int target_index = -1;
            for (int index = 0; index < graph->n_nodes; ++index)
                if (graph->nodes[index] == op) target_index = index;
            unsigned offenders = 0;
            for (int index = 0; root && index < graph->n_nodes && offenders < 3; ++index) {
                const auto * output = graph->nodes[index];
                if (!output || !output->data) continue;
                switch (output->op) {
                    case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE:
                    case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: continue;
                    default: break;
                }
                if (disjoint(root->data, ggml_nbytes(root), output->data, ggml_nbytes(output))) continue;
                ++offenders;
                std::fprintf(stderr, "OCR_D2T_ALIAS sequence=%u target_index=%d output_index=%d "
                    "output=%p name=%.63s op=%d data=%p bytes=%zu ne=[%lld,%lld,%lld,%lld]\n",
                    sequence, target_index, index, static_cast<const void *>(output), output->name,
                    int(output->op), output->data, ggml_nbytes(output),
                    static_cast<long long>(output->ne[0]), static_cast<long long>(output->ne[1]),
                    static_cast<long long>(output->ne[2]), static_cast<long long>(output->ne[3]));
            }
        }
        const ggml_tensor * visited[64]{};
        const ggml_tensor * current = ids;
        for (size_t depth = 0; current && depth < 64; ++depth) {
            bool cycle = false;
            for (size_t i = 0; i < depth; ++i) cycle |= visited[i] == current;
            if (cycle) {
                std::fprintf(stderr, "OCR_D2T_IDS_LEVEL sequence=%u depth=%zu cycle=%p\n",
                             sequence, depth, static_cast<const void *>(current));
                break;
            }
            visited[depth] = current;
            std::fprintf(stderr, "OCR_D2T_IDS_LEVEL sequence=%u depth=%zu tensor=%p name=%.63s "
                         "type=%d op=%d data=%p buffer=%p usage=%d view_src=%p view_offs=%zu "
                         "ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] contiguous=%d bytes=%zu\n",
                         sequence, depth, static_cast<const void *>(current), current->name,
                         int(current->type), int(current->op), current->data,
                         static_cast<const void *>(current->buffer),
                         current->buffer ? int(ggml_backend_buffer_get_usage(current->buffer)) : -1,
                         static_cast<const void *>(current->view_src), current->view_offs,
                         static_cast<long long>(current->ne[0]), static_cast<long long>(current->ne[1]),
                         static_cast<long long>(current->ne[2]), static_cast<long long>(current->ne[3]),
                         current->nb[0], current->nb[1], current->nb[2], current->nb[3],
                         int(ggml_is_contiguous(current)), ggml_nbytes(current));
            current = current->view_src;
        }
    }

    enum class rejection { none, other, shape, immutable_ids, alias };

    static bool eligible(const ggml_tensor * op, rejection * rejected = nullptr,
                         const ggml_cgraph * graph = nullptr, bool * graph_readonly_copy = nullptr) {
        const auto reject = [&](rejection why) { if (rejected) *rejected = why; return false; };
        if (rejected) *rejected = rejection::none;
        if (!op || op->op != GGML_OP_SET_ROWS || !op->src[0] || !op->src[1] || !op->src[2])
            return reject(rejection::other);
        const auto * values = op->src[0];
        const auto * ids = op->src[1];
        // Exact initial DSpark scope; generic SET_ROWS keeps all other cases.
        if (op->type != GGML_TYPE_F32 || values->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I64 ||
            op->ne[0] != 1 || op->ne[1] != 248100 || op->ne[3] != 1 ||
            (op->ne[2] != 1 && op->ne[2] != 15) || values->ne[0] != 1 || values->ne[1] != 114688 ||
            values->ne[2] != op->ne[2] || values->ne[3] != 1 ||
            ids->ne[0] != 114688 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            !ggml_is_contiguous(op) || !ggml_is_contiguous(values) || !ggml_is_contiguous(ids) ||
            !op->data || !values->data || !ids->data)
            return reject(rejection::shape);
        const ggml_tensor * ids_root = nullptr;
        if (!immutable_ids(ids, &ids_root, nullptr, graph, graph_readonly_copy)) return reject(rejection::immutable_ids);
        const size_t dst_bytes = size_t(op->ne[1]) * size_t(op->ne[2]) * sizeof(uint32_t);
        const size_t src_bytes = size_t(values->ne[1]) * size_t(values->ne[2]) * sizeof(uint32_t);
        const size_t ids_bytes = size_t(ids->ne[0]) * sizeof(int64_t);
        // A SET_ROWS result is the existing full tensor's view. Do not admit
        // aliasing that lets the scatter modify an input or the immutable IDs.
        if (op->src[2]->data != op->data ||
            !disjoint(values->data, src_bytes, op->data, dst_bytes) ||
            !disjoint(ids->data, ids_bytes, op->data, dst_bytes) ||
            !disjoint(ids_root->data, ggml_nbytes(ids_root), op->data, dst_bytes))
            return reject(rejection::alias);
        return true;
    }

    // Host-only, once per graph call, under the existing execution/stream locks.
    // Revalidating each launch avoids stale pointers/revisions across model loads.
    // The immutable WEIGHTS IDs must not be changed by another graph node.
    void prepare(const std::vector<spacemit_opnode> & nodes, unsigned nth, const ggml_cgraph * graph = nullptr) {
        plans.clear();
        mappings.clear();
        const char * trace_hits = std::getenv("GGML_SPACEMIT_OCR_D2T_SCATTER_TRACE");
        trace_worker_hits = trace_hits && std::strcmp(trace_hits, "1") == 0;
        const char * option = std::getenv("GGML_SPACEMIT_OCR_D2T_SCATTER");
        const unsigned mode = option && std::strcmp(option, "1") == 0 ? 1 :
                              option && std::strcmp(option, "2") == 0 ? 2 : 0;
        if (!mode || nth != 6) return;
        const auto begin = std::chrono::steady_clock::now();
        mappings.reserve(nodes.size());
        plans.reserve(nodes.size());
        size_t admitted = 0, duplicate = 0, invalid = 0, readonly_copies = 0;
        size_t scatter_ops = 0, rejected_shape = 0, rejected_ids = 0, rejected_alias = 0;
        for (const auto & node : nodes) {
            const auto * op = node.node;
            if (!op || op->op != GGML_OP_SET_ROWS) continue;
            ++scatter_ops;
            rejection why{};
            bool readonly_copy = false;
            if (!eligible(op, &why, graph, &readonly_copy)) {
                rejected_shape += why == rejection::shape || why == rejection::other;
                rejected_ids += why == rejection::immutable_ids;
                if (why == rejection::immutable_ids) diagnose_immutable_ids(op, graph);
                rejected_alias += why == rejection::alias;
                continue;
            }
            ++admitted;
            readonly_copies += readonly_copy;
            const auto * ids = op->src[1];
            mapping * selected = nullptr;
            for (auto & candidate : mappings)
                if (candidate.ids->data == ids->data && candidate.ids->ne[0] == ids->ne[0]) {
                    selected = &candidate; break;
                }
            if (!selected) {
                mappings.emplace_back();
                selected = &mappings.back();
                selected->ids = ids;
                selected->result = validate_i64(static_cast<const int64_t *>(ids->data),
                    size_t(ids->ne[0]), size_t(op->ne[1]), selected->offsets, seen);
                duplicate += selected->result == validation::duplicate;
                invalid += selected->result == validation::invalid;
            }
            if (selected->result != validation::valid) continue;
            plans.push_back({op, selected->offsets.data(), size_t(op->src[0]->ne[1]),
                             size_t(op->ne[1]), size_t(op->ne[2]), mode});
        }
        const char * trace = std::getenv("GGML_SPACEMIT_OCR_D2T_SCATTER_TRACE");
        if (trace && std::strcmp(trace, "1") == 0) {
            const double us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - begin).count();
            std::fprintf(stderr, "OCR_D2T_SCATTER mode=%u set_rows=%zu admitted=%zu selected=%zu mappings=%zu "
                         "reject_shape=%zu reject_ids=%zu reject_alias=%zu graph_readonly_ids=%zu "
                         "duplicate_maps=%zu invalid_maps=%zu prepare_us=%.3f\n",
                         mode, scatter_ops, admitted, plans.size(), mappings.size(),
                         rejected_shape, rejected_ids, rejected_alias, readonly_copies, duplicate, invalid, us);
        }
    }

    // Every node has the existing collective barrier. Lane zero is the only
    // counter writer, and report runs after the graph future has synchronized.
    void record_worker_hit(const plan * selected, unsigned ith) const {
        if (trace_worker_hits && ith == 0) ++selected->worker_hits;
    }

    void report_worker_hits() const {
        if (!trace_worker_hits) return;
        size_t calls = 0, columns = 0, hits1 = 0, hits15 = 0, missing = 0;
        for (const auto & selected : plans) {
            calls += selected.worker_hits;
            columns += selected.worker_hits * selected.columns;
            hits1 += selected.columns == 1 ? selected.worker_hits : 0;
            hits15 += selected.columns == 15 ? selected.worker_hits : 0;
            missing += selected.worker_hits == 0;
        }
        std::fprintf(stderr, "OCR_D2T_WORKER selected=%zu calls=%zu columns=%zu "
            "hits_m1=%zu hits_m15=%zu missing=%zu\n", plans.size(), calls,
            columns, hits1, hits15, missing);
    }

    const plan * find(const ggml_tensor * op) const {
        for (const auto & candidate : plans)
            if (candidate.op == op) return &candidate;
        return nullptr;
    }
};

} // namespace ocr_d2t_scatter
#endif
