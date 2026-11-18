#pragma once

// Opt-in graph-local destination forwarding for the *existing* K16 GDN
// recurrence. This does not fuse, approximate or reorder its arithmetic.
// The gather and both original graph nodes remain visible to the allocator.
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace ocr_gdn_snapshot_store {

constexpr int64_t dim = 128;
constexpr int64_t heads = 16;
constexpr int64_t slots = 16;
constexpr int64_t state_elems = dim * dim * heads;
constexpr size_t state_bytes = size_t(state_elems) * sizeof(float);
constexpr size_t attn_bytes = size_t(dim * heads * slots) * sizeof(float);

inline bool enabled_on_host() {
    const char * value = std::getenv("GGML_SPACEMIT_OCR_GDN_SNAPSHOT_STORE");
    const char * hashes = std::getenv("GGML_SPACEMIT_OP_HASH_TRACE_PATH");
    // The hash diagnostic samples the whole original tensor, including the
    // intentionally elided snapshot region. Keep that observer's contract.
    return value != nullptr && std::strcmp(value, "1") == 0 &&
           (hashes == nullptr || *hashes == '\0');
}

inline bool shape(const ggml_tensor * t, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    return t != nullptr && t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2 && t->ne[3] == n3;
}

// Positive, nonoverlapping F32 strides only. Check overflow before asking
// anything else to form a pointer or a byte interval.
inline bool span_bytes(const ggml_tensor * t, size_t & result) {
    if (t == nullptr || t->type != GGML_TYPE_F32 || t->data == nullptr ||
        reinterpret_cast<uintptr_t>(t->data) % alignof(float) != 0 || t->nb[0] != sizeof(float)) {
        return false;
    }
    size_t extent = sizeof(float);
    for (int d = 0; d < 4; ++d) {
        if (t->ne[d] < 1 || t->nb[d] % sizeof(float) != 0 ||
            (t->ne[d] > 1 && t->nb[d] < extent)) {
            return false;
        }
        const size_t count = size_t(t->ne[d] - 1);
        if (count != 0 && t->nb[d] > (std::numeric_limits<size_t>::max() - extent) / count) {
            return false;
        }
        extent += count * t->nb[d];
    }
    const uintptr_t address = reinterpret_cast<uintptr_t>(t->data);
    if (extent > std::numeric_limits<uintptr_t>::max() - address) {
        return false;
    }
    result = extent;
    return true;
}

inline bool overlaps(const void * a, size_t an, const void * b, size_t bn) {
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a);
    const uintptr_t bb = reinterpret_cast<uintptr_t>(b);
    return aa <= bb ? bb - aa < an : aa - bb < bn;
}

inline bool within(const void * child, size_t cn, const void * parent, size_t pn) {
    const uintptr_t c = reinterpret_cast<uintptr_t>(child);
    const uintptr_t p = reinterpret_cast<uintptr_t>(parent);
    return c >= p && c - p <= pn && cn <= pn - (c - p);
}

inline bool is_view_of(const ggml_tensor * t, const ggml_tensor * parent) {
    // GGML normally flattens views. Bound the walk so corrupt/cyclic metadata
    // can never trap a worker or a diagnostic fixture.
    for (unsigned depth = 0; t != nullptr && depth < 32; ++depth) {
        if (t == parent) {
            return true;
        }
        t = t->view_src;
    }
    return false;
}

inline bool attention_view(const ggml_tensor * t, const ggml_tensor * g) {
    size_t bytes = 0;
    return t != g && is_view_of(t, g) && span_bytes(t, bytes) &&
           within(t->data, bytes, g->data, attn_bytes);
}

// Also called on workers before both original op positions. It reads only
// immutable metadata / addresses, never input values, env, logs or allocations.
// A refusal takes the unchanged GDN + CPY path on *all* workers.
inline bool storage_eligible(const ggml_tensor * g, const ggml_tensor * store) {
    if (g == nullptr || store == nullptr || g->op != GGML_OP_GATED_DELTA_NET ||
        ggml_get_op_params_i32(g, 0) != slots || g->view_src != nullptr ||
        (g->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_is_contiguous(g) ||
        ggml_nelements(g) != int64_t(attn_bytes / sizeof(float)) + slots * state_elems ||
        store->op != GGML_OP_CPY || !shape(store, state_elems, 1, slots, 1) ||
        store->src[0] == nullptr || store->src[1] == nullptr) {
        return false;
    }
    size_t gn = 0, dn = 0;
    if (!span_bytes(g, gn) || !span_bytes(store, dn) || store->nb[2] < state_bytes ||
        overlaps(g->data, gn, store->data, dn)) {
        return false;
    }
    const ggml_tensor * from = store->src[0];
    const ggml_tensor * dest = store->src[1];
    size_t from_n = 0, dest_n = 0, cache_n = 0;
    if (from->view_src != g || from->view_offs != attn_bytes ||
        !shape(from, state_elems, 1, slots, 1) || !ggml_is_contiguous(from) ||
        !span_bytes(from, from_n) || from_n != slots * state_bytes ||
        reinterpret_cast<uintptr_t>(from->data) != reinterpret_cast<uintptr_t>(g->data) + attn_bytes ||
        !shape(dest, state_elems, 1, slots, 1) || !span_bytes(dest, dest_n) ||
        dest->data != store->data || dest_n != dn || dest->nb[2] != store->nb[2] ||
        dest->view_src == nullptr || !span_bytes(dest->view_src, cache_n) ||
        !within(dest->data, dest_n, dest->view_src->data, cache_n) ||
        dest->view_offs > cache_n || reinterpret_cast<uintptr_t>(dest->data) !=
            reinterpret_cast<uintptr_t>(dest->view_src->data) + dest->view_offs) {
        return false;
    }
    // Preserve the complete original state input and every model input. In
    // particular, direct writing slot0 must not destroy the gathered state.
    for (int j = 0; j < 6; ++j) {
        const ggml_tensor * input = g->src[j];
        size_t in_n = 0;
        if (!span_bytes(input, in_n) || overlaps(input->data, in_n, store->data, dn) ||
            overlaps(input->data, in_n, g->data, gn)) {
            return false;
        }
    }
    const ggml_tensor * q = g->src[0];
    const ggml_tensor * k = g->src[1];
    if (q->ne[1] < 1 || q->ne[1] > heads || heads % q->ne[1] != 0 ||
        k->ne[1] < 1 || k->ne[1] > heads || heads % k->ne[1] != 0 ||
        !shape(q, dim, q->ne[1], slots, 1) || !shape(k, dim, k->ne[1], slots, 1) ||
        !shape(g->src[2], dim, heads, slots, 1) ||
        !shape(g->src[3], 1, heads, slots, 1) ||
        !shape(g->src[4], 1, heads, slots, 1) ||
        !shape(g->src[5], dim, dim, heads, 1) || !ggml_is_contiguous(g->src[5])) {
        return false;
    }
    return true;
}

template<class Graph, class IsCompute>
inline ggml_tensor * match_store(const Graph * graph, int gi, IsCompute is_compute) {
    if (graph == nullptr || gi < 0 || gi >= graph->n_nodes) {
        return nullptr;
    }
    ggml_tensor * g = graph->nodes[gi];
    if (g == nullptr || g->op != GGML_OP_GATED_DELTA_NET ||
        ggml_get_op_params_i32(g, 0) != slots || (g->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return nullptr;
    }
    ggml_tensor * store = nullptr;
    int store_index = -1;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * t = graph->nodes[i];
        if (t == nullptr) {
            return nullptr;
        }
        if (is_view_of(t, g) && (t->flags & GGML_TENSOR_FLAG_OUTPUT) && !attention_view(t, g)) {
            return nullptr;
        }
        if (!is_compute(t)) {
            continue;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * x = t->src[j];
            if (x == nullptr || !is_view_of(x, g) || attention_view(x, g)) {
                continue;
            }
            // No consumer may observe the elided snapshots except this one
            // original CPY. Reading G as a whole is always refused.
            if (x == g || store != nullptr || t->op != GGML_OP_CPY || j != 0 ||
                x->view_src != g || x->view_offs != attn_bytes) {
                return nullptr;
            }
            store = t;
            store_index = i;
        }
    }
    if (store == nullptr || store_index <= gi || !storage_eligible(g, store)) {
        return nullptr;
    }
    // Forwarding writes the cache at G rather than C. Metadata-only nodes
    // between them cannot observe it; every intervening computation refuses.
    for (int i = gi + 1; i < store_index; ++i) {
        if (is_compute(graph->nodes[i])) {
            return nullptr;
        }
    }
    return store;
}

// Same reverse target_slot = n_tokens - 1 - t as the existing recurrence.
// Destination is proven row contiguous within one snapshot; only its outer
// snapshot stride differs from the materialized temporary.
inline float * snapshot_pointer(void * base, size_t stride_bytes, int64_t slot, int64_t head) {
    return reinterpret_cast<float *>(static_cast<char *>(base) + size_t(slot) * stride_bytes) +
           head * dim * dim;
}

} // namespace ocr_gdn_snapshot_store
