#pragma once

#include "ggml-impl.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// Graph guards are portable so the same matcher can be tested without IME2.
namespace ocr_norm_affine {
struct plan {
    ggml_tensor * residual = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * bias = nullptr;
    int last = -1;
    explicit operator bool() const { return last >= 0; }
};

inline bool dense(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

inline bool row_weight(const ggml_tensor * t, int64_t width) {
    return dense(t) && t->ne[0] == width && t->ne[1] == 1 &&
           t->ne[2] == 1 && t->ne[3] == 1;
}

inline bool layout(const plan & p) {
    if (!p.norm || !p.scale || !p.bias || p.norm->op != GGML_OP_NORM ||
        p.scale->op != GGML_OP_MUL || p.bias->op != GGML_OP_ADD ||
        p.scale->src[0] != p.norm || p.bias->src[0] != p.scale ||
        !dense(p.norm->src[0]) || !dense(p.norm) || !dense(p.scale) || !dense(p.bias) ||
        !ggml_are_same_shape(p.norm->src[0], p.norm) ||
        !ggml_are_same_shape(p.norm, p.scale) || !ggml_are_same_shape(p.norm, p.bias) ||
        !row_weight(p.scale->src[1], p.norm->ne[0]) ||
        !row_weight(p.bias->src[1], p.norm->ne[0])) return false;
    float epsilon;
    std::memcpy(&epsilon, p.norm->op_params, sizeof(epsilon));
    if (!(epsilon > 0.f) || !std::isfinite(epsilon)) return false;
    return !p.residual || (p.residual == p.norm->src[0] &&
        p.residual->op == GGML_OP_ADD && !p.residual->view_src &&
        dense(p.residual->src[0]) && dense(p.residual->src[1]) &&
        ggml_are_same_shape(p.residual, p.residual->src[0]) &&
        ggml_are_same_shape(p.residual, p.residual->src[1]));
}

inline int next_compute(const ggml_cgraph * graph, int index) {
    for (int i = index + 1; i < graph->n_nodes; ++i) {
        if (graph->nodes[i]->op == GGML_OP_NONE) continue;
        // Do not skip views: the allocator and external consumers may need them.
        return i;
    }
    return -1;
}

inline plan match(const ggml_cgraph * graph, int index) {
    if (!graph || index < 0 || index >= graph->n_nodes) return {};
    plan p;
    const auto first = graph->nodes[index];
    int ni = index;
    if (first->op == GGML_OP_ADD) {
        p.residual = first;
        ni = next_compute(graph, index);
    }
    if (ni < 0 || graph->nodes[ni]->op != GGML_OP_NORM) return {};
    const int si = next_compute(graph, ni);
    const int bi = si < 0 ? -1 : next_compute(graph, si);
    if (bi < 0) return {};
    p.norm = graph->nodes[ni]; p.scale = graph->nodes[si]; p.bias = graph->nodes[bi];
    const int indices[] = {ni, si, bi};
    const ggml_op ops[] = {GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD};
    if (!ggml_can_fuse_ext(graph, indices, ops, 3) || !layout(p)) return {};
    // Residual can have any number of consumers: it is still materialized at
    // its original graph position. Only NORM and MUL intermediates disappear.
    p.last = bi;
    return p;
}

inline bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) return false;
    const auto x = reinterpret_cast<uintptr_t>(a->data);
    const auto y = reinterpret_cast<uintptr_t>(b->data);
    return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
}

inline bool storage_safe(const plan & p) {
    if (!layout(p)) return false;
    const auto input = p.norm->src[0];
    if (overlaps(p.bias, p.scale->src[1]) || overlaps(p.bias, p.bias->src[1])) return false;
    if (p.residual) {
        // Keep the residual alive for later branches. The allocator may alias
        // its inputs exactly; the first pass reads a vector before storing it.
        if (overlaps(p.bias, p.residual) ||
            overlaps(p.bias, p.residual->src[0]) ||
            overlaps(p.bias, p.residual->src[1]) ||
            overlaps(p.residual, p.scale->src[1]) ||
            overlaps(p.residual, p.bias->src[1])) return false;
        for (int i = 0; i < 2; ++i)
            if (overlaps(p.residual, p.residual->src[i]) &&
                p.residual->data != p.residual->src[i]->data) return false;
    } else if (overlaps(p.bias, input) && p.bias->data != input->data) {
        return false;
    }
    return true;
}
} // namespace ocr_norm_affine
