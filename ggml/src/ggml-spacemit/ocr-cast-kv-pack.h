#pragma once
#include "ggml-impl.h"
#include "ocr-fa-pack-v40.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ocr_cast_kv_pack {
inline bool enabled() {
    static const bool on = [] {
        const char * p = std::getenv("GGML_SPACEMIT_OCR_CAST_KV_PACK");
        return p && !std::strcmp(p, "1");
    }();
    return on;
}
inline bool cast(const ggml_tensor * t) {
    return t && t->op == GGML_OP_CPY && !t->view_src &&
        t->type == GGML_TYPE_F16 && ggml_is_contiguous(t) && t->src[0] &&
        t->src[0]->type == GGML_TYPE_F32 && t->src[0]->nb[0] == sizeof(float) &&
        ggml_are_same_shape(t, t->src[0]);
}
inline bool disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) return false;
    const auto aa = reinterpret_cast<uintptr_t>(a->data);
    const auto bb = reinterpret_cast<uintptr_t>(b->data);
    return aa <= bb ? ggml_nbytes(a) <= bb-aa : ggml_nbytes(b) <= aa-bb;
}
struct plan {
    ggml_tensor * attention = nullptr;
    int value_index = -1;
    explicit operator bool() const { return attention != nullptr; }
};
inline int next_compute(const ggml_cgraph * g,int i) {
    while (++i < g->n_nodes) if (!ggml_op_is_empty(g->nodes[i]->op)) return i;
    return -1;
}
inline plan match(const ggml_cgraph * g, int i) {
    if (!enabled()) return {};
    const int vi=next_compute(g,i),fi=vi<0?-1:next_compute(g,vi);
    if(fi<0)return {};
    auto * k = g->nodes[i];
    auto * v = g->nodes[vi];
    auto * fa = g->nodes[fi];
    const char * lean = std::getenv("GGML_SPACEMIT_OCR_FA_LEAN_PREPARE");
    const char * direct = std::getenv("GGML_SPACEMIT_OCR_FA_FULL_PACK");
    if (!lean || std::strcmp(lean,"1") || !direct || std::strcmp(direct,"1") ||
        !cast(k) || !cast(v) || !ocr_fa_pack_v40::full_kv(fa) ||
        fa->src[1] != k || fa->src[2] != v ||
        !disjoint(k,v) || !disjoint(k,k->src[0]) || !disjoint(k,v->src[0]) ||
        !disjoint(v,k->src[0]) || !disjoint(v,v->src[0])) return {};
    // Only metadata views may intervene. Both original cast outputs remain
    // materialized; no computing operation or consumer is bypassed.
    const int idx[2] = {i,vi};
    const ggml_op ops[2] = {GGML_OP_CPY,GGML_OP_CPY};
    return ggml_can_fuse_subgraph_ext(g,idx,2,ops,idx,2) ? plan{fa,vi} : plan{};
}
}
