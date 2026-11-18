#pragma once
#include "ggml.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ocr_gdn_rollback_tcm {
inline constexpr size_t scratch_floats=128*128+3*128;
inline constexpr size_t scratch_bytes=scratch_floats*sizeof(float);
inline std::atomic<bool> enabled{false};
inline std::atomic<bool> debug_enabled{false};
inline std::atomic<bool> trace_enabled{false};
inline std::atomic<uint64_t> selected_calls{0};

inline void prepare_on_host(unsigned workers,size_t available,bool pinned_tcm) {
    const char * option=std::getenv("GGML_SPACEMIT_OCR_GDN_ROLLBACK_TCM");
    const char * trace=std::getenv("GGML_SPACEMIT_OCR_GDN_ROLLBACK_TCM_TRACE");
    const bool debug=std::getenv("GGML_SPACEMIT_GDN_DEBUG")!=nullptr;
    debug_enabled.store(debug,std::memory_order_relaxed);
    trace_enabled.store(trace&&std::strcmp(trace,"1")==0,std::memory_order_relaxed);
    // Capacity and runtime storage policy are decided on the host collectively.
    enabled.store(option&&std::strcmp(option,"1")==0&&workers==6&&pinned_tcm&&
        available>=scratch_bytes&&!debug,std::memory_order_relaxed);
    selected_calls.store(0,std::memory_order_relaxed);
}

inline bool valid_f32(const ggml_tensor * t) {
    if(!t||t->type!=GGML_TYPE_F32||!t->data||
        reinterpret_cast<uintptr_t>(t->data)%alignof(float)||t->nb[0]!=sizeof(float))return false;
    size_t extent=sizeof(float);
    for(unsigned d=0;d<4;++d){
        if(t->ne[d]<=0||!t->nb[d]||size_t(t->ne[d]-1)>(SIZE_MAX-extent)/t->nb[d])return false;
        if(t->nb[d]<extent)return false;
        extent+=size_t(t->ne[d]-1)*t->nb[d];
    }
    return extent<=UINTPTR_MAX-reinterpret_cast<uintptr_t>(t->data);
}
inline bool disjoint(const void * a,size_t an,const void * b,size_t bn) {
    const uintptr_t av=reinterpret_cast<uintptr_t>(a),bv=reinterpret_cast<uintptr_t>(b);
    if(an>UINTPTR_MAX-av||bn>UINTPTR_MAX-bv)return false;
    return av+an<=bv||bv+bn<=av;
}

inline float * workspace(const ggml_tensor * op,void * memory,size_t available,
                         unsigned ith,unsigned workers,bool pinned_tcm) {
    if(!enabled.load(std::memory_order_relaxed)||workers!=6||ith>=workers||!pinned_tcm||
        !memory||available<scratch_bytes||reinterpret_cast<uintptr_t>(memory)%alignof(float)||
        !valid_f32(op)||!ggml_is_contiguous(op))return nullptr;
    int32_t k=0;std::memcpy(&k,op->op_params,sizeof k);
    if(k!=16)return nullptr;
    for(unsigned i=0;i<6;++i)if(!valid_f32(op->src[i]))return nullptr;
    const auto * q=op->src[0],*key=op->src[1],*v=op->src[2],*g=op->src[3],*beta=op->src[4],*state=op->src[5];
    const int64_t tokens=v->ne[2];
    // Two-pass only. Small fused/K1/other layouts keep the exact DDR fallback.
    // AI6 ABBA found a regression for scalar-gate T16. Keep small scalar
    // gates on DDR; KDA and measured long prefill blocks remain eligible.
    if(tokens<5||tokens>256||(g->ne[0]!=128&&tokens<240)||
        v->ne[0]!=128||v->ne[1]!=16||v->ne[3]!=1||
        q->ne[0]!=128||q->ne[1]!=16||q->ne[2]!=tokens||q->ne[3]!=1||
        key->ne[0]!=128||key->ne[1]!=16||key->ne[2]!=tokens||key->ne[3]!=1||
        (g->ne[0]!=1&&g->ne[0]!=128)||g->ne[1]!=16||g->ne[2]!=tokens||g->ne[3]!=1||
        beta->ne[0]!=1||beta->ne[1]!=16||beta->ne[2]!=tokens||beta->ne[3]!=1||
        state->ne[0]!=128||state->ne[1]!=128||state->ne[2]!=16||state->ne[3]!=1||
        state->nb[1]!=128*sizeof(float)||state->nb[2]!=128*128*sizeof(float)||
        ggml_nelements(op)!=128*16*tokens+16*128*128*16)return nullptr;
    const size_t output_bytes=ggml_nbytes(op);
    if(!disjoint(memory,scratch_bytes,op->data,output_bytes))return nullptr;
    for(unsigned i=0;i<6;++i){
        const auto * source=op->src[i];const size_t bytes=ggml_nbytes(source);
        if(!disjoint(memory,scratch_bytes,source->data,bytes)||
            !disjoint(op->data,output_bytes,source->data,bytes))return nullptr;
    }
    if(ith==0&&trace_enabled.load(std::memory_order_relaxed))selected_calls.fetch_add(1,std::memory_order_relaxed);
    return static_cast<float *>(memory);
}

inline void drain_on_host() {
    const uint64_t selected=selected_calls.exchange(0,std::memory_order_relaxed);
    if(trace_enabled.load(std::memory_order_relaxed))
        std::fprintf(stderr,"OCR_GDN_ROLLBACK_TCM admitted=%d selected=%llu bytes_per_lane=%zu\n",
            int(enabled.load(std::memory_order_relaxed)),static_cast<unsigned long long>(selected),scratch_bytes);
}
} // namespace ocr_gdn_rollback_tcm
