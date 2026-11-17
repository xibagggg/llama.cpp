#pragma once
// Default-off scheduling/staging change only. Quantization, physical N32 packs,
// integer dot products and per-output FP32 K-block accumulation stay unchanged.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace paddle_llm_head_stage {
inline int requested_cols() {
    static const int value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_PADDLE_LLM_HEAD_STAGE_N");
        return p&&!std::strcmp(p,"64")?64:p&&!std::strcmp(p,"128")?128:0;
    }();
    return value;
}
inline bool shape(int64_t m,int64_t k,int64_t n,const char *name) {
    const bool output_name=name&&(!std::strcmp(name,"output.weight")||!std::strcmp(name,"lm_head.weight"));
    return m==1&&k==1024&&n==103424&&output_name;
}
struct plan {
    int64_t cols=32;
    size_t b_bytes=0;
    bool selected=false;
};
inline plan choose(int request,int64_t m,int64_t k,int64_t n,const char *name,
                   size_t row_stride_b,size_t tcm_bytes) {
    plan out;
    // The row stride must be the established K1024 physical Q8_0 layout:
    // 32 K-blocks * (32 bytes + one FP16 scale). No new pack is introduced.
    if((request!=64&&request!=128)||!shape(m,k,n,name)||row_stride_b!=1088||
       tcm_bytes/row_stride_b<size_t(request))return out;
    out.cols=request;out.b_bytes=size_t(request)*row_stride_b;out.selected=true;
    return out;
}
struct schedule {
    int64_t cols=32,nth=1;
    int64_t first(int64_t ith)const{return ith*cols;}
    int64_t pair_first(int64_t ith)const{return (ith&~int64_t(1))*cols;}
    int64_t lane_offset(int64_t ith)const{return (ith&1)*cols;}
    int64_t step()const{return nth*cols;}
    int64_t count(int64_t ni,int64_t n)const{return ni<n?std::min(n-ni,cols):0;}
};
} // namespace paddle_llm_head_stage
