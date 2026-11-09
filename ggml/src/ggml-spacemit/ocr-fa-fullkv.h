#pragma once
#include "ocr-pv-pipeline.h"

// Opt-in D72 full-KV PV helpers. Packed weights are transient per graph call;
// no input-address cache is used. The caller validates its TCM capacity.
namespace ocr_fa_fullkv {
constexpr int D=72, Q=16, PAD_V=96;
inline bool trust_bounds_enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_TRUST_BOUNDS");return p&&!std::strcmp(p,"1");}();return on;
}
inline bool head_key_cache_enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_HEAD_K_TCM");return p&&!std::strcmp(p,"1");}();return on;
}
inline bool all_key_cache_enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_ALL_K_TCM");return p&&!std::strcmp(p,"1");}();return on;
}
inline bool half_pack_enabled(){
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_HALF_PACK");return p&&!std::strcmp(p,"1");}();return on;
}
inline bool lean_prepare_enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_LEAN_PREPARE");
        return p&&!std::strcmp(p,"1");}();
    return on;
}
inline bool batch_check_enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_BATCH_CHECK");
        return p&&!std::strcmp(p,"1");}();
    return on;
}
inline bool fixed_index_enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_FIXED_INDEX");
        return p&&!std::strcmp(p,"1");}();
    return on;
}
inline bool p_store64_enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_P_STORE64");
        return p&&!std::strcmp(p,"1");}();
    return on;
}
inline bool fused_query_enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_FUSED_QUERY");
        return p&&!std::strcmp(p,"1");}();
    return on;
}
inline bool query32_enabled() {
    static const bool on=[] { const char *p=std::getenv("GGML_SPACEMIT_OCR_FA_Q32");
        return p&&std::strcmp(p,"1")==0; }();
    return on;
}
inline bool compact_enabled() {
    static const bool on=[] { const char *p=std::getenv("GGML_SPACEMIT_OCR_FA_COMPACT");
        return p&&std::strcmp(p,"1")==0; }();
    return on||query32_enabled();
}
inline int softmax_pack_mode() {
    static const int mode=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_SOFTMAX_PACK");
        return p&&!std::strcmp(p,"1")?1:p&&!std::strcmp(p,"2")?2:0;}();
    return mode;
}
inline bool direct_pack_enabled() {
    static const bool on=[] { const char *p=std::getenv("GGML_SPACEMIT_OCR_FA_FULL_PACK");
        return p&&std::strcmp(p,"1")==0; }();
    return on;
}
// Preserve BOTH original FP16 roundings: f32*attention_scale -> f16,
// followed by f32(f16)*1024 -> f16. Only the temporary row store is removed.
inline bool pack_query(const void*src,ptrdiff_t stride_bytes,int rows,
                       int padded_rows,float scale,_Float16*dst){
    if(!src||!dst||rows<1||rows>padded_rows||padded_rows%8)return false;
    if(!compact_enabled()||rows!=padded_rows)
        std::memset(dst,0,size_t(padded_rows)*D*sizeof(_Float16));
    constexpr size_t vl=D;
    const auto ch=__riscv_vid_v_u32m4(vl);
    const auto offsets=__riscv_vsll_vx_u32m4(__riscv_vadd_vv_u32m4(
        __riscv_vsll_vx_u32m4(__riscv_vsrl_vx_u32m4(ch,3,vl),6,vl),
        __riscv_vand_vx_u32m4(ch,7,vl),vl),1,vl);
    for(int row=0;row<rows;++row){
        const auto*input=reinterpret_cast<const float*>(static_cast<const char*>(src)+row*stride_bytes);
        const auto h=__riscv_vfncvt_f_f_w_f16m2(__riscv_vfmul_vf_f32m4(
            __riscv_vle32_v_f32m4(input,vl),scale,vl),vl);
        const auto f=__riscv_vfwcvt_f_f_v_f32m4(h,vl);
        const auto bad=__riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(f,f,vl),
            __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(f,vl),63.f,vl),vl);
        if(__riscv_vcpop_m_b8(bad,vl))return false;
        auto*out=dst+size_t(row/8)*8*D+(row%8)*8;
        __riscv_vsuxei32_v_f16m2(out,offsets,__riscv_vfncvt_f_f_w_f16m2(
            __riscv_vfmul_vf_f32m4(f,1024.f,vl),vl),vl);
    }
    return true;
}
// Load a complete contiguous D72 row instead of 9 strided eight-lane
// slices. Keep exactly the old power-of-two scale and FP16 rounding.
inline bool pack_rows(const _Float16 *src, ptrdiff_t stride_bytes, int rows,
                      int padded_rows, _Float16 *dst) {
    if(!src||!dst||rows<1||rows>padded_rows||padded_rows%8)return false;
    if(!compact_enabled()||rows!=padded_rows)
        std::memset(dst,0,size_t(padded_rows)*D*sizeof(_Float16));
    constexpr size_t vl=D;
    const auto ch=__riscv_vid_v_u32m4(vl);
    const auto offsets=__riscv_vsll_vx_u32m4(__riscv_vadd_vv_u32m4(
        __riscv_vsll_vx_u32m4(__riscv_vsrl_vx_u32m4(ch,3,vl),6,vl),
        __riscv_vand_vx_u32m4(ch,7,vl),vl),1,vl);
    auto bad_all=__riscv_vmclr_m_b8(vl);
    for(int row=0;row<rows;++row) {
        const auto *input=reinterpret_cast<const _Float16*>(reinterpret_cast<const char*>(src)+row*stride_bytes);
        const auto h=__riscv_vle16_v_f16m2(input,vl);
        auto *output=dst+size_t(row/8)*8*D+(row%8)*8;
        if(half_pack_enabled()){
            const auto bad=__riscv_vmor_mm_b8(__riscv_vmfne_vv_f16m2_b8(h,h,vl),
                __riscv_vmfgt_vf_f16m2_b8(__riscv_vfabs_v_f16m2(h,vl),(_Float16)63.f,vl),vl);
            bad_all=__riscv_vmor_mm_b8(bad_all,bad,vl);
            __riscv_vsuxei32_v_f16m2(output,offsets,__riscv_vfmul_vf_f16m2(h,(_Float16)1024.f,vl),vl);
        }else{
            const auto f=__riscv_vfwcvt_f_f_v_f32m4(h,vl);
            const auto bad=__riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(f,f,vl),
                __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(f,vl),63.f,vl),vl);
            if(__riscv_vcpop_m_b8(bad,vl))return false;
            __riscv_vsuxei32_v_f16m2(output,offsets,__riscv_vfncvt_f_f_w_f16m2(
                __riscv_vfmul_vf_f32m4(f,1024.f,vl),vl),vl);
        }
    }
    return !half_pack_enabled()||!__riscv_vcpop_m_b8(bad_all,vl);
}
inline bool pack_v(const _Float16 * src, ptrdiff_t stride_bytes, int n, int padded,
                   _Float16 * dst) {
    if(!src||!dst||n<1||padded<n||padded%8)return false;
    if(compact_enabled()&&n==padded)
        std::memset(dst+size_t(D)*padded,0,size_t(PAD_V-D)*padded*sizeof(_Float16));
    else std::memset(dst,0,size_t(PAD_V)*padded*sizeof(_Float16));
    constexpr size_t vl=D;
    const auto ch=__riscv_vid_v_u32m4(vl);
    const auto channel_offset=__riscv_vadd_vv_u32m4(
        __riscv_vmul_vx_u32m4(__riscv_vsrl_vx_u32m4(ch,3,vl),8*padded,vl),
        __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(ch,7,vl),3,vl),vl);
    auto bad_all=__riscv_vmclr_m_b8(vl);
    for(int k=0;k<n;++k) {
        const auto * row=reinterpret_cast<const _Float16*>(reinterpret_cast<const char*>(src)+k*stride_bytes);
        const auto h=__riscv_vle16_v_f16m2(row,vl);
        const auto offsets=__riscv_vsll_vx_u32m4(__riscv_vadd_vx_u32m4(
            channel_offset,(k/8)*64+k%8,vl),1,vl);
        if(half_pack_enabled()){
            const auto bad=__riscv_vmor_mm_b8(__riscv_vmfne_vv_f16m2_b8(h,h,vl),
                __riscv_vmfgt_vf_f16m2_b8(__riscv_vfabs_v_f16m2(h,vl),(_Float16)63.f,vl),vl);
            bad_all=__riscv_vmor_mm_b8(bad_all,bad,vl);
            __riscv_vsuxei32_v_f16m2(dst,offsets,__riscv_vfmul_vf_f16m2(h,(_Float16)1024.f,vl),vl);
        }else{
            const auto f=__riscv_vfwcvt_f_f_v_f32m4(h,vl);
            const auto bad=__riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(f,f,vl),
                __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(f,vl),63.f,vl),vl);
            if(__riscv_vcpop_m_b8(bad,vl))return false;
            __riscv_vsuxei32_v_f16m2(dst,offsets,__riscv_vfncvt_f_f_w_f16m2(
                __riscv_vfmul_vf_f32m4(f,1024.f,vl),vl),vl);
        }
    }
    return !half_pack_enabled()||!__riscv_vcpop_m_b8(bad_all,vl);
}
// A task owns disjoint sequence rows in every channel panel. The caller
// separately initializes only the 24 padded channels and the head marker.
inline bool pack_v_slice(const _Float16*src,ptrdiff_t stride_bytes,int n,int padded,
                         int begin,int end,_Float16*dst){
    constexpr size_t vl=D;
    const auto ch=__riscv_vid_v_u32m4(vl);
    const auto channel_offset=__riscv_vadd_vv_u32m4(
        __riscv_vmul_vx_u32m4(__riscv_vsrl_vx_u32m4(ch,3,vl),8*padded,vl),
        __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(ch,7,vl),3,vl),vl);
    bool valid=true;
    for(int k=begin;k<end;++k){
        const auto offsets=__riscv_vsll_vx_u32m4(__riscv_vadd_vx_u32m4(
            channel_offset,(k/8)*64+k%8,vl),1,vl);
        auto h=__riscv_vfmv_v_f_f16m2((_Float16)0,vl);
        if(k<n){
            const auto*row=reinterpret_cast<const _Float16*>(reinterpret_cast<const char*>(src)+k*stride_bytes);
            const auto f=__riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(row,vl),vl);
            const auto bad=__riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(f,f,vl),
                __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(f,vl),63.f,vl),vl);
            if(__riscv_vcpop_m_b8(bad,vl))valid=false;
            h=__riscv_vfncvt_f_f_w_f16m2(__riscv_vfmul_vf_f32m4(f,1024.f,vl),vl);
        }
        __riscv_vsuxei32_v_f16m2(dst,offsets,h,vl);
    }
    return valid;
}
inline bool pack_p(const float * p, int rows, int n, int padded,
                   _Float16 * high, _Float16 * low, int passes, int capacity=Q) {
    if(rows<1||rows>capacity||(capacity!=16&&capacity!=32)||n<1||padded<n||padded%8||(passes!=1&&passes!=2))return false;
    if(!compact_enabled()||rows!=capacity||n!=padded) {
        std::memset(high,0,size_t(capacity)*padded*sizeof(_Float16));
        if(passes==2)std::memset(low,0,size_t(capacity)*padded*sizeof(_Float16));
    }
    for(int row=0;row<rows;++row)for(int k=0;k<n;k+=64) {
        const size_t vl=std::min(64,n-k);
        const auto ids=__riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vl),k,vl);
        const auto offsets=__riscv_vsll_vx_u32m2(__riscv_vadd_vv_u32m2(
            __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(ids,3,vl),6,vl),
            __riscv_vand_vx_u32m2(ids,7,vl),vl),1,vl);
        const auto f=__riscv_vle32_v_f32m2(p+size_t(row)*n+k,vl);
        const auto bad=__riscv_vmor_mm_b16(__riscv_vmfne_vv_f32m2_b16(f,f,vl),
            __riscv_vmor_mm_b16(__riscv_vmflt_vf_f32m2_b16(f,0.f,vl),
                __riscv_vmfgt_vf_f32m2_b16(f,1.001f,vl),vl),vl);
        if(__riscv_vcpop_m_b16(bad,vl))return false;
        const auto h=__riscv_vfncvt_f_f_w_f16m1(__riscv_vfmul_vf_f32m2(f,1024.f,vl),vl);
        const size_t base=size_t(row/8)*8*padded+(row%8)*8;
        __riscv_vsuxei32_v_f16m1(high+base,offsets,h,vl);
        if(passes==2) {
            const auto restored=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(h,vl),1.f/1024.f,vl);
            const auto residual=__riscv_vfsub_vv_f32m2(f,restored,vl);
            __riscv_vsuxei32_v_f16m1(low+base,offsets,__riscv_vfncvt_f_f_w_f16m1(
                __riscv_vfmul_vf_f32m2(residual,16777216.f,vl),vl),vl);
        }
    }
    return true;
}

// 16x32, K multiple of eight; independently constrained pointer registers.
// Accumulators stay live across the complete K range, with no tile readback.
__attribute__((noinline)) static void pv16_reference(const _Float16*x,const _Float16*w,float*c,int k){
 const _Float16*x1=x+8*k,*w1=w+8*k,*w2=w+16*k,*w3=w+24*k;int blocks=k/8;
 asm volatile(
 "vsetvli t0,zero,e32,m2,ta,ma\n\t"
 "vmv.v.i v16,0\n\tvmv.v.i v18,0\n\tvmv.v.i v20,0\n\tvmv.v.i v22,0\n\t"
 "vmv.v.i v24,0\n\tvmv.v.i v26,0\n\tvmv.v.i v28,0\n\tvmv.v.i v30,0\n\t"
 "vsetvli t0,zero,e16,m1,ta,ma\n\t1:\n\t"
 "vle16.v v2,(%[x])\n\tvle16.v v3,(%[x1])\n\t"
 "vle16.v v8,(%[w])\n\tvle16.v v9,(%[w1])\n\tvle16.v v10,(%[w2])\n\tvle16.v v11,(%[w3])\n\t"
 "smt.vfwmadot v16,v2,v8\n\tsmt.vfwmadot v18,v2,v9\n\t"
 "smt.vfwmadot v20,v2,v10\n\tsmt.vfwmadot v22,v2,v11\n\t"
 "smt.vfwmadot v24,v3,v8\n\tsmt.vfwmadot v26,v3,v9\n\t"
 "smt.vfwmadot v28,v3,v10\n\tsmt.vfwmadot v30,v3,v11\n\t"
 "addi %[x],%[x],128\n\taddi %[x1],%[x1],128\n\t"
 "addi %[w],%[w],128\n\taddi %[w1],%[w1],128\n\t"
 "addi %[w2],%[w2],128\n\taddi %[w3],%[w3],128\n\t"
 "addi %[blocks],%[blocks],-1\n\tbnez %[blocks],1b\n\t"
 "vsetvli t0,zero,e32,m2,ta,ma\n\tvse32.v v16,(%[c])\n\t"
 "addi t0,%[c],256\n\tvse32.v v18,(t0)\n\taddi t0,%[c],512\n\tvse32.v v20,(t0)\n\t"
 "addi t0,%[c],768\n\tvse32.v v22,(t0)\n\taddi t0,%[c],1024\n\tvse32.v v24,(t0)\n\t"
 "addi t0,%[c],1280\n\tvse32.v v26,(t0)\n\taddi t0,%[c],1536\n\tvse32.v v28,(t0)\n\t"
 "addi t0,%[c],1792\n\tvse32.v v30,(t0)\n\t"
 :[x]"+&r"(x),[x1]"+&r"(x1),[w]"+&r"(w),[w1]"+&r"(w1),[w2]"+&r"(w2),[w3]"+&r"(w3),[blocks]"+&r"(blocks)
 :[c]"r"(c):"t0","v2","v3","v8","v9","v10","v11","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","memory");
}

// Keep the existing function signature for channel-panel dispatch. The new
// implementation remains opt-in; the eight-channel tail has its own flag.
static void pv16(const _Float16*x,const _Float16*w,float*c,int k){
 if(ocr_pv_pipeline::enabled()&&k>=8&&k%8==0)ocr_pv_pipeline::core16(x,w,c,k);
 else pv16_reference(x,w,c,k);
}

// The D72 tail has only eight output channels. Avoid retaining 24 zero
// channels in TCM, allowing twice as many queries without spilling to DDR.
__attribute__((noinline)) static void pv16x8_reference(const _Float16*x,const _Float16*w,float*c,int k){
 const _Float16*x1=x+8*k;int blocks=k/8;
 asm volatile(
 "vsetvli t0,zero,e32,m2,ta,ma\n\t"
 "vmv.v.i v16,0\n\tvmv.v.i v18,0\n\t"
 "vsetvli t0,zero,e16,m1,ta,ma\n\t1:\n\t"
 "vle16.v v2,(%[x])\n\tvle16.v v3,(%[x1])\n\tvle16.v v8,(%[w])\n\t"
 "smt.vfwmadot v16,v2,v8\n\tsmt.vfwmadot v18,v3,v8\n\t"
 "addi %[x],%[x],128\n\taddi %[x1],%[x1],128\n\taddi %[w],%[w],128\n\t"
 "addi %[blocks],%[blocks],-1\n\tbnez %[blocks],1b\n\t"
 "vsetvli t0,zero,e32,m2,ta,ma\n\tvse32.v v16,(%[c])\n\t"
 "addi t0,%[c],1024\n\tvse32.v v18,(t0)\n\t"
 :[x]"+&r"(x),[x1]"+&r"(x1),[w]"+&r"(w),[blocks]"+&r"(blocks)
 :[c]"r"(c):"t0","v2","v3","v8","v16","v17","v18","v19","memory");
}
static void pv16x8(const _Float16*x,const _Float16*w,float*c,int k){
 if(ocr_pv_pipeline::tail_enabled()&&k>=8&&k%8==0)ocr_pv_pipeline::core8(x,w,c,k);
 else pv16x8_reference(x,w,c,k);
}
} // namespace ocr_fa_fullkv
