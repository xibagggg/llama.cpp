#pragma once
// Generated from exact frozen backend functions; see fixture-excerpts.json.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <riscv_vector.h>
#define GGML_ASSERT(x) do { if(!(x)) { std::fprintf(stderr,"GGML_ASSERT failed: %s\n",#x);std::abort(); } } while(0)
static inline vfloat32m2_t rvv_expf_approx_f32m2(vfloat32m2_t x, size_t vl) {
    const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    const vfloat32m2_t b =
        __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl), 0x1.7f7d1cp-20f, n, vl);
    const vuint32m2_t  e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl));
    const vbool16_t    c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl), u, vl),
        u, vl);

    if (!__riscv_vcpop_m_b16(c, vl)) {
        return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    }

    const vbool16_t    dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    const vuint32m2_t  d  = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    const vfloat32m2_t r1 =
        __riscv_vmerge_vvm_f32m2(__riscv_vfmacc_vv_f32m2(k, k, j, vl),
                                 __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl), c, vl);
    return __riscv_vmerge_vvm_f32m2(r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
                                    __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl), vl);
}
static inline vfloat32m2_t rvv_tanh_approx_f32m2(vfloat32m2_t x, size_t vl) {
    const vfloat32m2_t abs_x       = __riscv_vfabs_v_f32m2(x, vl);
    const vfloat32m2_t neg_2_abs   = __riscv_vfmul_vf_f32m2(abs_x, -2.0f, vl);
    const vfloat32m2_t exp_term    = rvv_expf_approx_f32m2(neg_2_abs, vl);
    const vfloat32m2_t numerator   = __riscv_vfsub_vf_f32m2(exp_term, 1.0f, vl);
    const vfloat32m2_t denominator = __riscv_vfadd_vf_f32m2(exp_term, 1.0f, vl);
    const vfloat32m2_t tanh_abs    = __riscv_vfneg_v_f32m2(__riscv_vfdiv_vv_f32m2(numerator, denominator, vl), vl);
    const vbool16_t    neg_mask    = __riscv_vmflt_vf_f32m2_b16(x, 0.0f, vl);
    const vfloat32m2_t tanh_neg    = __riscv_vfneg_v_f32m2(tanh_abs, vl);
    return __riscv_vmerge_vvm_f32m2(tanh_abs, tanh_neg, neg_mask, vl);
}
namespace spacemit_kernels::rvv {
void forward_gelu_f32_tile(float *data,size_t elements) {
    static constexpr float GELU_ALPHA = 0.7978845608f;
    static constexpr float GELU_BETA  = 0.044715f;

    size_t i = 0;
    while (i < elements) {
        const size_t vl = __riscv_vsetvl_e32m2(elements - i);

        vfloat32m2_t x     = __riscv_vle32_v_f32m2(data + i, vl);
        vfloat32m2_t x2    = __riscv_vfmul_vv_f32m2(x, x, vl);
        vfloat32m2_t inner = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), GELU_BETA, x2, vl);
        inner              = __riscv_vfmul_vf_f32m2(__riscv_vfmul_vv_f32m2(x, inner, vl), GELU_ALPHA, vl);
        vfloat32m2_t th    = rvv_tanh_approx_f32m2(inner, vl);
        vfloat32m2_t out   = __riscv_vfmul_vf_f32m2(
                __riscv_vfmul_vv_f32m2(x, __riscv_vfadd_vf_f32m2(th, 1.0f, vl), vl), 0.5f, vl);
        __riscv_vse32_v_f32m2(data + i, out, vl);
        i += vl;
    }
}
}
namespace paddle_f16_ime2 {
inline bool activation_store64_enabled() {
    static const bool value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_PADDLE_F16_ACT_STORE64");
        return p&&!std::strcmp(p,"1");
    }();
    return value;
}
inline float scale_for(float maximum) {
    if(maximum==0)return 1;
    GGML_ASSERT(std::isfinite(maximum));
    int exponent=0;std::frexp(maximum,&exponent);
    return std::ldexp(1.0f,std::max(-100,std::min(100,14-exponent)));
}
inline void pack_activation64(const float *x,_Float16 *px,_Float16 *pl,
                              float scale,int group,int M,int K) {
    // For lane i, half-index=(i/8)*64+i%8 in the existing K8/M8 layout.
    // Maximum byte index is 910, so u16 offsets are sufficient.
    const auto ids=__riscv_vid_v_u16m1(64);
    const auto offsets=__riscv_vsll_vx_u16m1(__riscv_vadd_vv_u16m1(
        __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(ids,3,64),6,64),
        __riscv_vand_vx_u16m1(ids,7,64),64),1,64);
    const auto ids64=__riscv_vid_v_u64m1(16);
    const auto offsets64=__riscv_vsll_vx_u64m1(__riscv_vadd_vv_u64m1(
        __riscv_vsll_vx_u64m1(__riscv_vsrl_vx_u64m1(ids64,1,16),4,16),
        __riscv_vand_vx_u64m1(ids64,1,16),16),3,16);
    for(int r=0;r<8;++r)for(int k=0;k<K;k+=64) {
        const size_t vl=std::min(64,K-k);
        GGML_ASSERT(vl%4==0);
        const size_t off=size_t(group)*8*K+size_t(k)*8+r*8;
        if(group*8+r>=M) {
            const auto zero=__riscv_vfmv_v_f_f16m1(0,vl);
            if(activation_store64_enabled()) {
                const auto bits=__riscv_vreinterpret_v_u16m1_u64m1(__riscv_vreinterpret_v_f16m1_u16m1(zero));
                __riscv_vsuxei64_v_u64m1(reinterpret_cast<uint64_t*>(px+off),offsets64,bits,vl/4);
                __riscv_vsuxei64_v_u64m1(reinterpret_cast<uint64_t*>(pl+off),offsets64,bits,vl/4);
            } else {
                __riscv_vsuxei16_v_f16m1(px+off,offsets,zero,vl);
                __riscv_vsuxei16_v_f16m1(pl+off,offsets,zero,vl);
            }
            continue;
        }
        const auto a=__riscv_vfmul_vf_f32m2(
            __riscv_vle32_v_f32m2(x+size_t(group*8+r)*K+k,vl),scale,vl);
        auto h=__riscv_vfncvt_f_f_w_f16m1(a,vl);
        auto h32=__riscv_vfwcvt_f_f_v_f32m2(h,vl);
        const auto small=__riscv_vmflt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(h32,vl),0x1p-14f,vl);
        h32=__riscv_vfmerge_vfm_f32m2(h32,0.0f,small,vl);
        h=__riscv_vfncvt_f_f_w_f16m1(h32,vl);
        const auto lo=__riscv_vfncvt_f_f_w_f16m1(
            __riscv_vfmul_vf_f32m2(__riscv_vfsub_vv_f32m2(a,h32,vl),4096.0f,vl),vl);
        // All indices are distinct, so unordered stores retain exact layout.
        if(activation_store64_enabled()) {
            // Lane j holds the unmodified bits of F16 lanes 4*j..4*j+3.
            // Byte offset=(j/2)*128+(j%2)*8 in the original K8/M8 pack.
            __riscv_vsuxei64_v_u64m1(reinterpret_cast<uint64_t*>(pl+off),offsets64,
                __riscv_vreinterpret_v_u16m1_u64m1(__riscv_vreinterpret_v_f16m1_u16m1(lo)),vl/4);
            __riscv_vsuxei64_v_u64m1(reinterpret_cast<uint64_t*>(px+off),offsets64,
                __riscv_vreinterpret_v_u16m1_u64m1(__riscv_vreinterpret_v_f16m1_u16m1(h)),vl/4);
        } else {
            __riscv_vsuxei16_v_f16m1(pl+off,offsets,lo,vl);
            __riscv_vsuxei16_v_f16m1(px+off,offsets,h,vl);
        }
    }
}
inline void pack_gelu_group(const float *input,float *scratch,_Float16 *high,_Float16 *low,
                            float *scale_out,int rows,int K) {
    // The exact existing RVV GELU formula is evaluated once per element.
    // This buffer is reused only after the complete K of all real A8 rows.
    std::memcpy(scratch,input,size_t(rows)*K*sizeof(float));
    spacemit_kernels::rvv::forward_gelu_f32_tile(scratch,size_t(rows)*K);
    auto vmax=__riscv_vfmv_v_f_f32m2(0,64);
    for(int r=0;r<rows;++r)for(int k=0;k<K;k+=64) {
        const size_t vl=std::min(64,K-k);
        const auto a=__riscv_vfabs_v_f32m2(__riscv_vle32_v_f32m2(scratch+size_t(r)*K+k,vl),vl);
        vmax=__riscv_vfmax_vv_f32m2_tu(vmax,vmax,a,vl);
    }
    const float maximum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(
        vmax,__riscv_vfmv_v_f_f32m1(0,1),64));
    const float scale=scale_for(maximum);*scale_out=scale;
    pack_activation64(scratch,high,low,scale,0,rows,K);
}
}
inline void reference_gelu_partition(const float *src_ptr,float *dst_ptr,
    int64_t ne,int64_t ith,int64_t nth) {
    const int64_t dr=(ne+nth-1)/nth,i0=dr*ith,i1=std::min(i0+dr,ne);
    static constexpr float GELU_ALPHA = 0.7978845608f;
    static constexpr float GELU_BETA  = 0.044715f;

    int64_t i = i0;
    while (i < i1) {
        const size_t vl = __riscv_vsetvl_e32m2(i1 - i);

        vfloat32m2_t x     = __riscv_vle32_v_f32m2(src_ptr + i, vl);
        vfloat32m2_t x2    = __riscv_vfmul_vv_f32m2(x, x, vl);
        vfloat32m2_t inner = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(1.0f, vl), GELU_BETA, x2, vl);
        inner              = __riscv_vfmul_vf_f32m2(__riscv_vfmul_vv_f32m2(x, inner, vl), GELU_ALPHA, vl);
        vfloat32m2_t th    = rvv_tanh_approx_f32m2(inner, vl);
        vfloat32m2_t out   = __riscv_vfmul_vf_f32m2(
                __riscv_vfmul_vv_f32m2(x, __riscv_vfadd_vf_f32m2(th, 1.0f, vl), vl), 0.5f, vl);
        __riscv_vse32_v_f32m2(dst_ptr + i, out, vl);
        i += vl;
    }
}
inline float reference_group_maximum(const float *x,int g,int M,int K) {
    auto vmax=__riscv_vfmv_v_f_f32m2(0,64);
    for(int r=0;r<8&&g*8+r<M;++r)for(int k=0;k<K;k+=64) {
        const size_t vl=std::min(64,K-k);
        const auto a=__riscv_vfabs_v_f32m2(__riscv_vle32_v_f32m2(x+size_t(g*8+r)*K+k,vl),vl);
        vmax=__riscv_vfmax_vv_f32m2_tu(vmax,vmax,a,vl);
    }
    const float maximum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(vmax,__riscv_vfmv_v_f_f32m1(0,1),64));
    return maximum;
}
inline void reference_group_pack(const float *x,_Float16 *high,_Float16 *low,
    float *scale_out,int g,int M,int K) {
    float scale=paddle_f16_ime2::scale_for(reference_group_maximum(x,g,M,K));
    *scale_out=scale;paddle_f16_ime2::pack_activation64(x,high,low,scale,g,M,K);
}
