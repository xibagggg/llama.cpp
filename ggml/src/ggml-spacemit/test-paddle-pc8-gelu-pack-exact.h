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
inline void reference_token_pack(const float*x,int8_t*packed,float*scales,int M,int ith,int nth){
    constexpr int K=4352,input_k=4304;constexpr bool PAD=true;
    const int MP=(M+3)/4*4;
        for(int g=ith;g<MP/4;g+=nth)for(int r=0;r<4;++r){
            if(g*4+r>=M){scales[g*4+r]=0;for(int k=0;k<K;k+=32)std::memset(packed+size_t(g)*4*K+k*4+r*32,0,32);continue;}
            auto vmax=__riscv_vfmv_v_f_f32m2(0,64);
            for(int k=0;k<input_k;k+=64){const size_t vl=std::min(64,input_k-k);const auto a=__riscv_vfabs_v_f32m2(__riscv_vle32_v_f32m2(x+size_t(g*4+r)*input_k+k,vl),vl);vmax=__riscv_vfmax_vv_f32m2_tu(vmax,vmax,a,vl);}
            const float maximum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(vmax,__riscv_vfmv_v_f_f32m1(0,1),64));
            const float scale=maximum/127;GGML_ASSERT(std::isfinite(maximum));scales[g*4+r]=scale;const float inv=scale?1/scale:0;
            for(int k=0;k<K;k+=32){auto*target=packed+size_t(g)*4*K+k*4+r*32;
                const size_t vl=PAD?std::min(32,std::max(0,input_k-k)):32;
                if constexpr(PAD){if(vl<32)std::memset(target,0,32);if(!vl)continue;}
                const auto a=__riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(x+size_t(g*4+r)*input_k+k,vl),inv,vl);
                const auto q=__riscv_vncvt_x_x_w_i8mf4(__riscv_vfncvt_x_f_w_i16mf2(a,vl),vl);
                __riscv_vse8_v_i8mf4(target,q,vl);}
        }
}
