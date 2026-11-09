#pragma once
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

// Experimental, explicitly opt-in interpolation. Tables are initialized on
// the host when the backend DSO loads; no model or activation result is cached.
namespace ocr_nonlinear_lut {
struct tables_t {
    std::array<float,4098> gelu{};
    std::array<float,4098> exp{};
    tables_t(){
        for(size_t i=0;i<gelu.size();++i){
            const double x=-8.+double(i)/256.;
            gelu[i]=float(.5*x*(1.+std::tanh(.7978845608028654*x*(1.+.044715*x*x))));
        }
        for(size_t i=0;i<exp.size();++i)exp[i]=float(std::exp(-32.+double(i)/128.));
    }
};
static const tables_t tables;
inline bool gelu_enabled(){static const bool b=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_GELU_LUT");return p&&!std::strcmp(p,"1");}();return b;}
inline bool exp_enabled(){static const bool b=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_EXP_LUT");return p&&!std::strcmp(p,"1");}();return b;}
inline vfloat32m2_t interpolate(vfloat32m2_t position,const float*table,size_t vl){
    const auto index=__riscv_vfcvt_rtz_xu_f_v_u32m2(position,vl);
    const auto fraction=__riscv_vfsub_vv_f32m2(position,__riscv_vfcvt_f_xu_v_f32m2(index,vl),vl);
    const auto offsets=__riscv_vsll_vx_u32m2(index,2,vl);
    const auto low=__riscv_vluxei32_v_f32m2(table,offsets,vl);
    const auto high=__riscv_vluxei32_v_f32m2(table+1,offsets,vl);
    return __riscv_vfmacc_vv_f32m2(low,fraction,__riscv_vfsub_vv_f32m2(high,low,vl),vl);
}
inline vfloat32m2_t gelu(vfloat32m2_t x,size_t vl,const float*lookup){
    // Clamp before integer conversion/gather, including NaN/Inf. The final
    // masks restore signed zero, propagate NaNs, and match Inf -> NaN.
    const auto clipped=__riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(x,-8.f,vl),8.f,vl);
    const auto position=__riscv_vfmul_vf_f32m2(__riscv_vfadd_vf_f32m2(clipped,8.f,vl),256.f,vl);
    auto y=interpolate(position,lookup,vl);
    y=__riscv_vmerge_vvm_f32m2(y,__riscv_vfmv_v_f_f32m2(0.f,vl),__riscv_vmflt_vf_f32m2_b16(x,-8.f,vl),vl);
    y=__riscv_vmerge_vvm_f32m2(y,x,__riscv_vmfgt_vf_f32m2_b16(x,8.f,vl),vl);
    y=__riscv_vmerge_vvm_f32m2(y,x,__riscv_vmfeq_vf_f32m2_b16(x,0.f,vl),vl);
    const auto nonfinite=__riscv_vmor_mm_b16(__riscv_vmfne_vv_f32m2_b16(x,x,vl),
        __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(x,vl),3.402823466e38f,vl),vl);
    return __riscv_vmerge_vvm_f32m2(y,__riscv_vfsub_vv_f32m2(x,x,vl),nonfinite,vl);
}
// Called only with finite nonpositive softmax arguments after maximum checks.
// Below -32 the omitted mass is <1.27e-14 per element.
inline vfloat32m2_t exp_negative(vfloat32m2_t x,size_t vl,const float*lookup){
    const auto clipped=__riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(x,-32.f,vl),0.f,vl);
    const auto position=__riscv_vfmul_vf_f32m2(__riscv_vfadd_vf_f32m2(clipped,32.f,vl),128.f,vl);
    const auto y=interpolate(position,lookup,vl);
    return __riscv_vmerge_vvm_f32m2(y,__riscv_vfmv_v_f_f32m2(0.f,vl),__riscv_vmflt_vf_f32m2_b16(x,-32.f,vl),vl);
}
}
