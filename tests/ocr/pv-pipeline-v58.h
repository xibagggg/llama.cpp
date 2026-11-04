#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>
#pragma once
// Same FMA sequence for each output element; reuse K/V across eight queries.
static inline bool ocr_fa_x8_enabled(){
    static const bool enabled=[] {const char *v=std::getenv("GGML_SPACEMIT_OCR_FA_X8");return v&&std::strcmp(v,"1")==0;}();
    return enabled;
}
static inline void ocr_qk_f16_x8(float *dst,const _Float16 *q,const _Float16 *k,int dk,int stride,int count){
    const size_t vl=__riscv_vsetvl_e16m1(count);
    auto a0=__riscv_vfmv_v_f_f32m2(0,vl),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    for(int d=0;d<dk;++d){
        const auto kv=__riscv_vle16_v_f16m1(k+d*stride,vl);
        a0=__riscv_vfwmacc_vf_f32m2(a0,q[d],kv,vl);
        a1=__riscv_vfwmacc_vf_f32m2(a1,q[dk+d],kv,vl);
        a2=__riscv_vfwmacc_vf_f32m2(a2,q[2*dk+d],kv,vl);
        a3=__riscv_vfwmacc_vf_f32m2(a3,q[3*dk+d],kv,vl);
        a4=__riscv_vfwmacc_vf_f32m2(a4,q[4*dk+d],kv,vl);
        a5=__riscv_vfwmacc_vf_f32m2(a5,q[5*dk+d],kv,vl);
        a6=__riscv_vfwmacc_vf_f32m2(a6,q[6*dk+d],kv,vl);
        a7=__riscv_vfwmacc_vf_f32m2(a7,q[7*dk+d],kv,vl);
    }
    __riscv_vse32_v_f32m2(dst,a0,vl);__riscv_vse32_v_f32m2(dst+stride,a1,vl);
    __riscv_vse32_v_f32m2(dst+2*stride,a2,vl);__riscv_vse32_v_f32m2(dst+3*stride,a3,vl);
    __riscv_vse32_v_f32m2(dst+4*stride,a4,vl);__riscv_vse32_v_f32m2(dst+5*stride,a5,vl);
    __riscv_vse32_v_f32m2(dst+6*stride,a6,vl);__riscv_vse32_v_f32m2(dst+7*stride,a7,vl);
}
static inline void ocr_pv_f16_x8(float *dst,const float *p,const _Float16 *v,int dv,int stride,int count){
    for(int off=0;off<dv;){
        const size_t vl=__riscv_vsetvl_e16m1(dv-off);
        auto a0=__riscv_vle32_v_f32m2(dst+off,vl),a1=__riscv_vle32_v_f32m2(dst+dv+off,vl);
        auto a2=__riscv_vle32_v_f32m2(dst+2*dv+off,vl),a3=__riscv_vle32_v_f32m2(dst+3*dv+off,vl);
        auto a4=__riscv_vle32_v_f32m2(dst+4*dv+off,vl),a5=__riscv_vle32_v_f32m2(dst+5*dv+off,vl);
        auto a6=__riscv_vle32_v_f32m2(dst+6*dv+off,vl),a7=__riscv_vle32_v_f32m2(dst+7*dv+off,vl);
        for(int t=0;t<count;++t){
            const auto vv=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+t*dv+off,vl),vl);
            a0=__riscv_vfmacc_vf_f32m2(a0,p[t],vv,vl);
            a1=__riscv_vfmacc_vf_f32m2(a1,p[stride+t],vv,vl);
            a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t],vv,vl);
            a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t],vv,vl);
            a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t],vv,vl);
            a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t],vv,vl);
            a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t],vv,vl);
            a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t],vv,vl);
        }
        __riscv_vse32_v_f32m2(dst+off,a0,vl);__riscv_vse32_v_f32m2(dst+dv+off,a1,vl);
        __riscv_vse32_v_f32m2(dst+2*dv+off,a2,vl);__riscv_vse32_v_f32m2(dst+3*dv+off,a3,vl);
        __riscv_vse32_v_f32m2(dst+4*dv+off,a4,vl);__riscv_vse32_v_f32m2(dst+5*dv+off,a5,vl);
        __riscv_vse32_v_f32m2(dst+6*dv+off,a6,vl);__riscv_vse32_v_f32m2(dst+7*dv+off,a7,vl);
        off+=int(vl);
    }
}

__attribute__((noinline)) static void pv_u2(float *__restrict__ dst,const float *__restrict__ p,const _Float16 *__restrict__ v,int dv,int stride,int count){
 for(int off=0;off<dv;){
  const size_t vl=__riscv_vsetvl_e16m1(dv-off);
  auto a0=__riscv_vle32_v_f32m2(dst+0*dv+off,vl);
  auto a1=__riscv_vle32_v_f32m2(dst+1*dv+off,vl);
  auto a2=__riscv_vle32_v_f32m2(dst+2*dv+off,vl);
  auto a3=__riscv_vle32_v_f32m2(dst+3*dv+off,vl);
  auto a4=__riscv_vle32_v_f32m2(dst+4*dv+off,vl);
  auto a5=__riscv_vle32_v_f32m2(dst+5*dv+off,vl);
  auto a6=__riscv_vle32_v_f32m2(dst+6*dv+off,vl);
  auto a7=__riscv_vle32_v_f32m2(dst+7*dv+off,vl);
  int t=0;for(;t+1<count;t+=2){
   const auto v0=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+(t+0)*dv+off,vl),vl);
   const auto v1=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+(t+1)*dv+off,vl),vl);
   a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t+0],v0,vl);
   a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t+0],v0,vl);
   a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t+0],v0,vl);
   a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t+0],v0,vl);
   a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t+0],v0,vl);
   a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t+0],v0,vl);
   a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t+0],v0,vl);
   a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t+0],v0,vl);
   a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t+1],v1,vl);
   a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t+1],v1,vl);
   a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t+1],v1,vl);
   a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t+1],v1,vl);
   a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t+1],v1,vl);
   a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t+1],v1,vl);
   a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t+1],v1,vl);
   a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t+1],v1,vl);
  }
  for(;t<count;++t){const auto vv=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+t*dv+off,vl),vl);
   a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t],vv,vl);
   a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t],vv,vl);
   a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t],vv,vl);
   a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t],vv,vl);
   a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t],vv,vl);
   a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t],vv,vl);
   a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t],vv,vl);
   a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t],vv,vl);
  }
  __riscv_vse32_v_f32m2(dst+0*dv+off,a0,vl);
  __riscv_vse32_v_f32m2(dst+1*dv+off,a1,vl);
  __riscv_vse32_v_f32m2(dst+2*dv+off,a2,vl);
  __riscv_vse32_v_f32m2(dst+3*dv+off,a3,vl);
  __riscv_vse32_v_f32m2(dst+4*dv+off,a4,vl);
  __riscv_vse32_v_f32m2(dst+5*dv+off,a5,vl);
  __riscv_vse32_v_f32m2(dst+6*dv+off,a6,vl);
  __riscv_vse32_v_f32m2(dst+7*dv+off,a7,vl);
  off+=vl;
 }
}

__attribute__((noinline)) static void pv_u4(float *__restrict__ dst,const float *__restrict__ p,const _Float16 *__restrict__ v,int dv,int stride,int count){
 for(int off=0;off<dv;){
  const size_t vl=__riscv_vsetvl_e16m1(dv-off);
  auto a0=__riscv_vle32_v_f32m2(dst+0*dv+off,vl);
  auto a1=__riscv_vle32_v_f32m2(dst+1*dv+off,vl);
  auto a2=__riscv_vle32_v_f32m2(dst+2*dv+off,vl);
  auto a3=__riscv_vle32_v_f32m2(dst+3*dv+off,vl);
  auto a4=__riscv_vle32_v_f32m2(dst+4*dv+off,vl);
  auto a5=__riscv_vle32_v_f32m2(dst+5*dv+off,vl);
  auto a6=__riscv_vle32_v_f32m2(dst+6*dv+off,vl);
  auto a7=__riscv_vle32_v_f32m2(dst+7*dv+off,vl);
  int t=0;for(;t+3<count;t+=4){
   const auto v0=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+(t+0)*dv+off,vl),vl);
   const auto v1=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+(t+1)*dv+off,vl),vl);
   const auto v2=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+(t+2)*dv+off,vl),vl);
   const auto v3=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+(t+3)*dv+off,vl),vl);
   a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t+0],v0,vl);
   a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t+0],v0,vl);
   a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t+0],v0,vl);
   a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t+0],v0,vl);
   a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t+0],v0,vl);
   a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t+0],v0,vl);
   a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t+0],v0,vl);
   a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t+0],v0,vl);
   a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t+1],v1,vl);
   a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t+1],v1,vl);
   a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t+1],v1,vl);
   a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t+1],v1,vl);
   a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t+1],v1,vl);
   a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t+1],v1,vl);
   a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t+1],v1,vl);
   a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t+1],v1,vl);
   a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t+2],v2,vl);
   a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t+2],v2,vl);
   a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t+2],v2,vl);
   a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t+2],v2,vl);
   a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t+2],v2,vl);
   a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t+2],v2,vl);
   a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t+2],v2,vl);
   a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t+2],v2,vl);
   a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t+3],v3,vl);
   a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t+3],v3,vl);
   a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t+3],v3,vl);
   a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t+3],v3,vl);
   a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t+3],v3,vl);
   a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t+3],v3,vl);
   a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t+3],v3,vl);
   a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t+3],v3,vl);
  }
  for(;t<count;++t){const auto vv=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+t*dv+off,vl),vl);
   a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t],vv,vl);
   a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t],vv,vl);
   a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t],vv,vl);
   a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t],vv,vl);
   a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t],vv,vl);
   a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t],vv,vl);
   a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t],vv,vl);
   a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t],vv,vl);
  }
  __riscv_vse32_v_f32m2(dst+0*dv+off,a0,vl);
  __riscv_vse32_v_f32m2(dst+1*dv+off,a1,vl);
  __riscv_vse32_v_f32m2(dst+2*dv+off,a2,vl);
  __riscv_vse32_v_f32m2(dst+3*dv+off,a3,vl);
  __riscv_vse32_v_f32m2(dst+4*dv+off,a4,vl);
  __riscv_vse32_v_f32m2(dst+5*dv+off,a5,vl);
  __riscv_vse32_v_f32m2(dst+6*dv+off,a6,vl);
  __riscv_vse32_v_f32m2(dst+7*dv+off,a7,vl);
  off+=vl;
 }
}
