#pragma once
#include "ocr-down-compact-v43.h"
namespace ocr_down_prefetch_v48 {
template<int R, int C, bool BULK>
static __attribute__((always_inline)) inline void cell(
 const float *weights,const float *activations,int64_t K,int64_t stride,float *y0,float *y1){
 const size_t vlmax=__riscv_vsetvlmax_e32m4();
 vfloat32m4_t a00=__riscv_vfmv_v_f_f32m4(0.f,vlmax);
 vfloat32m4_t a01=__riscv_vfmv_v_f_f32m4(0.f,vlmax);
 vfloat32m4_t a10=__riscv_vfmv_v_f_f32m4(0.f,vlmax);
 vfloat32m4_t a11=__riscv_vfmv_v_f_f32m4(0.f,vlmax);
 if constexpr(BULK){
  int64_t k=0;
  for(;k+int64_t(vlmax)<=K;k+=vlmax){
   const size_t vl=vlmax;
  const vfloat32m4_t x0=__riscv_vle32_v_f32m4(activations+k,vl);
  vfloat32m4_t w0=__riscv_vle32_v_f32m4(weights+k,vl);
  a00=__riscv_vfmacc_vv_f32m4_tu(a00,w0,x0,vl);
  if constexpr(C==2){
   const vfloat32m4_t x1=__riscv_vle32_v_f32m4(activations+stride+k,vl);
   a01=__riscv_vfmacc_vv_f32m4_tu(a01,w0,x1,vl);
   if constexpr(R==2){
    w0=__riscv_vle32_v_f32m4(weights+stride+k,vl);
    a10=__riscv_vfmacc_vv_f32m4_tu(a10,w0,x0,vl);
    a11=__riscv_vfmacc_vv_f32m4_tu(a11,w0,x1,vl);
   }
  }else if constexpr(R==2){
   w0=__riscv_vle32_v_f32m4(weights+stride+k,vl);
   a10=__riscv_vfmacc_vv_f32m4_tu(a10,w0,x0,vl);
  }

  }
  if(k<K){const size_t vl=size_t(K-k);
  const vfloat32m4_t x0=__riscv_vle32_v_f32m4(activations+k,vl);
  vfloat32m4_t w0=__riscv_vle32_v_f32m4(weights+k,vl);
  a00=__riscv_vfmacc_vv_f32m4_tu(a00,w0,x0,vl);
  if constexpr(C==2){
   const vfloat32m4_t x1=__riscv_vle32_v_f32m4(activations+stride+k,vl);
   a01=__riscv_vfmacc_vv_f32m4_tu(a01,w0,x1,vl);
   if constexpr(R==2){
    w0=__riscv_vle32_v_f32m4(weights+stride+k,vl);
    a10=__riscv_vfmacc_vv_f32m4_tu(a10,w0,x0,vl);
    a11=__riscv_vfmacc_vv_f32m4_tu(a11,w0,x1,vl);
   }
  }else if constexpr(R==2){
   w0=__riscv_vle32_v_f32m4(weights+stride+k,vl);
   a10=__riscv_vfmacc_vv_f32m4_tu(a10,w0,x0,vl);
  }

  }
 }else{
 for(int64_t left=K;left>0;){
  const size_t vl=BULK?std::min<size_t>(vlmax,left):__riscv_vsetvl_e32m4(left);const int64_t k=K-left;
  const vfloat32m4_t x0=__riscv_vle32_v_f32m4(activations+k,vl);
  vfloat32m4_t w0=__riscv_vle32_v_f32m4(weights+k,vl);
  a00=__riscv_vfmacc_vv_f32m4_tu(a00,w0,x0,vl);
  if constexpr(C==2){
   const vfloat32m4_t x1=__riscv_vle32_v_f32m4(activations+stride+k,vl);
   a01=__riscv_vfmacc_vv_f32m4_tu(a01,w0,x1,vl);
   if constexpr(R==2){
    w0=__riscv_vle32_v_f32m4(weights+stride+k,vl);
    a10=__riscv_vfmacc_vv_f32m4_tu(a10,w0,x0,vl);
    a11=__riscv_vfmacc_vv_f32m4_tu(a11,w0,x1,vl);
   }
  }else if constexpr(R==2){
   w0=__riscv_vle32_v_f32m4(weights+stride+k,vl);
   a10=__riscv_vfmacc_vv_f32m4_tu(a10,w0,x0,vl);
  }
  left-=vl;
 }
 }

 const vfloat32m1_t zero=__riscv_vfmv_v_f_f32m1(0.f,1);
 y0[0]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a00,zero,vlmax));
 if constexpr(R==2)y0[1]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a10,zero,vlmax));
 if constexpr(C==2){
  y1[0]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a01,zero,vlmax));
  if constexpr(R==2)y1[1]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a11,zero,vlmax));
 }
}
template<int MODE>
static __attribute__((noipa)) void kernel(ggml::spacemit::context &ctx,ggml_tensor *op,float *cache){
 const auto *w=op->src[0],*x=op->src[1];
 const int64_t K=w->ne[0],M=w->ne[1],N=x->ne[1];
 constexpr bool PAD=MODE>=5,PAIR=MODE==3||MODE==4||MODE==7;
 constexpr bool BULK=MODE==2||MODE==4||MODE==6;
 const int64_t stride=PAD?4352:K;
 constexpr int TILE=20;
 const int64_t dr=(M+ctx.nth-1)/ctx.nth,r0=ctx.ith*dr,r1=std::min(r0+dr,M);
 const size_t used=size_t(2+TILE)*stride*sizeof(float);
 GGML_ASSERT(ctx.shared.size>=used+64);
 auto *guard=(unsigned char*)cache+used;std::memset(guard,0xA7,64);
 float *activation=cache+2*stride;
 for(int64_t n0=0;n0<N;n0+=TILE){
  const int64_t nn=std::min<int64_t>(TILE,N-n0);
  for(int64_t n=0;n<nn;++n)std::memcpy(activation+n*stride,(const char*)x->data+(n0+n)*x->nb[1],K*4);
  for(int64_t m=r0;m<r1;m+=2){
   const int rows=int(std::min<int64_t>(2,r1-m));
   if constexpr(PAIR){
    for(int64_t k=0;k<K;){
     const size_t vl=__riscv_vsetvl_e16m2(K-k);
     const auto *p0=(const _Float16*)((const char*)w->data+m*w->nb[1]);
     const vfloat16m2_t a=__riscv_vle16_v_f16m2(p0+k,vl);
     const vfloat32m4_t a32=__riscv_vfwcvt_f_f_v_f32m4(a,vl);
     if(rows==2){
      const auto *p1=(const _Float16*)((const char*)w->data+(m+1)*w->nb[1]);
      const vfloat16m2_t b=__riscv_vle16_v_f16m2(p1+k,vl);
      const vfloat32m4_t b32=__riscv_vfwcvt_f_f_v_f32m4(b,vl);
      __riscv_vse32_v_f32m4(cache+stride+k,b32,vl);
     }
     __riscv_vse32_v_f32m4(cache+k,a32,vl);k+=vl;
    }
   }else{
    for(int row=0;row<rows;++row){
     const auto *wp=(const _Float16*)((const char*)w->data+(m+row)*w->nb[1]);
     for(int64_t k=0;k<K;){
      const size_t vl=__riscv_vsetvl_e16m2(K-k);
      if constexpr(MODE>=8){
       constexpr int DIST=MODE==8?2:(MODE==9?4:8);
       if(m+row+DIST<r1){
        const auto *next=(const _Float16*)((const char*)w->data+(m+row+DIST)*w->nb[1]);
        __builtin_prefetch(next+k,0,3);
       }
      }
      const vfloat16m2_t a=__riscv_vle16_v_f16m2(wp+k,vl);
      __riscv_vse32_v_f32m4(cache+row*stride+k,__riscv_vfwcvt_f_f_v_f32m4(a,vl),vl);k+=vl;
     }
    }
   }
   for(int64_t n=0;n<nn;n+=2){
    auto *y0=(float*)((char*)op->data+(n0+n)*op->nb[1])+m;
    auto *y1=n+1<nn?(float*)((char*)op->data+(n0+n+1)*op->nb[1])+m:nullptr;
    if(rows==2){if(y1)cell<2,2,BULK>(cache,activation+n*stride,K,stride,y0,y1);else cell<2,1,BULK>(cache,activation+n*stride,K,stride,y0,nullptr);}
    else{if(y1)cell<1,2,BULK>(cache,activation+n*stride,K,stride,y0,y1);else cell<1,1,BULK>(cache,activation+n*stride,K,stride,y0,nullptr);}
   }
  }
 }
 for(size_t i=0;i<64;++i)GGML_ASSERT(guard[i]==0xA7);
}
inline int mode(){
 static const int value=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_DOWN_VECTOR");
  return p&&p[0]>='1'&&p[0]<='7'&&!p[1]?int(p[0]-'0'):0;}();return value;
}
inline bool forward(ggml::spacemit::context &ctx,ggml_tensor*op){
 const int m=mode();
 if(!m||ocr_down_compact_v43::tile()!=20||ocr_down_sched_v42::schedule()!=2||ocr_down_tcm_v41::tile()!=16||
    ocr_down_tcm_v41::phase_directory()||!op||!op->src[0]||!op->src[1])return false;
 const auto*w=op->src[0],*x=op->src[1];const size_t need=size_t(22)*(m>=5?4352:4304)*4+64;
 if(w->type!=GGML_TYPE_F16||x->type!=GGML_TYPE_F32||op->type!=GGML_TYPE_F32||
    w->ne[0]!=4304||w->ne[1]!=1152||x->ne[1]<988||
    w->nb[0]!=2||x->nb[0]!=4||op->nb[0]!=4||w->ne[2]!=1||w->ne[3]!=1||
    x->ne[2]!=1||x->ne[3]!=1||op->ne[2]!=1||op->ne[3]!=1||
    !ctx.shared.data||reinterpret_cast<uintptr_t>(ctx.shared.data)%alignof(float)||ctx.shared.size<need)return false;
 auto*c=static_cast<float*>(ctx.shared.data);
 switch(m){case 1:kernel<1>(ctx,op,c);break;case 2:kernel<2>(ctx,op,c);break;case 3:kernel<3>(ctx,op,c);break;
 case 4:kernel<4>(ctx,op,c);break;case 5:kernel<5>(ctx,op,c);break;case 6:kernel<6>(ctx,op,c);break;case 7:kernel<7>(ctx,op,c);break;default:return false;}
 static std::atomic<bool>announced{false};
 if(ctx.ith==0&&!announced.exchange(true))GGML_LOG_INFO("ocr-down-vector-v44: active mode=%d scratch=%zu bytes per worker\n",m,need);
 return true;
}
}
