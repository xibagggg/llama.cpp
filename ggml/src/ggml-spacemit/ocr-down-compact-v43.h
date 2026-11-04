#pragma once
// Reuse the unchanged v42 cell; only the TCM layout/tile width changes.
#include "ocr-down-sched-v42.h"
namespace ocr_down_compact_v43 {
using ocr_down_sched_v42::cell;
template<int TILE>
static __attribute__((noipa)) void compact_kernel(ggml::spacemit::context &ctx,ggml_tensor *op,float *cache){
 const auto *w=op->src[0],*x=op->src[1];
 const int64_t K=w->ne[0],M=w->ne[1],N=x->ne[1];
 const int64_t dr=(M+ctx.nth-1)/ctx.nth,r0=ctx.ith*dr,r1=std::min(r0+dr,M);
 const size_t used=size_t(2+TILE)*K*sizeof(float);
 GGML_ASSERT(ctx.shared.size>=used+64);
 auto *guard=(unsigned char*)cache+used;std::memset(guard,0xA7,64);
 float *activation=cache+2*K;
 for(int64_t n0=0;n0<N;n0+=TILE){
  const int64_t nn=std::min<int64_t>(TILE,N-n0);
  for(int64_t n=0;n<nn;++n)std::memcpy(activation+n*K,(const char*)x->data+(n0+n)*x->nb[1],K*4);
  for(int64_t m=r0;m<r1;m+=2){
   const int rows=int(std::min<int64_t>(2,r1-m));
   for(int row=0;row<rows;++row){
    const _Float16 *wp=(const _Float16*)((const char*)w->data+(m+row)*w->nb[1]);
    for(int64_t k=0;k<K;){
     const size_t vl=__riscv_vsetvl_e16m2(K-k);
     const vfloat16m2_t a=__riscv_vle16_v_f16m2(wp+k,vl);
     __riscv_vse32_v_f32m4(cache+row*K+k,__riscv_vfwcvt_f_f_v_f32m4(a,vl),vl);k+=vl;
    }
   }
   for(int64_t n=0;n<nn;n+=2){
    float *y0=(float*)((char*)op->data+(n0+n)*op->nb[1])+m;
    float *y1=n+1<nn?(float*)((char*)op->data+(n0+n+1)*op->nb[1])+m:nullptr;
    if(rows==2){if(y1)cell<2,2>(cache,activation+n*K,K,y0,y1);else cell<2,1>(cache,activation+n*K,K,y0,nullptr);}
    else {if(y1)cell<1,2>(cache,activation+n*K,K,y0,y1);else cell<1,1>(cache,activation+n*K,K,y0,nullptr);}
   }
  }
 }
 for(size_t i=0;i<64;++i)GGML_ASSERT(guard[i]==0xA7);
}
inline int tile(){
 static const int t=[] {const char *p=std::getenv("GGML_SPACEMIT_OCR_DOWN_COMPACT_TILE");
   if(!p)return 0;return std::strcmp(p,"16")==0?16:std::strcmp(p,"20")==0?20:0;}();return t;
}
inline bool forward(ggml::spacemit::context &ctx,ggml_tensor *op){
 const int t=tile();
 if(!t||ocr_down_sched_v42::schedule()!=2||ocr_down_tcm_v41::tile()!=16||
    ocr_down_tcm_v41::phase_directory()||!op||!op->src[0]||!op->src[1])return false;
 const auto *w=op->src[0],*x=op->src[1];
 const size_t used=size_t(2+t)*4304*4+64;
 if(w->type!=GGML_TYPE_F16||x->type!=GGML_TYPE_F32||op->type!=GGML_TYPE_F32||
    w->ne[0]!=4304||w->ne[1]!=1152||x->ne[1]<988||
    w->nb[0]!=2||x->nb[0]!=4||op->nb[0]!=4||
    w->ne[2]!=1||w->ne[3]!=1||x->ne[2]!=1||x->ne[3]!=1||op->ne[2]!=1||op->ne[3]!=1||
    !ctx.shared.data||reinterpret_cast<uintptr_t>(ctx.shared.data)%alignof(float)||ctx.shared.size<used)return false;
 auto *cache=static_cast<float*>(ctx.shared.data);
 if(t==20)compact_kernel<20>(ctx,op,cache);else compact_kernel<16>(ctx,op,cache);
 static std::atomic<bool> announced{false};
 if(ctx.ith==0&&!announced.exchange(true))GGML_LOG_INFO("ocr-down-compact-v43: active tile=%d scratch=%zu bytes per worker\n",t,used);
 return true;
}
}
