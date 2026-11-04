#pragma once
#include "spacemit-context.h"
#include <riscv_vector.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
namespace ocr_concat_vector_v49 {
inline int mode(){static const int v=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_CONCAT_TCM");
 return p&&p[0]>='1'&&p[0]<='4'&&!p[1]?int(p[0]-'0'):0;}();return v;}
template<int C,int T,bool VECTOR=false>
inline bool kernel(ggml::spacemit::context&ctx,ggml_tensor*op,int64_t col0){
 const auto*s0=op->src[0],*s1=op->src[1];
 const size_t used=size_t(C)*T*4;
 if(!ctx.shared.data||reinterpret_cast<uintptr_t>(ctx.shared.data)%4||ctx.shared.size<used+64)return false;
 auto*scratch=(float*)ctx.shared.data;auto*guard=(unsigned char*)scratch+used;std::memset(guard,0xA7,64);
 const int64_t dc=(op->ne[1]+ctx.nth-1)/ctx.nth,c0=dc*ctx.ith,c1=std::min(c0+dc,op->ne[1]);
 for(int64_t i3=0;i3<op->ne[3];++i3)for(int64_t i2=0;i2<op->ne[2];++i2){
  const char*a=(const char*)s0->data+i2*s0->nb[2]+i3*s0->nb[3];
  const char*b=(const char*)s1->data+i2*s1->nb[2]+i3*s1->nb[3];
  char*d=(char*)op->data+i2*op->nb[2]+i3*op->nb[3];
  if(col0<s0->ne[0])for(int64_t c=c0;c<c1;++c)
   std::memcpy(d+c*op->nb[1]+col0*4,a+c*s0->nb[1]+col0*4,size_t(s0->ne[0]-col0)*4);
  for(int64_t c=c0;c<c1;c+=C){const int64_t cols=std::min<int64_t>(C,c1-c);
   for(int64_t t0=std::max<int64_t>(0,col0-s0->ne[0]);t0<s1->ne[0];t0+=T){
    const int64_t tokens=std::min<int64_t>(T,s1->ne[0]-t0);
    for(int64_t t=0;t<tokens;++t)for(int64_t off=0;off<cols;){
     const size_t vl=__riscv_vsetvl_e32m1(cols-off);
     const auto x=__riscv_vle32_v_f32m1((const float*)(b+(t0+t)*s1->nb[0]+(c+off)*s1->nb[1]),vl);
     __riscv_vsse32_v_f32m1(scratch+off*T+t,T*4,x,vl);off+=vl;
    }
    for(int64_t off=0;off<cols;++off){
     auto*dst=(float*)(d+(c+off)*op->nb[1]+(s0->ne[0]+t0)*4);
     if constexpr(VECTOR){
      for(int64_t i=0;i<tokens;){const size_t vl=__riscv_vsetvl_e32m8(tokens-i);
       const auto v=__riscv_vle32_v_f32m8(scratch+off*T+i,vl);
       __riscv_vse32_v_f32m8(dst+i,v,vl);i+=vl;}
     }else std::memcpy(dst,scratch+off*T,size_t(tokens)*4);
    }
   }
  }
 }
 for(size_t i=0;i<64;++i)GGML_ASSERT(guard[i]==0xA7);
 return true;
}
inline bool forward(ggml::spacemit::context&ctx,ggml_tensor*op,int64_t col0,int selected=-1){
 const int m=selected<0?mode():selected;
 if(!m||!op||!op->src[0]||!op->src[1]||col0<0||col0>op->ne[0])return false;
 const auto*a=op->src[0],*b=op->src[1];
 // v47: measured 1-token and trailing-three writes regress with TCM staging.
 // Keep original handling for partial writes and fewer than four new tokens.
 if(col0!=0||b->ne[0]<4)return false;
 if(op->type!=GGML_TYPE_F32||a->type!=GGML_TYPE_F32||b->type!=GGML_TYPE_F32||
    ggml_get_op_params_i32(op,0)!=0||op->nb[0]!=4||a->nb[0]!=4||b->nb[1]!=4||b->nb[0]==4||
    op->ne[0]!=a->ne[0]+b->ne[0]||op->nb[1]<size_t(op->ne[0])*4||
    a->ne[1]!=op->ne[1]||b->ne[1]!=op->ne[1]||a->ne[2]!=op->ne[2]||b->ne[2]!=op->ne[2]||
    a->ne[3]!=op->ne[3]||b->ne[3]!=op->ne[3])return false;
 // CONCAT materialization must not overwrite either source during the tile.
 const uintptr_t out=(uintptr_t)op->data,end=out+ggml_nbytes(op);
 for(const auto*src:{a,b}){const uintptr_t begin=(uintptr_t)src->data,stop=begin+ggml_nbytes(src);
  if(out<stop&&begin<end)return false;}
 bool ok=false;
 switch(m){case 1:ok=kernel<16,32>(ctx,op,col0);break;case 2:ok=kernel<32,64>(ctx,op,col0);break;
 case 3:ok=kernel<32,128>(ctx,op,col0);break;case 4:ok=kernel<64,64>(ctx,op,col0);break;case 5:ok=kernel<64,64,true>(ctx,op,col0);break;default:return false;}
 static std::atomic<bool>announced{false};
 if(ok&&selected<0&&ctx.ith==0&&!announced.exchange(true))GGML_LOG_INFO("ocr-concat-tcm-v46: active mode=%d bounded transpose and contiguous writeback\n",m);
 return ok;
}
}
