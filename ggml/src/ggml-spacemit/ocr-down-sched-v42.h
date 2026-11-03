#pragma once
// v42 experiments retain each output's original K-lane FMA/reduction order.
#include "ocr-down-tcm-v41.h"
namespace ocr_down_sched_v42 {
using ocr_down_tcm_v41::phase_record;
using ocr_down_tcm_v41::now_ns;
template<int TILE, bool PROFILE>
static __attribute__((noipa)) void relaxed_kernel(ggml::spacemit::context & ctx, ggml_tensor * op, float * cache) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    GGML_ASSERT(w->type == GGML_TYPE_F16 && x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
    GGML_ASSERT(w->nb[0] == sizeof(_Float16) && x->nb[0] == sizeof(float) && op->nb[0] == sizeof(float));
    GGML_ASSERT(w->ne[2] == 1 && w->ne[3] == 1 && x->ne[2] == 1 && x->ne[3] == 1);
    GGML_ASSERT(op->ne[2] == 1 && op->ne[3] == 1 && cache != nullptr);
    const int64_t K = w->ne[0], M = w->ne[1], N = x->ne[1];
    const int64_t dr = (M + ctx.nth - 1) / ctx.nth;
    const int64_t r0 = ctx.ith * dr, r1 = MIN(r0 + dr, M);
    phase_record p(ctx,op,TILE);
    if constexpr(PROFILE)p.total_start=now_ns();
    constexpr size_t guard_bytes=64;
    const size_t used=size_t(4+TILE)*K*sizeof(float);
    auto *guard=reinterpret_cast<unsigned char*>(cache)+used;
    std::memset(guard,0xA7,guard_bytes);
    float *activation=cache+4*K;
    for(int64_t n0=0;n0<N;n0+=(TILE?TILE:N)){
      const int64_t nn=TILE?std::min<int64_t>(TILE,N-n0):N;
      if constexpr(TILE){
        uint64_t start=0;if constexpr(PROFILE)start=now_ns();
        for(int64_t n=0;n<nn;++n)
          std::memcpy(activation+n*K,static_cast<const char*>(x->data)+(n0+n)*x->nb[1],K*sizeof(float));
        if constexpr(PROFILE)p.activation_ns+=now_ns()-start;
        p.activation_bytes+=size_t(nn)*K*sizeof(float);
      }
    int64_t m = r0;
    for (; m + 4 <= r1; m += 4) {
        uint64_t pack_start=0;if constexpr(PROFILE)pack_start=now_ns();
        for (int row = 0; row < 4; ++row) {
            const _Float16 * wp = reinterpret_cast<const _Float16 *>(
                static_cast<const char *>(w->data) + (m + row) * w->nb[1]);
            int64_t k = 0;
            while (k < K) {
                const size_t vl = __riscv_vsetvl_e16m2(K - k);
                const vfloat16m2_t w16 = __riscv_vle16_v_f16m2(wp + k, vl);
                const vfloat32m4_t w32 = __riscv_vfwcvt_f_f_v_f32m4(w16, vl);
                __riscv_vse32_v_f32m4(cache + row * K + k, w32, vl);
                k += vl;
            }
        }
        if constexpr(PROFILE)p.weight_ns+=now_ns()-pack_start;
        p.weight_input_bytes+=size_t(4)*K*sizeof(_Float16);
        uint64_t core_start=0;if constexpr(PROFILE)core_start=now_ns();
        for (int64_t n = 0; n < nn; ++n) {
            const float * xp;
            if constexpr(TILE)xp=activation+n*K;
            else xp=reinterpret_cast<const float *>(static_cast<const char *>(x->data) + (n0+n) * x->nb[1]);
            const size_t vlmax = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            int64_t k_left = K;
            while (k_left > 0) {
                const size_t vl = __riscv_vsetvl_e16m2(k_left);
                const int64_t k = K - k_left;
                const vfloat32m4_t x32 = __riscv_vle32_v_f32m4(xp, vl);
                vfloat32m4_t w32 = __riscv_vle32_v_f32m4(cache + k, vl);
                acc0 = __riscv_vfmacc_vv_f32m4_tu(acc0, w32, x32, vl);
                w32 = __riscv_vle32_v_f32m4(cache + K + k, vl);
                acc1 = __riscv_vfmacc_vv_f32m4_tu(acc1, w32, x32, vl);
                w32 = __riscv_vle32_v_f32m4(cache + 2 * K + k, vl);
                acc2 = __riscv_vfmacc_vv_f32m4_tu(acc2, w32, x32, vl);
                w32 = __riscv_vle32_v_f32m4(cache + 3 * K + k, vl);
                acc3 = __riscv_vfmacc_vv_f32m4_tu(acc3, w32, x32, vl);
                xp += vl;
                k_left -= vl;
            }
            const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            float * out = reinterpret_cast<float *>(static_cast<char *>(op->data) + (n0+n) * op->nb[1]);
            out[m + 0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc0, zero, vlmax));
            out[m + 1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc1, zero, vlmax));
            out[m + 2] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc2, zero, vlmax));
            out[m + 3] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc3, zero, vlmax));
        }
        if constexpr(PROFILE)p.core_ns+=now_ns()-core_start;
    }
    // Worker partitions need not be divisible by four. Tail rows stay local.
    for (; m < r1; ++m) {
        const _Float16 * wp = reinterpret_cast<const _Float16 *>(static_cast<const char *>(w->data) + m * w->nb[1]);
        uint64_t pack_start=0;if constexpr(PROFILE)pack_start=now_ns();
        for (int64_t k = 0; k < K;) {
            const size_t vl = __riscv_vsetvl_e16m2(K - k);
            const vfloat16m2_t w16 = __riscv_vle16_v_f16m2(wp + k, vl);
            const vfloat32m4_t w32 = __riscv_vfwcvt_f_f_v_f32m4(w16, vl);
            __riscv_vse32_v_f32m4(cache + k, w32, vl);
            k += vl;
        }
        if constexpr(PROFILE)p.weight_ns+=now_ns()-pack_start;
        p.weight_input_bytes+=size_t(1)*K*sizeof(_Float16);
        uint64_t core_start=0;if constexpr(PROFILE)core_start=now_ns();
        for (int64_t n = 0; n < nn; ++n) {
            const float * xp;
            if constexpr(TILE)xp=activation+n*K;
            else xp=reinterpret_cast<const float *>(static_cast<const char *>(x->data) + (n0+n) * x->nb[1]);
            const size_t vlmax = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            int64_t k_left = K;
            while (k_left > 0) {
                const size_t vl = __riscv_vsetvl_e16m2(k_left);
                const vfloat32m4_t w32 = __riscv_vle32_v_f32m4(cache + (K - k_left), vl);
                const vfloat32m4_t x32 = __riscv_vle32_v_f32m4(xp, vl);
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, w32, x32, vl);
                xp += vl;
                k_left -= vl;
            }
            const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            float * out = reinterpret_cast<float *>(static_cast<char *>(op->data) + (n0+n) * op->nb[1]);
            out[m] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc, zero, vlmax));
        }
        if constexpr(PROFILE)p.core_ns+=now_ns()-core_start;
    }
    }
    for(size_t i=0;i<guard_bytes;++i)GGML_ASSERT(guard[i]==0xA7);
    if constexpr(PROFILE){
      p.total_ns=now_ns()-p.total_start;
      const uint64_t start=now_ns();
      GGML_ASSERT(ctx.sync()==spert::Status::Ok);
      p.final_barrier_wait_ns=now_ns()-start;
      p.write();
    }
}

// At most four M4 accumulators, two activation vectors and one weight vector.
// Each 2x2 cell uses four vector loads for four FMAs, versus five for M4xN1.
template<int R, int C>
static __attribute__((always_inline)) inline void cell(
 const float *weights,const float *activations,int64_t K,float *y0,float *y1){
 const size_t vlmax=__riscv_vsetvlmax_e32m4();
 vfloat32m4_t a00=__riscv_vfmv_v_f_f32m4(0.f,vlmax);
 vfloat32m4_t a01=__riscv_vfmv_v_f_f32m4(0.f,vlmax);
 vfloat32m4_t a10=__riscv_vfmv_v_f_f32m4(0.f,vlmax);
 vfloat32m4_t a11=__riscv_vfmv_v_f_f32m4(0.f,vlmax);
 for(int64_t left=K;left>0;){
  const size_t vl=__riscv_vsetvl_e16m2(left);const int64_t k=K-left;
  const vfloat32m4_t x0=__riscv_vle32_v_f32m4(activations+k,vl);
  vfloat32m4_t w0=__riscv_vle32_v_f32m4(weights+k,vl);
  a00=__riscv_vfmacc_vv_f32m4_tu(a00,w0,x0,vl);
  if constexpr(C==2){
   const vfloat32m4_t x1=__riscv_vle32_v_f32m4(activations+K+k,vl);
   a01=__riscv_vfmacc_vv_f32m4_tu(a01,w0,x1,vl);
   if constexpr(R==2){
    w0=__riscv_vle32_v_f32m4(weights+K+k,vl);
    a10=__riscv_vfmacc_vv_f32m4_tu(a10,w0,x0,vl);
    a11=__riscv_vfmacc_vv_f32m4_tu(a11,w0,x1,vl);
   }
  }else if constexpr(R==2){
   w0=__riscv_vle32_v_f32m4(weights+K+k,vl);
   a10=__riscv_vfmacc_vv_f32m4_tu(a10,w0,x0,vl);
  }
  left-=vl;
 }
 const vfloat32m1_t zero=__riscv_vfmv_v_f_f32m1(0.f,1);
 y0[0]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a00,zero,vlmax));
 if constexpr(R==2)y0[1]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a10,zero,vlmax));
 if constexpr(C==2){
  y1[0]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a01,zero,vlmax));
  if constexpr(R==2)y1[1]=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(a11,zero,vlmax));
 }
}
template<int TILE>
static __attribute__((noipa)) void paired_kernel(ggml::spacemit::context &ctx,ggml_tensor *op,float *cache){
 const auto *w=op->src[0],*x=op->src[1];
 const int64_t K=w->ne[0],M=w->ne[1],N=x->ne[1];
 const int64_t dr=(M+ctx.nth-1)/ctx.nth,r0=ctx.ith*dr,r1=std::min(r0+dr,M);
 const size_t used=size_t(4+TILE)*K*sizeof(float);
 GGML_ASSERT(ctx.shared.size>=used+64);
 auto *guard=(unsigned char*)cache+used;std::memset(guard,0xA7,64);
 float *activation=cache+4*K;
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
inline int schedule(){
 static const int mode=[] {const char *p=std::getenv("GGML_SPACEMIT_OCR_DOWN_SCHEDULE");
  if(!p)return 0;if(std::strcmp(p,"paired")==0)return 2;
  return std::strcmp(p,"relaxed")==0?1:0;}();return mode;
}
inline bool forward(ggml::spacemit::context &ctx,ggml_tensor *op){
 const int mode=schedule();
 if(!mode||ocr_down_tcm_v41::tile()!=16||ocr_down_tcm_v41::phase_directory()||
    !op||!op->src[0]||!op->src[1])return false;
 const auto *w=op->src[0],*x=op->src[1];
 constexpr size_t used=size_t(20)*4304*4+64;
 if(w->type!=GGML_TYPE_F16||x->type!=GGML_TYPE_F32||op->type!=GGML_TYPE_F32||
    w->ne[0]!=4304||w->ne[1]!=1152||x->ne[1]<988||
    w->nb[0]!=2||x->nb[0]!=4||op->nb[0]!=4||
    w->ne[2]!=1||w->ne[3]!=1||x->ne[2]!=1||x->ne[3]!=1||op->ne[2]!=1||op->ne[3]!=1||
    !ctx.shared.data||reinterpret_cast<uintptr_t>(ctx.shared.data)%alignof(float)||ctx.shared.size<used)return false;
 auto *cache=static_cast<float*>(ctx.shared.data);
 if(mode==2)paired_kernel<16>(ctx,op,cache);else relaxed_kernel<16,false>(ctx,op,cache);
 static std::atomic<bool> announced{false};
 if(ctx.ith==0&&!announced.exchange(true))GGML_LOG_INFO("ocr-down-sched-v42: active mode=%s tile=16 scratch=%zu bytes per worker\n",mode==2?"paired":"relaxed",used);
 return true;
}

}
