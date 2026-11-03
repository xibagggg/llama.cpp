#pragma once
#include "ggml.h"
#include "ggml-impl.h"
#include "spacemit-context.h"
#include <riscv_vector.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <cstdio>
namespace ocr_ime2_fp16_v35 {
constexpr int K=4304,M=1152;
constexpr size_t scratch_bytes=346624;
inline bool normalized(){
    static const bool value=[] {const char *v=std::getenv("GGML_SPACEMIT_IME2_FP16_NORMALIZE");return v&&std::strcmp(v,"1")==0;}();
    return value;
}
// Explicit experimental per-layer selection. Unset retains the previous path.
inline bool selected_layer(const ggml_tensor *w){
    static const uint32_t mask=[] {
        const char *text=std::getenv("GGML_SPACEMIT_IME2_FP16_DOWN_LAYERS");
        if(!text)return (uint32_t(1)<<27)-1;
        if(std::strcmp(text,"none")==0)return uint32_t(0);
        uint32_t result=0;
        for(const char *p=text;*p;){
            char *end=nullptr;long layer=std::strtol(p,&end,10);
            if(end==p||layer<0||layer>=27||(*end&&*end!=','))GGML_ABORT("Invalid OCR down layer list");
            result|=uint32_t(1)<<layer;
            if(*end==','&&!end[1])GGML_ABORT("Trailing comma in OCR down layer list");
            p=*end?end+1:end;
        }
        if(!*text)GGML_ABORT("Empty OCR down layer list; use none");
        return result;
    }();
    int layer=-1;char suffix=0;
    return std::sscanf(w->name,"v.blk.%d.ffn_down.weight%c",&layer,&suffix)==1&&
           layer>=0&&layer<27&&(mask&(uint32_t(1)<<layer));
}
inline bool eligible(const ggml_tensor *op) {
    static const bool enabled=[] {const char *v=std::getenv("GGML_SPACEMIT_IME2_FP16_DOWN");return v&&std::strcmp(v,"1")==0;}();
    if(!enabled||!op||op->op!=GGML_OP_MUL_MAT||!op->src[0]||!op->src[1])return false;
    const auto *w=op->src[0],*x=op->src[1];
    if(w->type!=GGML_TYPE_F16||x->type!=GGML_TYPE_F32||op->type!=GGML_TYPE_F32||
       w->ne[0]!=K||w->ne[1]!=M||w->ne[2]!=1||w->ne[3]!=1||
       x->ne[0]!=K||x->ne[1]<1||x->ne[2]!=1||x->ne[3]!=1||
       !ggml_is_contiguous(w)||!ggml_is_contiguous(x)||!ggml_is_contiguous(op)||
       std::strncmp(w->name,"v.blk.",6)!=0||std::strstr(w->name,".ffn_down.weight")==nullptr||!selected_layer(w))return false;
    static const auto info=spert::backend_info();
    return info.core_arch_id==0xa064&&info.vlen==128&&info.shared_mem_size>=scratch_bytes;
}
inline bool compensated(){
    static const bool value=[] {const char *v=std::getenv("GGML_SPACEMIT_IME2_FP16_COMPENSATE");return v&&std::strcmp(v,"1")==0;}();
    return value;
}
inline size_t padded_n(const ggml_tensor *op){return (size_t(op->src[1]->ne[1])+7)/8*8;}
inline size_t work_bytes(const ggml_tensor *op){return size_t(K)*(M+padded_n(op)*(compensated()?2:1))*sizeof(_Float16);}
// Four independent 8x8 accumulators; packed A is tokens, B is output rows.
__attribute__((noinline)) static void kernel32(const _Float16 *x,const _Float16 *w,float *c){
    const _Float16 *w1=w+8*K,*w2=w+16*K,*w3=w+24*K;int blocks=K/8;
    asm volatile(
      "vsetvli t0,zero,e32,m2,ta,ma\n\t"
      "vmv.v.i v16,0\n\tvmv.v.i v18,0\n\tvmv.v.i v20,0\n\tvmv.v.i v22,0\n\t"
      "vsetvli t0,zero,e16,m1,ta,ma\n\t"
      "1:\n\t"
      "vle16.v v2,(%[x])\n\tvle16.v v8,(%[w])\n\t"
      "vle16.v v9,(%[w1])\n\tvle16.v v10,(%[w2])\n\tvle16.v v11,(%[w3])\n\t"
      "smt.vfwmadot v16,v2,v8\n\tsmt.vfwmadot v18,v2,v9\n\t"
      "smt.vfwmadot v20,v2,v10\n\tsmt.vfwmadot v22,v2,v11\n\t"
      "addi %[x],%[x],128\n\taddi %[w],%[w],128\n\taddi %[w1],%[w1],128\n\t"
      "addi %[w2],%[w2],128\n\taddi %[w3],%[w3],128\n\t"
      "addi %[blocks],%[blocks],-1\n\tbnez %[blocks],1b\n\t"
      "vsetvli t0,zero,e32,m2,ta,ma\n\tvse32.v v16,(%[c])\n\t"
      "addi t0,%[c],256\n\tvse32.v v18,(t0)\n\t"
      "addi t0,%[c],512\n\tvse32.v v20,(t0)\n\t"
      "addi t0,%[c],768\n\tvse32.v v22,(t0)\n\t"
      : [x]"+&r"(x),[w]"+&r"(w),[w1]"+&r"(w1),[w2]"+&r"(w2),[w3]"+&r"(w3),[blocks]"+&r"(blocks)
      : [c]"r"(c)
      : "t0","v2","v8","v9","v10","v11","v16","v17","v18","v19","v20","v21","v22","v23","memory");
}
static void pack_w(const _Float16 *w,_Float16 *p,int group){
    for(int k=0;k<K;k+=8)for(int r=0;r<8;++r){
        auto *dst=p+group*8*K+k*8+r*8;
        const auto *src=w+(group*8+r)*K+k;
        if(!normalized()){std::memcpy(dst,src,16);continue;}
        auto a=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(src,8),8);
        // Every nonzero FP16 value becomes normal after multiplying by 2^10.
        // This opt-in kernel is bounded to weights <= 63 to avoid overflow.
        GGML_ASSERT(__riscv_vcpop_m_b16(__riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(a,8),63.0f,8),8)==0);
        a=__riscv_vfmul_vf_f32m2(a,1024.0f,8);
        __riscv_vse16_v_f16m1(dst,__riscv_vfncvt_f_f_w_f16m1(a,8),8);
    }
}
// Packing pads only physical token storage. Logical dimensions stay intact.
// high = fp16(x), low = fp16(x - fp32(high)); two independent FP32
// matrix products are summed after reduction, preserving small corrections.
static void pack_x(const float *x,_Float16 *hi,_Float16 *lo,int group,int N){
    for(int k=0;k<K;k+=8)for(int r=0;r<8;++r){
        const size_t offset=size_t(group)*8*K+k*8+r*8;
        if(group*8+r>=N){
            std::memset(hi+offset,0,16);
            if(lo)std::memset(lo+offset,0,16);
            continue;
        }
        const auto a=__riscv_vle32_v_f32m2(x+size_t(group*8+r)*K+k,8);
        auto h=__riscv_vfncvt_f_f_w_f16m1(a,8);
        if(normalized()){
            auto h32=__riscv_vfwcvt_f_f_v_f32m2(h,8);
            const auto small=__riscv_vmflt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(h32,8),0x1p-14f,8);
            // Match the effective IME high input, then place the entire small
            // value in the scaled residual rather than subtracting a lost high.
            h32=__riscv_vfmerge_vfm_f32m2(h32,0.0f,small,8);
            h=__riscv_vfncvt_f_f_w_f16m1(h32,8);
        }
        __riscv_vse16_v_f16m1(hi+offset,h,8);
        if(lo){
            const auto h32=__riscv_vfwcvt_f_f_v_f32m2(h,8);
            const auto residual=__riscv_vfsub_vv_f32m2(a,h32,8);
            // Scale residuals before narrowing so IME FP16 subnormal handling
            // cannot erase the correction. Undo the exact power-of-two later.
            const auto scaled=__riscv_vfmul_vf_f32m2(residual,4096.0f,8);
            const auto l=__riscv_vfncvt_f_f_w_f16m1(scaled,8);
            __riscv_vse16_v_f16m1(lo+offset,l,8);
        }
    }
}

inline bool forward(ggml::spacemit::context &ctx,ggml_tensor *op){
    if(!eligible(op))return false;
    GGML_ASSERT(ctx.shared.data&&ctx.shared.size>=scratch_bytes&&ctx.workspace&&ctx.workspace_size>=work_bytes(op));
    GGML_ASSERT(reinterpret_cast<uintptr_t>(ctx.shared.data)%128==0);
    const int N=int(op->src[1]->ne[1]),rank=int(ctx.ith),workers=int(ctx.nth);
    const int groups=(N+7)/8;
    const bool split=compensated();
    GGML_ASSERT(!normalized()||split);
    const auto *w=static_cast<const _Float16*>(op->src[0]->data);
    const auto *x=static_cast<const float*>(op->src[1]->data);
    auto *pw=static_cast<_Float16*>(ctx.workspace),*px=pw+M*K;
    auto *pl=split?px+size_t(groups)*8*K:nullptr;
    auto *base=static_cast<unsigned char*>(ctx.shared.data);
    auto *sw=reinterpret_cast<_Float16*>(base+128),*sx=sw+32*K;
    auto *temp=reinterpret_cast<float*>(sx+8*K);
    auto *low=temp+256;
    // Diagnostics keep canaries around worker-local TCM scratch.
    std::memset(base,0xa5,128);std::memset(reinterpret_cast<unsigned char*>(temp)+2048,0xa5,128);
    for(int g=rank;g<M/8;g+=workers)pack_w(w,pw,g);
    for(int g=rank;g<groups;g+=workers)pack_x(x,px,pl,g,N);
    GGML_ASSERT(ctx.sync()==spert::Status::Ok);
    auto *out=static_cast<float*>(op->data);
    for(int mb=rank;mb<M/32;mb+=workers){
        std::memcpy(sw,pw+mb*32*K,32*K*2);
        for(int nb=0;nb<groups;++nb){
            std::memcpy(sx,px+size_t(nb)*8*K,8*K*2);kernel32(sx,sw,temp);
            if(split){
                std::memcpy(sx,pl+size_t(nb)*8*K,8*K*2);kernel32(sx,sw,low);
                for(int i=0;i<256;++i)temp[i]+=low[i]*(1.0f/4096.0f);
            }
            if(normalized())for(int i=0;i<256;++i)temp[i]*=(1.0f/1024.0f);
            for(int g=0;g<4;++g)for(int n=0;n<8&&nb*8+n<N;++n)
                std::memcpy(out+(nb*8+n)*M+mb*32+g*8,temp+g*64+n*8,32);
        }
    }
    for(int i=0;i<128;++i)GGML_ASSERT(base[i]==0xa5&&reinterpret_cast<unsigned char*>(temp)[2048+i]==0xa5);
    static std::atomic<bool> announced{false};
    if(rank==0&&!announced.exchange(true))GGML_LOG_INFO("ocr-ime2-fp16-v35: active K=%d M=%d N=%d workers=%d scratch=%zu compensate=%d normalize=%d\n",K,M,N,workers,scratch_bytes,int(split),int(normalized()));
    return true;
}
} // namespace ocr_ime2_fp16_v35
