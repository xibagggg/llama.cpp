#pragma once

// Default-off F16xF16 patch embedding candidate. ggml conv2d lowers to
// src0=F16 image patches [K,M], src1=F16 constant weights [K,N], output F32
// [M,N]. Existing IME2 dense dispatch only admits F32 activations.
#include "spacemit-context.h"
#include "ggml-impl.h"
#include <algorithm>
#include <riscv_vector.h>
#include <cstdlib>
#include <cstring>

namespace paddle_f16_f16_patch {
constexpr int K = 588;
constexpr int KP = 592;
constexpr int N = 1152;
constexpr size_t scratch_bytes = 128 + size_t(32)*KP*2 + size_t(8)*KP*2 + size_t(32)*8*sizeof(float) + 256;

inline bool enabled() {
    static const bool value=[] { const char * p=std::getenv("GGML_SPACEMIT_PADDLE_F16_F16_PATCH"); return p && !std::strcmp(p,"1"); }();
    return value;
}
inline bool selected(const ggml_tensor * op) {
    if (!enabled() || !op || op->op != GGML_OP_MUL_MAT || !op->src[0] || !op->src[1]) return false;
    const auto * x=op->src[0], * w=op->src[1];
    static const auto info=spert::backend_info();
    return info.core_arch_id==0xa064 && info.vlen==128 && info.shared_mem_size>=scratch_bytes &&
        x->type==GGML_TYPE_F16 && w->type==GGML_TYPE_F16 && op->type==GGML_TYPE_F32 &&
        x->ne[0]==K && w->ne[0]==K && w->ne[1]==N && x->ne[1]>0 && x->ne[2]==1 && x->ne[3]==1 &&
        w->ne[2]==1 && w->ne[3]==1 && op->ne[0]==x->ne[1] && op->ne[1]==N &&
        op->ne[2]==1 && op->ne[3]==1 &&
        ggml_is_contiguous(x) && ggml_is_contiguous(w) && ggml_is_contiguous(op);
}
inline size_t work_bytes(const ggml_tensor *op) { return size_t((op->src[0]->ne[1]+7)/8)*8*KP*2; }

// Same K8 order and F32 accumulators as the existing Paddle F16 IME2 kernel.
// W is 32 output rows; X is 8 input rows. K588 is zero-padded to K592.
__attribute__((noinline)) static void kernel32(const _Float16 * x,const _Float16 * w,float * c) {
    const _Float16 *w1=w+8*KP,*w2=w+16*KP,*w3=w+24*KP; int blocks=KP/8;
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

inline bool forward(ggml::spacemit::context & ctx,ggml_tensor * op) {
    if (!selected(op)) return false;
    GGML_ASSERT(ctx.nth>0 && ctx.ith<ctx.nth && ctx.shared.data && ctx.shared.size>=scratch_bytes);
    GGML_ASSERT(ctx.workspace && ctx.workspace_size>=work_bytes(op));
    const auto * x=op->src[0], * w=op->src[1];
    const int rows=int(x->ne[1]), cols=int(w->ne[1]);
    auto * base=static_cast<unsigned char *>(ctx.shared.data);
    auto * wb=reinterpret_cast<_Float16 *>(base+128);
    auto * xb=wb+size_t(32)*KP;
    auto * acc=reinterpret_cast<float *>(xb+size_t(8)*KP);
    auto * dst=static_cast<float *>(op->data);
    const auto * xp=static_cast<const _Float16 *>(x->data);
    const auto * wp=static_cast<const _Float16 *>(w->data);
    const int tiles_n=(cols+31)/32, tiles_m=(rows+7)/8;
    auto * packed_x=static_cast<_Float16 *>(ctx.workspace);
    // Pack each dynamic patch once, then reuse its K8/M8 layout for all
    // output panels. The all-worker barrier publishes every packed group.
    for(int g=int(ctx.ith);g<tiles_m;g+=int(ctx.nth)) {
        auto * target=packed_x+size_t(g)*8*KP;
        std::memset(target,0,size_t(8)*KP*2);
        for(int k=0;k<K;k+=8)for(int r=0;r<8&&g*8+r<rows;++r)
            std::memcpy(target+size_t(k)*8+r*8,xp+size_t(g*8+r)*K+k,size_t(std::min(8,K-k))*2);
    }
    GGML_ASSERT(ctx.sync()==spert::Status::Ok);
    std::memset(base,0xa5,128);
    std::memset(acc+256,0xa5,256);
    for (int tile=int(ctx.ith); tile<tiles_n; tile+=int(ctx.nth)) {
        const int n0=tile*32, nc=std::min(32,cols-n0);
        std::memset(wb,0,size_t(32)*KP*2);
        for(int n=0;n<nc;++n)for(int k=0;k<K;k+=8)
            std::memcpy(wb+size_t(n/8)*8*KP+size_t(k)*8+(n%8)*8,
                        wp+size_t(n0+n)*K+k,size_t(std::min(8,K-k))*2);
        for(int g=0;g<tiles_m;++g) {
        const int m0=g*8,mc=std::min(8,rows-m0);
        std::memcpy(xb,packed_x+size_t(g)*8*KP,size_t(8)*KP*2);
        kernel32(xb,wb,acc);
        for (int n=0;n<nc;++n) for (int m=0;m<mc;++m)
            dst[size_t(n0+n)*rows+m0+m]=acc[(n/8)*64+size_t(m)*8+n%8];
        }
    }
    for (int i=0;i<128;++i) GGML_ASSERT(base[i]==0xa5);
    for (int i=0;i<256;++i) GGML_ASSERT(reinterpret_cast<unsigned char *>(acc+256)[i]==0xa5);
    return true;
}
}
