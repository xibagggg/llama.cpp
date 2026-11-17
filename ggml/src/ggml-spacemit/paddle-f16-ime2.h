#pragma once
// F16 vision weights packed once at load for K3 IME2. GGUF files stay intact.
// FP32 activations use a scaled high/residual split and FP32 accumulation.
// Enabled explicitly; all other shapes retain the existing RVV implementation.
#include "spacemit-context.h"
#include "spacemit-opnode.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

namespace paddle_f16_ime2 {
inline bool enabled() {
    static const bool value=[] {const char *p=std::getenv("GGML_SPACEMIT_PADDLE_F16_IME2");return p&&!std::strcmp(p,"1");}();
    return value;
}
// Diagnostic opt-in: number of this worker's W32 panels resident in TCM.
// K4304 and K4608 fall back to one panel because of TCM capacity.
inline int requested_panels() {
    static const int value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_PADDLE_F16_PANEL_TILES");
        return p?std::max(1,std::min(4,std::atoi(p))):1;
    }();
    return value;
}
// Diagnostic opt-in. M16 shares weight vector loads across two packed A8 groups.
// Each group retains its own high/residual scale and original K accumulation order.
inline bool m16_enabled() {
    static const bool value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_PADDLE_F16_M16");
        return p&&!std::strcmp(p,"1");
    }();
    return value;
}
// Independent opt-in for Paddle's two F16 patch-merge projector matrices.
// K4608 keeps M8/N32 with one shared high/residual A staging area.
inline bool projector_enabled() {
    static const bool value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_PADDLE_F16_PROJECTOR");
        return p&&!std::strcmp(p,"1");
    }();
    return value;
}
// Independent opt-in: K4304 uses compact K1024/N128/M16 staging.
// Each output continues its original FP32 K8 accumulator across K chunks.
inline bool kblock_enabled() {
    static const bool value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_PADDLE_F16_KBLOCK");
        return p&&!std::strcmp(p,"1024");
    }();
    return value;
}
inline bool kblock_selected(const ggml_tensor *op) {
    return kblock_enabled() && op && op->src[0] && op->src[0]->ne[0]==4304 &&
           op->ne[0]==1152;
}
inline size_t activation_workspace_bytes(const ggml_tensor *op) {
    const size_t groups=GGML_PAD(size_t(op->ne[1]),8)/8;
    return GGML_PAD(groups*8*size_t(op->src[0]->ne[0])*4,64)+GGML_PAD(groups*4,64);
}
inline size_t kblock_lane_bytes(const ggml_tensor *op) {
    // Four N32 panels, each with high and residual M8/N32 accumulators.
    const size_t groups=GGML_PAD(size_t(op->ne[1]),8)/8;
    return groups*4*256*sizeof(float)*2;
}
inline bool selected(const ggml_tensor *w) {
    if (!enabled() || !w || w->type != GGML_TYPE_F16 || !ggml_is_contiguous(w) ||
        w->ne[2] != 1 || w->ne[3] != 1) return false;
    const bool vision = !std::strncmp(w->name,"v.blk.",6) &&
        ((w->ne[0] == 1152 && (w->ne[1] == 1152 || w->ne[1] == 4304)) ||
         (w->ne[0] == 4304 && w->ne[1] == 1152));
    const bool projector = projector_enabled() && w->ne[0] == 4608 &&
        ((!std::strcmp(w->name,"mm.1.weight") && w->ne[1] == 4608) ||
         (!std::strcmp(w->name,"mm.2.weight") && w->ne[1] == 1024));
    static const auto info=spert::backend_info();
    // K4608: W32 294912B + shared A8 73728B + FP32 scratch/guards 2304B.
    const size_t minimum_shared = projector ? 370944 : 346624;
    return (vision || projector) && info.core_arch_id==0xa064 && info.vlen==128 &&
           info.shared_mem_size>=minimum_shared;
}
inline size_t nbytes(const ggml_tensor *w) {return 128+GGML_PAD(size_t(w->ne[1]),32)*size_t(w->ne[0])*2;}
inline float scale_for(float maximum) {
    if(maximum==0)return 1;
    GGML_ASSERT(std::isfinite(maximum));
    int exponent=0;std::frexp(maximum,&exponent);
    return std::ldexp(1.0f,std::max(-100,std::min(100,14-exponent)));
}
__attribute__((noinline)) static void kernel32(const _Float16 *x,const _Float16 *w,float *c,int K){
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
      :[x]"+&r"(x),[w]"+&r"(w),[w1]"+&r"(w1),[w2]"+&r"(w2),[w3]"+&r"(w3),[blocks]"+&r"(blocks)
      :[c]"r"(c)
      :"t0","v2","v8","v9","v10","v11","v16","v17","v18","v19","v20","v21","v22","v23","memory");
}
// Two existing M8 packs share W loads. FP32 output never narrows to F16.
__attribute__((noinline)) static void kernel16(
    const _Float16 *x0, const _Float16 *x1, const _Float16 *w,
    float *c0, float *c1, int K) {
    const _Float16 *w1=w+8*K, *w2=w+16*K, *w3=w+24*K;
    int blocks=K/8;
    asm volatile(
      "vsetvli t0,zero,e32,m8,ta,ma\n\t"
      "vmv.v.i v16,0\n\tvmv.v.i v24,0\n\t"
      "vsetvli t0,zero,e16,m1,ta,ma\n\t"
      "1:\n\t"
      "vle16.v v2,(%[x0])\n\tvle16.v v3,(%[x1])\n\t"
      "vle16.v v8,(%[w])\n\tvle16.v v9,(%[w1])\n\t"
      "vle16.v v10,(%[w2])\n\tvle16.v v11,(%[w3])\n\t"
      "smt.vfwmadot v16,v2,v8\n\tsmt.vfwmadot v24,v3,v8\n\t"
      "smt.vfwmadot v18,v2,v9\n\tsmt.vfwmadot v26,v3,v9\n\t"
      "smt.vfwmadot v20,v2,v10\n\tsmt.vfwmadot v28,v3,v10\n\t"
      "smt.vfwmadot v22,v2,v11\n\tsmt.vfwmadot v30,v3,v11\n\t"
      "addi %[x0],%[x0],128\n\taddi %[x1],%[x1],128\n\t"
      "addi %[w],%[w],128\n\taddi %[w1],%[w1],128\n\t"
      "addi %[w2],%[w2],128\n\taddi %[w3],%[w3],128\n\t"
      "addi %[blocks],%[blocks],-1\n\tbnez %[blocks],1b\n\t"
      "vsetvli t0,zero,e32,m2,ta,ma\n\t"
      "vse32.v v16,(%[c0])\n\taddi t0,%[c0],256\n\tvse32.v v18,(t0)\n\t"
      "addi t0,%[c0],512\n\tvse32.v v20,(t0)\n\t"
      "addi t0,%[c0],768\n\tvse32.v v22,(t0)\n\t"
      "vse32.v v24,(%[c1])\n\taddi t0,%[c1],256\n\tvse32.v v26,(t0)\n\t"
      "addi t0,%[c1],512\n\tvse32.v v28,(t0)\n\t"
      "addi t0,%[c1],768\n\tvse32.v v30,(t0)\n\t"
      : [x0]"+&r"(x0), [x1]"+&r"(x1), [w]"+&r"(w), [w1]"+&r"(w1),
        [w2]"+&r"(w2), [w3]"+&r"(w3), [blocks]"+&r"(blocks)
      : [c0]"r"(c0), [c1]"r"(c1)
      : "t0", "v2", "v3", "v8", "v9", "v10", "v11",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
}
// Chunked kernels reload existing FP32 accumulators instead of adding partial dot products.
__attribute__((noinline)) static void kernel32_continue(const _Float16 *x,const _Float16 *w,float *c,int K,int accumulate){
    const _Float16 *w1=w+8*K,*w2=w+16*K,*w3=w+24*K;int blocks=K/8;
    asm volatile(
      "vsetvli t0,zero,e32,m2,ta,ma\n\t"
      "beqz %[accumulate],2f\n\t"
      "vle32.v v16,(%[c])\n\taddi t0,%[c],256\n\tvle32.v v18,(t0)\n\t"
      "addi t0,%[c],512\n\tvle32.v v20,(t0)\n\t"
      "addi t0,%[c],768\n\tvle32.v v22,(t0)\n\tj 3f\n\t"
      "2:\n\tvmv.v.i v16,0\n\tvmv.v.i v18,0\n\tvmv.v.i v20,0\n\tvmv.v.i v22,0\n\t"
      "3:\n\t"
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
      :[x]"+&r"(x),[w]"+&r"(w),[w1]"+&r"(w1),[w2]"+&r"(w2),[w3]"+&r"(w3),[blocks]"+&r"(blocks)
      :[c]"r"(c),[accumulate]"r"(accumulate)
      :"t0","v2","v8","v9","v10","v11","v16","v17","v18","v19","v20","v21","v22","v23","memory");
}
__attribute__((noinline)) static void kernel16_continue(
    const _Float16 *x0, const _Float16 *x1, const _Float16 *w,
    float *c0, float *c1, int K, int accumulate) {
    const _Float16 *w1=w+8*K, *w2=w+16*K, *w3=w+24*K;
    int blocks=K/8;
    asm volatile(
      "beqz %[accumulate],2f\n\t"
      "vsetvli t0,zero,e32,m2,ta,ma\n\t"
      "vle32.v v16,(%[c0])\n\taddi t0,%[c0],256\n\tvle32.v v18,(t0)\n\t"
      "addi t0,%[c0],512\n\tvle32.v v20,(t0)\n\t"
      "addi t0,%[c0],768\n\tvle32.v v22,(t0)\n\t"
      "vle32.v v24,(%[c1])\n\taddi t0,%[c1],256\n\tvle32.v v26,(t0)\n\t"
      "addi t0,%[c1],512\n\tvle32.v v28,(t0)\n\t"
      "addi t0,%[c1],768\n\tvle32.v v30,(t0)\n\tj 3f\n\t"
      "2:\n\tvsetvli t0,zero,e32,m8,ta,ma\n\t"
      "vmv.v.i v16,0\n\tvmv.v.i v24,0\n\t"
      "3:\n\t"
      "vsetvli t0,zero,e16,m1,ta,ma\n\t"
      "1:\n\t"
      "vle16.v v2,(%[x0])\n\tvle16.v v3,(%[x1])\n\t"
      "vle16.v v8,(%[w])\n\tvle16.v v9,(%[w1])\n\t"
      "vle16.v v10,(%[w2])\n\tvle16.v v11,(%[w3])\n\t"
      "smt.vfwmadot v16,v2,v8\n\tsmt.vfwmadot v24,v3,v8\n\t"
      "smt.vfwmadot v18,v2,v9\n\tsmt.vfwmadot v26,v3,v9\n\t"
      "smt.vfwmadot v20,v2,v10\n\tsmt.vfwmadot v28,v3,v10\n\t"
      "smt.vfwmadot v22,v2,v11\n\tsmt.vfwmadot v30,v3,v11\n\t"
      "addi %[x0],%[x0],128\n\taddi %[x1],%[x1],128\n\t"
      "addi %[w],%[w],128\n\taddi %[w1],%[w1],128\n\t"
      "addi %[w2],%[w2],128\n\taddi %[w3],%[w3],128\n\t"
      "addi %[blocks],%[blocks],-1\n\tbnez %[blocks],1b\n\t"
      "vsetvli t0,zero,e32,m2,ta,ma\n\t"
      "vse32.v v16,(%[c0])\n\taddi t0,%[c0],256\n\tvse32.v v18,(t0)\n\t"
      "addi t0,%[c0],512\n\tvse32.v v20,(t0)\n\t"
      "addi t0,%[c0],768\n\tvse32.v v22,(t0)\n\t"
      "vse32.v v24,(%[c1])\n\taddi t0,%[c1],256\n\tvse32.v v26,(t0)\n\t"
      "addi t0,%[c1],512\n\tvse32.v v28,(t0)\n\t"
      "addi t0,%[c1],768\n\tvse32.v v30,(t0)\n\t"
      : [x0]"+&r"(x0), [x1]"+&r"(x1), [w]"+&r"(w), [w1]"+&r"(w1),
        [w2]"+&r"(w2), [w3]"+&r"(w3), [blocks]"+&r"(blocks)
      : [c0]"r"(c0), [c1]"r"(c1), [accumulate]"r"(accumulate)
      : "t0", "v2", "v3", "v8", "v9", "v10", "v11",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
}
// Diagnostic opt-in. The 64-wide path preserves the original per-value
// high/residual arithmetic and changes only vector width and packed stores.
inline bool wide_activation_pack() {
    static const bool value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH");
        return p&&!std::strcmp(p,"64");
    }();
    return value;
}
inline void pack_activation64(const float *x,_Float16 *px,_Float16 *pl,
                              float scale,int group,int M,int K) {
    // For lane i, half-index=(i/8)*64+i%8 in the existing K8/M8 layout.
    // Maximum byte index is 910, so u16 offsets are sufficient.
    const auto ids=__riscv_vid_v_u16m1(64);
    const auto offsets=__riscv_vsll_vx_u16m1(__riscv_vadd_vv_u16m1(
        __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(ids,3,64),6,64),
        __riscv_vand_vx_u16m1(ids,7,64),64),1,64);
    for(int r=0;r<8;++r)for(int k=0;k<K;k+=64) {
        const size_t vl=std::min(64,K-k);
        const size_t off=size_t(group)*8*K+size_t(k)*8+r*8;
        if(group*8+r>=M) {
            const auto zero=__riscv_vfmv_v_f_f16m1(0,vl);
            __riscv_vsuxei16_v_f16m1(px+off,offsets,zero,vl);
            __riscv_vsuxei16_v_f16m1(pl+off,offsets,zero,vl);
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
        __riscv_vsuxei16_v_f16m1(pl+off,offsets,lo,vl);
        __riscv_vsuxei16_v_f16m1(px+off,offsets,h,vl);
    }
}
inline bool supported(const ggml_tensor *op) {
    return op&&op->op==GGML_OP_MUL_MAT&&selected(op->src[0])&&op->src[1]&&
        op->src[1]->type==GGML_TYPE_F32&&op->type==GGML_TYPE_F32&&
        op->src[1]->ne[0]==op->src[0]->ne[0]&&op->src[1]->ne[2]==1&&op->src[1]->ne[3]==1&&
        ggml_is_contiguous(op->src[1])&&ggml_is_contiguous(op);
}
// Bias fusion is deliberately limited to the already selected, physically
// repacked F16 matrix path. The graph matcher additionally checks use/output/view
// ownership before it elides the intermediate matrix result.
inline bool bias_supported(const ggml_tensor *op,const ggml_tensor *add) {
    if(!supported(op)||!add||add->op!=GGML_OP_ADD||add->src[0]!=op||
       add->type!=GGML_TYPE_F32||!ggml_are_same_shape(op,add)||!ggml_is_contiguous(add))return false;
    const auto *bias=add->src[1];
    return bias&&bias->op==GGML_OP_NONE&&bias->src[0]==nullptr&&bias->type==GGML_TYPE_F32&&
        bias->ne[0]==op->ne[0]&&bias->ne[1]==1&&bias->ne[2]==1&&bias->ne[3]==1&&
        ggml_is_contiguous(bias)&&ggml_can_repeat(bias,op);
}
inline bool byte_ranges_overlap(const void *a,size_t an,const void *b,size_t bn) {
    const auto ap=reinterpret_cast<uintptr_t>(a),bp=reinterpret_cast<uintptr_t>(b);
    return ap<=bp?bp-ap<an:ap-bp<bn;
}
inline bool bias_storage_safe(const ggml_tensor *op,const ggml_tensor *add) {
    if(!bias_supported(op,add)||!add->data||!add->src[1]->data||!op->src[0]->data)return false;
    // Activations are fully packed before the existing all-worker barrier, so
    // their allocation may be reused by the destination. Bias and packed model
    // weights remain live throughout all stores and must never overlap it.
    return !byte_ranges_overlap(add->data,ggml_nbytes(add),add->src[1]->data,ggml_nbytes(add->src[1]))&&
        !byte_ranges_overlap(add->data,ggml_nbytes(add),op->src[0]->data,nbytes(op->src[0]));
}
inline void store_result_tile(float *out,const float *result,int M,int N,
                              int first_m,int first_n,const float *bias) {
    for(int o=0;o<4&&first_n+o*8<N;++o) {
        const size_t columns=size_t(std::min(8,N-first_n-o*8));
        for(int r=0;r<8&&first_m+r<M;++r) {
            auto *dst=out+size_t(first_m+r)*N+first_n+o*8;
            const auto *src=result+o*64+r*8;
            if(!bias) {
                std::memcpy(dst,src,columns*sizeof(float));
            } else {
                // Separate FP32 addition after the unchanged residual merge and
                // inverse scale. No FMA, narrower value or altered K reduction.
                const auto value=__riscv_vle32_v_f32m2(src,columns);
                const auto offset=__riscv_vle32_v_f32m2(bias+first_n+o*8,columns);
                __riscv_vse32_v_f32m2(dst,__riscv_vfadd_vv_f32m2(value,offset,columns),columns);
            }
        }
    }
}
struct traits final:ggml::spacemit::tensor_traits_base {
    bool work_size(int nth,const ggml_tensor *op,size_t &size)const override {
        if(!supported(op))return false;
        size=activation_workspace_bytes(op);
        if(kblock_selected(op))size+=size_t(nth)*kblock_lane_bytes(op);
        return true;
    }
    int repack(ggml_tensor *w,const void *data,size_t bytes)const override {
        GGML_ASSERT(selected(w)&&bytes==ggml_nbytes(w));
        const int K=int(w->ne[0]),N=int(w->ne[1]),NP=GGML_PAD(N,32);
        const auto *q=static_cast<const _Float16*>(data);
        float maximum=0;
        for(size_t i=0;i<bytes/2;++i) maximum=std::max(maximum,std::fabs(float(q[i])));
        const float scale=scale_for(maximum);
        std::memset(w->data,0,nbytes(w));std::memcpy(w->data,&scale,4);
        auto *p=reinterpret_cast<_Float16*>(static_cast<char*>(w->data)+128);
        for(int n=0;n<N;++n)for(int k=0;k<K;k+=8) {
            const auto f=__riscv_vfmul_vf_f32m2(
                __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(q+size_t(n)*K+k,8),8),scale,8);
            __riscv_vse16_v_f16m1(p+(n/8)*8*K+k*8+(n%8)*8,
                                 __riscv_vfncvt_f_f_w_f16m1(f,8),8);
        }
        GGML_LOG_DEBUG("paddle-f16-ime2: packed %s K=%d N=%d paddedN=%d scale=%g bytes=%zu\n",w->name,K,N,NP,scale,nbytes(w));
        return 0;
    }
    bool compute_forward(ggml::spacemit::context &ctx,ggml_tensor *op)const override {
        return compute(ctx,op,op,nullptr);
    }
    bool compute_forward_fused(ggml::spacemit::context &ctx,const spacemit_opnode &node)const override {
        if(node.opcode!=SPACEMIT_OP_PADDLE_F16_BIAS||node.fused.size()!=1||
           !node.extra_dsts.empty()||node.packed_im2col||node.elided_pad||
           !bias_storage_safe(node.node,node.fused[0])||node.node->src[0]->extra!=this)return false;
        return compute(ctx,node.node,node.fused[0],node.fused[0]->src[1]);
    }
    bool compute(ggml::spacemit::context &ctx,ggml_tensor *op,ggml_tensor *dst,
                 const ggml_tensor *bias)const {
        if(!supported(op))return false;
        const int K=int(op->src[0]->ne[0]),N=int(op->ne[0]),M=int(op->ne[1]),groups=(M+7)/8;
        constexpr int tile_n=32;
        size_t size=0;work_size(ctx.nth,op,size);
        GGML_ASSERT(ctx.workspace&&ctx.workspace_size>=size&&ctx.shared.data&&ctx.shared.size>=size_t((tile_n+8)*K*2+2304));
        auto *px=static_cast<_Float16*>(ctx.workspace);
        auto *pl=px+size_t(groups)*8*K;
        auto *scales=reinterpret_cast<float*>(static_cast<char*>(ctx.workspace)+GGML_PAD(size_t(groups)*8*K*4,64));
        const auto *x=static_cast<const float*>(op->src[1]->data);
        const auto *bias_data=bias?static_cast<const float*>(bias->data):nullptr;
        for(int g=int(ctx.ith);g<groups;g+=int(ctx.nth)) {
            auto vmax=__riscv_vfmv_v_f_f32m2(0,64);
            for(int r=0;r<8&&g*8+r<M;++r)for(int k=0;k<K;k+=64) {
                const size_t vl=std::min(64,K-k);
                const auto a=__riscv_vfabs_v_f32m2(__riscv_vle32_v_f32m2(x+size_t(g*8+r)*K+k,vl),vl);
                vmax=__riscv_vfmax_vv_f32m2_tu(vmax,vmax,a,vl);
            }
            const float maximum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(vmax,__riscv_vfmv_v_f_f32m1(0,1),64));
            const float scale=scale_for(maximum);scales[g]=scale;
            if(wide_activation_pack()) {
                pack_activation64(x,px,pl,scale,g,M,K);
                continue;
            }
            for(int k=0;k<K;k+=8)for(int r=0;r<8;++r) {
                const size_t off=size_t(g)*8*K+k*8+r*8;
                if(g*8+r>=M){std::memset(px+off,0,16);std::memset(pl+off,0,16);continue;}
                const auto a=__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(x+size_t(g*8+r)*K+k,8),scale,8);
                auto h=__riscv_vfncvt_f_f_w_f16m1(a,8);
                {
                    auto h32=__riscv_vfwcvt_f_f_v_f32m2(h,8);
                    const auto small=__riscv_vmflt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(h32,8),0x1p-14f,8);
                    h32=__riscv_vfmerge_vfm_f32m2(h32,0.0f,small,8);
                    h=__riscv_vfncvt_f_f_w_f16m1(h32,8);
                    const auto lo=__riscv_vfncvt_f_f_w_f16m1(__riscv_vfmul_vf_f32m2(__riscv_vfsub_vv_f32m2(a,h32,8),4096.0f,8),8);
                    __riscv_vse16_v_f16m1(pl+off,lo,8);
                }
                __riscv_vse16_v_f16m1(px+off,h,8);
            }
        }
        GGML_ASSERT(ctx.sync()==spert::Status::Ok);
        const auto *pw=reinterpret_cast<const _Float16*>(static_cast<const char*>(op->src[0]->data)+128);
        float wscale;std::memcpy(&wscale,op->src[0]->data,4);
        auto *base=static_cast<uint8_t*>(ctx.shared.data);
        if(kblock_selected(op)&&ctx.shared.size>=344320) {
            constexpr int block_k=1024, panel_count=4, panel_floats=512;
            constexpr size_t weight_bytes=size_t(panel_count)*tile_n*block_k*2;
            constexpr size_t activation_bytes=size_t(16)*block_k*2;
            constexpr size_t scratch_bytes=2*panel_count*panel_floats*sizeof(float)+256;
            static_assert(weight_bytes+2*activation_bytes+scratch_bytes==344320,
                          "K1024/N128/M16 TCM budget changed");
            auto *kw=reinterpret_cast<_Float16*>(base+128);
            auto *kx=kw+size_t(panel_count)*tile_n*block_k;
            auto *kl=kx+size_t(16)*block_k;
            auto *high=reinterpret_cast<float*>(kl+size_t(16)*block_k);
            auto *low=high+panel_count*panel_floats;
            std::memset(base,0xa5,128);std::memset(low+panel_count*panel_floats,0xa5,128);
            auto *partial=reinterpret_cast<float*>(
                static_cast<char*>(ctx.workspace)+activation_workspace_bytes(op)+
                size_t(ctx.ith)*kblock_lane_bytes(op));
            auto *partial_low=partial+size_t(panel_count)*groups*256;
            auto *out=static_cast<float*>(dst->data);
            const int ntiles=(N+tile_n-1)/tile_n;
            // Preserve the existing strided N32 ownership: with N1152/six
            // workers, each lane handles four panels then its final two.
            for(int first_nb=int(ctx.ith);first_nb<ntiles;first_nb+=panel_count*int(ctx.nth)) {
                const int active=std::min(panel_count,1+(ntiles-1-first_nb)/int(ctx.nth));
                for(int k0=0;k0<K;k0+=block_k) {
                    const int kb=std::min(block_k,K-k0); // final chunk is208.
                    GGML_ASSERT(kb>0&&kb%8==0);
                    for(int p=0;p<active;++p) {
                        const int nb=first_nb+p*int(ctx.nth);
                        // Existing packing is N8 x full-K. Slice each N8
                        // block independently and compact it to chunk-K.
                        for(int o=0;o<4;++o)
                            std::memcpy(kw+size_t(p*tile_n+o*8)*kb,
                                        pw+size_t(nb*tile_n+o*8)*K+size_t(k0)*8,
                                        size_t(8)*kb*2);
                    }
                    for(int g=0;g<groups;g+=2) {
                        const int batch_groups=std::min(2,groups-g);
                        for(int j=0;j<batch_groups;++j) {
                            std::memcpy(kx+size_t(j)*8*kb,
                                        px+size_t(g+j)*8*K+size_t(k0)*8,size_t(8)*kb*2);
                            std::memcpy(kl+size_t(j)*8*kb,
                                        pl+size_t(g+j)*8*K+size_t(k0)*8,size_t(8)*kb*2);
                        }
                        for(int p=0;p<active;++p) {
                            auto *ph=high+p*panel_floats,*lo=low+p*panel_floats;
                            auto *save=partial+(size_t(p)*groups+g)*256;
                            auto *save_low=partial_low+(size_t(p)*groups+g)*256;
                            if(k0) {
                                std::memcpy(ph,save,size_t(batch_groups)*256*sizeof(float));
                                std::memcpy(lo,save_low,size_t(batch_groups)*256*sizeof(float));
                            }
                            const auto *w=kw+size_t(p)*tile_n*kb;
                            if(batch_groups==2) {
                                kernel16_continue(kx,kx+8*kb,w,ph,ph+256,kb,k0!=0);
                                kernel16_continue(kl,kl+8*kb,w,lo,lo+256,kb,k0!=0);
                            } else {
                                kernel32_continue(kx,w,ph,kb,k0!=0);
                                kernel32_continue(kl,w,lo,kb,k0!=0);
                            }
                            if(k0+kb<K) {
                                std::memcpy(save,ph,size_t(batch_groups)*256*sizeof(float));
                                std::memcpy(save_low,lo,size_t(batch_groups)*256*sizeof(float));
                                continue;
                            }
                            const int first_n=(first_nb+p*int(ctx.nth))*tile_n;
                            for(int j=0;j<batch_groups;++j) {
                                auto *result=ph+j*256;
                                const auto *residual=lo+j*256;
                                for(int i=0;i<256;++i)result[i]+=residual[i]*(1.0f/4096);
                                const float inverse=1/(wscale*scales[g+j]);
                                for(int i=0;i<256;++i)result[i]*=inverse;
                                store_result_tile(out,result,M,N,(g+j)*8,first_n,bias_data);
                            }
                        }
                    }
                }
            }
            for(int i=0;i<128;++i)
                GGML_ASSERT(base[i]==0xa5&&reinterpret_cast<uint8_t*>(low+panel_count*panel_floats)[i]==0xa5);
            return true;
        }
        const size_t weight_bytes=size_t(tile_n)*K*2;
        const size_t activation_bytes=size_t(8)*K*2;
        // M16 is admitted only where both high and residual A16 staging plus
        // W32 and FP32 temporaries fit. K4304/K4608 retain M8 staging.
        const bool m16=m16_enabled()&&K==1152&&
            ctx.shared.size>=weight_bytes+4*activation_bytes+4352;
        const int staged_groups=m16?2:1;
        const int tile_floats=m16?512:256;
        const size_t staged_bytes=size_t(staged_groups)*activation_bytes;
        const size_t scratch_bytes=size_t(2)*tile_floats*sizeof(float)+256;
        const size_t fixed_bytes=2*staged_bytes+scratch_bytes;
        const int capacity=ctx.shared.size>fixed_bytes?int((ctx.shared.size-fixed_bytes)/weight_bytes):0;
        const int panels=std::max(1,std::min(requested_panels(),capacity));
        const bool resident_low=m16||panels>1;
        auto *sw=reinterpret_cast<_Float16*>(base+128),*sx=sw+size_t(panels)*tile_n*K;
        auto *sl=resident_low?sx+size_t(staged_groups)*8*K:sx;
        auto *temp=reinterpret_cast<float*>(sx+size_t(resident_low?2:1)*staged_groups*8*K),*low=temp+tile_floats;
        GGML_ASSERT(ctx.shared.size>=size_t(panels)*weight_bytes+(resident_low?2:1)*staged_bytes+scratch_bytes);
        std::memset(base,0xa5,128);std::memset(low+tile_floats,0xa5,128);
        auto *out=static_cast<float*>(dst->data);
        const int ntiles=(N+tile_n-1)/tile_n;
        // Group this worker's existing strided panels. This keeps the same
        // output ownership and avoids the load imbalance of global N96/N128.
        for(int first_nb=int(ctx.ith);first_nb<ntiles;first_nb+=panels*int(ctx.nth)) {
            const int active=std::min(panels,1+(ntiles-1-first_nb)/int(ctx.nth));
            for(int p=0;p<active;++p) {
                const int nb=first_nb+p*int(ctx.nth);
                std::memcpy(sw+size_t(p)*tile_n*K,pw+size_t(nb)*tile_n*K,weight_bytes);
            }
            for(int g=0;g<groups;) {
                // The final odd A8 group uses kernel32. Never read a second
                // group beyond the allocated packed high/residual workspace.
                const int batch_groups=m16&&g+1<groups?2:1;
                const size_t copy_bytes=size_t(batch_groups)*activation_bytes;
                std::memcpy(sx,px+size_t(g)*8*K,copy_bytes);
                if(resident_low)std::memcpy(sl,pl+size_t(g)*8*K,copy_bytes);
                for(int p=0;p<active;++p) {
                    const auto *weights=sw+size_t(p)*tile_n*K;
                    if(batch_groups==2)kernel16(sx,sx+8*K,weights,temp,temp+256,K);
                    else kernel32(sx,weights,temp,K);
                    if(!resident_low)std::memcpy(sl,pl+size_t(g)*8*K,copy_bytes);
                    if(batch_groups==2)kernel16(sl,sl+8*K,weights,low,low+256,K);
                    else kernel32(sl,weights,low,K);
                    const int first_n=(first_nb+p*int(ctx.nth))*tile_n;
                    for(int group=0;group<batch_groups;++group) {
                        auto *result=temp+group*256;
                        const auto *residual=low+group*256;
                        for(int i=0;i<256;++i)result[i]+=residual[i]*(1.0f/4096);
                        const float inverse=1/(wscale*scales[g+group]);
                        for(int i=0;i<256;++i)result[i]*=inverse;
                        store_result_tile(out,result,M,N,(g+group)*8,first_n,bias_data);
                    }
                }
                g+=batch_groups;
            }
        }
        for(int i=0;i<128;++i)GGML_ASSERT(base[i]==0xa5&&reinterpret_cast<uint8_t*>(low+tile_floats)[i]==0xa5);
        return true;
    }
};
inline const traits impl;
}
