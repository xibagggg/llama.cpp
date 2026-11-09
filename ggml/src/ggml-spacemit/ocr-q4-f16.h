#pragma once
// Experimental Q4 file -> persistent packed FP16 weights at model load.
// This changes runtime arithmetic and resident memory, not the GGUF file.
// No address-based activation cache; inputs are packed once per graph call.
#include "spacemit-context.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

namespace ocr_q4_f16 {
inline int mode(){static const int m=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_F16");return p?std::atoi(p):0;}();return m;}
inline bool split(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_F16_X2");return p&&std::strcmp(p,"1")==0;}();return x;}
inline int output_tiles(int k) {
    static const int n=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_F16_NTILE");const int x=p?std::atoi(p):1;return x==2||x==4?x:1;}();
    return k==1152?n:1;
}
inline bool selected(const ggml_tensor * w) {
    if(mode()<1||mode()>2||!w||w->type!=GGML_TYPE_Q4_0||!ggml_is_contiguous(w)||w->ne[2]!=1||w->ne[3]!=1)return false;
    if(std::strncmp(w->name,"v.blk.",6))return false;
    const bool shape=(w->ne[0]==1152&&(w->ne[1]==1152||w->ne[1]==4304))||
        (mode()==2&&w->ne[0]==4352&&w->ne[1]==1152);
    static const auto info=spert::backend_info();
    return shape&&info.core_arch_id==0xa064&&info.vlen==128&&info.shared_mem_size>=350464;
}
inline size_t nbytes(const ggml_tensor * w){return 128+GGML_PAD(size_t(w->ne[1]),32)*size_t(w->ne[0])*2;}
inline float scale_for(float maximum) {
    if(maximum==0)return 1;
    GGML_ASSERT(std::isfinite(maximum));
    int exponent=0;std::frexp(maximum,&exponent);
    return std::ldexp(1.0f,std::max(-100,std::min(100,14-exponent)));
}
template<int K> __attribute__((noinline)) static void kernel32(const _Float16 *x,const _Float16 *w,float *c){
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
inline bool supported(const ggml_tensor *op) {
    return op&&op->op==GGML_OP_MUL_MAT&&selected(op->src[0])&&op->src[1]&&
        op->src[1]->type==GGML_TYPE_F32&&op->type==GGML_TYPE_F32&&
        op->src[1]->ne[0]==op->src[0]->ne[0]&&op->src[1]->ne[2]==1&&op->src[1]->ne[3]==1&&
        ggml_is_contiguous(op->src[1])&&ggml_is_contiguous(op);
}
struct traits final:ggml::spacemit::tensor_traits_base {
    bool work_size(int,const ggml_tensor *op,size_t &size)const override {
        if(!supported(op))return false;
        const size_t groups=GGML_PAD(size_t(op->ne[1]),8)/8;
        size=GGML_PAD(groups*8*size_t(op->src[0]->ne[0])*2*(split()?2:1),64)+GGML_PAD(groups*4,64);
        return true;
    }
    int repack(ggml_tensor *w,const void *data,size_t bytes)const override {
        GGML_ASSERT(selected(w)&&bytes==ggml_nbytes(w));
        const int K=int(w->ne[0]),N=int(w->ne[1]),NP=GGML_PAD(N,32);
        const auto *q=static_cast<const block_q4_0*>(data);
        float maximum=0;
        for(size_t i=0;i<bytes/sizeof(block_q4_0);++i)
            maximum=std::max(maximum,std::fabs(ggml_fp16_to_fp32(q[i].d))*8);
        const float scale=scale_for(maximum);
        std::memset(w->data,0,nbytes(w));std::memcpy(w->data,&scale,4);
        auto *p=reinterpret_cast<_Float16*>(static_cast<char*>(w->data)+128);
        for(int n=0;n<N;++n)for(int k=0;k<K;k+=32) {
            const auto &b=q[size_t(n)*(K/32)+k/32];
            const float d=ggml_fp16_to_fp32(b.d)*scale;
            for(int j=0;j<32;++j) {
                const int v=(j<16?b.qs[j]&15:b.qs[j-16]>>4)-8;
                p[(n/8)*8*K+((k+j)/8)*64+(n%8)*8+j%8]=_Float16(d*v);
            }
        }
        GGML_LOG_DEBUG("ocr-q4-f16: packed %s K=%d N=%d paddedN=%d scale=%g bytes=%zu\n",w->name,K,N,NP,scale,nbytes(w));
        return 0;
    }
    bool compute_forward(ggml::spacemit::context &ctx,ggml_tensor *op)const override {
        if(!supported(op))return false;
        const int K=int(op->src[0]->ne[0]),N=int(op->ne[0]),M=int(op->ne[1]),groups=(M+7)/8;
        const int tiles=output_tiles(K),tile_n=32*tiles;
        const bool x2=split();size_t size=0;work_size(ctx.nth,op,size);
        GGML_ASSERT(ctx.workspace&&ctx.workspace_size>=size&&ctx.shared.data&&ctx.shared.size>=size_t((tile_n+8*(x2&&tiles>1?2:1))*K*2+2304));
        auto *px=static_cast<_Float16*>(ctx.workspace);
        auto *pl=x2?px+size_t(groups)*8*K:nullptr;
        auto *scales=reinterpret_cast<float*>(static_cast<char*>(ctx.workspace)+GGML_PAD(size_t(groups)*8*K*2*(x2?2:1),64));
        const auto *x=static_cast<const float*>(op->src[1]->data);
        for(int g=int(ctx.ith);g<groups;g+=int(ctx.nth)) {
            auto vmax=__riscv_vfmv_v_f_f32m2(0,64);
            for(int r=0;r<8&&g*8+r<M;++r)for(int k=0;k<K;k+=64) {
                const size_t vl=std::min(64,K-k);
                const auto a=__riscv_vfabs_v_f32m2(__riscv_vle32_v_f32m2(x+size_t(g*8+r)*K+k,vl),vl);
                vmax=__riscv_vfmax_vv_f32m2_tu(vmax,vmax,a,vl);
            }
            const float maximum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(vmax,__riscv_vfmv_v_f_f32m1(0,1),64));
            const float scale=scale_for(maximum);scales[g]=scale;
            for(int k=0;k<K;k+=8)for(int r=0;r<8;++r) {
                const size_t off=size_t(g)*8*K+k*8+r*8;
                if(g*8+r>=M){std::memset(px+off,0,16);if(pl)std::memset(pl+off,0,16);continue;}
                const auto a=__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(x+size_t(g*8+r)*K+k,8),scale,8);
                auto h=__riscv_vfncvt_f_f_w_f16m1(a,8);
                if(pl) {
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
        auto *sw=reinterpret_cast<_Float16*>(base+128),*sx=sw+tile_n*K;
        auto *sl=x2?(tiles>1?sx+8*K:sx):nullptr;
        auto *temp=reinterpret_cast<float*>(sx+8*K*(x2&&tiles>1?2:1)),*low=temp+256;
        std::memset(base,0xa5,128);std::memset(low+256,0xa5,128);
        auto *out=static_cast<float*>(op->data);
        for(int nb=int(ctx.ith);nb<(N+tile_n-1)/tile_n;nb+=int(ctx.nth)) {
            const int real_tiles=std::min(tiles,(N+31)/32-nb*tiles);
            std::memcpy(sw,pw+size_t(nb)*tile_n*K,size_t(real_tiles)*32*K*2);
            for(int g=0;g<groups;++g) {
                std::memcpy(sx,px+size_t(g)*8*K,8*K*2);
                if(x2&&tiles>1)std::memcpy(sl,pl+size_t(g)*8*K,8*K*2);
                for(int tile=0;tile<real_tiles;++tile) {
                    const auto *weights=sw+size_t(tile)*32*K;
                    if(K==1152)kernel32<1152>(sx,weights,temp);else kernel32<4352>(sx,weights,temp);
                    if(x2) {
                        if(tiles==1)std::memcpy(sl,pl+size_t(g)*8*K,8*K*2);
                        if(K==1152)kernel32<1152>(sl,weights,low);else kernel32<4352>(sl,weights,low);
                        for(int i=0;i<256;++i)temp[i]+=low[i]*(1.0f/4096);
                    }
                    const float inverse=1/(wscale*scales[g]);
                    for(int i=0;i<256;++i)temp[i]*=inverse;
                    const int first_n=nb*tile_n+tile*32;
                    for(int o=0;o<4&&first_n+o*8<N;++o)for(int r=0;r<8&&g*8+r<M;++r)
                        std::memcpy(out+size_t(g*8+r)*N+first_n+o*8,temp+o*64+r*8,
                                    size_t(std::min(8,N-first_n-o*8))*4);
                }
            }
        }
        for(int i=0;i<128;++i)GGML_ASSERT(base[i]==0xa5&&reinterpret_cast<uint8_t*>(low+256)[i]==0xa5);
        return true;
    }
};
inline const traits impl;
}
