#pragma once
// Q4 GGUF stays unchanged. Experimental load-time requantization to one INT8
// scale per output channel; runtime has one INT8 scale per input token.
// Unlike Q8_0 this accumulates the entire K dimension in INT32, rescaling once.
#include "spacemit-context.h"
#include "spacemit-opnode.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <riscv_vector.h>
#include "ocr-pc8-load.h"
namespace ocr_q4_pc8 {
inline int mode(){static const int m=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_PC8");return p?std::atoi(p):0;}();return m;}
inline bool rows8(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_PC8_M8");return p&&std::strcmp(p,"1")==0;}();return x;}
inline bool down_rows8(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_PC8_DOWN_M8");return p&&!std::strcmp(p,"1");}();return x;}
inline bool multi_n(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_PC8_MULTI_N");return p&&std::strcmp(p,"1")==0;}();return x;}
// Prepare each immutable weight panel once, rather than for every M tile.
inline bool prepack(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_PC8_PREPACK");return p&&std::strcmp(p,"1")==0;}();return x;}
__attribute__((noinline)) static void prepack_panel(int8_t*p,int k){
    int blocks=k/32;
    asm volatile(
        "vsetvli t0,zero,e32,m1\n\t1:\n\t"
        "vl8r.v v8,(%[P])\n\t"
        "vupack.vv v0,v8,v9,1\n\tvupack.vv v2,v10,v11,1\n\t"
        "vupack.vv v4,v12,v13,1\n\tvupack.vv v6,v14,v15,1\n\t"
        "vs8r.v v0,(%[P])\n\taddi %[P],%[P],1024\n\t"
        "addi %[BK],%[BK],-1\n\tbnez %[BK],1b\n\t"
        :[P]"+&r"(p),[BK]"+&r"(blocks)::"t0","memory",
         "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","v12","v13","v14","v15");
}
inline bool selected(const ggml_tensor*w){
    if(mode()<1||mode()>2||!w||w->type!=GGML_TYPE_Q4_0||!ggml_is_contiguous(w)||w->ne[2]!=1||w->ne[3]!=1||std::strncmp(w->name,"v.blk.",6))return false;
    const bool shape=(w->ne[0]==1152&&(w->ne[1]==1152||w->ne[1]==4304))||(mode()==2&&w->ne[0]==4352&&w->ne[1]==1152);
    static const auto info=spert::backend_info();
    return shape&&info.core_arch_id==0xa064&&info.vlen==128&&info.shared_mem_size>=160000;
}
inline size_t nbytes(const ggml_tensor*w){return 128+GGML_PAD(size_t(w->ne[1]),32)*(4+size_t(w->ne[0]));}
__attribute__((noinline)) static void kernel(const int8_t*a,const int8_t*b,int32_t*c,int k){
    int blocks=k/32;
    asm volatile(
        "vsetvli t0,zero,e32,m1\n\t"
        "vxor.vv v16,v16,v16\n\tvxor.vv v18,v18,v18\n\tvxor.vv v20,v20,v20\n\tvxor.vv v22,v22,v22\n\t"
        ".align 4\n\t1:\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "vl1r.v v0,(%[A])\n\taddi %[A],%[A],128\n\t"
        "vl4r.v v4,(%[B])\n\taddi %[B],%[B],512\n\t"
        "vl4r.v v8,(%[B])\n\taddi %[B],%[B],512\n\t"
        "vsetvli t0,zero,e32,m1\n\t"
        "vupack.vv v2,v0,v0,1\n\t"
        "vupack.vv v24,v4,v5,1\n\tvupack.vv v26,v6,v7,1\n\t"
        "vupack.vv v4,v8,v9,1\n\tvupack.vv v6,v10,v11,1\n\t"
        "vmadot v16,v2,v24,i8\n\tvmadot v18,v2,v26,i8\n\t"
        "vmadot v20,v2,v4,i8\n\tvmadot v22,v2,v6,i8\n\t"
        "vmadot v16,v3,v25,i8\n\tvmadot v18,v3,v27,i8\n\t"
        "vmadot v20,v3,v5,i8\n\tvmadot v22,v3,v7,i8\n\t"
        "addi %[BK],%[BK],-1\n\tbnez %[BK],1b\n\t"
        "vpack.vv v0,v16,v18,2\n\tvpack.vv v2,v20,v22,2\n\t"
        "vpack.vv v16,v0,v2,3\n\tvpack.vv v18,v1,v3,3\n\t"
        "vse32.v v16,(%[C])\n\taddi t0,%[C],128\n\tvse32.v v17,(t0)\n\t"
        "addi t0,%[C],256\n\tvse32.v v18,(t0)\n\taddi t0,%[C],384\n\tvse32.v v19,(t0)\n\t"
        :[A]"+&r"(a),[B]"+&r"(b),[BK]"+&r"(blocks)
        :[C]"r"(c)
        :"memory","t0","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
         "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27");
}
// Two independently accumulated M4 groups reuse each loaded K32 weight panel.
// Integer accumulation order and the existing FP32 epilogue stay unchanged.
__attribute__((noinline)) static void kernel8(const int8_t*a0,const int8_t*a1,const int8_t*b,int32_t*c,int k){
    int blocks=k/32;
#define OCR_PC8_DOT(A,C0,C1,C2,C3) \
    "vsetvli t0,zero,e8,m1\n\t" \
    "vl1r.v v0,(%[" A "])\n\taddi %[" A "],%[" A "],128\n\t" \
    "vsetvli t0,zero,e32,m1\n\tvupack.vv v2,v0,v0,1\n\t" \
    "vmadot " C0 ",v2,v12,i8\n\tvmadot " C1 ",v2,v14,i8\n\t" \
    "vmadot " C2 ",v2,v4,i8\n\tvmadot " C3 ",v2,v6,i8\n\t" \
    "vmadot " C0 ",v3,v13,i8\n\tvmadot " C1 ",v3,v15,i8\n\t" \
    "vmadot " C2 ",v3,v5,i8\n\tvmadot " C3 ",v3,v7,i8\n\t"
    asm volatile(
        "vsetvli t0,zero,e32,m1\n\t"
        "vxor.vv v16,v16,v16\n\tvxor.vv v18,v18,v18\n\tvxor.vv v20,v20,v20\n\tvxor.vv v22,v22,v22\n\t"
        "vxor.vv v24,v24,v24\n\tvxor.vv v26,v26,v26\n\tvxor.vv v28,v28,v28\n\tvxor.vv v30,v30,v30\n\t"
        ".align 4\n\t1:\n\t"
        "vsetvli t0,zero,e8,m1\n\t"
        "vl4r.v v4,(%[B])\n\taddi %[B],%[B],512\n\t"
        "vl4r.v v8,(%[B])\n\taddi %[B],%[B],512\n\t"
        "vsetvli t0,zero,e32,m1\n\t"
        "vupack.vv v12,v4,v5,1\n\tvupack.vv v14,v6,v7,1\n\t"
        "vupack.vv v4,v8,v9,1\n\tvupack.vv v6,v10,v11,1\n\t"
        OCR_PC8_DOT("A0","v16","v18","v20","v22")
        OCR_PC8_DOT("A1","v24","v26","v28","v30")
        "addi %[BK],%[BK],-1\n\tbnez %[BK],1b\n\t"
        "vpack.vv v0,v16,v18,2\n\tvpack.vv v2,v20,v22,2\n\t"
        "vpack.vv v16,v0,v2,3\n\tvpack.vv v18,v1,v3,3\n\t"
        "vse32.v v16,(%[C])\n\taddi t0,%[C],128\n\tvse32.v v17,(t0)\n\t"
        "addi t0,%[C],256\n\tvse32.v v18,(t0)\n\taddi t0,%[C],384\n\tvse32.v v19,(t0)\n\t"
        "vpack.vv v0,v24,v26,2\n\tvpack.vv v2,v28,v30,2\n\t"
        "vpack.vv v24,v0,v2,3\n\tvpack.vv v26,v1,v3,3\n\t"
        "addi t0,%[C],512\n\tvse32.v v24,(t0)\n\taddi t0,%[C],640\n\tvse32.v v25,(t0)\n\t"
        "addi t0,%[C],768\n\tvse32.v v26,(t0)\n\taddi t0,%[C],896\n\tvse32.v v27,(t0)\n\t"
        :[A0]"+&r"(a0),[A1]"+&r"(a1),[B]"+&r"(b),[BK]"+&r"(blocks)
        :[C]"r"(c)
        :"memory","t0","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
         "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23",
         "v24","v25","v26","v27","v28","v29","v30","v31");
#undef OCR_PC8_DOT
}
__attribute__((noinline)) static void kernel_prepacked(const int8_t*a,const int8_t*b,int32_t*c,int k){
    int blocks=k/64;
    asm volatile(
        "vsetvli t0,zero,e32,m1\n\t"
        "vxor.vv v16,v16,v16\n\tvxor.vv v18,v18,v18\n\tvxor.vv v20,v20,v20\n\tvxor.vv v22,v22,v22\n\t"
        ".align 4\n\t1:\n\t"
        "vl1r.v v0,(%[A])\n\taddi %[A],%[A],128\n\t"
        "vl2r.v v24,(%[B])\n\taddi %[B],%[B],256\n\t"
        "vl2r.v v26,(%[B])\n\taddi %[B],%[B],256\n\t"
        "vl2r.v v4,(%[B])\n\taddi %[B],%[B],256\n\t"
        "vl2r.v v6,(%[B])\n\taddi %[B],%[B],256\n\t"
        "vupack.vv v2,v0,v0,1\n\t"
        "vmadot v16,v2,v24,i8\n\tvmadot v18,v2,v26,i8\n\t"
        "vmadot v20,v2,v4,i8\n\tvmadot v22,v2,v6,i8\n\t"
        "vmadot v16,v3,v25,i8\n\tvmadot v18,v3,v27,i8\n\t"
        "vmadot v20,v3,v5,i8\n\tvmadot v22,v3,v7,i8\n\t"
        "vl1r.v v0,(%[A])\n\taddi %[A],%[A],128\n\t"
        "vl2r.v v24,(%[B])\n\taddi %[B],%[B],256\n\t"
        "vl2r.v v26,(%[B])\n\taddi %[B],%[B],256\n\t"
        "vl2r.v v4,(%[B])\n\taddi %[B],%[B],256\n\t"
        "vl2r.v v6,(%[B])\n\taddi %[B],%[B],256\n\t"
        "vupack.vv v2,v0,v0,1\n\t"
        "vmadot v16,v2,v24,i8\n\tvmadot v18,v2,v26,i8\n\t"
        "vmadot v20,v2,v4,i8\n\tvmadot v22,v2,v6,i8\n\t"
        "vmadot v16,v3,v25,i8\n\tvmadot v18,v3,v27,i8\n\t"
        "vmadot v20,v3,v5,i8\n\tvmadot v22,v3,v7,i8\n\t"
        "addi %[BK],%[BK],-1\n\tbnez %[BK],1b\n\t"
        "vpack.vv v0,v16,v18,2\n\tvpack.vv v2,v20,v22,2\n\t"
        "vpack.vv v16,v0,v2,3\n\tvpack.vv v18,v1,v3,3\n\t"
        "vse32.v v16,(%[C])\n\taddi t0,%[C],128\n\tvse32.v v17,(t0)\n\t"
        "addi t0,%[C],256\n\tvse32.v v18,(t0)\n\taddi t0,%[C],384\n\tvse32.v v19,(t0)\n\t"
        :[A]"+&r"(a),[B]"+&r"(b),[BK]"+&r"(blocks)
        :[C]"r"(c)
        :"memory","t0","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
         "v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27");
}
// Two independently accumulated M4 groups reuse each loaded K32 weight panel.
// Integer accumulation order and the existing FP32 epilogue stay unchanged.
__attribute__((noinline)) static void kernel8_prepacked(const int8_t*a0,const int8_t*a1,const int8_t*b,int32_t*c,int k){
    int blocks=k/64;
#define OCR_PC8_DOT(A,C0,C1,C2,C3) \
    "vl1r.v v0,(%[" A "])\n\taddi %[" A "],%[" A "],128\n\t" \
    "vupack.vv v2,v0,v0,1\n\t" \
    "vmadot " C0 ",v2,v12,i8\n\tvmadot " C1 ",v2,v14,i8\n\t" \
    "vmadot " C2 ",v2,v4,i8\n\tvmadot " C3 ",v2,v6,i8\n\t" \
    "vmadot " C0 ",v3,v13,i8\n\tvmadot " C1 ",v3,v15,i8\n\t" \
    "vmadot " C2 ",v3,v5,i8\n\tvmadot " C3 ",v3,v7,i8\n\t"
    asm volatile(
        "vsetvli t0,zero,e32,m1\n\t"
        "vxor.vv v16,v16,v16\n\tvxor.vv v18,v18,v18\n\tvxor.vv v20,v20,v20\n\tvxor.vv v22,v22,v22\n\t"
        "vxor.vv v24,v24,v24\n\tvxor.vv v26,v26,v26\n\tvxor.vv v28,v28,v28\n\tvxor.vv v30,v30,v30\n\t"
        ".align 4\n\t1:\n\t"
        "vl4r.v v12,(%[B])\n\taddi %[B],%[B],512\n\t"
        "vl4r.v v4,(%[B])\n\taddi %[B],%[B],512\n\t"
        OCR_PC8_DOT("A0","v16","v18","v20","v22")
        OCR_PC8_DOT("A1","v24","v26","v28","v30")
        "vl4r.v v12,(%[B])\n\taddi %[B],%[B],512\n\t"
        "vl4r.v v4,(%[B])\n\taddi %[B],%[B],512\n\t"
        OCR_PC8_DOT("A0","v16","v18","v20","v22")
        OCR_PC8_DOT("A1","v24","v26","v28","v30")
        "addi %[BK],%[BK],-1\n\tbnez %[BK],1b\n\t"
        "vpack.vv v0,v16,v18,2\n\tvpack.vv v2,v20,v22,2\n\t"
        "vpack.vv v16,v0,v2,3\n\tvpack.vv v18,v1,v3,3\n\t"
        "vse32.v v16,(%[C])\n\taddi t0,%[C],128\n\tvse32.v v17,(t0)\n\t"
        "addi t0,%[C],256\n\tvse32.v v18,(t0)\n\taddi t0,%[C],384\n\tvse32.v v19,(t0)\n\t"
        "vpack.vv v0,v24,v26,2\n\tvpack.vv v2,v28,v30,2\n\t"
        "vpack.vv v24,v0,v2,3\n\tvpack.vv v26,v1,v3,3\n\t"
        "addi t0,%[C],512\n\tvse32.v v24,(t0)\n\taddi t0,%[C],640\n\tvse32.v v25,(t0)\n\t"
        "addi t0,%[C],768\n\tvse32.v v26,(t0)\n\taddi t0,%[C],896\n\tvse32.v v27,(t0)\n\t"
        :[A0]"+&r"(a0),[A1]"+&r"(a1),[B]"+&r"(b),[BK]"+&r"(blocks)
        :[C]"r"(c)
        :"memory","t0","v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
         "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23",
         "v24","v25","v26","v27","v28","v29","v30","v31");
#undef OCR_PC8_DOT
}
inline bool supported(const ggml_tensor*op){return op&&op->op==GGML_OP_MUL_MAT&&selected(op->src[0])&&op->src[1]&&op->src[1]->type==GGML_TYPE_F32&&op->type==GGML_TYPE_F32&&op->src[1]->ne[0]==op->src[0]->ne[0]&&op->src[1]->ne[2]==1&&op->src[1]->ne[3]==1&&ggml_is_contiguous(op->src[1])&&ggml_is_contiguous(op);}
inline bool pad_supported(const ggml_tensor*op){
    if(!supported(op)||op->src[0]->ne[0]!=4352)return false;
    const auto*p=op->src[1];const auto*x=p->src[0];
    if(p->op!=GGML_OP_PAD||!x||x->type!=GGML_TYPE_F32||x->ne[0]!=4304||
       !ggml_is_contiguous(x)||x->ne[1]!=p->ne[1]||x->ne[2]!=1||x->ne[3]!=1)return false;
    for(int i=0;i<8;++i)if(p->op_params[i]!=(i==1?48:0))return false;
    return true;
}
struct traits final:ggml::spacemit::tensor_traits_base {
    bool work_size(int,const ggml_tensor*op,size_t&size)const override {
        if(!supported(op))return false;
        size=GGML_PAD(size_t(op->ne[1]),4)*(4+size_t(op->src[0]->ne[0]));return true;
    }
    int repack(ggml_tensor*w,const void*data,size_t bytes)const override {
        GGML_ASSERT(selected(w)&&bytes==ggml_nbytes(w));
        const int K=w->ne[0],N=w->ne[1],NP=GGML_PAD(N,32);
        const auto*q=static_cast<const block_q4_0*>(data);
        static const int load_threads=[] {
            const char * value=std::getenv("GGML_SPACEMIT_OCR_PC8_LOAD_THREADS");
            return value?std::max(1,std::min(8,std::atoi(value))):1;
        }();
        if(load_threads>1) {
            ocr_pc8_load::repack(q,w->data,K,N,load_threads,
                [](ggml_half h){return ggml_fp16_to_fp32(h);});
            return 0;
        }
        std::memset(w->data,0,nbytes(w));auto*scales=reinterpret_cast<float*>(static_cast<char*>(w->data)+128);
        auto*p=reinterpret_cast<int8_t*>(scales+NP);std::vector<float> row(K);
        for(int r=0;r<N;++r){
            float maximum=0;
            for(int k=0;k<K;k+=32){const auto&b=q[size_t(r)*(K/32)+k/32];const float d=ggml_fp16_to_fp32(b.d);
                for(int j=0;j<32;++j){const int v=(j<16?b.qs[j]&15:b.qs[j-16]>>4)-8;row[k+j]=d*v;maximum=std::max(maximum,std::fabs(row[k+j]));}}
            GGML_ASSERT(std::isfinite(maximum));const float scale=maximum/127;scales[r]=scale;
            const float inv=scale?1/scale:0;
            for(int k=0;k<K;k+=32)for(int j=0;j<32;++j)
                p[size_t(r/32)*32*K+(k/32)*1024+(r%32)*32+j]=int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(row[k+j]*inv))));
        }return 0;
    }
    bool compute_forward(ggml::spacemit::context&ctx,ggml_tensor*op)const override {
        if(!supported(op))return false;
        return compute<false>(ctx,op,op->src[1]);
    }
    bool compute_forward_fused(ggml::spacemit::context&ctx,const spacemit_opnode&node)const override {
        if(node.opcode==SPACEMIT_OP_PC8_MUL_MAT_BIAS) {
            if(!supported(node.node)||(node.fused.size()!=1&&node.fused.size()!=2))return false;
            if(node.fused.size()==2)return compute<false,true>(ctx,node.node,node.node->src[1],node.fused[1],node.fused[0]->src[1]);
            return compute<false>(ctx,node.node,node.node->src[1],node.fused[0],node.fused[0]->src[1]);
        }
        if(node.opcode!=SPACEMIT_OP_PC8_PAD_MUL_MAT||node.elided_pad!=node.node->src[1]||
           !pad_supported(node.node))return false;
        if(node.fused.empty())return compute<true>(ctx,node.node,node.elided_pad->src[0]);
        const auto*bias=node.fused[0]->src[1];
        if(node.fused.size()==1)return compute<true>(ctx,node.node,node.elided_pad->src[0],node.fused[0],bias);
        if(node.fused.size()!=2)return false;
        auto*res=node.fused[1];const bool left=res->src[0]!=node.fused[0];
        return compute<true>(ctx,node.node,node.elided_pad->src[0],res,bias,res->src[left?0:1],left);
    }
    template<bool PAD,bool GELU=false>
    bool compute(ggml::spacemit::context&ctx,ggml_tensor*op,const ggml_tensor*input,ggml_tensor*dst=nullptr,const ggml_tensor*bias=nullptr,const ggml_tensor*residual=nullptr,bool residual_left=false)const {
        const int K=op->src[0]->ne[0],N=op->ne[0],M=op->ne[1],MP=GGML_PAD(M,4),NP=GGML_PAD(N,32);
        const int input_k=PAD?int(input->ne[0]):K;
        const bool m8=rows8()&&(K==1152||(K==4352&&down_rows8()))&&M>=8&&
            ctx.shared.size>=size_t(40*K+8*32*4+256);
        const int scratch_rows=m8?8:4,temp_elements=scratch_rows*32;
        size_t size=0;GGML_ASSERT(work_size(ctx.nth,op,size));GGML_ASSERT(ctx.workspace&&ctx.workspace_size>=size&&ctx.shared.data&&ctx.shared.size>=size_t((32+scratch_rows)*K+temp_elements*4+256));
        auto*packed=static_cast<int8_t*>(ctx.workspace);auto*scales=reinterpret_cast<float*>(packed+size_t(MP)*K);
        const auto*x=static_cast<const float*>(input->data);
        static const bool reuse_enabled=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_PC8_REUSE_INPUT");return p&&!std::strcmp(p,"1");}();
        const bool reuse=reuse_enabled&&!PAD&&ctx.pc8_packed_source==input;
        static const bool trace_reuse=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_PC8_REUSE_TRACE");return p&&!std::strcmp(p,"1");}();
        if(reuse&&trace_reuse&&ctx.ith==0)GGML_LOG_INFO("ocr-pc8-reuse: K=%d M=%d\n",K,M);
        ctx.mark_subphase(100); // activation quantization and packing
        if(!reuse)for(int g=ctx.ith;g<MP/4;g+=ctx.nth)for(int r=0;r<4;++r){
            if(g*4+r>=M){scales[g*4+r]=0;for(int k=0;k<K;k+=32)std::memset(packed+size_t(g)*4*K+k*4+r*32,0,32);continue;}
            auto vmax=__riscv_vfmv_v_f_f32m2(0,64);
            for(int k=0;k<input_k;k+=64){const size_t vl=std::min(64,input_k-k);const auto a=__riscv_vfabs_v_f32m2(__riscv_vle32_v_f32m2(x+size_t(g*4+r)*input_k+k,vl),vl);vmax=__riscv_vfmax_vv_f32m2_tu(vmax,vmax,a,vl);}
            const float maximum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(vmax,__riscv_vfmv_v_f_f32m1(0,1),64));
            const float scale=maximum/127;GGML_ASSERT(std::isfinite(maximum));scales[g*4+r]=scale;const float inv=scale?1/scale:0;
            for(int k=0;k<K;k+=32){auto*target=packed+size_t(g)*4*K+k*4+r*32;
                const size_t vl=PAD?std::min(32,std::max(0,input_k-k)):32;
                if constexpr(PAD){if(vl<32)std::memset(target,0,32);if(!vl)continue;}
                const auto a=__riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(x+size_t(g*4+r)*input_k+k,vl),inv,vl);
                const auto q=__riscv_vncvt_x_x_w_i8mf4(__riscv_vfncvt_x_f_w_i16mf2(a,vl),vl);
                __riscv_vse8_v_i8mf4(target,q,vl);}
        }
        ctx.mark_subphase(101); // grid barrier before consuming packed input
        GGML_ASSERT(ctx.sync()==spert::Status::Ok);
        ctx.mark_subphase(102); // scratch setup
        const auto*ws=reinterpret_cast<const float*>(static_cast<const char*>(op->src[0]->data)+128);
        const auto*weights=reinterpret_cast<const int8_t*>(ws+NP);
        const int tiles=NP/32;
        const int simultaneous=multi_n()&&M>=32?std::min((tiles+int(ctx.nth)-1)/int(ctx.nth),
            int((ctx.shared.size-scratch_rows*K-temp_elements*4-256)/(32*K))):1;
        GGML_ASSERT(simultaneous>=1);
        auto*base=static_cast<uint8_t*>(ctx.shared.data);auto*sw=reinterpret_cast<int8_t*>(base+128);auto*sx=sw+simultaneous*32*K;auto*temp=reinterpret_cast<int32_t*>(sx+scratch_rows*K);
        std::memset(base,0xa5,128);std::memset(temp+temp_elements,0xa5,128);auto*out=static_cast<float*>((dst?dst:op)->data);
        for(int nb=ctx.ith;nb<tiles;nb+=simultaneous*ctx.nth){
            const int batch=std::min(simultaneous,(tiles-1-nb)/int(ctx.nth)+1);
            ctx.mark_subphase(200 + uint64_t(nb) * 16); // weight panel copy
            for(int t=0;t<batch;++t){
                std::memcpy(sw+size_t(t)*32*K,weights+size_t(nb+t*ctx.nth)*32*K,32*K);
                ctx.mark_subphase(201 + uint64_t(nb) * 16); // panel permutation
                if(prepack())prepack_panel(sw+size_t(t)*32*K,K);
            }
            for(int g=0;g<MP/4;){const int count=m8&&g+1<MP/4?8:4;
                ctx.mark_subphase(0x100000000ull | (uint64_t(nb)<<24) | (uint64_t(g)<<4));
                std::memcpy(sx,packed+size_t(g)*4*K,count*K);
                for(int t=0;t<batch;++t){const int tile=nb+t*ctx.nth,cols=std::min(32,N-tile*32);
                const auto*w=sw+size_t(t)*32*K;
                ctx.mark_subphase(0x200000000ull | (uint64_t(tile)<<24) | (uint64_t(g)<<4));
                if(prepack()) {if(count==8)kernel8_prepacked(sx,sx+4*K,w,temp,K);else kernel_prepacked(sx,w,temp,K);}
                else {if(count==8)kernel8(sx,sx+4*K,w,temp,K);else kernel(sx,w,temp,K);}
                ctx.mark_subphase(0x300000000ull | (uint64_t(tile)<<24) | (uint64_t(g)<<4));
                if constexpr(GELU)for(int r=0;r<count;++r){
                    const int valid=r+g*4<M?cols:0;
                    std::memset(reinterpret_cast<float*>(temp)+r*32+valid,0,(32-valid)*sizeof(float));
                }
                const auto wscale=__riscv_vle32_v_f32m1(ws+tile*32,cols);
                for(int r=0;r<count&&g*4+r<M;++r){auto c=__riscv_vfcvt_f_x_v_f32m1(__riscv_vle32_v_i32m1(temp+r*32,cols),cols);
                    c=__riscv_vfmul_vv_f32m1(c,wscale,cols);c=__riscv_vfmul_vf_f32m1(c,scales[g*4+r],cols);
                    if(bias)c=__riscv_vfadd_vv_f32m1(c,__riscv_vle32_v_f32m1(static_cast<const float*>(bias->data)+tile*32,cols),cols);
                    if(residual){
                        const auto skip=__riscv_vle32_v_f32m1(static_cast<const float*>(residual->data)+size_t(g*4+r)*N+tile*32,cols);
                        c=residual_left?__riscv_vfadd_vv_f32m1(skip,c,cols):__riscv_vfadd_vv_f32m1(c,skip,cols);
                    }
                    if constexpr(GELU)__riscv_vse32_v_f32m1(reinterpret_cast<float*>(temp)+r*32,c,cols);
                    else __riscv_vse32_v_f32m1(out+size_t(g*4+r)*N+tile*32,c,cols);
                }
                if constexpr(GELU){
                    spacemit_kernels::rvv::forward_gelu_f32_tile(reinterpret_cast<float*>(temp),count*32);
                    for(int r=0;r<count&&g*4+r<M;++r)
                        __riscv_vse32_v_f32m1(out+size_t(g*4+r)*N+tile*32,
                            __riscv_vle32_v_f32m1(reinterpret_cast<float*>(temp)+r*32,cols),cols);
                }}
                g+=count/4;}
        }
        ctx.mark_subphase(400); // scratch canaries
        for(int i=0;i<128;++i)GGML_ASSERT(base[i]==0xa5&&reinterpret_cast<uint8_t*>(temp+temp_elements)[i]==0xa5);
        // A previous output aliasing the input invalidates it, even if tensor
        // identity and shape match. This is a transient activation pack, never
        // a cache of matrix results, and every graph launch resets the tag.
        const auto in_begin=reinterpret_cast<uintptr_t>(input->data);
        const auto out_begin=reinterpret_cast<uintptr_t>((dst?dst:op)->data);
        const bool overlaps=in_begin<out_begin+ggml_nbytes(dst?dst:op)&&out_begin<in_begin+ggml_nbytes(input);
        ctx.pc8_packed_source=reuse_enabled&&!PAD&&!overlaps?input:nullptr;
        return true;
    }
};
inline const traits impl;
}
