#pragma once
// Paddle F16 M16/N32 K8 software pipeline. Arithmetic and physical layouts
// remain those of kernel16/kernel16_continue; only load scheduling changes.
#include <cstdlib>
#include <cstring>

namespace paddle_f16_pipeline {
inline bool enabled() {
    static const bool value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_PADDLE_F16_PIPELINE16");
        return p&&!std::strcmp(p,"1");
    }();
    return value;
}

// Two explicit M8 activation and output pointers: continuation callers may
// provide disjoint c0/c1 tiles rather than adjacent storage.
__attribute__((noinline)) static void core16(
    const _Float16 *x0,const _Float16 *x1,const _Float16 *w,
    float *c0,float *c1,int K,int accumulate) {
    GGML_ASSERT(K>0&&K%8==0);
    const _Float16 *w1=w+8*K,*w2=w+16*K,*w3=w+24*K;
    const int blocks=K/8;
    int pairs=(blocks-1)/2;
    const int even=(blocks&1)==0;
#define PADDLE_F16_PIPE_LOAD(A0,A1,W0,W1,W2,W3) \
    "vle16.v " A0 ",(%[x0])\n\t" \
    "vle16.v " A1 ",(%[x1])\n\t" \
    "vle16.v " W0 ",(%[w])\n\t" \
    "vle16.v " W1 ",(%[w1])\n\t" \
    "vle16.v " W2 ",(%[w2])\n\t" \
    "vle16.v " W3 ",(%[w3])\n\t"
#define PADDLE_F16_PIPE_DOT(A0,A1,W0,W1,W2,W3) \
    "smt.vfwmadot v16," A0 "," W0 "\n\t" \
    "smt.vfwmadot v24," A1 "," W0 "\n\t" \
    "smt.vfwmadot v18," A0 "," W1 "\n\t" \
    "smt.vfwmadot v26," A1 "," W1 "\n\t" \
    "smt.vfwmadot v20," A0 "," W2 "\n\t" \
    "smt.vfwmadot v28," A1 "," W2 "\n\t" \
    "smt.vfwmadot v22," A0 "," W3 "\n\t" \
    "smt.vfwmadot v30," A1 "," W3 "\n\t"
#define PADDLE_F16_PIPE_ADVANCE \
    "addi %[x0],%[x0],128\n\taddi %[x1],%[x1],128\n\t" \
    "addi %[w],%[w],128\n\taddi %[w1],%[w1],128\n\t" \
    "addi %[w2],%[w2],128\n\taddi %[w3],%[w3],128\n\t"
    asm volatile(
        "beqz %[accumulate],5f\n\t"
        "vsetvli t0,zero,e32,m2,ta,ma\n\t"
        "vle32.v v16,(%[c0])\n\taddi t0,%[c0],256\n\tvle32.v v18,(t0)\n\t"
        "addi t0,%[c0],512\n\tvle32.v v20,(t0)\n\t"
        "addi t0,%[c0],768\n\tvle32.v v22,(t0)\n\t"
        "vle32.v v24,(%[c1])\n\taddi t0,%[c1],256\n\tvle32.v v26,(t0)\n\t"
        "addi t0,%[c1],512\n\tvle32.v v28,(t0)\n\t"
        "addi t0,%[c1],768\n\tvle32.v v30,(t0)\n\tj 6f\n\t"
        "5:\n\tvsetvli t0,zero,e32,m8,ta,ma\n\t"
        "vmv.v.i v16,0\n\tvmv.v.i v24,0\n\t"
        "6:\n\tvsetvli t0,zero,e16,m1,ta,ma\n\t"
        PADDLE_F16_PIPE_LOAD("v2","v3","v8","v9","v10","v11")
        PADDLE_F16_PIPE_ADVANCE
        "beqz %[pairs],2f\n\t"
        ".align 4\n\t1:\n\t"
        PADDLE_F16_PIPE_LOAD("v0","v1","v4","v5","v6","v7")
        PADDLE_F16_PIPE_DOT("v2","v3","v8","v9","v10","v11")
        PADDLE_F16_PIPE_ADVANCE
        PADDLE_F16_PIPE_LOAD("v2","v3","v8","v9","v10","v11")
        PADDLE_F16_PIPE_DOT("v0","v1","v4","v5","v6","v7")
        PADDLE_F16_PIPE_ADVANCE
        "addi %[pairs],%[pairs],-1\n\tbnez %[pairs],1b\n\t"
        "2:\n\tbeqz %[even],3f\n\t"
        PADDLE_F16_PIPE_LOAD("v0","v1","v4","v5","v6","v7")
        PADDLE_F16_PIPE_DOT("v2","v3","v8","v9","v10","v11")
        PADDLE_F16_PIPE_DOT("v0","v1","v4","v5","v6","v7")
        "j 4f\n\t3:\n\t"
        PADDLE_F16_PIPE_DOT("v2","v3","v8","v9","v10","v11")
        "4:\n\tvsetvli t0,zero,e32,m2,ta,ma\n\t"
        "vse32.v v16,(%[c0])\n\taddi t0,%[c0],256\n\tvse32.v v18,(t0)\n\t"
        "addi t0,%[c0],512\n\tvse32.v v20,(t0)\n\t"
        "addi t0,%[c0],768\n\tvse32.v v22,(t0)\n\t"
        "vse32.v v24,(%[c1])\n\taddi t0,%[c1],256\n\tvse32.v v26,(t0)\n\t"
        "addi t0,%[c1],512\n\tvse32.v v28,(t0)\n\t"
        "addi t0,%[c1],768\n\tvse32.v v30,(t0)\n\t"
        : [x0]"+&r"(x0),[x1]"+&r"(x1),[w]"+&r"(w),[w1]"+&r"(w1),
          [w2]"+&r"(w2),[w3]"+&r"(w3),[pairs]"+&r"(pairs)
        : [c0]"r"(c0),[c1]"r"(c1),[accumulate]"r"(accumulate),[even]"r"(even)
        : "t0","memory","v0","v1","v2","v3","v4","v5","v6","v7",
          "v8","v9","v10","v11","v16","v17","v18","v19","v20","v21",
          "v22","v23","v24","v25","v26","v27","v28","v29","v30","v31");
#undef PADDLE_F16_PIPE_LOAD
#undef PADDLE_F16_PIPE_DOT
#undef PADDLE_F16_PIPE_ADVANCE
}
} // namespace paddle_f16_pipeline
