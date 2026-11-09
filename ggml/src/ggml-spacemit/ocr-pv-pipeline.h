#pragma once
#include <cstdlib>
#include <cstring>

namespace ocr_pv_pipeline {

inline bool enabled() {
    static const bool on = [] {
        const char * value = std::getenv("GGML_SPACEMIT_OCR_PV_PIPELINE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return on;
}

// Independent flag preserves r37's 32-channel-only comparison.
inline bool tail_enabled() {
    static const bool on = [] {
        const char * value = std::getenv("GGML_SPACEMIT_OCR_PV_TAIL_PIPELINE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return on;
}

// Variable K, multiple of eight. Pipeline paired K8 steps, then execute one
// or two final steps without reading beyond the input panels. The original
// per-accumulator multiply-add order and output panel layout are preserved.
__attribute__((noinline)) static void core16(const _Float16 * x, const _Float16 * w, float * c, int k) {
    const _Float16 * x1=x+8*k, * w1=w+8*k, * w2=w+16*k, * w3=w+24*k;
    int pairs=(k/8-1)/2;
    const int even=(k/8)%2==0;
#define OCR_PV_LOAD(Q0,Q1,K0,K1,K2,K3) \
    "vle16.v " Q0 ",(%[x])\n\t" \
    "vle16.v " Q1 ",(%[x1])\n\t" \
    "vle16.v " K0 ",(%[w])\n\t" \
    "vle16.v " K1 ",(%[w1])\n\t" \
    "vle16.v " K2 ",(%[w2])\n\t" \
    "vle16.v " K3 ",(%[w3])\n\t"
#define OCR_PV_DOT(Q0,Q1,K0,K1,K2,K3) \
    "smt.vfwmadot v16," Q0 "," K0 "\n\t" \
    "smt.vfwmadot v18," Q0 "," K1 "\n\t" \
    "smt.vfwmadot v20," Q0 "," K2 "\n\t" \
    "smt.vfwmadot v22," Q0 "," K3 "\n\t" \
    "smt.vfwmadot v24," Q1 "," K0 "\n\t" \
    "smt.vfwmadot v26," Q1 "," K1 "\n\t" \
    "smt.vfwmadot v28," Q1 "," K2 "\n\t" \
    "smt.vfwmadot v30," Q1 "," K3 "\n\t"
#define OCR_PV_ADVANCE \
    "addi %[x],%[x],128\n\taddi %[x1],%[x1],128\n\t" \
    "addi %[w],%[w],128\n\taddi %[w1],%[w1],128\n\t" \
    "addi %[w2],%[w2],128\n\taddi %[w3],%[w3],128\n\t"
    asm volatile(
        "vsetvli t0,zero,e32,m2,ta,ma\n\t"
        "vmv.v.i v16,0\n\tvmv.v.i v18,0\n\tvmv.v.i v20,0\n\tvmv.v.i v22,0\n\t"
        "vmv.v.i v24,0\n\tvmv.v.i v26,0\n\tvmv.v.i v28,0\n\tvmv.v.i v30,0\n\t"
        "vsetvli t0,zero,e16,m1,ta,ma\n\t"
        OCR_PV_LOAD("v2","v3","v8","v9","v10","v11")
        OCR_PV_ADVANCE
        "beqz %[pairs],2f\n\t"
        ".align 4\n\t1:\n\t"
        OCR_PV_LOAD("v0","v1","v4","v5","v6","v7")
        OCR_PV_DOT("v2","v3","v8","v9","v10","v11")
        OCR_PV_ADVANCE
        OCR_PV_LOAD("v2","v3","v8","v9","v10","v11")
        OCR_PV_DOT("v0","v1","v4","v5","v6","v7")
        OCR_PV_ADVANCE
        "addi %[pairs],%[pairs],-1\n\tbnez %[pairs],1b\n\t"
        "2:\n\tbeqz %[even],3f\n\t"
        OCR_PV_LOAD("v0","v1","v4","v5","v6","v7")
        OCR_PV_DOT("v2","v3","v8","v9","v10","v11")
        OCR_PV_DOT("v0","v1","v4","v5","v6","v7")
        "j 4f\n\t3:\n\t"
        OCR_PV_DOT("v2","v3","v8","v9","v10","v11")
        "4:\n\t"
        "vsetvli t0,zero,e32,m2,ta,ma\n\t"
        "vse32.v v16,(%[c])\n\taddi t0,%[c],256\n\tvse32.v v18,(t0)\n\t"
        "addi t0,%[c],512\n\tvse32.v v20,(t0)\n\taddi t0,%[c],768\n\tvse32.v v22,(t0)\n\t"
        "addi t0,%[c],1024\n\tvse32.v v24,(t0)\n\taddi t0,%[c],1280\n\tvse32.v v26,(t0)\n\t"
        "addi t0,%[c],1536\n\tvse32.v v28,(t0)\n\taddi t0,%[c],1792\n\tvse32.v v30,(t0)\n\t"
        : [x]"+&r"(x), [x1]"+&r"(x1), [w]"+&r"(w), [w1]"+&r"(w1),
          [w2]"+&r"(w2), [w3]"+&r"(w3), [pairs]"+&r"(pairs)
        : [c]"r"(c), [even]"r"(even)
        : "t0", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#undef OCR_PV_LOAD
#undef OCR_PV_DOT
#undef OCR_PV_ADVANCE
}

// D72's final eight channels: same two FP32 accumulators and sparse output
// panel layout as pv16x8_reference. No speculative load past the last K8.
__attribute__((noinline)) static void core8(const _Float16 * x, const _Float16 * w, float * c, int k) {
    const _Float16 * x1 = x + 8*k;
    const int blocks = k/8;
    int pairs = (blocks-1)/2;
    const int even = 1-(blocks&1);
#define OCR_PV8_LOAD(Q0,Q1,V) \
    "vle16.v " Q0 ",(%[x])\n\t" \
    "vle16.v " Q1 ",(%[x1])\n\t" \
    "vle16.v " V ",(%[w])\n\t"
#define OCR_PV8_DOT(Q0,Q1,V) \
    "smt.vfwmadot v16," Q0 "," V "\n\t" \
    "smt.vfwmadot v18," Q1 "," V "\n\t"
#define OCR_PV8_ADVANCE \
    "addi %[x],%[x],128\n\taddi %[x1],%[x1],128\n\taddi %[w],%[w],128\n\t"
    asm volatile(
        "vsetvli t0,zero,e32,m2,ta,ma\n\t"
        "vmv.v.i v16,0\n\tvmv.v.i v18,0\n\t"
        "vsetvli t0,zero,e16,m1,ta,ma\n\t"
        OCR_PV8_LOAD("v2","v3","v8")
        OCR_PV8_ADVANCE
        "beqz %[pairs],2f\n\t"
        ".align 4\n\t1:\n\t"
        OCR_PV8_LOAD("v0","v1","v4")
        OCR_PV8_DOT("v2","v3","v8")
        OCR_PV8_ADVANCE
        OCR_PV8_LOAD("v2","v3","v8")
        OCR_PV8_DOT("v0","v1","v4")
        OCR_PV8_ADVANCE
        "addi %[pairs],%[pairs],-1\n\tbnez %[pairs],1b\n\t"
        "2:\n\tbeqz %[even],3f\n\t"
        OCR_PV8_LOAD("v0","v1","v4")
        OCR_PV8_DOT("v2","v3","v8")
        OCR_PV8_DOT("v0","v1","v4")
        "j 4f\n\t3:\n\t"
        OCR_PV8_DOT("v2","v3","v8")
        "4:\n\t"
        "vsetvli t0,zero,e32,m2,ta,ma\n\t"
        "vse32.v v16,(%[c])\n\taddi t0,%[c],1024\n\tvse32.v v18,(t0)\n\t"
        : [x]"+&r"(x), [x1]"+&r"(x1), [w]"+&r"(w), [pairs]"+&r"(pairs)
        : [c]"r"(c), [even]"r"(even)
        : "t0", "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v16", "v17", "v18", "v19");
#undef OCR_PV8_LOAD
#undef OCR_PV8_DOT
#undef OCR_PV8_ADVANCE
}

} // namespace ocr_pv_pipeline
