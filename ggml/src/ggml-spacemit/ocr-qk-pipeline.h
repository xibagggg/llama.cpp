#pragma once

namespace ocr_qk_pipeline {

inline bool enabled() {
    static const bool on = [] {
        const char * value = std::getenv("GGML_SPACEMIT_OCR_QK_PIPELINE");
        return value && std::strcmp(value, "1") == 0;
    }();
    return on;
}

// D72 = nine K8 steps. Alternate two sets of Q/K registers so loads for
// the next step precede the current step's dot products. Each accumulator
// still receives exactly the original nine dot products in original order.
// No additional scratch, TCM capacity or out-of-range look-ahead is needed.
__attribute__((noinline)) static void core16(const _Float16 * x, const _Float16 * w, float * c) {
    const _Float16 * x1=x+8*72, * w1=w+8*72, * w2=w+16*72, * w3=w+24*72;
    int pairs=4;
#define OCR_QK_LOAD(Q0,Q1,K0,K1,K2,K3) \
    "vle16.v " Q0 ",(%[x])\n\t" \
    "vle16.v " Q1 ",(%[x1])\n\t" \
    "vle16.v " K0 ",(%[w])\n\t" \
    "vle16.v " K1 ",(%[w1])\n\t" \
    "vle16.v " K2 ",(%[w2])\n\t" \
    "vle16.v " K3 ",(%[w3])\n\t"
#define OCR_QK_DOT(Q0,Q1,K0,K1,K2,K3) \
    "smt.vfwmadot v16," Q0 "," K0 "\n\t" \
    "smt.vfwmadot v18," Q0 "," K1 "\n\t" \
    "smt.vfwmadot v20," Q0 "," K2 "\n\t" \
    "smt.vfwmadot v22," Q0 "," K3 "\n\t" \
    "smt.vfwmadot v24," Q1 "," K0 "\n\t" \
    "smt.vfwmadot v26," Q1 "," K1 "\n\t" \
    "smt.vfwmadot v28," Q1 "," K2 "\n\t" \
    "smt.vfwmadot v30," Q1 "," K3 "\n\t"
#define OCR_QK_ADVANCE \
    "addi %[x],%[x],128\n\taddi %[x1],%[x1],128\n\t" \
    "addi %[w],%[w],128\n\taddi %[w1],%[w1],128\n\t" \
    "addi %[w2],%[w2],128\n\taddi %[w3],%[w3],128\n\t"
    asm volatile(
        "vsetvli t0,zero,e32,m2,ta,ma\n\t"
        "vmv.v.i v16,0\n\tvmv.v.i v18,0\n\tvmv.v.i v20,0\n\tvmv.v.i v22,0\n\t"
        "vmv.v.i v24,0\n\tvmv.v.i v26,0\n\tvmv.v.i v28,0\n\tvmv.v.i v30,0\n\t"
        "vsetvli t0,zero,e16,m1,ta,ma\n\t"
        OCR_QK_LOAD("v2","v3","v8","v9","v10","v11")
        OCR_QK_ADVANCE
        ".align 4\n\t1:\n\t"
        OCR_QK_LOAD("v0","v1","v4","v5","v6","v7")
        OCR_QK_DOT("v2","v3","v8","v9","v10","v11")
        OCR_QK_ADVANCE
        OCR_QK_LOAD("v2","v3","v8","v9","v10","v11")
        OCR_QK_DOT("v0","v1","v4","v5","v6","v7")
        OCR_QK_ADVANCE
        "addi %[pairs],%[pairs],-1\n\tbnez %[pairs],1b\n\t"
        OCR_QK_DOT("v2","v3","v8","v9","v10","v11")
        "vsetvli t0,zero,e32,m2,ta,ma\n\t"
        "vse32.v v16,(%[c])\n\taddi t0,%[c],256\n\tvse32.v v18,(t0)\n\t"
        "addi t0,%[c],512\n\tvse32.v v20,(t0)\n\taddi t0,%[c],768\n\tvse32.v v22,(t0)\n\t"
        "addi t0,%[c],1024\n\tvse32.v v24,(t0)\n\taddi t0,%[c],1280\n\tvse32.v v26,(t0)\n\t"
        "addi t0,%[c],1536\n\tvse32.v v28,(t0)\n\taddi t0,%[c],1792\n\tvse32.v v30,(t0)\n\t"
        : [x]"+&r"(x), [x1]"+&r"(x1), [w]"+&r"(w), [w1]"+&r"(w1),
          [w2]"+&r"(w2), [w3]"+&r"(w3), [pairs]"+&r"(pairs)
        : [c]"r"(c)
        : "t0", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21",
          "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#undef OCR_QK_LOAD
#undef OCR_QK_DOT
#undef OCR_QK_ADVANCE
}

} // namespace ocr_qk_pipeline
