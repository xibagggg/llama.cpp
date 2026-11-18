#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Optional Q4_0 output-head M1 preparation. This does not requantize: it
// materializes the original kernel's exact A nibble expansion and F16 initial
// correction once per K32 block, then reuses those values across N32 panels.
namespace ocr_q4_head_prepared {
inline std::atomic<bool> active{false};
inline constexpr size_t block_bytes = 512;
inline void prepare_on_host() {
    static const bool enabled=[] {
        const char * e=std::getenv("GGML_SPACEMIT_Q4_HEAD_PREPARED_A");
        return e&&std::strcmp(e,"1")==0;
    }();
    active.store(enabled,std::memory_order_relaxed);
}

#if defined(RISCV64_SPACEMIT_IME2)
inline void prepare(uint8_t * dst,const uint8_t * a,size_t blocks) {
    // Existing INTER_SIZE32 IME2 Q8 layout is {F32 scale,I16 sum,I8[32]}.
    // Three full-register payloads are low-nibble A, high-nibble A and the
    // original F16(asum*8) initializer. The F32 scale stays unchanged at +384.
    for(size_t k=0;k<blocks;++k) {
        uint8_t * p=dst+k*block_bytes;
        const uint8_t * q=a+k*38;
        std::memcpy(p+384,q,4);
        asm volatile(
            "addi t1,%[A],6\n\t"
            "lh t2,4(%[A])\n\t"
            "vsetvli t0,zero,e8,mf4\n\t"
            "vle8.v v3,(t1)\n\t"
            "vsetvli t0,zero,e16,m1\n\t"
            "vmv.v.i v28,8\n\t"
            "vsetvli t0,zero,e8,m1\n\t"
            "vsrl.vi v24,v3,4\n\t"
            "vsetvli t0,zero,e16,m1\n\t"
            "vmul.vx v26,v28,t2\n\t"
            "vnpack4.vv v8,v3,v3,3\n\t"
            "vnpack4.vv v10,v24,v24,3\n\t"
            "vfcvt.f.x.v v16,v26\n\t"
            "vs1r.v v8,(%[D])\n\t"
            "addi t1,%[D],128\n\t"
            "vs1r.v v10,(t1)\n\t"
            "addi t1,%[D],256\n\t"
            "vs1r.v v16,(t1)\n\t"
            : :[A]"r"(q),[D]"r"(p)
            :"t0","t1","t2","v3","v8","v10","v16","v24","v26","v28","memory");
    }
}

inline size_t compute(const uint8_t * prepared,const uint8_t * b,float * out,
                      size_t count_n,size_t blocks) {
    alignas(64) float tail[32];
    for(size_t n=0;n<count_n;n+=32) {
        const size_t cols=std::min<size_t>(32,count_n-n);
        const uint8_t * panel=b+(n/32)*blocks*576;
        float * c=cols==32?out+n:tail;
        asm volatile(
            "mv t3,%[BK]\n\t"
            "mv s2,%[A]\n\t"
            "mv s4,%[B]\n\t"
            "addi s5,%[B],64\n\t"
            "vsetvli t0,zero,e16,m1\n\t"
            "vmv.v.i v0,1\n\t"
            "vsll.vi v1,v0,4\n\t"
            "vxor.vv v2,v0,v0\n\t"
            "vfcvt.f.x.v v0,v0\n\t"
            "vfcvt.f.x.v v1,v1\n\t"
            ".align 4\n\t"
            "1:\n\t"
            "vsetvli t0,zero,e8,m1\n\t"
            "vl4r.v v4,(s5)\n\t"
            "addi s5,s5,576\n\t"
            "vsetvli t0,zero,e8,mf2\n\t"
            "vle8.v v30,(s4)\n\t"
            "addi s4,s4,576\n\t"
            "vl1r.v v8,(s2)\n\t"
            "addi t1,s2,128\n\t"
            "vl1r.v v10,(t1)\n\t"
            "addi t1,s2,256\n\t"
            "vl1r.v v16,(t1)\n\t"
            "flw f0,384(s2)\n\t"
            "addi s2,s2,512\n\t"
            "vsetvli t0,zero,e16,m1\n\t"
            "vadd.vi v18,v16,0\n\t"
            "vadd.vi v20,v16,0\n\t"
            "vadd.vi v22,v16,0\n\t"
            "vmadotsu.hp v16,v10,v4,v1,0,i4\n\t"
            "vmadotsu.hp v18,v10,v5,v1,0,i4\n\t"
            "vmadotsu.hp v20,v10,v6,v1,0,i4\n\t"
            "vmadotsu.hp v22,v10,v7,v1,0,i4\n\t"
            "vmadotu.hp v16,v8,v4,v0,0,i4\n\t"
            "vmadotu.hp v18,v8,v5,v0,0,i4\n\t"
            "vmadotu.hp v20,v8,v6,v0,0,i4\n\t"
            "vmadotu.hp v22,v8,v7,v0,0,i4\n\t"
            "vpack.vv v24,v16,v18,1\n\t"
            "vpack.vv v26,v20,v22,1\n\t"
            "vpack.vv v16,v24,v26,2\n\t"
            "vsetvli t0,zero,e16,mf2\n\t"
            "vfwmul.vv v31,v30,v16\n\t"
            "vsetvli t0,zero,e32,m1\n\t"
            "vfmacc.vf v2,f0,v31\n\t"
            "addi t3,t3,-1\n\t"
            "bgtz t3,1b\n\t"
            "vse32.v v2,(%[C])\n\t"
            : :[A]"r"(prepared),[B]"r"(panel),[C]"r"(c),[BK]"r"(blocks)
            :"t0","t1","t3","s2","s4","s5","f0","memory",
             "v0","v1","v2","v4","v5","v6","v7","v8","v10",
             "v16","v18","v20","v22","v24","v26","v30","v31");
        if(cols<32)std::memcpy(out+n,tail,cols*sizeof(float));
    }
    return 1;
}
#endif
} // namespace ocr_q4_head_prepared
