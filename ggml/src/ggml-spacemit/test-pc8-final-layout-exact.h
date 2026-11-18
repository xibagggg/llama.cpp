#pragma once
// Byte-exact old production instruction bodies. Never call on the host CPU.
#include <cstdint>
namespace pc8_native_reference {
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
}
