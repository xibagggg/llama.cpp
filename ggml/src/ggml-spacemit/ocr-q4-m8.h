#pragma once
// Included inside spacemit_kernels::ime2. The original K32 Q4 arithmetic is
// retained, with two independent groups of four rows sharing each weight tile.
// Q4 weights, Q8 activation packing, K accumulation order and FP32 epilogue
// match the M4 kernel. Only the bounded OCR dispatch below selects this path.
static size_t ocr_q4_m8(size_t blk_len,const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,const uint8_t * quant_b_zp,
                        float * c_ptr,size_t count_m,size_t count_n,size_t k_blks,size_t ldc) {
    if(blk_len!=32||quant_b_zp||count_m<8||count_n>32||k_blks!=36)
        return gemm_kernel_i8i4(blk_len,quant_a_ptr,quant_b_data,quant_b_zp,c_ptr,count_m,count_n,k_blks,ldc);
    GGML_ASSERT(count_n>0);
    const uint8_t * a0=quant_a_ptr;
    const uint8_t * a1=quant_a_ptr+4*k_blks*q8_blk_size(32,true);
    const uint8_t * b=quant_b_data;
    alignas(64) float tail[8*32];
    float * dst=count_n==32?c_ptr:tail;
    const size_t stride=(count_n==32?ldc:32)*sizeof(float);
#define OCR_Q4_M8_GROUP(A,C0,C1,C2,C3) \
    "flw fa0, (%[" A "])\n\t" \
    "flw fa1, 4(%[" A "])\n\t" \
    "flw fa2, 8(%[" A "])\n\t" \
    "flw fa3, 12(%[" A "])\n\t" \
    "addi %[" A "], %[" A "], 16\n\t" \
    "vsetivli t0, 4, e16, mf2\n\t" \
    "vle16.v v8, (%[" A "])\n\t" \
    "addi %[" A "], %[" A "], 8\n\t" \
    "vsll.vi v8, v8, 3\n\t" \
    "vfcvt.f.x.v v9, v8\n\t" \
    "vsetvli t0, x0, e64, m1\n\t" \
    "vrgather.vi v10, v9, 0\n\t" \
    "vsetvli t0, x0, e8, m1\n\t" \
    "vl1r.v v16, (%[" A "])\n\t" \
    "addi %[" A "], %[" A "], 128\n\t" \
    "vsrl.vi v17, v16, 4\n\t" \
    "vnpack4.vv v12, v16, v17, 3\n\t" \
    "vupack.vv v2, v12, v12, 2\n\t" \
    "vsetvli t0, x0, e16, m1\n\t" \
    "vpack.vv v16, v10, v10, 0\n\t" \
    "vsetvli t0, x0, e32, m1\n\t" \
    "vpack.vv v20, v16, v16, 0\n\t" \
    "vsetvli t0, x0, e64, m1\n\t" \
    "vpack.vv v18, v20, v20, 0\n\t" \
    "vor.vv v20, v18, v18\n\t" \
    "vor.vv v21, v18, v18\n\t" \
    "vsetvli t0, x0, e16, m1\n\t" \
    "vmadotsu.hp v18, v3, v4, v1, 0, i4\n\t" \
    "vmadotsu.hp v19, v3, v5, v1, 0, i4\n\t" \
    "vmadotsu.hp v20, v3, v6, v1, 0, i4\n\t" \
    "vmadotsu.hp v21, v3, v7, v1, 0, i4\n\t" \
    "vmadotu.hp v18, v2, v4, v0, 0, i4\n\t" \
    "vmadotu.hp v19, v2, v5, v0, 0, i4\n\t" \
    "vmadotu.hp v20, v2, v6, v0, 0, i4\n\t" \
    "vmadotu.hp v21, v2, v7, v0, 0, i4\n\t" \
    "vpack.vv v8, v18, v19, 1\n\t" \
    "vpack.vv v12, v20, v21, 1\n\t" \
    "vpack.vv v20, v8, v12, 2\n\t" \
    "vfwmul.vv v16, v20, v14\n\t" \
    "vfwmul.vv v18, v21, v14\n\t" \
    "vsetvli t0, x0, e32, m1\n\t" \
    "vfmacc.vf " C0 ", fa0, v16\n\t" \
    "vfmacc.vf " C1 ", fa1, v17\n\t" \
    "vfmacc.vf " C2 ", fa2, v18\n\t" \
    "vfmacc.vf " C3 ", fa3, v19\n\t"
    asm volatile(
        "vsetvli t0, x0, e16, m1\n\t"
        "vxor.vv v24,v24,v24\n\t" "vxor.vv v25,v25,v25\n\t"
        "vxor.vv v26,v26,v26\n\t" "vxor.vv v27,v27,v27\n\t"
        "vxor.vv v28,v28,v28\n\t" "vxor.vv v29,v29,v29\n\t"
        "vxor.vv v30,v30,v30\n\t" "vxor.vv v31,v31,v31\n\t"
        "vmv.v.i v0,1\n\t" "vsll.vi v1,v0,4\n\t"
        "vfcvt.f.x.v v0,v0\n\t" "vfcvt.f.x.v v1,v1\n\t"
        "mv t4,%[BK]\n\t"
        ".align 4\n\t"
        "OCR_Q4_M8_LOOP%=:\n\t"
        "vsetvli t0,x0,e16,mf2\n\t"
        "vle16.v v12,(%[B])\n\t" "addi %[B],%[B],64\n\t"
        "vsetvli t0,x0,e16,m1\n\t" "vpack.vv v14,v12,v12,3\n\t"
        "vl4r.v v4,(%[B])\n\t" "addi %[B],%[B],512\n\t"
        OCR_Q4_M8_GROUP("A0","v24","v25","v26","v27")
        OCR_Q4_M8_GROUP("A1","v28","v29","v30","v31")
        "addi t4,t4,-1\n\t" "bgtz t4,OCR_Q4_M8_LOOP%=\n\t"
        "vse32.v v24,(%[DST])\n\t" "add t2,%[DST],%[LDC]\n\t"
        "vse32.v v25,(t2)\n\t" "add t2,t2,%[LDC]\n\t"
        "vse32.v v26,(t2)\n\t" "add t2,t2,%[LDC]\n\t"
        "vse32.v v27,(t2)\n\t" "add t2,t2,%[LDC]\n\t"
        "vse32.v v28,(t2)\n\t" "add t2,t2,%[LDC]\n\t"
        "vse32.v v29,(t2)\n\t" "add t2,t2,%[LDC]\n\t"
        "vse32.v v30,(t2)\n\t" "add t2,t2,%[LDC]\n\t"
        "vse32.v v31,(t2)\n\t"
        :[A0]"+&r"(a0),[A1]"+&r"(a1),[B]"+&r"(b)
        :[DST]"r"(dst),[LDC]"r"(stride),[BK]"r"(k_blks)
        :"memory","t0","t2","t4","fa0","fa1","fa2","fa3",
         "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11",
         "v12","v13","v14","v15","v16","v17","v18","v19","v20","v21","v22","v23",
         "v24","v25","v26","v27","v28","v29","v30","v31");
#undef OCR_Q4_M8_GROUP
    if(count_n!=32)for(size_t r=0;r<8;++r)memcpy(c_ptr+r*ldc,tail+r*32,count_n*sizeof(float));
    return 8;
}
