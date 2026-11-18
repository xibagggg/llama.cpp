// Frozen baseline instructions, ime2 SHA256 6fdcbbbae9fdd10ed119b1ca8846df8bf18728b67aa5309464c6ff09d1b6a53c.
#pragma once
static void q6_head_n32_reference(size_t          blk_len,
                         const uint8_t * quant_a_ptr,
                         const uint8_t * quant_b_data,
                         const uint8_t * quant_b_zp,
                         float *         c_ptr,
                         size_t          count_m,
                         size_t          count_n,
                         size_t          k_blks,
                         size_t          ldc) {
    for (size_t n = 0; n < count_n; n += 32) {
        size_t    nblks         = (count_n - n) > 32 ? 32 : count_n - n;
        uint8_t * QuantBDataPtr = (uint8_t *) quant_b_data +      //
                                  n * k_blks * blk_len +          // b data
                                  n * k_blks * sizeof(_Float16);  // scale
        float * CPtr = c_ptr + n;
        size_t  cnt  = k_blks;

        // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
        // A M1K32 int8    256bit
        // Ascale fp32 * 1  32bit
        // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
        // || Element                          || Element                          || ...
        // B format
        // B N8K32 int4    2048bit
        //   4VRF, N32K32, 8192bit
        // Bscale fp16 * N32 512bit;
        // || scl*32..(fp16) | blk0 blk1 ... blk31 || scl*32..(fp16) | blk0 blk1 ... blk31 || ...
        // || Element                              || Element                              || ...

        //bias always be nullptr
        __asm__ volatile(

            // t3 = k/32
            "mv           t3, %[BCK]              \n\t"
            "mv           t4, %[NBLKS]            \n\t"
            "mv           s2, %[pA]               \n\t"  // s2 = pASCL
            "addi         s3, %[pA], 4+2          \n\t"  // s3 = pAData, (pA+AScl+ASum)
            "mv           s4, %[pB]               \n\t"  // s4 = pBSCL
            "addi         s5, %[pB], 32*2         \n\t"  // s5 = pBdata;
            "mv           s6, %[pC]               \n\t"

            "vsetvli      t0, x0, e32, m1         \n\t"
            "vxor.vv      v2, v0, v0              \n\t"  // clear acc

            // ordinary vmadot: vle*6 flw*1 vecIns*64 vmadot*8
            ".align 4                             \n\t"
            "_K_LPST%=:                           \n\t"

            "vsetvli      t0, x0, e8, m1          \n\t"
            "vl4r.v       v4, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
            "addi         s5, s5, 128*4           \n\t"
            "vl4r.v       v8, (s5)                \n\t"  // B Data 4VRF * 8Row * 32
            "addi         s5, s5, 128*4+64        \n\t"

            "vsetvli      t0, x0, e8, mf2         \n\t"
            "vle8.v       v0, (s4)                \n\t"  // B Scale 4VRF*8Row*FP16 = 512bit
            "addi         s4, s4, 64+128*8        \n\t"

            "vsetvli      t0, x0, e8, mf4         \n\t"
            "vle8.v       v3, (s3)                \n\t"  // A Data M1*K32*int8 = 256bit
            "addi         s3, s3, 32+6            \n\t"

            "flw          f0, (s2)                \n\t"  // A Scale fp32
            "addi         s2, s2, 6+32            \n\t"  // AScale + Asum(FP32+i16)

            "vsetvli      t0, zero, e32, m1       \n\t"
            "vupack.vv    v24, v4, v5, 1          \n\t"
            "vupack.vv    v26, v6, v7, 1          \n\t"
            "vupack.vv    v28, v8, v9, 1          \n\t"
            "vupack.vv    v30, v10, v11, 1        \n\t"

            "vslidedown.vi  v4, v3, 4             \n\t"

            "vxor.vv      v16, v16, v16           \n\t"
            "vxor.vv      v18, v16, v16           \n\t"
            "vxor.vv      v20, v16, v16           \n\t"
            "vxor.vv      v22, v16, v16           \n\t"

            "vmadot       v16, v3, v24, i8         \n\t"  // M0 N0 - N7 INT32(256bit)
            "vmadot       v18, v3, v26, i8         \n\t"  // M0 N8 - N15
            "vmadot       v20, v3, v28, i8         \n\t"  // M0 N16 - N23
            "vmadot       v22, v3, v30, i8         \n\t"  // M0 N24 - N31

            "vmadot       v16, v4, v25, i8         \n\t"
            "vmadot       v18, v4, v27, i8         \n\t"
            "vmadot       v20, v4, v29, i8         \n\t"
            "vmadot       v22, v4, v31, i8         \n\t"

            "vpack.vv     v24, v16, v18, 2        \n\t"
            "vpack.vv     v26, v20, v22, 2        \n\t"
            "vpack.vv     v16, v24, v26, 3        \n\t"

            // b_scale fp16 -> fp32
            "vsetvli      t0, x0, e16, mf2        \n\t"
            "vfwcvt.f.f.v v24, v0                 \n\t"
            // mac result i32 -> fp32
            "vsetvli      t0, x0, e32, m1         \n\t"
            "vfcvt.f.x.v  v26, v16                \n\t"
            // a_scale * b_scale;
            "vfmul.vf     v1, v24, f0             \n\t"
            // static_cast<float>(qsum) * a_scale * b_scale;
            "vfmacc.vv    v2, v1, v26             \n\t"

            "addi         t3, t3, -1              \n\t"
            "bgtz         t3, _K_LPST%=           \n\t"
            "_K_LPND%=:                           \n\t"

            //-----------------------------------------
            // STORE Equal 32N-------------------------
            "_ST32%=:                             \n\t"
            "vsetvli      t0, t4, e32, m1         \n\t"
            "vse32.v      v2, (s6)                \n\t"  // M0 [N0 : N32]; FP32(1024bit)

            "_FUNC_END%=:                         \n\t"

            :
            : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
            : "cc", "memory", "t0", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    }
}
