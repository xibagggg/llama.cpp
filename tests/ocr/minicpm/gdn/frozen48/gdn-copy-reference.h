namespace gdn_copy {
void memcpy1d(void * dst, const void * src, int64_t size) {
    if (size <= 0) {
        return;
    }
    size_t byte_size_all = size;
    size_t vlen          = __riscv_vlenb() * 8;
    if (vlen == 256) {
        // 1024 bytes
        __asm__ volatile(
            //
            "srli           t0, %[size], 10             \n\t"
            "blez           t0, memcpy_tail%=           \n\t"
            "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
            "memcpy_main_loop%=:                        \n\t"
            "addi           t0, t0, -1                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v8, (%[s])                  \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v16, (%[s])                 \n\t"
            "addi           %[s], %[s], 256             \n\t"
            "vle8.v         v24, (%[s])                 \n\t"
            "addi           %[s], %[s], 256             \n\t"
            //
            "vse8.v         v0, (%[d])                  \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v8, (%[d])                  \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v16, (%[d])                 \n\t"
            "addi           %[d], %[d], 256             \n\t"
            "vse8.v         v24, (%[d])                 \n\t"
            "addi           %[d], %[d], 256             \n\t"
            //
            "bnez           t0, memcpy_main_loop%=      \n\t"
            "memcpy_tail%=:                             \n\t"
            "andi           t1, %[size], 1023           \n\t"
            "blez           t1, out%=                   \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            "out%=:                                     \n\t"
            : [s] "+&r"(src), [d] "+&r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "memory");
    } else if (vlen == 1024) {
        // 2048 bytes
        __asm__ volatile(
            //
            "srli           t0, %[size], 11             \n\t"
            "blez           t0, memcpy_tail%=           \n\t"
            "vsetvli        t1, x0, e8, m8, tu, mu      \n\t"
            "addi           t2, %[s], 1024              \n\t"
            "addi           t3, %[d], 1024              \n\t"
            "li             t5, 2048                    \n\t"
            "memcpy_main_loop%=:                        \n\t"
            "addi           t0, t0, -1                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t5              \n\t"
            "vle8.v         v8, (t2)                    \n\t"
            "add            t2, t2, t5                  \n\t"
            //
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t5              \n\t"
            "vse8.v         v8, (t3)                    \n\t"
            "add            t3, t3, t5                  \n\t"
            //
            "bnez           t0, memcpy_main_loop%=      \n\t"
            "memcpy_tail%=:                             \n\t"
            "andi           t1, %[size], 2047           \n\t"
            "blez           t1, out%=                   \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m2, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            "out%=:                                     \n\t"
            : [s] "+&r"(src), [d] "+&r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1", "t2", "t3", "t5",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "memory");
    } else {
        __asm__ volatile(
            //
            "add            t1, %[size], zero           \n\t"
            "memcpy_tail_loop%=:                        \n\t"
            "vsetvli        t0, t1, e8, m8, tu, mu      \n\t"
            "sub            t1, t1, t0                  \n\t"
            "vle8.v         v0, (%[s])                  \n\t"
            "add            %[s], %[s], t0              \n\t"
            "vse8.v         v0, (%[d])                  \n\t"
            "add            %[d], %[d], t0              \n\t"
            "bnez           t1, memcpy_tail_loop%=      \n\t"
            : [s] "+&r"(src), [d] "+&r"(dst)
            : [size] "r"(byte_size_all)
            : "cc", "t0", "t1", "t2", "t4", "t3",
              "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
    }
}
void forward_cpy_f32_row_strided(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(op->type == GGML_TYPE_F32 && src0->type == GGML_TYPE_F32);
    GGML_ASSERT(op->nb[0] == sizeof(float) && src0->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_nelements(src0) == ggml_nelements(dst));

    const int64_t nc  = src0->ne[0];
    const int64_t nr1 = src0->ne[1];
    const int64_t nr2 = src0->ne[2];
    const int64_t nr3 = src0->ne[3];
    const int64_t nrows = nr1 * nr2 * nr3; // rows of nc floats

    const int64_t ith = ctx.ith;
    const int64_t nth = ctx.nth;

    // rows per thread, flattened over the outer dims
    const int64_t dr = (nrows + nth - 1) / nth;
    const int64_t r0 = dr * ith;
    const int64_t r1 = MIN(r0 + dr, nrows);
    if (r0 >= r1) {
        return;
    }

    // each row is contiguous (nb[0] == sizeof(float)); the row strides (nb[1..3]) are arbitrary on
    // either side, e.g. the GDN rollback-snapshot copies into row-strided cache_s views
    const size_t row_bytes = nc * sizeof(float);
    for (int64_t r = r0; r < r1; ++r) {
        int64_t rem = r;
        const int64_t i3 = rem / (nr1 * nr2); rem -= i3 * nr1 * nr2;
        const int64_t i2 = rem / nr1;
        const int64_t i1 = rem - i2 * nr1;

        const char * sp = (const char *) src0->data + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3];
        char *       dp = (char *)       dst->data + i1 * dst->nb[1]  + i2 * dst->nb[2]  + i3 * dst->nb[3];
        memcpy1d(dp, sp, row_bytes);
    }
}

}
