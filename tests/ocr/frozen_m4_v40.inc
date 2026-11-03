static __attribute__((noipa)) void forward_mul_mat_f16_f32_m4_cache(ggml::spacemit::context & ctx, ggml_tensor * op, float * cache) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * x = op->src[1];
    GGML_ASSERT(w->type == GGML_TYPE_F16 && x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
    GGML_ASSERT(w->nb[0] == sizeof(_Float16) && x->nb[0] == sizeof(float) && op->nb[0] == sizeof(float));
    GGML_ASSERT(w->ne[2] == 1 && w->ne[3] == 1 && x->ne[2] == 1 && x->ne[3] == 1);
    GGML_ASSERT(op->ne[2] == 1 && op->ne[3] == 1 && cache != nullptr);
    const int64_t K = w->ne[0], M = w->ne[1], N = x->ne[1];
    const int64_t dr = (M + ctx.nth - 1) / ctx.nth;
    const int64_t r0 = ctx.ith * dr, r1 = MIN(r0 + dr, M);
    int64_t m = r0;
    for (; m + 4 <= r1; m += 4) {
        for (int row = 0; row < 4; ++row) {
            const _Float16 * wp = reinterpret_cast<const _Float16 *>(
                static_cast<const char *>(w->data) + (m + row) * w->nb[1]);
            int64_t k = 0;
            while (k < K) {
                const size_t vl = __riscv_vsetvl_e16m2(K - k);
                const vfloat16m2_t w16 = __riscv_vle16_v_f16m2(wp + k, vl);
                const vfloat32m4_t w32 = __riscv_vfwcvt_f_f_v_f32m4(w16, vl);
                __riscv_vse32_v_f32m4(cache + row * K + k, w32, vl);
                k += vl;
            }
        }
        for (int64_t n = 0; n < N; ++n) {
            const float * xp = reinterpret_cast<const float *>(static_cast<const char *>(x->data) + n * x->nb[1]);
            const size_t vlmax = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            int64_t k_left = K;
            while (k_left > 0) {
                const size_t vl = __riscv_vsetvl_e16m2(k_left);
                const int64_t k = K - k_left;
                const vfloat32m4_t x32 = __riscv_vle32_v_f32m4(xp, vl);
                vfloat32m4_t w32 = __riscv_vle32_v_f32m4(cache + k, vl);
                acc0 = __riscv_vfmacc_vv_f32m4_tu(acc0, w32, x32, vl);
                __asm__ __volatile__("" : "+vr"(acc0) : : "memory");
                w32 = __riscv_vle32_v_f32m4(cache + K + k, vl);
                acc1 = __riscv_vfmacc_vv_f32m4_tu(acc1, w32, x32, vl);
                __asm__ __volatile__("" : "+vr"(acc1) : : "memory");
                w32 = __riscv_vle32_v_f32m4(cache + 2 * K + k, vl);
                acc2 = __riscv_vfmacc_vv_f32m4_tu(acc2, w32, x32, vl);
                __asm__ __volatile__("" : "+vr"(acc2) : : "memory");
                w32 = __riscv_vle32_v_f32m4(cache + 3 * K + k, vl);
                acc3 = __riscv_vfmacc_vv_f32m4_tu(acc3, w32, x32, vl);
                xp += vl;
                k_left -= vl;
            }
            const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            float * out = reinterpret_cast<float *>(static_cast<char *>(op->data) + n * op->nb[1]);
            out[m + 0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc0, zero, vlmax));
            out[m + 1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc1, zero, vlmax));
            out[m + 2] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc2, zero, vlmax));
            out[m + 3] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc3, zero, vlmax));
        }
    }
    // Worker partitions need not be divisible by four. Tail rows stay local.
    for (; m < r1; ++m) {
        const _Float16 * wp = reinterpret_cast<const _Float16 *>(static_cast<const char *>(w->data) + m * w->nb[1]);
        for (int64_t k = 0; k < K;) {
            const size_t vl = __riscv_vsetvl_e16m2(K - k);
            const vfloat16m2_t w16 = __riscv_vle16_v_f16m2(wp + k, vl);
            const vfloat32m4_t w32 = __riscv_vfwcvt_f_f_v_f32m4(w16, vl);
            __riscv_vse32_v_f32m4(cache + k, w32, vl);
            k += vl;
        }
        for (int64_t n = 0; n < N; ++n) {
            const float * xp = reinterpret_cast<const float *>(static_cast<const char *>(x->data) + n * x->nb[1]);
            const size_t vlmax = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            int64_t k_left = K;
            while (k_left > 0) {
                const size_t vl = __riscv_vsetvl_e16m2(k_left);
                const vfloat32m4_t w32 = __riscv_vle32_v_f32m4(cache + (K - k_left), vl);
                const vfloat32m4_t x32 = __riscv_vle32_v_f32m4(xp, vl);
                acc = __riscv_vfmacc_vv_f32m4_tu(acc, w32, x32, vl);
                xp += vl;
                k_left -= vl;
            }
            const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            float * out = reinterpret_cast<float *>(static_cast<char *>(op->data) + n * op->nb[1]);
            out[m] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc, zero, vlmax));
        }
    }
}
