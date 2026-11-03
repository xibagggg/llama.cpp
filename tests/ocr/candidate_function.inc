template <bool CACHE_WEIGHT>
static void forward_mul_mat_f16_f32_impl(ggml::spacemit::context & ctx, ggml_tensor * op, float * cached_weights) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    ggml_tensor *       dst  = op;

    GGML_ASSERT(src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(_Float16) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    const int64_t M    = src0->ne[1];
    const int64_t K    = src0->ne[0];
    const int64_t N    = src1->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const int64_t bn2  = src1->ne[2];
    const int64_t bn3  = src1->ne[3];
    const int64_t ith  = ctx.ith;
    const int64_t nth  = ctx.nth;

    // one work unit per (batch3, batch2, m): a-rows across all b batches
    const int64_t nr = M * bn2 * bn3;
    const int64_t dr = (nr + nth - 1) / nth;
    const int64_t r0 = ith * dr;
    const int64_t r1 = MIN(r0 + dr, nr);

    for (int64_t r = r0; r < r1; ++r) {
        const int64_t i3 = r / (M * bn2);
        const int64_t i2 = (r - i3 * M * bn2) / M;
        const int64_t m  = r - i3 * M * bn2 - i2 * M;
        // src0 batch dims broadcast by block-stretch like ggml-cpu: i02 = i12/r2
        const int64_t r2 = bn2 / ne02;
        const int64_t r3 = bn3 / ne03;
        const int64_t a2 = i2 / r2;
        const int64_t a3 = i3 / r3;

        const _Float16 * w_row = (const _Float16 *)
            ((const char *) src0->data + a2 * src0->nb[2] + a3 * src0->nb[3] + m * src0->nb[1]);

        if constexpr (CACHE_WEIGHT) {
            // This row is reused by every activation column. Widen exactly once
            // into worker-local TCM; preserve the original FMA/reduction order.
            int64_t k = 0;
            while (k < K) {
                const size_t vl = __riscv_vsetvl_e16m2(K - k);
                const vfloat16m2_t w16 = __riscv_vle16_v_f16m2(w_row + k, vl);
                const vfloat32m4_t w32 = __riscv_vfwcvt_f_f_v_f32m4(w16, vl);
                __riscv_vse32_v_f32m4(cached_weights + k, w32, vl);
                k += vl;
            }
        }

        int64_t n = 0;
        // Four independent activation columns share each weight load/convert.
        // Each accumulator retains the single-column K loop and reduction order.
        for (; n + 4 <= N; n += 4) {
            const char * x_batch = (const char *) src1->data + i2 * src1->nb[2] + i3 * src1->nb[3];
            const float * xp0 = (const float *) (x_batch + (n + 0) * src1->nb[1]);
            const float * xp1 = (const float *) (x_batch + (n + 1) * src1->nb[1]);
            const float * xp2 = (const float *) (x_batch + (n + 2) * src1->nb[1]);
            const float * xp3 = (const float *) (x_batch + (n + 3) * src1->nb[1]);
            const size_t vlmax = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            vfloat32m4_t acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            vfloat32m4_t acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            vfloat32m4_t acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            const _Float16 * wp = w_row;
            int64_t k_left = K;
            while (k_left > 0) {
                const size_t vl = __riscv_vsetvl_e16m2(k_left);
                vfloat32m4_t w32;
                if constexpr (CACHE_WEIGHT) {
                    w32 = __riscv_vle32_v_f32m4(cached_weights + (K - k_left), vl);
                } else {
                    const vfloat16m2_t w16 = __riscv_vle16_v_f16m2(wp, vl);
                    w32 = __riscv_vfwcvt_f_f_v_f32m4(w16, vl);
                }
                // The final short chunk must retain previous full-chunk sums
                // in inactive lanes before the later VLMAX reduction.
                vfloat32m4_t x32 = __riscv_vle32_v_f32m4(xp0, vl);
                acc0 = __riscv_vfmacc_vv_f32m4_tu(acc0, w32, x32, vl);
                // Keep GCC from hoisting later column loads and spilling vectors.
                __asm__ __volatile__("" : "+vr"(acc0) : : "memory");
                x32 = __riscv_vle32_v_f32m4(xp1, vl);
                acc1 = __riscv_vfmacc_vv_f32m4_tu(acc1, w32, x32, vl);
                __asm__ __volatile__("" : "+vr"(acc1) : : "memory");
                x32 = __riscv_vle32_v_f32m4(xp2, vl);
                acc2 = __riscv_vfmacc_vv_f32m4_tu(acc2, w32, x32, vl);
                __asm__ __volatile__("" : "+vr"(acc2) : : "memory");
                x32 = __riscv_vle32_v_f32m4(xp3, vl);
                acc3 = __riscv_vfmacc_vv_f32m4_tu(acc3, w32, x32, vl);
                wp += vl;
                xp0 += vl; xp1 += vl; xp2 += vl; xp3 += vl;
                k_left -= vl;
            }
            const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            const float sum0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc0, zero, vlmax));
            const float sum1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc1, zero, vlmax));
            const float sum2 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc2, zero, vlmax));
            const float sum3 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc3, zero, vlmax));
            char * d_batch = (char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3];
            ((float *) (d_batch + (n + 0) * dst->nb[1]))[m] = sum0;
            ((float *) (d_batch + (n + 1) * dst->nb[1]))[m] = sum1;
            ((float *) (d_batch + (n + 2) * dst->nb[1]))[m] = sum2;
            ((float *) (d_batch + (n + 3) * dst->nb[1]))[m] = sum3;
        }
        for (; n < N; ++n) {
            const float * x_row = (const float *)
                ((const char *) src1->data + i2 * src1->nb[2] + i3 * src1->nb[3] + n * src1->nb[1]);

            vfloat32m4_t     acc    = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
            int64_t          k_left = K;
            const _Float16 * wp     = w_row;
            const float *    xp     = x_row;

            while (k_left > 0) {
                const size_t vl  = __riscv_vsetvl_e16m2(k_left);
                vfloat32m4_t w32;
                if constexpr (CACHE_WEIGHT) {
                    w32 = __riscv_vle32_v_f32m4(cached_weights + (K - k_left), vl);
                } else {
                    const vfloat16m2_t w16 = __riscv_vle16_v_f16m2(wp, vl);
                    w32 = __riscv_vfwcvt_f_f_v_f32m4(w16, vl);
                }
                vfloat32m4_t x32 = __riscv_vle32_v_f32m4(xp, vl);
                // Keep prior chunk sums in inactive lanes for the VLMAX reduction.
                acc              = __riscv_vfmacc_vv_f32m4_tu(acc, w32, x32, vl);
                wp     += vl;
                xp     += vl;
                k_left -= vl;
            }

            vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            vfloat32m1_t sum  = __riscv_vfredusum_vs_f32m4_f32m1(acc, zero, __riscv_vsetvlmax_e32m4());
            // dst is [M, N, bn2, bn3]: row n of batch (i2, i3) holds M results
            float * d_row = (float *) ((char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3] + n * dst->nb[1]);
            d_row[m]      = __riscv_vfmv_f_s_f32m1_f32(sum);
        }
    }
}

// Diagnostic m4/n1 path: widen four weight rows once into worker-local scratch.
// Each output keeps the frozen-v4 VL, FMA operand/order, tail and reduction.
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

static void candidate_forward_mul_mat_f16_f32(ggml::spacemit::context & ctx, ggml_tensor * op) {
    static const bool m4_cache_enabled = [] {
        const char * value = std::getenv("GGML_SPACEMIT_F16_M4_CACHE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    const ggml_tensor * m4_w = op->src[0];
    const ggml_tensor * m4_x = op->src[1];
    if (m4_cache_enabled && m4_w->type == GGML_TYPE_F16 && m4_x->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
        m4_w->nb[0] == sizeof(_Float16) && m4_x->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) &&
        m4_w->ne[0] == 4304 && m4_w->ne[1] == 1152 && m4_x->ne[1] >= 1056 &&
        m4_w->ne[2] == 1 && m4_w->ne[3] == 1 && m4_x->ne[2] == 1 && m4_x->ne[3] == 1 &&
        op->ne[2] == 1 && op->ne[3] == 1 && ctx.shared.data != nullptr &&
        reinterpret_cast<uintptr_t>(ctx.shared.data) % alignof(float) == 0 &&
        static_cast<uint64_t>(m4_w->ne[0]) <= ctx.shared.size / (4 * sizeof(float))) {
        forward_mul_mat_f16_f32_m4_cache(ctx, op, static_cast<float *>(ctx.shared.data));
        return;
    }
    // Diagnostic opt-in. Default execution retains the frozen-v4 direct path.
    static const bool row_cache_enabled = [] {
        const char * value = std::getenv("GGML_SPACEMIT_F16_ROW_CACHE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    const int64_t K = op->src[0]->ne[0];
    if (row_cache_enabled && op->src[1]->ne[1] >= 8 && K > 0 &&
        ctx.shared.data != nullptr &&
        reinterpret_cast<uintptr_t>(ctx.shared.data) % alignof(float) == 0 &&
        static_cast<uint64_t>(K) <= ctx.shared.size / sizeof(float)) {
        forward_mul_mat_f16_f32_impl<true>(ctx, op, static_cast<float *>(ctx.shared.data));
        return;
    }
    forward_mul_mat_f16_f32_impl<false>(ctx, op, nullptr);
}
