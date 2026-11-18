#pragma once

// Included inside rvv_kernels.cpp's anonymous namespace, after the original
// m1/mrow helpers. Exact mode preserves m1's per-row arithmetic and order.
// The legacy packed mrow route is a separate caller mode and is not bitwise.
static void flash_attn_ext_gqa4_exact_inner(
    float ** pq, const char * k_data_row, const char * v_data_row,
    const ggml_fp16_t * mp, float ** sinks, float ** dst,
    float scale, float logit_softcap, float slope, int64_t nek1,
    int64_t nbk1, int64_t nbv1, int64_t DV, int64_t DK,
    void * tcm_buffer, size_t tcm_buffer_size) {
    GGML_ASSERT(DK == 256 && DV == 256);
    GGML_ASSERT(tcm_buffer && tcm_buffer_size >= 4 * (DK + DV) * sizeof(_Float16));
    auto * VKQ16 = static_cast<_Float16 *>(tcm_buffer);
    auto * Q16 = VKQ16 + 4 * DV;
    float S[4] = {0.f, 0.f, 0.f, 0.f};
    float M[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    const int64_t seg = (int64_t) __riscv_vsetvlmax_e16m2();
    for (int row = 0; row < 4; ++row) {
        rvv_fill_f16(VKQ16 + row * DV, 0.f, DV);
        for (int64_t d0 = 0; d0 < DK; d0 += seg) {
            const size_t vl = __riscv_vsetvl_e16m2(DK - d0);
            __riscv_vse16_v_f16m2(Q16 + row * DK + d0,
                __riscv_vfncvt_f_f_w_f16m2(__riscv_vle32_v_f32m4(pq[row] + d0, vl), vl), vl);
        }
    }
    for (int64_t ic = 0; ic < nek1; ++ic) {
        const float mv = mp ? slope * ((const _Float16 *) mp)[ic] : 0.f;
        if (mv == -INFINITY) continue;
        const auto * k_data = (const _Float16 *) (k_data_row + ic * nbk1);
        const auto * v_data = (const _Float16 *) (v_data_row + ic * nbv1);
        float score[4] = {0.f, 0.f, 0.f, 0.f};
        // Exactly rvv_dot_f16_f32's segment VL, product and reduction tree.
        // Only the K vector load is shared across the four independent rows.
        for (int64_t d0 = 0; d0 < DK;) {
            const size_t vl = __riscv_vsetvl_e16m2(DK - d0);
            const vfloat16m2_t kv = __riscv_vle16_v_f16m2(k_data + d0, vl);
            for (int row = 0; row < 4; ++row) {
                const vfloat32m4_t product = __riscv_vfwmul_vv_f32m4(kv,
                    __riscv_vle16_v_f16m2(Q16 + row * DK + d0, vl), vl);
                const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, 1);
                score[row] += __riscv_vfmv_f_s_f32m1_f32(
                    __riscv_vfredusum_vs_f32m4_f32m1(product, zero, vl));
            }
            d0 += vl;
        }
        float ms[4], vs[4];
        for (int row = 0; row < 4; ++row) {
            float s = score[row];
            s = s * scale;
            if (logit_softcap != 0.f) s = logit_softcap * tanhf(s);
            s += mv;
            const float Mold = M[row];
            ms[row] = 1.f; vs[row] = 1.f;
            if (s > M[row]) {
                M[row] = s;
                ms[row] = expf(Mold - M[row]);
            } else {
                vs[row] = expf(s - M[row]);
            }
        }
        // Exactly rvv_scale_mad_f16's per-channel FP16 multiply/round/FMA.
        // Each row still updates at every visible KV position, in order.
        for (int64_t d0 = 0; d0 < DV;) {
            const size_t vl = __riscv_vsetvl_e16m2(DV - d0);
            const vfloat16m2_t vv = __riscv_vle16_v_f16m2(v_data + d0, vl);
            for (int row = 0; row < 4; ++row) {
                auto * out = VKQ16 + row * DV + d0;
                vfloat16m2_t acc = __riscv_vle16_v_f16m2(out, vl);
                if (ms[row] != 1.f) acc = __riscv_vfmul_vf_f16m2(acc, ms[row], vl);
                acc = __riscv_vfmacc_vf_f16m2(acc, vs[row], vv, vl);
                __riscv_vse16_v_f16m2(out, acc, vl);
            }
            d0 += vl;
        }
        for (int row = 0; row < 4; ++row) S[row] = S[row] * ms[row] + vs[row];
    }
    for (int row = 0; row < 4; ++row)
        flash_attn_ext_finalize_row_f16(dst[row], VKQ16 + row * DV, DV, S[row], M[row], sinks[row]);
}
