#pragma once

// Opt-in D72 PV: reuse the existing, compiler-constrained IME2 8x32/16x32
// primitive. Weight storage stays Q4; these are attention intermediates.
// A two-half probability decomposition avoids throwing away the FP32
// softmax mantissa in a single FP16 cast. It changes the accumulation order
// and therefore still requires numerical and OCR acceptance.
namespace ocr_pv_ime2 {
constexpr int D = 72, MAX_Q = 64, MAX_K = 64, PAD_V = 96;
constexpr size_t scratch_bytes = (2 * MAX_Q * D + PAD_V * D) * sizeof(_Float16) +
                                 2 * 512 * sizeof(float);
inline int mode() {
    static const int value = [] {
        const char * p = std::getenv("GGML_SPACEMIT_OCR_PV_IME2");
        return p && std::strcmp(p, "1") == 0 ? 1 :
               p && std::strcmp(p, "2") == 0 ? 2 : 0;
    }();
    return value;
}

inline bool pack_weight(const _Float16 * v, int count, _Float16 * weight) {
    if (!v || !weight || count < 1 || count > MAX_K) return false;
    std::memset(weight, 0, PAD_V * D * sizeof(_Float16));
    const size_t vl = D;
    const auto channels = __riscv_vid_v_u16m2(vl);
    const auto channel_offsets = __riscv_vadd_vv_u16m2(
        __riscv_vmul_vx_u16m2(__riscv_vsrl_vx_u16m2(channels, 3, vl), 8 * D, vl),
        __riscv_vsll_vx_u16m2(__riscv_vand_vx_u16m2(channels, 7, vl), 3, vl), vl);
    for (int k = 0; k < count; ++k) {
        const auto f = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(v + k * D, vl), vl);
        const auto bad = __riscv_vmor_mm_b8(
            __riscv_vmfne_vv_f32m4_b8(f, f, vl),
            __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(f, vl), 63.f, vl), vl);
        if (__riscv_vcpop_m_b8(bad, vl)) return false;
        const auto h = __riscv_vfncvt_f_f_w_f16m2(__riscv_vfmul_vf_f32m4(f, 1024.f, vl), vl);
        const auto indices = __riscv_vsll_vx_u16m2(__riscv_vadd_vx_u16m2(
            channel_offsets, (k / 8) * 64 + k % 8, vl), 1, vl);
        __riscv_vsuxei16_v_f16m2(weight, indices, h, vl);
    }
    return true;
}

inline bool forward(float * out, const float * probability, const _Float16 * v,
                    int rows, int count, int stride, void * scratch, int passes,
                    const _Float16 * prepacked_weight = nullptr) {
    if (!out || !probability || !v || !scratch || rows < 1 || rows > MAX_Q ||
        count < 1 || count > MAX_K || stride < count || (passes != 1 && passes != 2)) return false;
    auto * high = static_cast<_Float16 *>(scratch);
    auto * low = high + MAX_Q * D;
    auto * local_weight = low + MAX_Q * D;
    const auto * weight = prepacked_weight ? prepacked_weight : local_weight;
    auto * acc_high = reinterpret_cast<float *>(local_weight + PAD_V * D);
    auto * acc_low = acc_high + 512;
    // Canonical probabilities [Q,KV] -> [Q/8,K/8,8,8], K padded to 72.
    const size_t padded_bytes = ((rows + 7) / 8) * 8 * D * sizeof(_Float16);
    std::memset(high, 0, padded_bytes);
    if (passes == 2) std::memset(low, 0, padded_bytes);
    const size_t valid = count;
    const auto ks = __riscv_vid_v_u16m1(valid);
    const auto offsets = __riscv_vsll_vx_u16m1(__riscv_vadd_vv_u16m1(
        __riscv_vsll_vx_u16m1(__riscv_vsrl_vx_u16m1(ks, 3, valid), 6, valid),
        __riscv_vand_vx_u16m1(ks, 7, valid), valid), 1, valid);
    for (int row = 0; row < rows; ++row) {
                auto * ph = high + (row / 8) * 8 * D + (row % 8) * 8;
                auto * pl = low + (row / 8) * 8 * D + (row % 8) * 8;
                const auto p = __riscv_vle32_v_f32m2(probability + row * stride, valid);
                const auto bad = __riscv_vmor_mm_b16(
                    __riscv_vmfne_vv_f32m2_b16(p, p, valid),
                    __riscv_vmor_mm_b16(__riscv_vmflt_vf_f32m2_b16(p, 0.f, valid),
                                        __riscv_vmfgt_vf_f32m2_b16(p, 1.001f, valid), valid), valid);
                if (__riscv_vcpop_m_b16(bad, valid)) return false;
                const auto h = __riscv_vfncvt_f_f_w_f16m1(
                    __riscv_vfmul_vf_f32m2(p, 1024.f, valid), valid);
                __riscv_vsuxei16_v_f16m1(ph, offsets, h, valid);
                if (passes == 2) {
                    const auto restored = __riscv_vfmul_vf_f32m2(
                        __riscv_vfwcvt_f_f_v_f32m2(h, valid), 1.f / 1024.f, valid);
                    const auto residual = __riscv_vfsub_vv_f32m2(p, restored, valid);
                    __riscv_vsuxei16_v_f16m1(pl, offsets, __riscv_vfncvt_f_f_w_f16m1(
                        __riscv_vfmul_vf_f32m2(residual, 16777216.f, valid), valid), valid);
                }
    }
    // V [KV,72] -> output-channel panels, padded channels/K initialized.
    // Scaling by 2^10 also preserves all finite FP16 subnormals on IME2.
    if (!prepacked_weight && !pack_weight(v, count, local_weight)) return false;
    // Every fallback decision above precedes the first output write.
    for (int q = 0; q < rows; q += 16) {
        const int qr = std::min(16, rows - q);
        for (int channel = 0; channel < D; channel += 32) {
            const auto compute = [&](const _Float16 * a, float * c) {
                if (qr > 8) ocr_qk_ime2_v53::core16(a + q * D, weight + channel * D, c);
                else ocr_qk_ime2_v53::core(a + q * D, weight + channel * D, c);
            };
            compute(high, acc_high);
            if (passes == 2) compute(low, acc_low);
            const int columns = std::min(32, D - channel);
            const auto ids = __riscv_vid_v_u32m1(columns);
            const auto indices = __riscv_vsll_vx_u32m1(__riscv_vadd_vv_u32m1(
                __riscv_vsll_vx_u32m1(__riscv_vsrl_vx_u32m1(ids, 3, columns), 6, columns),
                __riscv_vand_vx_u32m1(ids, 7, columns), columns), 2, columns);
            for (int r = 0; r < qr; ++r) {
                const int offset = (r / 8) * 256 + (r % 8) * 8;
                auto value = __riscv_vfmul_vf_f32m1(
                    __riscv_vluxei32_v_f32m1(acc_high + offset, indices, columns), 1.f / 1048576.f, columns);
                if (passes == 2) value = __riscv_vfmacc_vf_f32m1(value, 1.f / 17179869184.f,
                    __riscv_vluxei32_v_f32m1(acc_low + offset, indices, columns), columns);
                auto * dst = out + (q + r) * D + channel;
                value = __riscv_vfadd_vv_f32m1(__riscv_vle32_v_f32m1(dst, columns), value, columns);
                __riscv_vse32_v_f32m1(dst, value, columns);
            }
        }
    }
    return true;
}
} // namespace ocr_pv_ime2
