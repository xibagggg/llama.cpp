#pragma once

#include "ocr-norm-affine.h"
#include <riscv_vector.h>

namespace ocr_norm_affine {
inline float reduce_m4(vfloat32m4_t value) {
    const size_t vl = __riscv_vsetvlmax_e32m1();
    auto v = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m4_f32m1(value, 0),
                                   __riscv_vget_v_f32m4_f32m1(value, 1), vl);
    v = __riscv_vfadd_vv_f32m1(v, __riscv_vget_v_f32m4_f32m1(value, 2), vl);
    v = __riscv_vfadd_vv_f32m1(v, __riscv_vget_v_f32m4_f32m1(value, 3), vl);
    v = __riscv_vfredusum_vs_f32m1_f32m1(v, __riscv_vfmv_v_f_f32m1(0.f, vl), vl);
    return __riscv_vfmv_f_s_f32m1_f32(v);
}

// Same m4 lane accumulation, sum-of-squares FMA, reduction tree and scalar
// variance expression as forward_norm_f32. MUL then ADD remain separate RVV
// instructions; introducing affine FMA would change the unfused rounding.
inline bool forward(const plan & p, int ith, int nth) {
    if (!storage_safe(p) || nth <= 0 || ith < 0 || ith >= nth) return false;
    const auto input = p.norm->src[0];
    const int64_t width = input->ne[0], rows = ggml_nrows(input);
    const int64_t chunk = (rows + nth - 1) / nth;
    const int64_t begin = chunk * ith, end = std::min(rows, begin + chunk);
    float epsilon;
    std::memcpy(&epsilon, p.norm->op_params, sizeof(epsilon));
    const auto * weight = static_cast<const float *>(p.scale->src[1]->data);
    const auto * bias = static_cast<const float *>(p.bias->src[1]->data);
    for (int64_t row = begin; row < end; ++row) {
        const auto * x = static_cast<const float *>(input->data) + row * width;
        auto * y = static_cast<float *>(p.bias->data) + row * width;
        auto sum = __riscv_vfmv_v_f_f32m4(0.f, __riscv_vsetvlmax_e32m4());
        auto square = __riscv_vfmv_v_f_f32m4(0.f, __riscv_vsetvlmax_e32m4());
        for (int64_t col = 0; col < width;) {
            const size_t vl = __riscv_vsetvl_e32m4(width - col);
            vfloat32m4_t value;
            if (p.residual) {
                const auto * a = static_cast<const float *>(p.residual->src[0]->data) + row * width + col;
                const auto * b = static_cast<const float *>(p.residual->src[1]->data) + row * width + col;
                value = __riscv_vfadd_vv_f32m4(__riscv_vle32_v_f32m4(a, vl),
                                               __riscv_vle32_v_f32m4(b, vl), vl);
                __riscv_vse32_v_f32m4(static_cast<float *>(p.residual->data) + row * width + col, value, vl);
            } else {
                value = __riscv_vle32_v_f32m4(x + col, vl);
            }
            sum = __riscv_vfadd_vv_f32m4(sum, value, vl);
            square = __riscv_vfmacc_vv_f32m4(square, value, value, vl);
            col += vl;
        }
        const float mean = reduce_m4(sum) / width;
        const float mean_square = reduce_m4(square) / width;
        // Use sqrt(double) followed by the original float store, matching the
        // existing norm rather than silently replacing it with sqrtf.
        float denominator = std::sqrt(double(mean_square - mean * mean + epsilon));
        const float inverse = 1.f / denominator;
        for (int64_t col = 0; col < width;) {
            const size_t vl = __riscv_vsetvl_e32m4(width - col);
            auto value = __riscv_vle32_v_f32m4(x + col, vl);
            value = __riscv_vfsub_vf_f32m4(value, mean, vl);
            value = __riscv_vfmul_vf_f32m4(value, inverse, vl);
            value = __riscv_vfmul_vv_f32m4(value, __riscv_vle32_v_f32m4(weight + col, vl), vl);
            value = __riscv_vfadd_vv_f32m4(value, __riscv_vle32_v_f32m4(bias + col, vl), vl);
            __riscv_vse32_v_f32m4(y + col, value, vl);
            col += vl;
        }
    }
    return true;
}
} // namespace ocr_norm_affine
