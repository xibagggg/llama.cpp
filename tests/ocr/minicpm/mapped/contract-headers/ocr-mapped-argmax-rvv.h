#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace ocr_mapped_argmax_rvv {
// Equivalent to the compact callback's non-NaN scan. IDs are narrowed exactly
// as in that callback, and ties are resolved in target-ID order, not row order.
// With NaNs, leave result untouched so the original full-order scan runs.
inline bool compute(const float * values, const int64_t * ids, int64_t count,
                    int64_t vocabulary, int32_t & result) {
#if defined(__riscv_vector)
    if (count <= 0 || vocabulary <= 0 || vocabulary > INT32_MAX) return false;
    const size_t full = __riscv_vsetvlmax_e32m2();
    auto maximum = __riscv_vfmv_v_f_f32m2(-std::numeric_limits<float>::infinity(), full);
    for (int64_t i = 0; i < count;) {
        const size_t vl = __riscv_vsetvl_e32m2(size_t(count-i));
        const auto value = __riscv_vle32_v_f32m2(values+i, vl);
        if (__riscv_vcpop_m_b16(__riscv_vmfne_vv_f32m2_b16(value, value, vl), vl)) return false;
        maximum = __riscv_vfmax_vv_f32m2_tu(maximum, maximum, value, vl);
        i += int64_t(vl);
    }
    const auto seed = __riscv_vfmv_v_f_f32m1(-std::numeric_limits<float>::infinity(), 1);
    const float greatest = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(maximum, seed, full));
    const int32_t initial = greatest == -std::numeric_limits<float>::infinity()
                          ? int32_t(vocabulary-1) : INT32_MIN;
    auto index = __riscv_vmv_v_x_i32m2(initial, full);
    for (int64_t i = 0; i < count;) {
        const size_t vl = __riscv_vsetvl_e32m2(size_t(count-i));
        const auto value = __riscv_vle32_v_f32m2(values+i, vl);
        const auto equal = __riscv_vmfeq_vf_f32m2_b16(value, greatest, vl);
        const auto mapped64 = __riscv_vle64_v_i64m4(ids+i, vl);
        const auto mapped = __riscv_vncvt_x_x_w_i32m2(mapped64, vl);
        const auto candidate = __riscv_vmax_vv_i32m2(index, mapped, vl);
        index = __riscv_vmerge_vvm_i32m2_tu(index, index, candidate, equal, vl);
        i += int64_t(vl);
    }
    result = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m2_i32m1(
            index, __riscv_vmv_v_x_i32m1(initial, 1), full));
    return true;
#else
    (void)values; (void)ids; (void)count; (void)vocabulary; (void)result;
    return false;
#endif
}
} // namespace ocr_mapped_argmax_rvv
