#pragma once
// Optional exact DSpark scalar-row fast paths. All environment parsing stays on
// the host before launch. Workers allocate nothing and add no barriers.
#include "ggml.h"
#include "spacemit-context.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace ocr_dspark_sampling {
struct options {
    bool contiguous_fill = false;
    bool vector_argmax = false;
    bool skip_topk = false;
};
inline options & state() { static options value; return value; }
inline bool selected(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}
inline void prepare_on_host() {
    auto & value = state();
    value.contiguous_fill = selected("GGML_SPACEMIT_DSPARK_CONTIG_FILL");
    value.vector_argmax = selected("GGML_SPACEMIT_DSPARK_VECTOR_ARGMAX");
    value.skip_topk = selected("GGML_SPACEMIT_DSPARK_SKIP_TOPK");
}
inline bool is_nan_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);
}

inline bool contiguous_fill(ggml::spacemit::context & ctx, ggml_tensor * op, float value) {
    if (!state().contiguous_fill || op->type != GGML_TYPE_F32 ||
        op->ne[0] != 1 || op->ne[1] != 248100 || op->ne[3] != 1 ||
        (op->ne[2] != 1 && op->ne[2] != 15) || !ggml_is_contiguous(op)) return false;
    const int64_t size = ggml_nelements(op);
    const int64_t begin = size * ctx.ith / ctx.nth;
    const int64_t end = size * (ctx.ith + 1) / ctx.nth;
    float * dst = static_cast<float *>(op->data);
#if defined(__riscv_vector)
    for (int64_t i = begin; i < end;) {
        const size_t vl = __riscv_vsetvl_e32m4(size_t(end-i));
        __riscv_vse32_v_f32m4(dst+i, __riscv_vfmv_v_f_f32m4(value, vl), vl);
        i += int64_t(vl);
    }
#else
    std::fill(dst + begin, dst + end, value);
#endif
    return true;
}

// Returns false on any NaN so the caller executes its original scalar scan.
// The non-NaN path uses equality with the maximum to select the largest index,
// preserving ties between +0/-0 and ties between infinities exactly.
inline bool vector_argmax_row(const float * x, int32_t n, int32_t & result) {
#if defined(__riscv_vector)
    const size_t full = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t maximum = __riscv_vfmv_v_f_f32m4(-std::numeric_limits<float>::infinity(), full);
    for (int32_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m4(size_t(n-i));
        const vfloat32m4_t values = __riscv_vle32_v_f32m4(x+i, vl);
        if (__riscv_vcpop_m_b8(__riscv_vmfne_vv_f32m4_b8(values, values, vl), vl)) return false;
        maximum = __riscv_vfmax_vv_f32m4_tu(maximum, maximum, values, vl);
        i += int32_t(vl);
    }
    const vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(-std::numeric_limits<float>::infinity(), 1);
    const float greatest = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(maximum, seed, full));
    vuint32m4_t last = __riscv_vmv_v_x_u32m4(0, full);
    for (int32_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e32m4(size_t(n-i));
        const vfloat32m4_t values = __riscv_vle32_v_f32m4(x+i, vl);
        const vbool8_t equal = __riscv_vmfeq_vf_f32m4_b8(values, greatest, vl);
        const vuint32m4_t index = __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(vl), uint32_t(i), vl);
        last = __riscv_vmerge_vvm_u32m4_tu(last, last, index, equal, vl);
        i += int32_t(vl);
    }
    result = int32_t(__riscv_vmv_x_s_u32m1_u32(__riscv_vredmaxu_vs_u32m4_u32m1(
        last, __riscv_vmv_v_x_u32m1(0, 1), full)));
    return true;
#else
    (void)x; (void)n; (void)result;
    return false;
#endif
}
inline bool vector_argmax(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    if (!state().vector_argmax || !src || src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_I32 ||
        src->ne[0] != 248100 || ggml_nrows(src) != 1 || ggml_nelements(op) != 1 ||
        !ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
    if (ctx.ith != 0) return true;
    int32_t index;
    if (!vector_argmax_row(static_cast<const float *>(src->data), int32_t(src->ne[0]), index)) return false;
    *static_cast<int32_t *>(op->data) = index;
    return true;
}

// Preserve the exact libstdc++ heap operations used by std::partial_sort.
// Only skip indices whose value cannot beat the current minimum of the heap.
// With NaNs the comparator is not a strict weak order; discard the private
// heap and fall back to the untouched original implementation in that case.
inline bool skip_topk_row(const float * x, int32_t n, int32_t * output) {
#if defined(__GLIBCXX__)
    constexpr int k = 10;
    int32_t heap[k];
    for (int32_t i = 0; i < k; ++i) {
        if (is_nan_bits(x[i])) return false;
        heap[i] = i;
    }
    auto compare = [x](int32_t a, int32_t b) { return x[a] > x[b]; };
    auto wrapped = __gnu_cxx::__ops::__iter_comp_iter(compare);
    std::make_heap(heap, heap+k, compare);
    for (int32_t cursor = k; cursor < n;) {
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m4(size_t(n-cursor));
        const vfloat32m4_t values = __riscv_vle32_v_f32m4(x+cursor, vl);
        if (__riscv_vcpop_m_b8(__riscv_vmfne_vv_f32m4_b8(values, values, vl), vl)) return false;
        const vbool8_t greater = __riscv_vmfgt_vf_f32m4_b8(values, x[heap[0]], vl);
        const long first = __riscv_vfirst_m_b8(greater, vl);
        if (first < 0) {
            cursor += int32_t(vl);
            continue;
        }
        cursor += int32_t(first);
#else
        if (is_nan_bits(x[cursor])) return false;
        if (!(x[cursor] > x[heap[0]])) { ++cursor; continue; }
#endif
        int32_t candidate = cursor;
        std::__pop_heap(heap, heap+k, &candidate, wrapped);
        ++cursor;
    }
    std::__sort_heap(heap, heap+k, wrapped);
    std::copy_n(heap, k, output);
    std::swap(output[0], output[1]);
    return true;
#else
    (void)x; (void)n; (void)output;
    return false;
#endif
}
inline bool skip_topk(ggml::spacemit::context & ctx, ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    if (!state().skip_topk || !src || src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_I32 ||
        src->ne[0] != 248100 || op->ne[0] != 10 || ggml_nrows(src) != 1 || ggml_nrows(op) != 1 ||
        !ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
    if (ctx.ith != 0) return true;
    return skip_topk_row(static_cast<const float *>(src->data), int32_t(src->ne[0]),
                         static_cast<int32_t *>(op->data));
}
} // namespace ocr_dspark_sampling
