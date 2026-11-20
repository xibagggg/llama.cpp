#pragma once

// Top-1 token selection without the candidate array.
//
// With top-k(1) as the first active sampler, a chain keeps one token: the one
// with the largest logit. common_sampler normally gets there by copying every
// logit into a llama_token_data array (12 bytes per vocabulary entry) and
// partially sorting it. The helpers here find that token in the raw logits.
//
// unique_max() accepts an input only when the full path cannot depend on the
// order std::partial_sort visits the candidates in: no element is NaN and
// exactly one element equals the maximum. A tied maximum (+0 and -0 tie), a NaN
// or an empty input is refused, and the caller builds the full array as before.

#include "llama.h"

#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

namespace ocr_top1_argmax {

// The kernels reduce the input block by block. This keeps the largest block
// maximum, the block it came from and whether another block reached it too.
struct block_scan {
    float  best       = 0.0f;
    size_t best_begin = 0;
    bool   have_best  = false;
    bool   tie        = false;

    void add(float block_max, size_t begin) {
        if (!have_best || block_max > best) {
            best       = block_max;
            best_begin = begin;
            have_best  = true;
            tie        = false;
        } else if (block_max == best) {
            tie = true;
        }
    }

    // The maximum is unique when no other block reached it and the winning
    // block holds it exactly once.
    bool finish(const float * x, size_t n, size_t block, bool has_nan, size_t * index) const {
        if (!have_best || has_nan || tie) {
            return false;
        }
        const size_t end = n - best_begin < block ? n : best_begin + block;
        size_t count = 0;
        size_t first = 0;
        for (size_t i = best_begin; i < end; i++) {
            if (x[i] == best) {
                if (count == 0) {
                    first = i;
                }
                count++;
            }
        }
        if (count != 1) {
            return false;
        }
        *index = first;
        return true;
    }
};

// Portable kernel. Four independent fmax chains per block; fmax drops a NaN
// operand, so NaN is tracked separately (v == v is false only for NaN).
inline bool unique_max_scalar(const float * x, size_t n, size_t * index) {
    const size_t block = 256;
    block_scan scan;
    unsigned   ordered = 1;
    for (size_t begin = 0; begin < n; begin += block) {
        const size_t end = n - begin < block ? n : begin + block;
        float m0 = -INFINITY;
        float m1 = -INFINITY;
        float m2 = -INFINITY;
        float m3 = -INFINITY;
        size_t i = begin;
        for (; i + 4 <= end; i += 4) {
            const float a = x[i];
            const float b = x[i + 1];
            const float c = x[i + 2];
            const float d = x[i + 3];
            m0 = std::fmax(m0, a);
            m1 = std::fmax(m1, b);
            m2 = std::fmax(m2, c);
            m3 = std::fmax(m3, d);
            ordered &= unsigned(a == a) & unsigned(b == b) & unsigned(c == c) & unsigned(d == d);
        }
        for (; i < end; i++) {
            m0 = std::fmax(m0, x[i]);
            ordered &= unsigned(x[i] == x[i]);
        }
        scan.add(std::fmax(std::fmax(m0, m1), std::fmax(m2, m3)), begin);
    }
    return scan.finish(x, n, block, ordered == 0, index);
}

#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
// RVV kernel. A block is 16 full LMUL=2 vectors, reduced once per block; vfmax
// drops a NaN operand like fmax does, so NaN is collected in a mask. What is
// left after the last full block is scanned in scalar code.
inline bool unique_max_rvv(const float * x, size_t n, size_t * index) {
    const size_t       vl      = __riscv_vsetvlmax_e32m2();
    const size_t       block   = vl * 16;
    const vfloat32m1_t neg_inf = __riscv_vfmv_v_f_f32m1(-INFINITY, __riscv_vsetvlmax_e32m1());
    block_scan scan;
    vbool16_t  nan   = __riscv_vmclr_m_b16(vl);
    size_t     begin = 0;
    for (; n - begin >= block; begin += block) {
        vfloat32m2_t acc = __riscv_vle32_v_f32m2(x + begin, vl);
        nan = __riscv_vmor_mm_b16(nan, __riscv_vmfne_vv_f32m2_b16(acc, acc, vl), vl);
        for (size_t j = 1; j < 16; j++) {
            const vfloat32m2_t v = __riscv_vle32_v_f32m2(x + begin + j * vl, vl);
            nan = __riscv_vmor_mm_b16(nan, __riscv_vmfne_vv_f32m2_b16(v, v, vl), vl);
            acc = __riscv_vfmax_vv_f32m2(acc, v, vl);
        }
        scan.add(__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(acc, neg_inf, vl)), begin);
    }
    bool has_nan = __riscv_vfirst_m_b16(nan, vl) >= 0;
    if (begin < n) {
        float m = -INFINITY;
        for (size_t i = begin; i < n; i++) {
            has_nan |= x[i] != x[i];
            m = std::fmax(m, x[i]);
        }
        scan.add(m, begin);
    }
    return scan.finish(x, n, block, has_nan, index);
}
#endif

// Index of the single largest element of x[0..n). Returns false when there is a
// NaN, when the maximum is not unique or when n is 0.
inline bool unique_max(const float * x, size_t n, size_t * index) {
#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
    return unique_max_rvv(x, n, index);
#else
    return unique_max_scalar(x, n, index);
#endif
}

// True when top-k heads the active part of the chain. Samplers named "?..." are
// the empty placeholders of disabled stages and leave the candidates alone; any
// other sampler ahead of top-k (logit-bias, penalties, dry, ...) may change the
// logits. The caller checks that k is 1.
inline bool chain_starts_with_top_k(struct llama_sampler * chain) {
    const int n = llama_sampler_chain_n(chain);
    for (int i = 0; i < n; i++) {
        const char * name = llama_sampler_name(llama_sampler_chain_get(chain, i));
        if (name == nullptr) {
            return false;
        }
        if (std::strcmp(name, "top-k") == 0) {
            return true;
        }
        if (name[0] != '?') {
            return false;
        }
    }
    return false;
}

} // namespace ocr_top1_argmax
