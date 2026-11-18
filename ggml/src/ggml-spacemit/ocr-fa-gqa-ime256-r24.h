#pragma once

// Include inside rvv_kernels.cpp's anonymous namespace, after its original
// m1/finalize helpers. No standard headers may be included from this location.
// D256 IME attention is approximate: mode 1 changes QK reduction only; mode 2
// also uses tile-online F32 PV with a two-pass representation of probabilities.
// Mode 3 uses IME QK + vector softmax + four-row register-resident FP16 PV.
// All three modes are approximate. Modes 1/2 retain their frozen arithmetic.
// Selection/capacity admission happens once on the host, before any AI launch.
namespace ocr_fa_gqa_ime256_r24 {
constexpr int D = 256, R = 24, C = 64;
constexpr size_t qk_bytes = 67072, full_bytes = 88576;
constexpr unsigned guard_byte = 0xa5;
inline std::atomic<unsigned> selected_mode{0};
inline std::atomic<bool> selected_vector_visibility{false};
inline std::atomic<uint64_t> selected_panels{0}, fallback_panels{0};

inline size_t required_bytes(unsigned mode) { return mode == 1 || mode == 3 ? qk_bytes : full_bytes; }

// Host only. No getenv, logging, initialization, allocation or barrier occurs
// in a worker. A disabled/admission-rejected flag uses the existing dispatcher.
inline void prepare(unsigned workers, size_t bytes, bool pinned_tcm) {
    const char * value = std::getenv("GGML_SPACEMIT_OCR_FA_GQA_IME_R24");
    unsigned mode = value && std::strcmp(value, "1") == 0 ? 1 :
                    value && std::strcmp(value, "2") == 0 ? 2 :
                    value && std::strcmp(value, "3") == 0 ? 3 : 0;
    if (workers != 6 || !pinned_tcm || (mode && bytes < required_bytes(mode))) mode = 0;
    const char * masks = std::getenv("GGML_SPACEMIT_OCR_FA_GQA_IME_MASK_VECTOR");
    selected_vector_visibility.store(mode == 3 && masks && std::strcmp(masks, "1") == 0,
                                     std::memory_order_relaxed);
    selected_mode.store(mode, std::memory_order_relaxed);
    selected_panels.store(0, std::memory_order_relaxed);
    fallback_panels.store(0, std::memory_order_relaxed);
}

struct rows {
    int count = 0;
    const float * query[R]{};
    const ggml_fp16_t * mask[R]{};
    const float * sink[R]{};
    float * output[R]{};
};

struct buffers {
    _Float16 * q;
    void * o;
    _Float16 * kv;
    float * scores;
    _Float16 * high;
    _Float16 * low;
    float * temp;
    float * temp_low;
    float * maximum;
    float * sum;
    unsigned char * guard_front;
    unsigned char * guard_back;
};

inline buffers scratch(unsigned mode, void * memory, size_t bytes) {
    GGML_ASSERT((mode == 1 || mode == 2 || mode == 3) && memory && bytes >= required_bytes(mode));
    GGML_ASSERT((reinterpret_cast<uintptr_t>(memory) & 127) == 0);
    auto * p = static_cast<unsigned char *>(memory);
    buffers b{};
    b.guard_front = p;
    b.q = reinterpret_cast<_Float16 *>(p + 128);
    b.o = p + 12416;
    if (mode == 1 || mode == 3) {
        b.kv = reinterpret_cast<_Float16 *>(p + 24704);
        b.scores = reinterpret_cast<float *>(p + 57472);
        b.temp = reinterpret_cast<float *>(p + 63616);
        b.maximum = reinterpret_cast<float *>(p + 66688);
        b.guard_back = p + 66944;
    } else {
        b.kv = reinterpret_cast<_Float16 *>(p + 36992);
        b.scores = reinterpret_cast<float *>(p + 69760);
        b.high = reinterpret_cast<_Float16 *>(p + 75904);
        b.low = reinterpret_cast<_Float16 *>(p + 78976);
        b.temp = reinterpret_cast<float *>(p + 82048);
        b.temp_low = reinterpret_cast<float *>(p + 85120);
        b.maximum = reinterpret_cast<float *>(p + 88192);
        b.guard_back = p + 88448;
    }
    b.sum = b.maximum + R;
    return b;
}

inline void check_guards(const buffers & b) {
    for (int i = 0; i < 128; ++i) {
        GGML_ASSERT(b.guard_front[i] == guard_byte && b.guard_back[i] == guard_byte);
    }
}


// Three independent query panels and four key/channel panels occupy seven
// input registers; twelve widening 8x8 accumulators occupy v8..v31. V0 is an
// ordinary unmasked data operand here. Every output is a native 24x32 tile.
__attribute__((noinline)) static void pv24_reference(
    const _Float16 * x, const _Float16 * w, float * c, int k) {
    const _Float16 * x1=x+8*k, * x2=x+16*k;
    const _Float16 * w1=w+8*k, * w2=w+16*k, * w3=w+24*k;
    int blocks=k/8;
    asm volatile(
    "vsetvli t0,zero,e32,m2,ta,ma\n\t"
    "vmv.v.i v8,0\n\tvmv.v.i v10,0\n\tvmv.v.i v12,0\n\tvmv.v.i v14,0\n\t"
    "vmv.v.i v16,0\n\tvmv.v.i v18,0\n\tvmv.v.i v20,0\n\tvmv.v.i v22,0\n\t"
    "vmv.v.i v24,0\n\tvmv.v.i v26,0\n\tvmv.v.i v28,0\n\tvmv.v.i v30,0\n\t"
    "vsetvli t0,zero,e16,m1,ta,ma\n\t1:\n\t"
    "vle16.v v0,(%[x])\n\tvle16.v v1,(%[x1])\n\tvle16.v v2,(%[x2])\n\t"
    "vle16.v v3,(%[w])\n\tvle16.v v4,(%[w1])\n\tvle16.v v5,(%[w2])\n\tvle16.v v6,(%[w3])\n\t"
    "smt.vfwmadot v8,v0,v3\n\tsmt.vfwmadot v10,v0,v4\n\t"
    "smt.vfwmadot v12,v0,v5\n\tsmt.vfwmadot v14,v0,v6\n\t"
    "smt.vfwmadot v16,v1,v3\n\tsmt.vfwmadot v18,v1,v4\n\t"
    "smt.vfwmadot v20,v1,v5\n\tsmt.vfwmadot v22,v1,v6\n\t"
    "smt.vfwmadot v24,v2,v3\n\tsmt.vfwmadot v26,v2,v4\n\t"
    "smt.vfwmadot v28,v2,v5\n\tsmt.vfwmadot v30,v2,v6\n\t"
    "addi %[x],%[x],128\n\taddi %[x1],%[x1],128\n\taddi %[x2],%[x2],128\n\t"
    "addi %[w],%[w],128\n\taddi %[w1],%[w1],128\n\t"
    "addi %[w2],%[w2],128\n\taddi %[w3],%[w3],128\n\t"
    "addi %[blocks],%[blocks],-1\n\tbnez %[blocks],1b\n\t"
    "vsetvli t0,zero,e32,m2,ta,ma\n\tvse32.v v8,(%[c])\n\t"
    "addi t0,%[c],256\n\tvse32.v v10,(t0)\n\taddi t0,%[c],512\n\tvse32.v v12,(t0)\n\t"
    "addi t0,%[c],768\n\tvse32.v v14,(t0)\n\taddi t0,%[c],1024\n\tvse32.v v16,(t0)\n\t"
    "addi t0,%[c],1280\n\tvse32.v v18,(t0)\n\taddi t0,%[c],1536\n\tvse32.v v20,(t0)\n\t"
    "addi t0,%[c],1792\n\tvse32.v v22,(t0)\n\t"
    // Base+2048 exceeds signed 12-bit addi; advance t0 incrementally.
    "addi t0,t0,256\n\tvse32.v v24,(t0)\n\taddi t0,t0,256\n\tvse32.v v26,(t0)\n\t"
    "addi t0,t0,256\n\tvse32.v v28,(t0)\n\taddi t0,t0,256\n\tvse32.v v30,(t0)\n\t"
    : [x]"+&r"(x),[x1]"+&r"(x1),[x2]"+&r"(x2),[w]"+&r"(w),[w1]"+&r"(w1),
      [w2]"+&r"(w2),[w3]"+&r"(w3),[blocks]"+&r"(blocks)
    : [c]"r"(c)
    : "t0","v0","v1","v2","v3","v4","v5","v6",
      "v8","v9","v10","v11","v12","v13","v14","v15",
      "v16","v17","v18","v19","v20","v21","v22","v23",
      "v24","v25","v26","v27","v28","v29","v30","v31","memory");
}

inline size_t native_index(int row, int column) {
    return size_t(row / 8) * 256 + (column / 8) * 64 + (row % 8) * 8 + column % 8;
}

// Preserve the original first Q narrowing: f32 -> f16 -> (f32 * 1024) -> f16.
// Scaling by a power of two lifts all nonzero FP16 subnormals into IME's normal
// range. Reject values that could overflow the scaled half representation.
inline bool pack_query(const rows & input, _Float16 * dst) {
    if (input.count != R) std::memset(dst, 0, R * D * sizeof(_Float16));
    for (int row = 0; row < input.count; ++row) {
        for (int d = 0; d < D;) {
            const size_t vl = __riscv_vsetvl_e16m2(D - d);
            const auto channels = __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(vl), d, vl);
            const auto offsets = __riscv_vsll_vx_u32m4(__riscv_vadd_vv_u32m4(
                __riscv_vsll_vx_u32m4(__riscv_vsrl_vx_u32m4(channels, 3, vl), 6, vl),
                __riscv_vand_vx_u32m4(channels, 7, vl), vl), 1, vl);
            const auto half = __riscv_vfncvt_f_f_w_f16m2(
                __riscv_vle32_v_f32m4(input.query[row] + d, vl), vl);
            const auto value = __riscv_vfwcvt_f_f_v_f32m4(half, vl);
            const auto bad = __riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(value, value, vl),
                __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(value, vl), 63.f, vl), vl);
            if (__riscv_vcpop_m_b8(bad, vl)) return false;
            auto * output = dst + size_t(row / 8) * 8 * D + (row % 8) * 8;
            __riscv_vsuxei32_v_f16m2(output, offsets, __riscv_vfncvt_f_f_w_f16m2(
                __riscv_vfmul_vf_f32m4(value, 1024.f, vl), vl), vl);
            d += int(vl);
        }
    }
    return true;
}

// A union-visible position is read once even if only one query can see it.
// Fully masked/uninitialized KV cache positions and the physical tail are never
// read. This is distinct from each query's own per-position mask below.
inline bool pack_key(const char * input, size_t stride, const bool * visible,
                     int count, bool dense, _Float16 * dst) {
    if (!dense) std::memset(dst, 0, C * D * sizeof(_Float16));
    for (int row = 0; row < count; ++row) {
        if (!visible[row]) continue;
        const auto * source = reinterpret_cast<const _Float16 *>(input + size_t(row) * stride);
        for (int d = 0; d < D;) {
            const size_t vl = __riscv_vsetvl_e16m2(D - d);
            const auto channels = __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(vl), d, vl);
            const auto offsets = __riscv_vsll_vx_u32m4(__riscv_vadd_vv_u32m4(
                __riscv_vsll_vx_u32m4(__riscv_vsrl_vx_u32m4(channels, 3, vl), 6, vl),
                __riscv_vand_vx_u32m4(channels, 7, vl), vl), 1, vl);
            const auto value = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(source + d, vl), vl);
            const auto bad = __riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(value, value, vl),
                __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(value, vl), 63.f, vl), vl);
            if (__riscv_vcpop_m_b8(bad, vl)) return false;
            auto * output = dst + size_t(row / 8) * 8 * D + (row % 8) * 8;
            __riscv_vsuxei32_v_f16m2(output, offsets, __riscv_vfncvt_f_f_w_f16m2(
                __riscv_vfmul_vf_f32m4(value, 1024.f, vl), vl), vl);
            d += int(vl);
        }
    }
    return true;
}

// Transpose V into 256 output-channel rows, reduction length 64. K and V
// share one 32-KiB allocation; all QK results must exist before this overwrite.
inline bool pack_value(const char * input, size_t stride, const bool * visible,
                       int count, bool dense, _Float16 * dst) {
    if (!dense) std::memset(dst, 0, D * C * sizeof(_Float16));
    for (int k = 0; k < count; ++k) {
        if (!visible[k]) continue;
        const auto * source = reinterpret_cast<const _Float16 *>(input + size_t(k) * stride);
        for (int d = 0; d < D;) {
            const size_t vl = __riscv_vsetvl_e16m2(D - d);
            const auto channels = __riscv_vadd_vx_u32m4(__riscv_vid_v_u32m4(vl), d, vl);
            const auto offset = __riscv_vadd_vv_u32m4(
                __riscv_vmul_vx_u32m4(__riscv_vsrl_vx_u32m4(channels, 3, vl), 8 * C, vl),
                __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(channels, 7, vl), 3, vl), vl);
            const auto offsets = __riscv_vsll_vx_u32m4(
                __riscv_vadd_vx_u32m4(offset, (k / 8) * 64 + k % 8, vl), 1, vl);
            const auto value = __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(source + d, vl), vl);
            const auto bad = __riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(value, value, vl),
                __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(value, vl), 63.f, vl), vl);
            if (__riscv_vcpop_m_b8(bad, vl)) return false;
            __riscv_vsuxei32_v_f16m2(dst, offsets, __riscv_vfncvt_f_f_w_f16m2(
                __riscv_vfmul_vf_f32m4(value, 1024.f, vl), vl), vl);
            d += int(vl);
        }
    }
    return true;
}

inline void pack_probability(const float * input, int count, _Float16 * high, _Float16 * low) {
    if (count != R) {
        std::memset(high, 0, R * C * sizeof(_Float16));
        std::memset(low, 0, R * C * sizeof(_Float16));
    }
    constexpr size_t vl = C;
    const auto k = __riscv_vid_v_u32m4(vl);
    const auto offsets = __riscv_vsll_vx_u32m4(__riscv_vadd_vv_u32m4(
        __riscv_vsll_vx_u32m4(__riscv_vsrl_vx_u32m4(k, 3, vl), 6, vl),
        __riscv_vand_vx_u32m4(k, 7, vl), vl), 1, vl);
    for (int row = 0; row < count; ++row) {
        const auto value = __riscv_vle32_v_f32m4(input + row * C, vl);
        const auto rounded_hi = __riscv_vfncvt_f_f_w_f16m2(__riscv_vfmul_vf_f32m4(value, 1024.f, vl), vl);
        const auto bits = __riscv_vreinterpret_v_f16m2_u16m2(rounded_hi);
        const auto subnormal = __riscv_vmseq_vx_u16m2_b8(__riscv_vand_vx_u16m2(bits, 0x7c00, vl), 0, vl);
        // IME flushes a subnormal operand. Clear it BEFORE subtracting the
        // restored high part, so low also represents this tiny-P contribution.
        const auto hi = __riscv_vreinterpret_v_u16m2_f16m2(
            __riscv_vmerge_vxm_u16m2(bits, 0, subnormal, vl));
        const auto residual = __riscv_vfsub_vv_f32m4(value,
            __riscv_vfmul_vf_f32m4(__riscv_vfwcvt_f_f_v_f32m4(hi, vl), 0x1p-10f, vl), vl);
        const size_t base = size_t(row / 8) * 8 * C + (row % 8) * 8;
        __riscv_vsuxei32_v_f16m2(high + base, offsets, hi, vl);
        __riscv_vsuxei32_v_f16m2(low + base, offsets, __riscv_vfncvt_f_f_w_f16m2(
            __riscv_vfmul_vf_f32m4(residual, 0x1p24f, vl), vl), vl);
    }
}

inline void finalize_f32(float * dst, const float * output, float S, float M, const float * sink) {
    float ms = 1.f;
    if (sink) {
        const float s = *sink;
        float vs = 1.f;
        if (s > M) ms = expf(M - s);
        else vs = expf(s - M);
        S = S * ms + vs;
    }
    const float inv = S == 0.f ? 0.f : 1.f / S;
    for (int d = 0; d < D;) {
        const size_t vl = __riscv_vsetvl_e32m4(D - d);
        auto value = __riscv_vle32_v_f32m4(output + d, vl);
        if (ms != 1.f) value = __riscv_vfmul_vf_f32m4(value, ms, vl);
        __riscv_vse32_v_f32m4(dst + d, __riscv_vfmul_vf_f32m4(value, inv, vl), vl);
        d += int(vl);
    }
}


// Mode3 gathers the native IME tile directly into row-major scores. Only the
// active physical key range may load mask data. No scalar native-index loop.
inline bool unpack_qk_vector(const buffers & b, const rows & input,
                             int64_t start, int count, int first, float scale) {
    constexpr size_t vl = 32;
    const auto columns = __riscv_vid_v_u32m1(vl);
    const auto offsets = __riscv_vsll_vx_u32m1(__riscv_vadd_vv_u32m1(
        __riscv_vsll_vx_u32m1(__riscv_vsrl_vx_u32m1(columns, 3, vl), 6, vl),
        __riscv_vand_vx_u32m1(columns, 7, vl), vl), 2, vl);
    const int live = std::max(0, std::min(32, count - first));
    const auto physical = __riscv_vmsltu_vx_u32m1_b32(columns, live, vl);
    for (int row = 0; row < input.count; ++row) {
        const size_t base = size_t(row / 8) * 256 + (row % 8) * 8;
        auto score = __riscv_vfmul_vf_f32m1(
            __riscv_vluxei32_v_f32m1(b.temp + base, offsets, vl), 0x1p-20f, vl);
        score = __riscv_vfmul_vf_f32m1(score, scale, vl);
        auto active = physical;
        if (input.mask[row] && live) {
            const auto bias = __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2_m(
                physical, reinterpret_cast<const _Float16 *>(input.mask[row]) + start + first, vl), vl);
            active = __riscv_vmand_mm_b32(physical,
                __riscv_vmfne_vf_f32m1_b32(bias, -INFINITY, vl), vl);
            score = __riscv_vfadd_vv_f32m1(score, bias, vl);
        }
        const auto bad = __riscv_vmand_mm_b32(active, __riscv_vmor_mm_b32(
            __riscv_vmfne_vv_f32m1_b32(score, score, vl),
            __riscv_vmfgt_vf_f32m1_b32(__riscv_vfabs_v_f32m1(score, vl), __FLT_MAX__, vl), vl), vl);
        if (__riscv_vcpop_m_b32(bad, vl)) return false;
        __riscv_vse32_v_f32m1(b.scores + row * C + first,
            __riscv_vmerge_vvm_f32m1(__riscv_vfmv_v_f_f32m1(-INFINITY, vl), score, active, vl), vl);
    }
    return true;
}

// K is dead when this runs; reuse its 32 KiB as raw row-major V. Values do
// not enter IME and need neither scaling nor transpose nor range rejection.
// A fully hidden or physical tail row must never read the source cache.
inline bool copy_value_finite(_Float16 * dst, const _Float16 * source, int count) {
    for (int i = 0; i < count;) {
        const size_t vl = __riscv_vsetvl_e16m8(count - i);
        const auto value = __riscv_vle16_v_f16m8(source + i, vl);
        const auto bits = __riscv_vreinterpret_v_f16m8_u16m8(value);
        const auto special = __riscv_vmseq_vx_u16m8_b2(
            __riscv_vand_vx_u16m8(bits, 0x7c00, vl), 0x7c00, vl);
        if (__riscv_vcpop_m_b2(special, vl)) return false;
        __riscv_vse16_v_f16m8(dst + i, value, vl);
        i += int(vl);
    }
    return true;
}

inline bool pack_value_raw(const char * input, size_t stride, const bool * visible,
                           int count, bool dense, _Float16 * dst) {
    if (dense && stride == D * sizeof(_Float16)) {
        return copy_value_finite(dst, reinterpret_cast<const _Float16 *>(input), C * D);
    }
    for (int row = 0; row < C; ++row) {
        if (row >= count || !visible[row]) {
            rvv_fill_f16(dst + row * D, 0.f, D);
        } else {
            if (!copy_value_finite(dst + row * D,
                    reinterpret_cast<const _Float16 *>(input + size_t(row) * stride), D)) return false;
        }
    }
    return true;
}

inline float softmax_vector_mode3(float * row, float maximum) {
    constexpr size_t vl = C;
    const auto score = __riscv_vle32_v_f32m2(row, vl);
    const auto masked = __riscv_vmfeq_vf_f32m2_b16(score, -INFINITY, vl);
    auto probability = rvv_expf_approx_f32m2(__riscv_vfsub_vf_f32m2(score, maximum, vl), vl);
    probability = __riscv_vmerge_vvm_f32m2(probability, __riscv_vfmv_v_f_f32m2(0.f, vl), masked, vl);
    __riscv_vse32_v_f32m2(row, probability, vl);
    return reduce_sum_f32m2_vlen1024(probability, vl);
}

// FA2's essential PV schedule: V is loaded once for four query rows; their
// FP16 accumulators remain registers for all 64 keys, and store once/tile.
// Padded panel rows have P=0 and O=0. They never write public destinations.
inline void pv_vector_mode3(const buffers & b, int count) {
    for (int row = 0; row < count; ++row) {
        const auto scores = __riscv_vle32_v_f32m2(b.scores + row * C, C);
        auto seed = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);
        seed = __riscv_vfredmax_vs_f32m2_f32m1(scores, seed, C);
        const float tile_max = __riscv_vfmv_f_s_f32m1_f32(seed);
        if (tile_max == -INFINITY) {
            __riscv_vse32_v_f32m2(b.scores + row * C, __riscv_vfmv_v_f_f32m2(0.f, C), C);
            continue;
        }
        if (tile_max > b.maximum[row]) {
            const float ms = expf(b.maximum[row] - tile_max);
            b.maximum[row] = tile_max;
            b.sum[row] *= ms;
            rvv_scale_f16(static_cast<_Float16 *>(b.o) + row * D, ms, D);
        }
        b.sum[row] += softmax_vector_mode3(b.scores + row * C, b.maximum[row]);
    }
    for (int d = 0; d < D;) {
        const size_t vl = __riscv_vsetvl_e16m2(D - d);
        for (int row = 0; row < count; row += 4) {
            auto * o0 = static_cast<_Float16 *>(b.o) + (row + 0) * D + d;
            auto * o1 = static_cast<_Float16 *>(b.o) + (row + 1) * D + d;
            auto * o2 = static_cast<_Float16 *>(b.o) + (row + 2) * D + d;
            auto * o3 = static_cast<_Float16 *>(b.o) + (row + 3) * D + d;
            const auto * p0 = b.scores + (row + 0) * C;
            const auto * p1 = b.scores + (row + 1) * C;
            const auto * p2 = b.scores + (row + 2) * C;
            const auto * p3 = b.scores + (row + 3) * C;
            auto a0 = __riscv_vle16_v_f16m2(o0, vl);
            auto a1 = __riscv_vle16_v_f16m2(o1, vl);
            auto a2 = __riscv_vle16_v_f16m2(o2, vl);
            auto a3 = __riscv_vle16_v_f16m2(o3, vl);
            for (int k = 0; k < C; ++k) {
                const auto v = __riscv_vle16_v_f16m2(b.kv + k * D + d, vl);
                a0 = __riscv_vfmacc_vf_f16m2(a0, (_Float16) p0[k], v, vl);
                a1 = __riscv_vfmacc_vf_f16m2(a1, (_Float16) p1[k], v, vl);
                a2 = __riscv_vfmacc_vf_f16m2(a2, (_Float16) p2[k], v, vl);
                a3 = __riscv_vfmacc_vf_f16m2(a3, (_Float16) p3[k], v, vl);
            }
            __riscv_vse16_v_f16m2(o0, a0, vl);
            __riscv_vse16_v_f16m2(o1, a1, vl);
            __riscv_vse16_v_f16m2(o2, a2, vl);
            __riscv_vse16_v_f16m2(o3, a3, vl);
        }
        d += int(vl);
    }
}


// Exact mask classification by binary16 bits, before any K/V cache read.
// GQA rows often share the same mask pointer. Deduplicate only identical
// pointers; arbitrary distinct per-query masks retain their own validation.
inline bool vector_visibility(const rows & input, int64_t start, int count,
                              bool * visible, bool & any, bool & dense) {
    const size_t vl = __riscv_vsetvl_e16m1(count);
    GGML_ASSERT(vl == size_t(count));
    auto united = __riscv_vmclr_m_b16(vl);
    for (int row = 0; row < input.count; ++row) {
        if (row && input.mask[row] == input.mask[row - 1]) continue;
        if (!input.mask[row]) { united = __riscv_vmset_m_b16(vl); continue; }
        const auto half = __riscv_vle16_v_f16m1(
            reinterpret_cast<const _Float16 *>(input.mask[row]) + start, vl);
        const auto bits = __riscv_vreinterpret_v_f16m1_u16m1(half);
        const auto active = __riscv_vmsne_vx_u16m1_b16(bits, 0xfc00, vl);
        const auto nonfinite = __riscv_vmseq_vx_u16m1_b16(
            __riscv_vand_vx_u16m1(bits, 0x7c00, vl), 0x7c00, vl);
        if (__riscv_vcpop_m_b16(__riscv_vmand_mm_b16(active, nonfinite, vl), vl)) return false;
        united = __riscv_vmor_mm_b16(united, active, vl);
    }
    const size_t live = __riscv_vcpop_m_b16(united, vl);
    any = live != 0; dense = count == C && live == size_t(count);
    uint16_t expanded[C];
    __riscv_vse16_v_u16m1(expanded, __riscv_vmerge_vxm_u16m1(
        __riscv_vmv_v_x_u16m1(0, vl), 1, united, vl), vl);
    for (int j = 0; j < count; ++j) visible[j] = expanded[j] != 0;
    return true;
}

// Returns false before writing any public output. Caller then recomputes ONLY
// its owned rows with original m1, including failures discovered in late tiles.
// Guards are checked on both success and fallback, so an overrun is never hidden.
inline bool compute_panel(unsigned mode, const rows & input, const char * key, const char * value,
                          int64_t length, size_t key_stride, size_t value_stride,
                          float scale, float cap, void * memory, size_t bytes) {
    GGML_ASSERT(input.count > 0 && input.count <= R && length > 0);
    const buffers b = scratch(mode, memory, bytes);
    std::memset(b.guard_front, guard_byte, 128);
    std::memset(b.guard_back, guard_byte, 128);
    auto reject = [&] { check_guards(b); return false; };
    if (!pack_query(input, b.q)) return reject();
    std::memset(b.o, 0, size_t(R) * D * (mode == 2 ? sizeof(float) : sizeof(_Float16)));
    if (mode == 3) std::memset(b.scores, 0, R * C * sizeof(float));
    for (int row = 0; row < R; ++row) { b.maximum[row] = -INFINITY; b.sum[row] = 0.f; }
    const bool use_vector_visibility = mode == 3 && selected_vector_visibility.load(std::memory_order_relaxed);
    for (int64_t start = 0; start < length; start += C) {
        const int count = int(std::min<int64_t>(C, length - start));
        bool visible[C]{};
        bool any = false, dense = count == C;
        if (use_vector_visibility) {
            if (!vector_visibility(input, start, count, visible, any, dense)) return reject();
        } else {
        for (int j = 0; j < count; ++j) {
            for (int row = 0; row < input.count; ++row) {
                const float bias = input.mask[row] ? float(reinterpret_cast<const _Float16 *>(input.mask[row])[start + j]) : 0.f;
                if (bias != -INFINITY && !std::isfinite(bias)) return reject();
                visible[j] |= bias != -INFINITY;
            }
            any |= visible[j]; dense &= visible[j];
        }
        }
        if (!any) continue;
        if (!pack_key(key + size_t(start) * key_stride, key_stride, visible, count, dense, b.kv)) return reject();
        // The primitive writes a native 24x32 tile. Convert to row-major scores;
        // neither primitive nor packing contains D72-specific assumptions.
        for (int first = 0; first < C; first += 32) {
            pv24_reference(b.q, b.kv + first * D, b.temp, D);
            if (mode == 3 && cap == 0.f) {
                if (!unpack_qk_vector(b, input, start, count, first, scale)) return reject();
                continue;
            }
            for (int row = 0; row < input.count; ++row) {
                for (int j = 0; j < 32; ++j) {
                    const int position = first + j;
                    const float bias = position < count && input.mask[row] ?
                        float(reinterpret_cast<const _Float16 *>(input.mask[row])[start + position]) : 0.f;
                    float score = -INFINITY;
                    if (position < count && bias != -INFINITY) {
                        score = b.temp[native_index(row, j)] * 0x1p-20f;
                        score = score * scale;
                        if (cap != 0.f) score = cap * tanhf(score);
                        score += bias;
                        if (!std::isfinite(score)) return reject();
                    }
                    b.scores[row * C + position] = score;
                }
            }
        }
        if (mode == 1) {
            // Original m1's per-position FP16 PV update, including its half
            // scalar conversion, rounding, expf and S/M operation order.
            for (int row = 0; row < input.count; ++row) {
                for (int j = 0; j < count; ++j) {
                    const float score = b.scores[row * C + j];
                    if (score == -INFINITY) continue;
                    const float old = b.maximum[row];
                    float ms = 1.f, vs = 1.f;
                    if (score > b.maximum[row]) {
                        b.maximum[row] = score; ms = expf(old - b.maximum[row]);
                    } else vs = expf(score - b.maximum[row]);
                    rvv_scale_mad_f16(static_cast<_Float16 *>(b.o) + row * D, ms, vs,
                        reinterpret_cast<const _Float16 *>(value + size_t(start + j) * value_stride), D);
                    b.sum[row] = b.sum[row] * ms + vs;
                }
            }
        } else if (mode == 3) {
            if (!pack_value_raw(value + size_t(start) * value_stride, value_stride, visible, count, dense, b.kv)) return reject();
            pv_vector_mode3(b, input.count);
        } else {
            for (int row = 0; row < input.count; ++row) {
                float maximum = -INFINITY;
                for (int j = 0; j < C; ++j) maximum = std::max(maximum, b.scores[row * C + j]);
                if (maximum == -INFINITY) {
                    std::memset(b.scores + row * C, 0, C * sizeof(float));
                    continue;
                }
                maximum = std::max(maximum, b.maximum[row]);
                const float alpha = b.maximum[row] == -INFINITY ? 0.f : expf(b.maximum[row] - maximum);
                float sum = 0.f;
                for (int j = 0; j < C; ++j) {
                    const float score = b.scores[row * C + j];
                    const float probability = score == -INFINITY ? 0.f : expf(score - maximum);
                    b.scores[row * C + j] = probability; sum += probability;
                }
                b.maximum[row] = maximum;
                b.sum[row] = b.sum[row] * alpha + sum;
                auto * output = static_cast<float *>(b.o) + row * D;
                if (alpha != 1.f) {
                    for (int d = 0; d < D;) {
                        const size_t vl = __riscv_vsetvl_e32m4(D - d);
                        __riscv_vse32_v_f32m4(output + d,
                            __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(output + d, vl), alpha, vl), vl);
                        d += int(vl);
                    }
                }
            }
            pack_probability(b.scores, input.count, b.high, b.low);
            if (!pack_value(value + size_t(start) * value_stride, value_stride, visible, count, dense, b.kv)) return reject();
            for (int first = 0; first < D; first += 32) {
                pv24_reference(b.high, b.kv + first * C, b.temp, C);
                pv24_reference(b.low, b.kv + first * C, b.temp_low, C);
                for (int row = 0; row < input.count; ++row) {
                    auto * output = static_cast<float *>(b.o) + row * D + first;
                    for (int j = 0; j < 32; ++j) {
                        const size_t index = native_index(row, j);
                        output[j] += b.temp[index] * 0x1p-20f + b.temp_low[index] * 0x1p-34f;
                    }
                }
            }
        }
    }
    check_guards(b);
    // Commit only after all tiles succeeded. A following unit can reuse TCM.
    for (int row = 0; row < input.count; ++row) {
        if (mode == 1 || mode == 3) flash_attn_ext_finalize_row_f16(input.output[row],
            static_cast<const _Float16 *>(b.o) + row * D, D, b.sum[row], b.maximum[row], input.sink[row]);
        else finalize_f32(input.output[row], static_cast<const float *>(b.o) + row * D,
            b.sum[row], b.maximum[row], input.sink[row]);
    }
    return true;
}

inline bool supports(const ggml_tensor * dst) {
    if (!dst) return false;
    const auto * q = dst->src[0]; const auto * k = dst->src[1]; const auto * v = dst->src[2];
    const auto * mask = dst->src[3]; const auto * sinks = dst->src[4];
    const auto * parameters = reinterpret_cast<const float *>(dst->op_params);
    if (!q || !k || !v || !q->data || !k->data || !v->data || !dst->data ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 ||
        q->ne[0] != D || k->ne[0] != D || v->ne[0] != D ||
        (q->ne[1] != 15 && q->ne[1] != 16) || (q->ne[2] != 8 && q->ne[2] != 16) ||
        q->ne[2] != 4 * k->ne[2] || k->ne[2] != v->ne[2] ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        k->ne[1] <= 0 || k->ne[1] != v->ne[1] || q->nb[0] != 4 || k->nb[0] != 2 || v->nb[0] != 2 ||
        dst->ne[0] != D || dst->ne[1] != q->ne[2] || dst->ne[2] != q->ne[1] || dst->ne[3] != 1 ||
        dst->nb[0] != 4 || dst->nb[1] != D * sizeof(float) ||
        dst->nb[2] != dst->nb[1] * size_t(dst->ne[1]) ||
        parameters[1] != 0.f || !std::isfinite(parameters[0]) || !std::isfinite(parameters[2]) ||
        (parameters[2] != 0.f && !std::isfinite(parameters[0] / parameters[2])) ||
        (mask && (!mask->data || mask->type != GGML_TYPE_F16 || mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1] ||
                  mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[0] != sizeof(_Float16))) ||
        (sinks && (!sinks->data || sinks->type != GGML_TYPE_F32 || !ggml_is_contiguous(sinks) ||
                   sinks->ne[0] != q->ne[2] || ggml_nelements(sinks) != q->ne[2]))) return false;
    return true;
}

inline bool forward(ggml::spacemit::context & ctx, ggml_tensor * dst) {
    const unsigned mode = selected_mode.load(std::memory_order_relaxed);
    if (!mode || !supports(dst)) return false;
    const auto * q = dst->src[0]; const auto * k = dst->src[1]; const auto * v = dst->src[2];
    const auto * mask = dst->src[3]; const auto * sinks = dst->src[4];
    const auto * parameters = reinterpret_cast<const float *>(dst->op_params);
    GGML_ASSERT(ctx.nth == 6 && ctx.shared.size >= required_bytes(mode) && __riscv_vlenb() == 128);
    GGML_ASSERT(ctx.shared.data && (reinterpret_cast<uintptr_t>(ctx.shared.data) & 127) == 0);
    const float cap = parameters[2], scale = cap == 0.f ? parameters[0] : parameters[0] / cap;
    const int panels = int((q->ne[1] * 4 + R - 1) / R);
    for (int64_t unit = ctx.ith; unit < k->ne[2] * panels; unit += ctx.nth) {
        const int64_t kv_head = unit / panels;
        const int panel = int(unit % panels);
        const int first = int(q->ne[1] * 4 * panel / panels);
        rows input{};
        input.count = int(q->ne[1] * 4 * (panel + 1) / panels) - first;
        for (int row = 0; row < input.count; ++row) {
            const int token = (first + row) / 4;
            const int64_t head = 4 * kv_head + (first + row) % 4;
            input.query[row] = reinterpret_cast<const float *>(static_cast<const char *>(q->data) + token * q->nb[1] + head * q->nb[2]);
            input.output[row] = reinterpret_cast<float *>(static_cast<char *>(dst->data) + token * dst->nb[2] + head * dst->nb[1]);
            input.mask[row] = mask ? reinterpret_cast<const ggml_fp16_t *>(static_cast<const char *>(mask->data) + token * mask->nb[1]) : nullptr;
            input.sink[row] = sinks ? static_cast<const float *>(sinks->data) + head : nullptr;
        }
        const char * key = static_cast<const char *>(k->data) + size_t(kv_head) * k->nb[2];
        const char * value = static_cast<const char *>(v->data) + size_t(kv_head) * v->nb[2];
        selected_panels.fetch_add(1, std::memory_order_relaxed);
        if (!compute_panel(mode, input, key, value, k->ne[1], k->nb[1], v->nb[1], scale, cap, ctx.shared.data, ctx.shared.size)) {
            fallback_panels.fetch_add(1, std::memory_order_relaxed);
            for (int row = 0; row < input.count; ++row) {
                flash_attn_ext_f16_one_chunk_inner_vlen1024_vf16_m1(input.query[row], key, value,
                    input.mask[row], input.sink[row], input.output[row], scale, cap, 1.f,
                    k->ne[1], k->nb[1], v->nb[1], D, D);
            }
        }
    }
    return true;
}
} // namespace ocr_fa_gqa_ime256_r24
