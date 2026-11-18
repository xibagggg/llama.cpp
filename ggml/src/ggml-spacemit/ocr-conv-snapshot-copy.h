#pragma once
#include "ggml.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#if defined(__riscv_v)
#include <riscv_vector.h>
#endif

namespace ocr_conv_snapshot_copy {
inline std::atomic<bool> enabled{false};

// Dispatching host only. The worker has no first-use guard/environment read.
inline void prepare_on_host() {
    const char * option = std::getenv("GGML_SPACEMIT_OCR_CONV_SNAPSHOT_COPY");
    enabled.store(option && std::strcmp(option,"1") == 0, std::memory_order_relaxed);
}

inline bool copy(const ggml_tensor * src, ggml_tensor * dst, unsigned ith, unsigned nth) {
    if (!enabled.load(std::memory_order_relaxed) || nth != 6 || ith >= nth ||
        !src || !dst || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !src->data || !dst->data || src->ne[0] != 3 || src->ne[1] <= 0 ||
        src->ne[2] != 1 || src->ne[3] != 1 || src->nb[0] != sizeof(float) ||
        src->nb[1] <= 4*sizeof(float) || src->nb[1] % sizeof(float) ||
        src->nb[1] > size_t(PTRDIFF_MAX) || !ggml_is_contiguous(dst)) return false;
    const size_t rows = size_t(src->ne[1]), stride = src->nb[1];
    if (rows > size_t(INT64_MAX)/3 || rows > SIZE_MAX/(3*sizeof(float)) ||
        rows-1 > (SIZE_MAX-3*sizeof(float))/stride ||
        ggml_nelements(dst) != int64_t(3*rows)) return false;
    const size_t src_bytes = (rows-1)*stride+3*sizeof(float);
    const size_t dst_bytes = rows*3*sizeof(float);
    const uintptr_t from = reinterpret_cast<uintptr_t>(src->data);
    const uintptr_t to = reinterpret_cast<uintptr_t>(dst->data);
    if (from % alignof(float) || to % alignof(float) ||
        src_bytes > UINTPTR_MAX-from || dst_bytes > UINTPTR_MAX-to ||
        !(from+src_bytes <= to || to+dst_bytes <= from)) return false;
    const size_t per_lane = (rows+nth-1)/nth;
    size_t row = per_lane*ith;
    const size_t end = row < rows ? (per_lane < rows-row ? row+per_lane : rows) : rows;
    for (; row < end;) {
#if defined(__riscv_v)
        const size_t vl = __riscv_vsetvl_e32m1(end-row);
        const float * in = reinterpret_cast<const float *>(static_cast<const char *>(src->data)+row*stride);
        float * out = static_cast<float *>(dst->data)+3*row;
        const vfloat32m1x3_t fields = __riscv_vlsseg3e32_v_f32m1x3(in,ptrdiff_t(stride),vl);
        __riscv_vsseg3e32_v_f32m1x3(out,fields,vl);
        row += vl;
#else
        // Portable fixture path only; production is built with __riscv_v.
        std::memcpy(static_cast<char *>(dst->data)+row*3*sizeof(float),
                    static_cast<const char *>(src->data)+row*stride,3*sizeof(float));
        ++row;
#endif
    }
    return true;
}
} // namespace ocr_conv_snapshot_copy
