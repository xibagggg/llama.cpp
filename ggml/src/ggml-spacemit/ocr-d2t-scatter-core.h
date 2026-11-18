#ifndef OCR_D2T_SCATTER_CORE_H
#define OCR_D2T_SCATTER_CORE_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__riscv_v)
#include <riscv_vector.h>
#endif

namespace ocr_d2t_scatter {

enum class validation { valid, duplicate, invalid };

// Host-only. Validate signed I64 values before narrowing or shifting. Unique
// destination addresses are required for parallel/unordered vector stores.
inline validation validate_i64(const int64_t * ids, size_t rows, size_t full_rows,
                               std::vector<uint32_t> & byte_offsets,
                               std::vector<uint8_t> & seen) {
    byte_offsets.clear();
    if (!ids || full_rows == 0 || full_rows > UINT32_MAX / sizeof(uint32_t))
        return validation::invalid;
    seen.assign(full_rows, 0);
    byte_offsets.resize(rows);
    for (size_t row = 0; row < rows; ++row) {
        const int64_t target = ids[row];
        if (target < 0 || uint64_t(target) >= uint64_t(full_rows)) {
            byte_offsets.clear();
            return validation::invalid;
        }
        if (seen[size_t(target)]) {
            byte_offsets.clear();
            return validation::duplicate;
        }
        seen[size_t(target)] = 1;
        byte_offsets[row] = uint32_t(target) * uint32_t(sizeof(uint32_t));
    }
    return validation::valid;
}

// The launch catalog guarantees unique, in-range offsets and disjoint src/dst.
// U32 loads/stores copy every F32 bit, including signed zero and NaN payloads.
// mode=1: RVV indexed scatter, mode=2: scalar rows-parallel control candidate.
// All storage and options were prepared on the host; no worker allocation,
// environment lookup, lock, logging, or extra collective barrier occurs here.
inline void copy_unique(const uint32_t * src, uint32_t * dst,
                        const uint32_t * byte_offsets,
                        size_t rows, size_t columns, size_t full_rows,
                        unsigned ith, unsigned nth, unsigned mode) {
    const size_t total = rows * columns;
    const size_t per_lane = (total + nth - 1) / nth;
    size_t pos = per_lane * ith;
    const size_t end = pos < total ? (per_lane < total - pos ? pos + per_lane : total) : total;
    while (pos < end) {
        const size_t column = pos / rows;
        const size_t row = pos % rows;
        const size_t count = rows - row < end - pos ? rows - row : end - pos;
        const uint32_t * in = src + column * rows + row;
        uint32_t * out = dst + column * full_rows;
        size_t copied = 0;
#if defined(__riscv_v)
        if (mode == 1) {
            while (copied < count) {
                const size_t vl = __riscv_vsetvl_e32m4(count - copied);
                const vuint32m4_t offsets = __riscv_vle32_v_u32m4(byte_offsets + row + copied, vl);
                const vuint32m4_t bits = __riscv_vle32_v_u32m4(in + copied, vl);
                __riscv_vsuxei32_v_u32m4(out, offsets, bits, vl);
                copied += vl;
            }
        }
#else
        (void) mode;
#endif
        for (; copied < count; ++copied)
            std::memcpy(reinterpret_cast<uint8_t *>(out) + byte_offsets[row + copied],
                        reinterpret_cast<const uint8_t *>(in + copied), sizeof(uint32_t));
        pos += count;
    }
}

} // namespace ocr_d2t_scatter
#endif
