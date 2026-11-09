#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Generated from the production function bodies by test-rvv-copy-contract.sh.
#include "rvv-copy-under-test.h"

template <typename T>
__attribute__((noinline)) static bool check_transpose(int rows, int cols, int padding) {
    const int source_stride = cols + padding;
    const int target_stride = rows + padding;
    std::vector<T> source(size_t(rows) * source_stride + 64, T(0x5a));
    std::vector<T> target(size_t(cols) * target_stride + 64, T(0xa5));
    auto expected = target;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            source[size_t(r) * source_stride + c] = T(r * 1009 + c * 37 + 11);
            expected[size_t(c) * target_stride + r] = source[size_t(r) * source_stride + c];
        }
    }
    const auto original = source;
    const size_t vl = __riscv_vsetvl_e32m1(32);
    uint32_t sentinel[32], restored[32];
    for (size_t i = 0; i < 32; ++i) sentinel[i] = 0x12340000u + uint32_t(i);
    auto live = __riscv_vle32_v_u32m1(sentinel, vl);
    // Keep a vector live across the inlined assembly. Undeclared clobbers may
    // corrupt values in a caller even when the transpose output looks right.
    asm volatile("" : "+vr"(live) : : "memory");
    if constexpr (sizeof(T) == 4) {
        rvv_transposed_s32_mn_to_nm(reinterpret_cast<int8_t *>(target.data()), target_stride * sizeof(T),
                                  reinterpret_cast<int8_t *>(source.data()), source_stride * sizeof(T), rows, cols);
    } else {
        rvv_transposed_s16_mn_to_nm(reinterpret_cast<int8_t *>(target.data()), target_stride * sizeof(T),
                                  reinterpret_cast<int8_t *>(source.data()), source_stride * sizeof(T), rows, cols);
    }
    asm volatile("" : "+vr"(live) : : "memory");
    __riscv_vse32_v_u32m1(restored, live, vl);
    if (target != expected || source != original || std::memcmp(sentinel, restored, vl * sizeof(uint32_t))) {
        std::fprintf(stderr, "transpose failed: bits=%zu rows=%d cols=%d padding=%d data=%d input=%d live=%d\n",
                     sizeof(T) * 8, rows, cols, padding, target == expected, source == original,
                     !std::memcmp(sentinel, restored, vl * sizeof(uint32_t)));
        return false;
    }
    return true;
}

__attribute__((noinline)) static bool check_copy(int bytes, int source_offset, int target_offset) {
    if (bytes < 0 || bytes > 65539 || source_offset < 0 || source_offset > 17 ||
        target_offset < 0 || target_offset > 31) return false;
    std::vector<uint8_t> source(size_t(bytes) + 256), target(size_t(bytes) + 256, 0xa5);
    for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i * 17 + 31);
    const auto original = source;
    auto expected = target;
    std::memcpy(expected.data() + target_offset, source.data() + source_offset, bytes);
    memcpy1d(target.data() + target_offset, source.data() + source_offset, bytes);
    if (target != expected || source != original) {
        std::fprintf(stderr, "copy failed: bytes=%d src_offset=%d dst_offset=%d\n", bytes, source_offset, target_offset);
        return false;
    }
    return true;
}

int main() {
    size_t passed = 0;
    for (int rows : {1, 7, 8, 9, 16, 17, 72}) {
        for (int cols : {1, 7, 8, 9, 31, 32, 33, 72, 252}) {
            for (int padding : {0, 3}) {
                if (!check_transpose<uint16_t>(rows, cols, padding) ||
                    !check_transpose<uint32_t>(rows, cols, padding)) return 1;
                passed += 2;
            }
        }
    }
    for (int bytes : {0, 1, 7, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 4097, 65539}) {
        for (int source_offset : {0, 1, 17}) {
            for (int target_offset : {0, 1, 31}) {
                if (!check_copy(bytes, source_offset, target_offset)) return 2;
                ++passed;
            }
        }
    }
    std::printf("PASS vlen_bits=%zu cases=%zu\n", size_t(__riscv_vlenb()) * 8, passed);
    return 0;
}
