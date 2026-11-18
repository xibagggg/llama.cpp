#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ocr_gdn_tcm {
inline std::atomic<bool> use_tcm{false};
inline void prepare() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_SPACEMIT_OCR_GDN_TCM");
        return value && std::strcmp(value, "1") == 0;
    }();
    use_tcm.store(enabled, std::memory_order_relaxed);
}
inline bool enabled() { return use_tcm.load(std::memory_order_relaxed); }
inline size_t scratch_bytes(int64_t d, int64_t cs = 16) {
    return size_t(d * d + 7 * cs * d + 3 * cs * cs + 4 * cs + 64) * sizeof(float);
}
} // namespace ocr_gdn_tcm
