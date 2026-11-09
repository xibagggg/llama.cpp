#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>

namespace ocr_fa_schedule {
struct range { int64_t begin, end; };
inline bool separate_barrier_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_SPACEMIT_OCR_FA_SEPARATE_BARRIER");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
inline int64_t configured_rows(int64_t tile_rows) {
    static const int64_t value = [tile_rows] {
        const char * text = std::getenv("GGML_SPACEMIT_OCR_FA_DYNAMIC_ROWS");
        if (!text || std::strcmp(text, "0") == 0) return int64_t(0);
        char * end = nullptr;
        const long rows = std::strtol(text, &end, 10);
        if (!*text || *end || rows <= 0 || rows > 65536 || rows % tile_rows) {
            std::fprintf(stderr, "GGML_SPACEMIT_OCR_FA_DYNAMIC_ROWS must be a positive tile multiple <=65536, or 0\n");
            std::abort();
        }
        return int64_t(rows);
    }();
    return value;
}
inline int64_t units(int64_t rows, int64_t heads, int64_t chunk) {
    assert(rows > 0 && heads > 0 && chunk > 0);
    return ((rows + chunk - 1) / chunk) * heads;
}
inline range task(int64_t unit, int64_t rows, int64_t chunk) {
    const int64_t per_head = (rows + chunk - 1) / chunk;
    const int64_t head = unit / per_head, offset = (unit % per_head) * chunk;
    return {head * rows + offset, head * rows + std::min(rows, offset + chunk)};
}
} // namespace ocr_fa_schedule
