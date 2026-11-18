#pragma once
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <cstring>

// Default off. Unlike the rejected v1, this never broadens full_kv(): old
// small shapes and every large-N legacy/IME-PV preparation route stay intact.
namespace ocr_fa_serial_p2_v3 {
// Independent default-off follow-up: mode1/2 v3 remain unchanged when off.
inline bool pair_qk_enabled() {
    static const bool value=[] {
        const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_SERIAL_P2_PAIR_QK");
        return p&&!std::strcmp(p,"1");
    }();
    return value;
}
inline bool pair_qk_report_enabled() {
    static const bool value=[] {
        const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_SERIAL_P2_PAIR_QK_REPORT");
        return p&&!std::strcmp(p,"1");
    }();
    return value;
}
inline std::atomic<uint64_t>& pair_qk_calls() {
    static std::atomic<uint64_t> count{0};return count;
}
inline int mode() {
    static const int value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_OCR_FA_SERIAL_P2");
        return p&&!std::strcmp(p,"1")?1:p&&!std::strcmp(p,"2")?2:0;
    }();
    return value;
}
inline bool eligible(int n,int passes) {
    if(!mode()||passes!=2||n<=1120||n>1536)return false;
    static const bool packing=[] {
        const auto one=[](const char*name) {
            const char*p=std::getenv(name);return p&&!std::strcmp(p,"1");
        };
        return (one("GGML_SPACEMIT_OCR_FA_COMPACT")||one("GGML_SPACEMIT_OCR_FA_Q32"))&&
            one("GGML_SPACEMIT_OCR_FA_FULL_PACK")&&one("GGML_SPACEMIT_OCR_FA_SOFTMAX_PACK");
    }();
    return packing;
}
inline size_t bytes(int n) {
    const size_t padded=(size_t(n)+7)/8*8;
    return 2*16*72*sizeof(_Float16)+(64*72*sizeof(_Float16)+512*sizeof(float))+
        16*size_t(n)*sizeof(float)+16*padded*sizeof(_Float16)+
        72*padded*sizeof(_Float16)+4*512*sizeof(float);
}
// Separate cache lines hold immutable pre-barrier local Q checks and one
// post-barrier ready bit that is written/read only by its owning worker.
constexpr size_t lane_bytes=128;
} // namespace ocr_fa_serial_p2_v3
