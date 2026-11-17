#pragma once
// Independent default-off gate for Paddle language-model D128 x8 attention.
// Both paths use the existing RVV x8 routines and original scalar/x4 tails.
#include <cstdlib>
#include <cstring>
#include <cstdint>
namespace paddle_llm_fa_x8 {
inline bool enabled() {
    static const bool value = [] {
        const char * p = std::getenv("GGML_SPACEMIT_PADDLE_LLM_FA_X8");
        return p && std::strcmp(p, "1") == 0;
    }();
    return value;
}
inline bool eligible(int64_t dk, int64_t dv) {
    return dk == 128 && dv == 128 && enabled();
}
}
