#pragma once
#include <cstdlib>
#include <cstring>

namespace ocr_q4_pc8 {
inline int mode(){static const int m=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_PC8");return p?std::atoi(p):0;}();return m;}
inline bool rows8(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_PC8_M8");return p&&std::strcmp(p,"1")==0;}();return x;}
inline bool down_rows8(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_PC8_DOWN_M8");return p&&!std::strcmp(p,"1");}();return x;}
inline bool multi_n(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_Q4_PC8_MULTI_N");return p&&std::strcmp(p,"1")==0;}();return x;}
// Prepare each immutable weight panel once, rather than for every M tile.
inline bool prepack(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_PC8_PREPACK");return p&&std::strcmp(p,"1")==0;}();return x;}
inline bool reuse_input(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_PC8_REUSE_INPUT");return p&&!std::strcmp(p,"1");}();return x;}
inline bool trace_reuse(){static const bool x=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_PC8_REUSE_TRACE");return p&&!std::strcmp(p,"1");}();return x;}

// Resolve function-local guards on the dispatching host before publishing tiles.
// AI workers only read the cached options; no first-use getenv or guard waits.
inline void prepare_options_on_host() {
    (void)mode();
    (void)rows8();
    (void)down_rows8();
    (void)multi_n();
    (void)prepack();
    (void)reuse_input();
    (void)trace_reuse();
}
} // namespace ocr_q4_pc8
