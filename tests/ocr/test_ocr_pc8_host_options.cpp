#include "ocr-pc8-options.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static thread_local bool ai_worker = false;
static std::atomic<unsigned> worker_environment_reads{0}, host_environment_reads{0};
extern "C" char * __real_getenv(const char *);
extern "C" char * __wrap_getenv(const char * name) {
    if (ai_worker) ++worker_environment_reads;
    else ++host_environment_reads;
    return __real_getenv(name);
}

int main() {
    setenv("GGML_SPACEMIT_OCR_Q4_PC8", "2", 1);
    const char * enabled[] = {"GGML_SPACEMIT_OCR_Q4_PC8_M8", "GGML_SPACEMIT_OCR_Q4_PC8_DOWN_M8",
                             "GGML_SPACEMIT_OCR_Q4_PC8_MULTI_N", "GGML_SPACEMIT_OCR_PC8_PREPACK",
                             "GGML_SPACEMIT_OCR_PC8_REUSE_INPUT"};
    for (const auto * name : enabled) setenv(name, "1", 1);
    setenv("GGML_SPACEMIT_OCR_PC8_REUSE_TRACE", "0", 1);
    ocr_q4_pc8::prepare_options_on_host();
    if (host_environment_reads != 7) return 1;
    // Every compute specialization must share the same cached input-reuse guard.
    std::atomic<unsigned> bad{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < 7; ++i) workers.emplace_back([&] {
        ai_worker = true;
        for (int repeat = 0; repeat < 100; ++repeat)
            if (ocr_q4_pc8::mode() != 2 || !ocr_q4_pc8::rows8() || !ocr_q4_pc8::down_rows8() ||
                !ocr_q4_pc8::multi_n() || !ocr_q4_pc8::prepack() || !ocr_q4_pc8::reuse_input() ||
                ocr_q4_pc8::trace_reuse()) ++bad;
    });
    for (auto & worker : workers) worker.join();
    if (worker_environment_reads || bad) return 2;
    std::puts("PASS all PC8 option initialization on host; zero worker getenv calls");
}
