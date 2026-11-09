// SPDX-License-Identifier: MIT
// Diagnostic client for validating EP precision and steady-state latency.
#include "ocr-ep.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

namespace {
void fault_trace(int signal) {
    void * stack[64];
    const int count = backtrace(stack, 64);
    backtrace_symbols_fd(stack, count, STDERR_FILENO);
    _exit(128 + signal);
}
}

int main(int argc, char ** argv) {
    std::signal(SIGSEGV, fault_trace);
    std::signal(SIGABRT, fault_trace);
    try {
        if (argc != 7) {
            throw std::invalid_argument("Usage: ocr-ep-probe layout|vision WIDTH HEIGHT INPUT_F32 OUTPUT_PREFIX REPEATS");
        }
        const std::string mode = argv[1];
        const int width = std::stoi(argv[2]);
        const int height = std::stoi(argv[3]);
        const int repeats = std::stoi(argv[6]);
        if ((mode != "layout" && mode != "vision") || width < 1 || height < 1 || width > 8192 || height > 8192 || repeats < 1 || repeats > 100) {
            throw std::invalid_argument("Invalid probe parameters");
        }
        std::vector<float> pixels(static_cast<size_t>(width) * height * 3);
        std::ifstream file(argv[4], std::ios::binary);
        if (!file.read(reinterpret_cast<char *>(pixels.data()), pixels.size() * sizeof(float)) || file.peek() != EOF) {
            throw std::runtime_error("Input must contain exactly width*height*3 float32 values");
        }
        for (int iteration = 0; iteration < repeats; ++iteration) {
            if (mode == "vision") {
                std::vector<float> features;
                ocr_ep_encode_hwc(pixels.data(), width, height, features);
                if (iteration == 0) {
                    std::ofstream out(std::string(argv[5])+".features.f32", std::ios::binary);
                    out.write(reinterpret_cast<const char *>(features.data()), features.size() * sizeof(float));
                    if (!out) { throw std::runtime_error("Cannot save visual features"); }
                }
            } else {
                if (width != 800 || height != 800) { throw std::invalid_argument("Layout requires 800x800 CHW"); }
                double seconds = 0;
                const auto tensors = ocr_ep_layout(pixels.data(), pixels.size(), seconds);
                if (iteration == 0) {
                    for (const auto & tensor : tensors) {
                        std::ofstream out(std::string(argv[5])+"."+tensor.name+"."+tensor.dtype, std::ios::binary);
                        out.write(reinterpret_cast<const char *>(tensor.data.data()), tensor.data.size());
                        if (!out) { throw std::runtime_error("Cannot save layout outputs"); }
                    }
                }
            }
        }
        std::puts("PASS: EP probe completed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "EP_PROBE_ERROR: %s\n", error.what());
        return 1;
    }
}
