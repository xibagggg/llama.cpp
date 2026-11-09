// SPDX-License-Identifier: MIT
// Diagnostic client for validating EP precision and steady-state latency.
#include "ocr-ep.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <chrono>

namespace {
void fault_trace(int signal) {
    void * stack[64];
    const int count = backtrace(stack, 64);
    backtrace_symbols_fd(stack, count, STDERR_FILENO);
    _exit(128 + signal);
}

void vision_file(int width, int height, const std::string & path, const std::string & output) {
    if (width < 1 || height < 1 || width > 8192 || height > 8192) {
        throw std::invalid_argument("Invalid sequence dimensions");
    }
    std::vector<float> pixels(static_cast<size_t>(width) * height * 3);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(pixels.data()), pixels.size() * sizeof(float)) || file.peek() != EOF) {
        throw std::runtime_error("Sequence input must contain exactly width*height*3 float32 values");
    }
    std::vector<float> features;
    const auto start = std::chrono::steady_clock::now();
    ocr_ep_encode_hwc(pixels.data(), width, height, features);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr, "EP_PROBE_SEQUENCE shape=%dx%d wall_seconds=%.6f\n", width, height, seconds);
    std::ofstream out(output, std::ios::binary);
    out.write(reinterpret_cast<const char *>(features.data()), features.size() * sizeof(float));
    if (!out) { throw std::runtime_error("Cannot save sequence visual features"); }
}
}

int main(int argc, char ** argv) {
    std::signal(SIGSEGV, fault_trace);
    std::signal(SIGABRT, fault_trace);
    try {
        if (argc == 5 && std::string(argv[1]) == "vision-sequence") {
            const int repeats = std::stoi(argv[4]);
            if (repeats < 1 || repeats > 100) { throw std::invalid_argument("Invalid sequence repeats"); }
            for (int iteration = 0; iteration < repeats; ++iteration) {
                std::ifstream manifest(argv[2]);
                if (!manifest) { throw std::runtime_error("Cannot open sequence manifest"); }
                int width, height, index = 0;
                std::string path;
                while (manifest >> width >> height >> path) {
                    vision_file(width, height, path, std::string(argv[3])+"."+std::to_string(iteration)+"."+std::to_string(index++)+".f32");
                }
                if (!manifest.eof() || index == 0) { throw std::runtime_error("Invalid sequence manifest"); }
            }
            std::puts("PASS: EP sequence completed");
            return 0;
        }
        if (argc != 7) {
            throw std::invalid_argument("Usage: ocr-ep-probe layout|vision WIDTH HEIGHT INPUT_F32 OUTPUT_PREFIX REPEATS; or vision-sequence MANIFEST OUTPUT_PREFIX REPEATS");
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
