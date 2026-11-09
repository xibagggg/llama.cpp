#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_TEST_SPACEMIT
#include "ggml-spacemit.h"
#else
#include "ggml-cpu.h"
#endif
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

struct test_shape { int64_t k, n, m; };

static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16);
}

static void fill(std::vector<float> & values, uint32_t seed, float scale) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = (static_cast<int32_t>(mix(static_cast<uint32_t>(i) + seed) % 20001) - 10000) * scale;
    }
}

static float scale_from_env(const char * name, float fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char * end = nullptr;
    const float parsed = std::strtof(value, &end);
    return end != value && *end == '\0' && std::isfinite(parsed) && parsed >= 0.0f ? parsed : fallback;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 4) return 1;
    const int selected_case = argc >= 3 ? std::atoi(argv[2]) : -1;
    const int repeats = argc >= 4 ? std::atoi(argv[3]) : 2;
    if (selected_case < -1 || selected_case >= 6 || repeats < 1 || repeats > 3) return 1;
    const test_shape shapes[] = {
        {256, 128, 1},       // decode/GEMV, including the direct-DRAM Q4 route when eligible
        {1024, 1152, 7},     // short prompt / four-row quantization tail
        {4352, 1152, 252},   // OCR visual down projection, real feature token count
        {4352, 1152, 1008},  // larger batch for full-tile scheduling
        {4352, 4304, 252},   // synthetic non-model shape, generic kernel check only
        {4352, 4304, 1008},  // synthetic non-model shape, generic kernel check only
    };
    std::ofstream output(argv[1], std::ios::binary);
    if (!output) return 2;
#ifdef GGML_TEST_SPACEMIT
    ggml_backend_t backend = ggml_backend_spacemit_init();
#else
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend) {
        const char * threads_env = std::getenv("GGML_TEST_CPU_THREADS");
        ggml_backend_cpu_set_n_threads(backend, threads_env ? std::max(1, std::atoi(threads_env)) : 16);
    }
#endif
    if (!backend) return 3;

    for (size_t ci = 0; ci < sizeof(shapes) / sizeof(shapes[0]); ++ci) {
        if (selected_case >= 0 && static_cast<int>(ci) != selected_case) continue;
        const auto s = shapes[ci];
        ggml_init_params params{};
        params.mem_size = 64 * 1024 * 1024;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return 4;

        ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, s.k, s.n);
        ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.m);
        ggml_tensor * bias    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.n, 1);
        ggml_set_input(weights); ggml_set_output(weights);
        ggml_set_input(input);   ggml_set_output(input);
        ggml_set_input(bias);    ggml_set_output(bias);

        ggml_tensor * matmul = ggml_mul_mat(ctx, weights, input);
        ggml_tensor * result = ggml_add(ctx, matmul, bias);
        ggml_set_output(result);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, result);

        if (!ggml_backend_supports_op(backend, matmul) || !ggml_backend_supports_op(backend, result)) {
            std::fprintf(stderr, "unsupported test graph %zu\n", ci);
            return 5;
        }
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, graph)) return 6;

        std::vector<float> weight_data(static_cast<size_t>(s.k * s.n));
        std::vector<uint8_t> quantized(ggml_nbytes(weights));
        std::vector<float> input_data(static_cast<size_t>(s.k * s.m));
        std::vector<float> bias_data(static_cast<size_t>(s.n));
        const float weight_scale = scale_from_env("GGML_TEST_WEIGHT_SCALE", 0.003f);
        const float input_scale  = scale_from_env("GGML_TEST_INPUT_SCALE", 0.01f);
        const float bias_scale   = scale_from_env("GGML_TEST_BIAS_SCALE", 0.02f);
        fill(weight_data, 91u + static_cast<uint32_t>(ci), weight_scale);
        fill(input_data, 177u + static_cast<uint32_t>(ci), input_scale);
        fill(bias_data, 311u + static_cast<uint32_t>(ci), bias_scale);
        // The OCR down projection is logically K=4304 padded with 48 zeros to
        // K=4352 before quantization. Keep the synthetic fixture's padded tail
        // faithful to that model conversion contract.
        if (s.k == 4352 && s.n == 1152) {
            for (int64_t row = 0; row < s.n; ++row) {
                std::fill(weight_data.begin() + row * s.k + 4304, weight_data.begin() + (row + 1) * s.k, 0.0f);
            }
            for (int64_t row = 0; row < s.m; ++row) {
                std::fill(input_data.begin() + row * s.k + 4304, input_data.begin() + (row + 1) * s.k, 0.0f);
            }
        }
        const size_t quantized_bytes = ggml_quantize_chunk(GGML_TYPE_Q4_0, weight_data.data(), quantized.data(),
                                                            0, s.n, s.k, nullptr);
        if (quantized_bytes != quantized.size()) return 7;
        ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
        ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
        ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));

        std::vector<float> result_data(static_cast<size_t>(s.n * s.m));
        std::vector<float> reference;
        double formal_ms = 0.0;
        for (int run = -1; run < repeats; ++run) {
            const auto begin = std::chrono::steady_clock::now();
            const ggml_status status = ggml_backend_graph_compute(backend, graph);
            if (status != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "graph compute failed case=%zu run=%d status=%d\n", ci, run,
                             static_cast<int>(status));
                return 8;
            }
            ggml_backend_synchronize(backend);
            const auto end = std::chrono::steady_clock::now();
            const char * diag = getenv("GGML_SPACEMIT_MUL_MAT_ADD_DIAG");
            if (diag != nullptr && std::strcmp(diag, "1") == 0) {
                std::vector<float> matmul_data(static_cast<size_t>(ggml_nelements(matmul)));
                ggml_backend_tensor_get(matmul, matmul_data.data(), 0, matmul_data.size() * sizeof(float));
                std::fprintf(stderr, "matmul sample case=%zu run=%d first=%g second=%g\n", ci, run,
                             matmul_data.empty() ? 0.0f : matmul_data[0],
                             matmul_data.size() < 2 ? 0.0f : matmul_data[1]);
            }
            ggml_backend_tensor_get(result, result_data.data(), 0, result_data.size() * sizeof(float));
            for (size_t i = 0; i < result_data.size(); ++i) {
                if (!std::isfinite(result_data[i])) {
                    std::fprintf(stderr, "non-finite output case=%zu run=%d index=%zu value=%g\n", ci, run, i,
                                 result_data[i]);
                    return 9;
                }
            }
            if (run == -1) {
                reference = result_data;
            } else {
                if (std::memcmp(reference.data(), result_data.data(), result_data.size() * sizeof(float))) {
                    std::fprintf(stderr, "repeat output mismatch case=%zu run=%d\n", ci, run);
                    return 12;
                }
                formal_ms += std::chrono::duration<double, std::milli>(end - begin).count();
            }
        }
        std::vector<float> input_check(input_data.size());
        ggml_backend_tensor_get(input, input_check.data(), 0, input_check.size() * sizeof(float));
        if (std::memcmp(input_check.data(), input_data.data(), input_data.size() * sizeof(float))) {
            std::fprintf(stderr, "input modified case=%zu\n", ci);
            return 13;
        }
        output.write(reinterpret_cast<const char *>(result_data.data()), result_data.size() * sizeof(float));
        if (!output) return 10;
        std::printf("{\"case\":%zu,\"K\":%lld,\"N\":%lld,\"M\":%lld,\"weight_scale\":%.8g,\"input_scale\":%.8g,\"bias_scale\":%.8g,\"mean_ms\":%.6f,\"warmup_and_repeats_exact\":true,\"input_immutable\":true}\n", ci,
                    static_cast<long long>(s.k), static_cast<long long>(s.n), static_cast<long long>(s.m),
                    weight_scale, input_scale, bias_scale, formal_ms / repeats);
        std::fflush(stdout);

        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return output ? 0 : 11;
}
