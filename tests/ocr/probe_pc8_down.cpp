// One synthetic down node with the last observed freeze shape. Compilation and
// --plan never run inference; --execute is a separate, explicit opt-in.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-spacemit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static constexpr int K = 4352, INPUT_K = 4304, N = 1152, M = 1080;

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--plan") == 0) {
        std::puts("{\"K\":4352,\"input_k\":4304,\"N\":1152,\"M\":1080,"
                  "\"graph_calls\":1,\"synthetic\":true,\"executed\":false}");
        return 0;
    }
    if (argc != 3 || std::strcmp(argv[1], "--execute") != 0) {
        std::fprintf(stderr, "Usage: %s --plan | --execute NEW_OUTPUT_PREFIX\n", argv[0]);
        return 1;
    }
    const std::string prefix = argv[2];
    // Never truncate a previous run's evidence.
    if (std::ifstream(prefix + ".f32").good()) return 2;
    auto backend = ggml_backend_spacemit_init();
    if (!backend) return 3;
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    auto ctx = ggml_init(params);
    if (!ctx) { ggml_backend_free(backend); return 4; }
    auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, INPUT_K, M);
    auto bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    auto residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);
    ggml_set_name(w, "v.blk.7.ffn_down.weight");
    for (auto * input : {w, x, bias, residual}) {
        ggml_set_input(input);
        ggml_set_output(input);
    }
    auto padded = ggml_pad(ctx, x, K - INPUT_K, 0, 0, 0);
    auto product = ggml_mul_mat(ctx, w, padded);
    ggml_set_name(product, "ffn_down-7");
    auto y = ggml_add(ctx, ggml_add(ctx, product, bias), residual);
    ggml_set_name(y, "down-probe-result");
    ggml_set_output(y);
    auto graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, y);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    int code = 5;
    if (ggml_gallocr_alloc_graph(alloc, graph)) {
        std::vector<float> weights(size_t(K) * N, 0);
        std::vector<float> input(size_t(INPUT_K) * M);
        std::vector<float> biases(N), skip(size_t(N) * M);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < INPUT_K; ++k)
                weights[size_t(n) * K + k] = float((n * 13 + k * 7) % 127 - 63) / 512;
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 257) - 128) / 128;
        for (int i = 0; i < N; ++i) biases[i] = float(i % 31 - 15) / 64;
        for (size_t i = 0; i < skip.size(); ++i) skip[i] = float(int(i % 67) - 33) / 128;
        std::vector<unsigned char> quantized(ggml_nbytes(w));
        const auto bytes = ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(),
                                               quantized.data(), 0, N, K, nullptr);
        if (bytes != quantized.size()) { code = 6; }
        else {
            ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
            ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(bias, biases.data(), 0, biases.size() * sizeof(float));
            ggml_backend_tensor_set(residual, skip.data(), 0, skip.size() * sizeof(float));
            std::puts("probe: dispatching one synthetic PAD/down/bias/residual graph");
            std::fflush(stdout);
            code = 7;
            if (ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS) {
                std::vector<float> output(size_t(N) * M), after(input.size());
                ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
                ggml_backend_tensor_get(x, after.data(), 0, after.size() * sizeof(float));
                const bool finite = std::all_of(output.begin(), output.end(),
                                                [](float v) { return std::isfinite(v); });
                const bool immutable = !std::memcmp(input.data(), after.data(), input.size() * sizeof(float));
                std::ofstream file(prefix + ".f32", std::ios::binary);
                file.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
                file.close();
                code = finite && immutable && bool(file) ? 0 : 8;
                std::printf("{\"graph_calls\":1,\"finite\":%s,\"input_immutable\":%s,"
                            "\"synthetic\":true,\"output_bytes\":%zu}\n",
                            finite ? "true" : "false", immutable ? "true" : "false",
                            output.size() * sizeof(float));
            }
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return code;
}
