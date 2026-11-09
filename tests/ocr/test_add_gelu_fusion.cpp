#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-spacemit.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// Run in separate OFF/ON processes. Compare the concatenated output files
// byte-for-byte, including guarded patterns that must remain unfused.
struct test_case { int width, rows, bias, guard; };
static unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16);
}

int main(int argc, char ** argv) {
    if (argc != 2) return 1;
    const test_case cases[] = {
        {1, 1, 0, 0}, {31, 3, 1, 0}, {129, 7, 0, 0},
        {1152, 252, 1, 0}, {4304, 252, 1, 0}, {4304, 1008, 1, 0},
        {4352, 3, 0, 0}, {129, 7, 1, 1}, {129, 7, 1, 2},
        {129, 7, 2, 0}, {129, 7, 1, 3}, {129, 7, 1, 4},
    };
    std::ofstream output(argv[1], std::ios::binary);
    if (!output) return 2;
    auto backend = ggml_backend_spacemit_init();
    if (!backend) return 3;
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
        const auto c = cases[ci];
        ggml_init_params params{}; params.mem_size = 2 * 1024 * 1024; params.no_alloc = true;
        auto ctx = ggml_init(params);
        if (!ctx) return 4;
        auto xbase = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.guard == 3 ? c.rows : c.width,
                                      c.guard == 3 ? c.width : c.rows);
        auto x = c.guard == 3 ? ggml_transpose(ctx, xbase) : xbase;
        auto b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.bias == 0 ? 1 : c.width,
                                   c.bias == 2 ? c.rows : 1);
        ggml_set_input(xbase); ggml_set_input(b);
        // Preserve input buffers across repeated graph executions. INPUT marks
        // allocation order; OUTPUT prevents gallocr from reusing their storage.
        ggml_set_output(xbase); ggml_set_output(b);
        auto add = ggml_add(ctx, x, b);
        if (c.guard == 1) ggml_set_output(add);
        auto gelu = c.guard == 4 ? ggml_gelu_inplace(ctx, add) : ggml_gelu(ctx, add);
        auto y = c.guard == 2 ? ggml_add(ctx, gelu, add) : gelu;
        ggml_set_output(y);
        auto graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, y);
        if (!ggml_backend_supports_op(backend, add)) {
            std::printf("{\"case\":%zu,\"skipped\":\"base backend does not support this ADD layout\"}\n", ci);
            std::fflush(stdout);
            ggml_free(ctx);
            continue;
        }
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, graph)) return 5;
        std::vector<float> input(ggml_nelements(xbase)), bias(ggml_nelements(b));
        for (size_t i = 0; i < input.size(); ++i) input[i] = (int(mix(i + 31) % 24001) - 12000) * .001f;
        for (size_t i = 0; i < bias.size(); ++i) bias[i] = (int(mix(i + 71) % 2001) - 1000) * .0003f;
        if (input.size() > 5) { input[0] = 0.f; input[1] = -0.f; input[2] = 1.e-38f; input[3] = -1.e-38f; }
        ggml_backend_tensor_set(xbase, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
        std::vector<float> result(ggml_nelements(y)), first;
        double total_ms = 0;
        for (int repeat = 0; repeat < 4; ++repeat) {
            const auto begin = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 6;
            ggml_backend_synchronize(backend);
            const auto end = std::chrono::steady_clock::now();
            ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
            for (float value : result) if (!std::isfinite(value)) return 7;
            if (repeat == 0) first = result;
            else {
                if (std::memcmp(first.data(), result.data(), result.size() * sizeof(float))) {
                    std::fprintf(stderr, "repeat mismatch case=%zu repeat=%d\n", ci, repeat);
                    return 8;
                }
                total_ms += std::chrono::duration<double, std::milli>(end - begin).count();
            }
        }
        std::vector<float> check_x(input.size()), check_b(bias.size());
        ggml_backend_tensor_get(xbase, check_x.data(), 0, check_x.size() * sizeof(float));
        ggml_backend_tensor_get(b, check_b.data(), 0, check_b.size() * sizeof(float));
        if (std::memcmp(input.data(), check_x.data(), input.size() * sizeof(float)) ||
            std::memcmp(bias.data(), check_b.data(), bias.size() * sizeof(float))) return 9;
        output.write((const char *) result.data(), result.size() * sizeof(float));
        if (c.guard == 1) {
            std::vector<float> extra(ggml_nelements(add));
            ggml_backend_tensor_get(add, extra.data(), 0, extra.size() * sizeof(float));
            output.write((const char *) extra.data(), extra.size() * sizeof(float));
        }
        std::printf("{\"case\":%zu,\"width\":%d,\"rows\":%d,\"bias\":%d,\"guard\":%d,\"mean_ms\":%.6f,\"repeat_exact\":true,\"input_immutable\":true}\n",
                    ci, c.width, c.rows, c.bias, c.guard, total_ms / 3);
        std::fflush(stdout);
        ggml_gallocr_free(alloc); ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return output ? 0 : 10;
}
