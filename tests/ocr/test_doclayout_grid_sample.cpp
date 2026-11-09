#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "doclayout-ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
static std::vector<float> load(const char * path, size_t count) {
    std::vector<float> values(count);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char *>(values.data()), count * sizeof(float));
    check(bool(input) && input.peek() == EOF, "input length mismatch");
    return values;
}

int main(int argc, char ** argv) {
    try {
        check(argc == 13, "usage: test-doclayout-grid N C H W Hout Wout image.f32 grid.f32 reference.f32 output.f32 threads repeats");
        const int64_t n = std::stoll(argv[1]), c = std::stoll(argv[2]), h = std::stoll(argv[3]), w = std::stoll(argv[4]);
        const int64_t ho = std::stoll(argv[5]), wo = std::stoll(argv[6]);
        const int threads = std::stoi(argv[11]), repeats = std::stoi(argv[12]);
        check(n > 0 && c > 0 && h > 0 && w > 0 && ho > 0 && wo > 0 && threads > 0 && threads <= 16
              && repeats > 0 && repeats <= 10, "invalid arguments");
        auto image_values = load(argv[7], n * c * h * w), grid_values = load(argv[8], n * ho * wo * 2);
        auto reference = load(argv[9], n * c * ho * wo);
        auto * ctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true});
        auto backend = ggml_backend_cpu_init();
        check(ctx && backend, "initialization failed");
        ggml_backend_cpu_set_n_threads(backend, threads);
        auto * image = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, c, n);
        auto * grid = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 2, wo, ho, n);
        ggml_set_input(image); ggml_set_output(image);
        ggml_set_input(grid); ggml_set_output(grid);
        auto * result = ocr_grid_sample_2d(ctx, image, grid);
        check(result && result->ne[0] == wo && result->ne[1] == ho && result->ne[2] == c && result->ne[3] == n,
              "incorrect operator shape");
        auto * invalid_grid = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2, wo, ho, n);
        check(ocr_grid_sample_2d(ctx, image, invalid_grid) == nullptr, "invalid grid type was accepted");
        ggml_set_output(result);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        check(ggml_gallocr_alloc_graph(allocator, graph), "allocation failed");
        std::vector<float> first, actual(reference.size()), image_check(image_values.size()), grid_check(grid_values.size());
        float max_abs = 0.f;
        for (int r = 0; r < repeats; ++r) {
            ggml_backend_tensor_set(image, image_values.data(), 0, image_values.size() * sizeof(float));
            ggml_backend_tensor_set(grid, grid_values.data(), 0, grid_values.size() * sizeof(float));
            check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "execution failed");
            ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
            ggml_backend_tensor_get(image, image_check.data(), 0, image_check.size() * sizeof(float));
            ggml_backend_tensor_get(grid, grid_check.data(), 0, grid_check.size() * sizeof(float));
            check(image_values == image_check && grid_values == grid_check, "input corrupted");
            if (r == 0) first = actual;
            check(!std::memcmp(first.data(), actual.data(), actual.size() * sizeof(float)), "repeat changed");
            for (size_t i = 0; i < actual.size(); ++i) {
                check(std::isfinite(actual[i]), "nonfinite output");
                const float delta = std::abs(actual[i] - reference[i]);
                max_abs = std::max(max_abs, delta);
                check(delta <= 2e-5f + 2e-5f * std::abs(reference[i]), "reference mismatch");
            }
        }
        std::ofstream output(argv[10], std::ios::binary);
        output.write(reinterpret_cast<const char *>(actual.data()), actual.size() * sizeof(float));
        check(bool(output), "output write failed");
        std::printf("{\"passed\":true,\"max_abs\":%.9g,\"repeats_exact\":true,\"inputs_preserved\":true}\n", max_abs);
        ggml_gallocr_free(allocator); ggml_free(ctx); ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
