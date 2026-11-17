#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <utility>
#include <vector>

// The option is read at graph construction/dispatch; use separate processes for
// off-mode fixtures and on-mode comparisons. Every output byte is compared.
// FP64 verification is sampled and does not replace the byte comparison.
enum class graph_kind : uint32_t {
    eligible, matmul_output, extra_consumer, external_view,
    computed_bias, matrix_bias, scalar_bias, bias_view, ordinary_f16,
};

static const char * kind_name(graph_kind kind) {
    switch (kind) {
        case graph_kind::eligible: return "eligible";
        case graph_kind::matmul_output: return "matmul-output";
        case graph_kind::extra_consumer: return "extra-consumer";
        case graph_kind::external_view: return "external-view";
        case graph_kind::computed_bias: return "computed-bias";
        case graph_kind::matrix_bias: return "matrix-bias";
        case graph_kind::scalar_bias: return "scalar-bias";
        case graph_kind::bias_view: return "bias-view";
        case graph_kind::ordinary_f16: return "ordinary-f16";
    }
    return "invalid";
}

struct trace_state {
    std::atomic<int> expected{0};
    std::atomic<int> lines{0};
    std::atomic<bool> mismatch{false};
    ggml_log_callback previous = nullptr;
    void * previous_data = nullptr;
};

static void capture_log(enum ggml_log_level level, const char * text, void * opaque) {
    auto & state = *static_cast<trace_state *>(opaque);
    const char * marker = std::strstr(text, "PADDLE_F16 GEMM+BIAS fused=");
    int fused = -1;
    if (marker && std::sscanf(marker, "PADDLE_F16 GEMM+BIAS fused=%d", &fused) == 1) {
        state.lines.fetch_add(1);
        if (fused != state.expected.load()) state.mismatch.store(true);
    }
    if (state.previous) state.previous(level, text, state.previous_data);
}

static bool fixture_output(FILE * file, bool write, graph_kind kind,
                           int k, int n, int m, int pass, int slot,
                           const std::vector<float> & output) {
    const uint32_t header[] = {0x50424941, 1, uint32_t(kind), uint32_t(k), uint32_t(n),
                              uint32_t(m), uint32_t(pass), uint32_t(slot), uint32_t(output.size())};
    if (write) {
        return std::fwrite(header, sizeof(header), 1, file) == 1 &&
               std::fwrite(output.data(), sizeof(float), output.size(), file) == output.size();
    }
    uint32_t actual_header[9];
    if (std::fread(actual_header, sizeof(actual_header), 1, file) != 1 ||
        std::memcmp(header, actual_header, sizeof(header))) {
        std::fprintf(stderr, "Reference header mismatch: %s K=%d N=%d M=%d pass=%d slot=%d\n",
                     kind_name(kind), k, n, m, pass, slot);
        return false;
    }
    std::vector<float> expected(output.size());
    if (std::fread(expected.data(), sizeof(float), expected.size(), file) != expected.size()) {
        std::fprintf(stderr, "Truncated reference\n");
        return false;
    }
    if (!std::memcmp(output.data(), expected.data(), output.size() * sizeof(float))) return true;
    for (size_t i = 0; i < output.size(); ++i) {
        uint32_t have, want;
        std::memcpy(&have, &output[i], sizeof(have));
        std::memcpy(&want, &expected[i], sizeof(want));
        if (have != want) {
            std::fprintf(stderr,
                         "Reference differs: %s K=%d N=%d M=%d pass=%d slot=%d element=%zu 0x%08x != 0x%08x\n",
                         kind_name(kind), k, n, m, pass, slot, i, unsigned(have), unsigned(want));
            break;
        }
    }
    return false;
}

struct test_case { int k, n, m; graph_kind kind; };

static bool run_case(const test_case & test, ggml_backend_t backend, trace_state & trace,
                     bool fusion_enabled, std::mt19937 & rng, FILE * fixture, bool write) {
    const int k = test.k, n = test.n, m = test.m;
    const auto kind = test.kind;
    ggml_init_params params{32 * 1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(params);
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n);
    ggml_set_name(w, kind == graph_kind::ordinary_f16 ? "ordinary.weight" :
                     k == 4608 ? (n == 4608 ? "mm.1.weight" : "mm.2.weight") :
                     "v.blk.0.attn_q.weight");
    ggml_set_param(w);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_set_input(x);
    auto * matmul = ggml_mul_mat(ctx, w, x);
    const int bias_n = kind == graph_kind::scalar_bias ? 1 : n;
    const int bias_m = kind == graph_kind::matrix_bias ? m : 1;
    auto * raw_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bias_n, bias_m);
    ggml_tensor * bias = raw_bias;
    if (kind == graph_kind::computed_bias) bias = ggml_scale(ctx, raw_bias, 0.5f);
    if (kind == graph_kind::bias_view) bias = ggml_view_1d(ctx, raw_bias, n, 0);
    auto * y = ggml_add(ctx, matmul, bias);
    ggml_set_output(y);
    ggml_tensor * probe = nullptr;
    if (kind == graph_kind::matmul_output) {
        ggml_set_output(matmul);
        probe = matmul;
    } else if (kind == graph_kind::extra_consumer) {
        probe = ggml_scale(ctx, matmul, 0.5f);
        ggml_set_output(probe);
    } else if (kind == graph_kind::external_view) {
        probe = ggml_view_2d(ctx, matmul, n, m - 1, matmul->nb[1], 0);
        ggml_set_output(probe);
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    if (probe) ggml_build_forward_expand(graph, probe);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::fprintf(stderr, "Tensor allocation failed\n");
        ggml_free(ctx);
        return false;
    }
    bool ok = true;
    if (kind != graph_kind::ordinary_f16 && !w->extra) {
        std::fprintf(stderr, "Paddle F16 trait is inactive\n");
        ok = false;
    }
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<ggml_fp16_t> weights(size_t(k) * n);
    std::vector<float> values(size_t(k) * m), biases(size_t(bias_n) * bias_m);
    std::vector<float> output(size_t(n) * m);
    for (auto & value : weights) value = ggml_fp32_to_fp16(dist(rng) * 0.1f);
    for (size_t i = 0; i < weights.size(); i += 31)
        weights[i] = ggml_fp32_to_fp16(i % 62 ? -0x1p-23f : 0x1p-23f);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = i % 11 ? dist(rng) * 2.f : dist(rng) * 1e-5f;
    if (m == 17) for (auto & value : values) value *= 1e-10f;
    if (m == 8) std::fill(values.begin(), values.end(), 0.f);
    if (m == 9) std::fill_n(values.begin(), k, 0.f);
    for (size_t i = 0; i < biases.size(); ++i)
        biases[i] = i % 9 ? float(int(i % 41) - 20) * 0.0078125f : 0.f;
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(ggml_fp16_t));
    const void * const input_address = x->data;
    const void * const bias_address = raw_bias->data;
    const int expected_fusions = fusion_enabled && kind == graph_kind::eligible ? 1 : 0;
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 1) {
            const float step = m == 17 ? 1e-12f : 0.015625f;
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = -0.5f * values[i] + float(int(i % 257) - 128) * step;
        }
        if (pass == 2) {
            for (size_t i = 0; i < biases.size(); ++i)
                biases[i] = i % 7 ? -2.f * biases[i] + 0.00390625f : -0.f;
        }
        ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(raw_bias, biases.data(), 0, biases.size() * sizeof(float));
        trace.expected.store(expected_fusions);
        trace.lines.store(0);
        trace.mismatch.store(false);
        const auto status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "Graph execution failed\n");
            ok = false;
            break;
        }
        const bool matching = !trace.mismatch.load() && (pass != 0 || trace.lines.load() > 0);
        const bool stable = x->data == input_address && raw_bias->data == bias_address;
        ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
        bool finite = true, zero_bias = true;
        for (float value : output) finite = finite && std::isfinite(value);
        const int zero_rows = pass == 0 ? (m == 8 ? m : m == 9 ? 1 : 0) : 0;
        for (int row = 0; row < zero_rows; ++row) for (int col = 0; col < n; ++col) {
            const size_t b = size_t(row % bias_m) * bias_n + col % bias_n;
            const float expected = kind == graph_kind::computed_bias ? biases[b] * 0.5f : biases[b];
            zero_bias = zero_bias && output[size_t(row) * n + col] == expected;
        }
        std::vector<int> columns;
        for (int j = 0; j < 64; ++j) columns.push_back((j * 71) % n);
        // All valid channels of the last N32 panel, including N4304's tail16.
        for (int col = std::max(0, n - 32); col < n; ++col) columns.push_back(col);
        std::sort(columns.begin(), columns.end());
        columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
        double sqerr = 0, sqref = 0, maxabs = 0;
        for (int row = 0; row < m; ++row) for (int col : columns) {
            double ref = 0;
            for (int z = 0; z < k; ++z)
                ref += double(ggml_fp16_to_fp32(weights[size_t(col) * k + z])) * values[size_t(row) * k + z];
            const size_t b = size_t(row % bias_m) * bias_n + col % bias_n;
            ref += kind == graph_kind::computed_bias ? double(biases[b] * 0.5f) : double(biases[b]);
            const double error = output[size_t(row) * n + col] - ref;
            sqerr += error * error;
            sqref += ref * ref;
            maxabs = std::max(maxabs, std::abs(error));
        }
        const double relative = std::sqrt(sqerr / std::max(sqref, 1e-30));
        bool same = !fixture || fixture_output(fixture, write, kind, k, n, m, pass, 0, output);
        if (probe) {
            std::vector<float> probe_output(size_t(ggml_nelements(probe)));
            ggml_backend_tensor_get(probe, probe_output.data(), 0, probe_output.size() * sizeof(float));
            for (float value : probe_output) finite = finite && std::isfinite(value);
            same = (!fixture || fixture_output(fixture, write, kind, k, n, m, pass, 1, probe_output)) && same;
        }
        // The ordinary fallback is a rejection test, with its own FP32 reduction
        // error. The strict all-byte off/on check still applies to every case.
        const double tolerance = kind == graph_kind::ordinary_f16 ? 2e-4 : 2e-5;
        const bool valid = finite && zero_bias && same && stable && matching &&
                           relative < tolerance && maxabs < 0.002;
        ok = ok && valid;
        std::printf("%s K=%d N=%d M=%d pass=%d relative_l2=%.9g max_abs=%.9g finite=%d zero_bias=%d stable=%d fusion=%d traces=%d matcher=%d fixture=%s %s\n",
                    kind_name(kind), k, n, m, pass, relative, maxabs, int(finite), int(zero_bias), int(stable),
                    expected_fusions, trace.lines.load(), int(matching),
                    !fixture ? "none" : write ? "write" : same ? "identical" : "different", valid ? "PASS" : "FAIL");
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 3 || (std::strcmp(argv[1], "--write-reference") &&
                      std::strcmp(argv[1], "--compare-reference"))) {
        std::fprintf(stderr, "Usage: %s --write-reference FILE | --compare-reference FILE\n", argv[0]);
        return 2;
    }
    const bool write = !std::strcmp(argv[1], "--write-reference");
    const char * flag = std::getenv("GGML_SPACEMIT_PADDLE_F16_BIAS_FUSE");
    const bool fusion_enabled = flag && !std::strcmp(flag, "1");
    if (write && fusion_enabled) {
        std::fprintf(stderr, "Write the reference with GGML_SPACEMIT_PADDLE_F16_BIAS_FUSE=0\n");
        return 2;
    }
    std::unique_ptr<FILE, decltype(&std::fclose)> fixture(std::fopen(argv[2], write ? "wb" : "rb"), &std::fclose);
    if (!fixture) { std::perror(argv[2]); return 2; }
    setenv("GGML_SPACEMIT_PADDLE_F16_IME2", "1", 1);
    setenv("GGML_SPACEMIT_PADDLE_F16_PROJECTOR", "1", 1);
    setenv("GGML_SPACEMIT_PADDLE_F16_BIAS_TRACE", "1", 1);
    // Keep other independent fusions out of the byte fixture/matcher checks.
    setenv("GGML_SPACEMIT_ADD_GELU", "0", 1);
    setenv("GGML_SPACEMIT_NORM_AFFINE", "0", 1);
    trace_state trace;
    ggml_log_get(&trace.previous, &trace.previous_data);
    ggml_log_set(capture_log, &trace);
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_name("SPACEMIT0");
    if (!dev) { std::fprintf(stderr, "SPACEMIT0 device not available\n"); return 2; }
    auto backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 3;
    std::vector<test_case> cases;
    for (auto shape : {std::pair<int, int>{1152, 1152}, {1152, 4304}, {4304, 1152}, {4608, 4608}, {4608, 1024}}) {
        for (int m : {1, 7, 8, 9, 17, 33}) cases.push_back({shape.first, shape.second, m, graph_kind::eligible});
        cases.push_back({shape.first, shape.second, shape.first == 4608 ? 212 : 848, graph_kind::eligible});
    }
    for (auto kind : {graph_kind::matmul_output, graph_kind::extra_consumer, graph_kind::external_view,
                      graph_kind::computed_bias, graph_kind::matrix_bias, graph_kind::scalar_bias,
                      graph_kind::bias_view, graph_kind::ordinary_f16})
        cases.push_back({1152, 1152, 9, kind});
    std::mt19937 rng(176);
    bool ok = true;
    for (const auto & test : cases) ok = run_case(test, backend, trace, fusion_enabled, rng, fixture.get(), write) && ok;
    if (write ? std::fflush(fixture.get()) != 0 :
                (std::fgetc(fixture.get()) != EOF || std::ferror(fixture.get()))) {
        std::fprintf(stderr, "Reference I/O error or trailing data\n");
        ok = false;
    }
    ggml_backend_free(backend);
    ggml_log_set(trace.previous, trace.previous_data);
    return ok ? 0 : 1;
}
