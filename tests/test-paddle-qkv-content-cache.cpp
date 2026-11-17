#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

// This is a numerical/lifetime fixture for the optional QKV content cache.
// The graph always uses real backend allocations.  The reference process has
// the cache disabled; the comparison process enables it through its CTest
// environment.  No private cache is substituted in this test.
namespace {
constexpr uint32_t MAGIC = 0x514b5643; // QKVC
constexpr uint32_t VERSION = 3;
constexpr int K = 1152, N = 1152;
const int M_VALUES[] = {1, 7, 8, 9, 17, 848};
constexpr int OUTPUTS = 3;

struct fixture_header { uint32_t magic, version, suite, m, pass, slot, count, input_bytes, weight_bytes; uint64_t input_hash, weight_hash; };

uint64_t hash_bytes(const void * data,size_t bytes) {
    uint64_t hash=UINT64_C(14695981039346656037);
    const auto * p=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<bytes;++i)hash=(hash^p[i])*UINT64_C(1099511628211);
    return hash;
}

bool fixture(FILE * file, bool write, int suite, int m, int pass, int slot,
             const std::vector<float> & values,const std::vector<float>&input,
             const std::vector<ggml_fp16_t>&weights) {
    fixture_header h{};
    h.magic=MAGIC;h.version=VERSION;h.suite=uint32_t(suite);h.m=uint32_t(m);
    h.pass=uint32_t(pass);h.slot=uint32_t(slot);h.count=uint32_t(values.size());
    h.input_bytes=uint32_t(input.size()*sizeof(float));h.weight_bytes=uint32_t(weights.size()*sizeof(ggml_fp16_t));
    h.input_hash=hash_bytes(input.data(),h.input_bytes);h.weight_hash=hash_bytes(weights.data(),h.weight_bytes);
    if (write) {
        return std::fwrite(&h, sizeof(h), 1, file) == 1 &&
               std::fwrite(values.data(), sizeof(float), values.size(), file) == values.size();
    }
    fixture_header got{};
    if (std::fread(&got, sizeof(got), 1, file) != 1 || std::memcmp(&h, &got, sizeof(h))) {
        std::fprintf(stderr, "QKV fixture header mismatch M=%d pass=%d slot=%d\n", m, pass, slot);
        return false;
    }
    std::vector<float> expected(values.size());
    if (std::fread(expected.data(), sizeof(float), expected.size(), file) != expected.size()) {
        std::fprintf(stderr, "QKV fixture truncated M=%d pass=%d slot=%d\n", m, pass, slot);
        return false;
    }
    if (std::memcmp(expected.data(), values.data(), values.size() * sizeof(float)) == 0) return true;
    for (size_t i = 0; i < values.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &values[i], sizeof(a));
        std::memcpy(&b, &expected[i], sizeof(b));
        if (a != b) {
            std::fprintf(stderr, "QKV fixture differs M=%d pass=%d slot=%d element=%zu 0x%08x != 0x%08x\n",
                         m, pass, slot, i, unsigned(a), unsigned(b));
            break;
        }
    }
    return false;
}

struct graph_case {
    ggml_context * ctx = nullptr;
    ggml_tensor * x[2]{};
    ggml_tensor * w[OUTPUTS]{};
    ggml_tensor * y[2][OUTPUTS]{};
    ggml_tensor * noise_input = nullptr;
    ggml_tensor * noise_weight = nullptr;
    ggml_tensor * noise = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph[2][OUTPUTS]{};
    ggml_cgraph * noise_graph = nullptr;
    std::vector<ggml_fp16_t> original_weights[OUTPUTS];
};

void fill_input(std::vector<float> & values, int m, int pass) {
    if (pass >= 2) {
        // Pass 2 differs from pass 1 only by zero sign bits. Pass 3 differs
        // from pass 2 in exactly ONE byte. Pass 4 reuses pass 3 content at
        // x[1], a distinct live backend allocation.
        if (pass == 2) for (size_t i = 0; i < values.size(); i += 113)
            values[i] = std::signbit(values[i]) ? 0.f : -0.f;
        if (pass == 3) {
            uint32_t bits;
            std::memcpy(&bits, &values[3], sizeof(bits));
            bits ^= 0x00200000u; // one mantissa byte; stays finite, changes value
            std::memcpy(&values[3], &bits, sizeof(bits));
        }
        return;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        const uint32_t q = uint32_t(i * 2654435761u + m * 97 + pass * 313);
        const float base = float(int(q % 131071u) - 65535) * 0x1p-15f;
        values[i] = base * (i % 13 ? 1.f : 1e-5f);
    }
    if (pass == 0 && m == 8) std::fill(values.begin(), values.end(), 0.f);
    if (pass == 0 && m == 9) std::fill_n(values.begin(), K, 0.f);
    if (pass == 1) {
        for (size_t i = 0; i < values.size(); i += 113) values[i] = (i & 1) ? -0.f : 0.f;
        values[3] = 0.75f;
    }
}

void fill_weight(std::vector<ggml_fp16_t> & values, int slot, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const float scale = slot == 0 ? 0.03125f : (slot == 1 ? 0.0625f : 0.125f);
    for (size_t i = 0; i < values.size(); ++i) values[i] = ggml_fp32_to_fp16(dist(rng) * scale);
    for (size_t i = slot + 1; i < values.size(); i += 97) {
        values[i] = ggml_fp32_to_fp16((i & 1) ? -0x1p-23f : 0x1p-23f);
    }
}

bool make_graph(graph_case & c, ggml_backend_t backend, int m, std::mt19937 & rng) {
    ggml_init_params p{96u * 1024u * 1024u, nullptr, true};
    c.ctx = ggml_init(p);
    if (!c.ctx) return false;
    for (int a = 0; a < 2; ++a) {
        c.x[a] = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, K, m);
        ggml_set_input(c.x[a]);
    }
    for (int s = 0; s < OUTPUTS; ++s) {
        c.w[s] = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, K, N);
        const char * names[] = {"v.blk.0.attn_q.weight", "v.blk.0.attn_k.weight", "v.blk.0.attn_v.weight"};
        ggml_set_name(c.w[s], names[s]);
        ggml_set_param(c.w[s]);
        for (int a = 0; a < 2; ++a) {
            c.y[a][s] = ggml_mul_mat(c.ctx, c.w[s], c.x[a]);
            ggml_set_output(c.y[a][s]);
        }
    }
    // Each Q/K/V is a DISTINCT graph_compute call sharing the same input.
    // An unrelated non-cache F16 GEMM deliberately clobbers the ordinary
    // activation workspace between Q->K and K->V.
    c.noise_input = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, K, m);
    ggml_set_input(c.noise_input);
    c.noise_weight = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F16, K, N);
    ggml_set_name(c.noise_weight, "v.blk.0.ffn_up.weight");
    ggml_set_param(c.noise_weight);
    c.noise = ggml_mul_mat(c.ctx, c.noise_weight, c.noise_input);
    ggml_set_output(c.noise);
    for (int a = 0; a < 2; ++a) for (int s = 0; s < OUTPUTS; ++s) {
        c.graph[a][s] = ggml_new_graph(c.ctx);
        ggml_build_forward_expand(c.graph[a][s], c.y[a][s]);
    }
    c.noise_graph = ggml_new_graph(c.ctx);
    ggml_build_forward_expand(c.noise_graph, c.noise);
    c.buffer = ggml_backend_alloc_ctx_tensors(c.ctx, backend);
    if (!c.buffer) return false;
    for (int s = 0; s < OUTPUTS; ++s) if (!c.w[s]->extra) return false;
    if (!c.noise_weight->extra) return false;
    std::vector<ggml_fp16_t> weights(size_t(K) * N);
    for (int s = 0; s < OUTPUTS; ++s) {
        fill_weight(weights, s, rng);
        c.original_weights[s] = weights;
        ggml_backend_tensor_set(c.w[s], weights.data(), 0, weights.size() * sizeof(ggml_fp16_t));
    }
    fill_weight(weights, 1, rng);
    ggml_backend_tensor_set(c.noise_weight, weights.data(), 0, weights.size() * sizeof(ggml_fp16_t));
    std::vector<float> noise(size_t(K) * m);
    for (size_t i = 0; i < noise.size(); ++i) noise[i] = float(int(i % 113)-56)*0.03125f;
    ggml_backend_tensor_set(c.noise_input, noise.data(), 0, noise.size() * sizeof(float));
    return true;
}

bool run_case(ggml_backend_t backend, int suite, int m, FILE * file, bool write, std::mt19937 & rng) {
    graph_case c;
    if (!make_graph(c, backend, m, rng)) return false;
    std::vector<float> input(size_t(K) * m), output(size_t(N) * m), alternate(output.size());
    const void * addresses[2] = {c.x[0]->data, c.x[1]->data};
    bool ok = true;
    for (int pass = 0; pass < 5; ++pass) {
        const std::vector<float> previous = input;
        fill_input(input, m, pass);
        if (pass == 3) {
            size_t changed = 0;
            const auto *a = reinterpret_cast<const uint8_t*>(input.data());
            const auto *b = reinterpret_cast<const uint8_t*>(previous.data());
            for (size_t i = 0; i < input.size()*sizeof(float); ++i) changed += a[i] != b[i];
            if (changed != 1) { std::fprintf(stderr,"one-byte input setup failed\n"); ok = false; }
        }
        std::vector<float> input_alt = input; // equal content at a different allocation
        ggml_backend_tensor_set(c.x[0], input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(c.x[1], input_alt.data(), 0, input_alt.size() * sizeof(float));
        ok = ok && c.x[0]->data == addresses[0] && c.x[1]->data == addresses[1] && addresses[0] != addresses[1];
        for (int s = 0; s < OUTPUTS; ++s) {
            const int address = pass == 4 ? 1 : 0;
            std::printf("qkv-case suite=%d M=%d pass=%d slot=%d address=%d before-compute\n", suite,m,pass,s,address);
            std::fflush(stdout);
            if (ggml_backend_graph_compute(backend,c.graph[address][s]) != GGML_STATUS_SUCCESS) { ok=false; break; }
            ggml_backend_tensor_get(c.y[address][s], output.data(), 0, output.size() * sizeof(float));
            bool finite = true;
            for (float v : output) finite = finite && std::isfinite(v);
            double sqerr = 0, sqref = 0, maxabs = 0;
            for (int row : {0, std::max(0, m/2), m-1}) for (int col : {0, 1, 31, 127, 511, 1023, 1151}) {
                if (row < 0 || row >= m || col >= N) continue;
                double ref = 0;
                for (int z = 0; z < K; ++z)
                    ref += double(ggml_fp16_to_fp32(c.original_weights[s][size_t(col)*K + z])) * input[size_t(row)*K + z];
                const double err = double(output[size_t(row)*N + col]) - ref;
                sqerr += err * err; sqref += ref * ref; maxabs = std::max(maxabs, std::abs(err));
            }
            const double relative = std::sqrt(sqerr / std::max(sqref, 1e-30));
            ok = ok && finite && relative < 2e-5 && maxabs < 0.002;
            std::printf("qkv-content-cache sample M=%d pass=%d slot=%d relative_l2=%.9g max_abs=%.9g finite=%d\n",
                        m, pass, s, relative, maxabs, int(finite));
            const bool same = fixture(file, write, suite, m, pass, s, output,input,c.original_weights[s]);
            ok = same && ok;
            if (pass == 4) {
                // x0 still has identical bytes and its previous output is
                // retained in a distinct output allocation from pass3.
                ggml_backend_tensor_get(c.y[0][s],alternate.data(),0,alternate.size()*sizeof(float));
                if (std::memcmp(output.data(),alternate.data(),output.size()*sizeof(float))) {
                    std::fprintf(stderr,"different-address same-content mismatch M=%d slot=%d\n",m,s);ok=false;
                }
            }
            if (s < OUTPUTS-1) {
                std::printf("qkv-case suite=%d M=%d pass=%d slot=%d pollution-before-next\n",suite,m,pass,s);
                std::fflush(stdout);
                if (ggml_backend_graph_compute(backend,c.noise_graph) != GGML_STATUS_SUCCESS) { ok=false;break; }
            }
        }
        std::vector<float> input_after(input.size());
        for (int address = 0; address < 2; ++address) {
            ggml_backend_tensor_get(c.x[address],input_after.data(),0,input_after.size()*sizeof(float));
            if (std::memcmp(input_after.data(),input.data(),input.size()*sizeof(float))) {
                std::fprintf(stderr,"input bytes changed M=%d pass=%d address=%d\n",m,pass,address);ok=false;
            }
        }
        std::printf("qkv-content-cache M=%d pass=%d address_alias=%d %s\n", m, pass,
                    int(addresses[0] == addresses[1]), ok ? "PASS" : "FAIL");
    }
    ggml_backend_buffer_free(c.buffer);
    ggml_free(c.ctx);
    return ok;
}
}

int main(int argc, char ** argv) {
    if (argc != 3 || (std::strcmp(argv[1], "--write-reference") && std::strcmp(argv[1], "--compare-reference"))) {
        std::fprintf(stderr, "Usage: %s --write-reference FILE | --compare-reference FILE\n", argv[0]);
        return 2;
    }
    const bool write = !std::strcmp(argv[1], "--write-reference");
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(argv[2], write ? "wb" : "rb"), &std::fclose);
    if (!file) { std::perror(argv[2]); return 2; }
    ggml_backend_load_all();
    auto dev = ggml_backend_dev_by_name("SPACEMIT0");
    if (!dev) return 3;
    auto backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) return 4;
    std::mt19937 rng(0x51c0ffee);
    bool ok = true;
    for (int m : M_VALUES) ok = run_case(backend, 0, m, file.get(), write, rng) && ok;
    // Return to a small shape after M848 while preserving one backend session.
    ok = run_case(backend,1,9,file.get(),write,rng) && ok;
    ggml_backend_free(backend);
    // A fresh backend owner cannot consume the former session's private pack.
    // This is sequential backend independence, not concurrent-session testing.
    backend = ggml_backend_dev_init(dev,nullptr);
    if (!backend) return 4;
    ok = run_case(backend,2,9,file.get(),write,rng) && ok;
    ggml_backend_free(backend);
    if (write ? std::fflush(file.get()) != 0 :
        (std::fgetc(file.get()) != EOF || std::ferror(file.get()))) ok = false;
    return ok ? 0 : 1;
}
