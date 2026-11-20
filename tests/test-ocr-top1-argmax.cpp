// Tests for common/ocr-top1-argmax.h, the top-1 argmax shortcut of common_sampler.
//
// 1. unique_max() and its kernels accept exactly the inputs that have no NaN and a single
//    largest element, and return the index of that element.
// 2. For every accepted input, a sampler chain headed by top-k(1) ends with the same candidates,
//    bit for bit, whether it starts from the full candidate list or from the one candidate the
//    shortcut builds, and the two chains stay in step from token to token.
// 3. chain_starts_with_top_k() refuses a chain in which a sampler ahead of top-k can change the
//    logits.

#include "ocr-top1-argmax.h"

#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            if (n_failed < 20) {                            \
                printf("FAIL %s:%d: ", __FILE__, __LINE__); \
                printf(__VA_ARGS__);                        \
                printf("\n");                               \
            }                                               \
            n_failed++;                                     \
        }                                                   \
    } while (0)

//
// kernels
//

// The condition under which top-k(1) cannot depend on the order the candidates are visited in:
// no NaN and exactly one element equal to the maximum.
static bool reference(const float * x, size_t n, size_t * index) {
    if (n == 0) {
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        if (x[i] != x[i]) {
            return false;
        }
    }
    float best = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > best) {
            best = x[i];
        }
    }
    size_t count = 0;
    size_t first = 0;
    for (size_t i = 0; i < n; i++) {
        if (x[i] == best) {
            if (count == 0) {
                first = i;
            }
            count++;
        }
    }
    if (count != 1) {
        return false;
    }
    *index = first;
    return true;
}

typedef bool (*kernel_fn)(const float *, size_t, size_t *);

struct kernel {
    const char * name;
    kernel_fn    fn;
};

static const kernel kernels[] = {
    { "unique_max",        ocr_top1_argmax::unique_max        },
    { "unique_max_scalar", ocr_top1_argmax::unique_max_scalar },
#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
    { "unique_max_rvv",    ocr_top1_argmax::unique_max_rvv    },
#endif
};

// Compares every kernel with the reference. Returns the reference decision.
static bool check_kernels(const std::vector<float> & x, const char * what) {
    size_t ref_index = 0;
    const bool ref_ok = reference(x.data(), x.size(), &ref_index);
    for (const auto & k : kernels) {
        size_t index = ~size_t(0);
        const bool ok = k.fn(x.data(), x.size(), &index);
        CHECK(ok == ref_ok && (!ok || index == ref_index),
              "%s, %s, n = %zu: reference %d/%zu, kernel %d/%zu",
              k.name, what, x.size(), int(ref_ok), ref_index, int(ok), index);
    }
    return ref_ok;
}

// The same, for an input whose outcome is known.
static void expect(const std::vector<float> & x, bool want_ok, size_t want_index, const char * what) {
    size_t ref_index = 0;
    const bool ref_ok = reference(x.data(), x.size(), &ref_index);
    CHECK(ref_ok == want_ok && (!ref_ok || ref_index == want_index),
          "reference, %s, n = %zu: got %d/%zu", what, x.size(), int(ref_ok), ref_index);
    check_kernels(x, what);
}

// Maximum, tie and NaN at the first and last elements and next to every boundary a kernel block
// can have (a block is 256 elements in the scalar kernel and 16 LMUL=2 vectors in the RVV one).
static void test_kernels_planted() {
    expect({}, false, 0, "empty");

    const size_t sizes[] = {
        1, 2, 3, 4, 5, 7, 8, 9, 127, 128, 129, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 2047, 2048, 2049, 4097,
    };
    for (const size_t n : sizes) {
        std::vector<size_t> pos = { 0, 1, n / 2, n - 1 };
        if (n >= 2) {
            pos.push_back(n - 2);
        }
        for (size_t m = 128; m <= 2048; m *= 2) {
            pos.push_back(m - 1);
            pos.push_back(m);
            pos.push_back(m + 1);
        }
        std::sort(pos.begin(), pos.end());
        pos.erase(std::unique(pos.begin(), pos.end()), pos.end());
        while (pos.back() >= n) {
            pos.pop_back();
        }

        std::vector<float> x(n, NAN);
        expect(x, false, 0, "all NaN");

        std::fill(x.begin(), x.end(), -INFINITY);
        expect(x, n == 1, 0, "all -inf");

        for (const size_t p : pos) {
            std::fill(x.begin(), x.end(), -1.0f);
            x[p] = 2.0f;
            expect(x, true, p, "single maximum");

            for (const size_t q : pos) {
                if (q == p) {
                    continue;
                }
                x[q] = 2.0f;
                expect(x, false, 0, "tied maximum");

                x[p] = 0.0f;
                x[q] = -0.0f;
                expect(x, false, 0, "+0 and -0");

                x[p] = 2.0f;
                x[q] = NAN;
                expect(x, false, 0, "NaN");

                x[q] = std::nextafter(2.0f, 0.0f);
                expect(x, true, p, "runner-up one step below");

                x[q] = -1.0f;
            }
        }
    }
}

static const float specials[] = {
    0.0f, -0.0f, INFINITY, -INFINITY, 1.0f, -1.0f, FLT_MAX, -FLT_MAX, FLT_MIN,
};

enum fill_mode {
    FILL_NORMAL,  // logit-like
    FILL_GRID,    // five distinct values, ties everywhere
    FILL_SPECIAL, // zeros, infinities and extremes only
    FILL_MIXED,   // logit-like with a special value now and then
    FILL_TINY,    // small magnitudes
    FILL_NEG_INF, // every element -inf
    FILL_MODE_COUNT,
};

// Fills x. No element is NaN.
static void fill(std::vector<float> & x, int mode, std::mt19937 & rng) {
    const size_t n_specials = sizeof(specials) / sizeof(specials[0]);
    std::normal_distribution<float> normal(-2.0f, 4.0f);
    for (auto & v : x) {
        switch (mode) {
            case FILL_NORMAL:  v = normal(rng);                                                  break;
            case FILL_GRID:    v = float(int(rng() % 5)) - 2.0f;                                 break;
            case FILL_SPECIAL: v = specials[rng() % n_specials];                                 break;
            case FILL_MIXED:   v = rng() % 50 == 0 ? specials[rng() % n_specials] : normal(rng); break;
            case FILL_TINY:    v = normal(rng) * 1e-3f;                                          break;
            default:           v = -INFINITY;                                                    break;
        }
    }
}

// Makes one element the single maximum, as close to the runner-up as a float gets. Leaves x
// alone when +inf is already there. x has no NaN and is not empty.
static void plant_max(std::vector<float> & x, std::mt19937 & rng) {
    const float top = *std::max_element(x.begin(), x.end());
    if (top < INFINITY) {
        x[rng() % x.size()] = std::nextafter(top, INFINITY);
    }
}

// Makes +inf the single maximum. x is not empty.
static void plant_inf(std::vector<float> & x, std::mt19937 & rng) {
    for (auto & v : x) {
        if (v == INFINITY) {
            v = FLT_MAX;
        }
    }
    x[rng() % x.size()] = INFINITY;
}

// Repeats the maximum at a second position. x has no NaN.
static void plant_tie(std::vector<float> & x, std::mt19937 & rng) {
    if (x.size() < 2) {
        return;
    }
    const size_t at = std::max_element(x.begin(), x.end()) - x.begin();
    size_t other = rng() % (x.size() - 1);
    if (other >= at) {
        other++;
    }
    x[other] = x[at];
}

static void test_kernels_random(std::mt19937 & rng) {
    const int n_iter = 20000;
    int n_accepted = 0;
    std::vector<float> x;
    for (int iter = 0; iter < n_iter; iter++) {
        // every length up to 1099 twice, then random lengths
        const size_t n = iter < 2200 ? size_t(iter % 1100) : size_t(1 + rng() % 3000);
        x.resize(n);
        fill(x, int(rng() % FILL_MODE_COUNT), rng);
        if (n > 0) {
            switch (rng() % 8) {
                case 0: plant_max(x, rng);                        break;
                case 1: plant_inf(x, rng);                        break;
                case 2: plant_tie(x, rng);                        break;
                case 3: x[rng() % n] = NAN;                       break;
                case 4: plant_max(x, rng); x[rng() % n] = -NAN;   break;
                default:                                          break;
            }
        }
        n_accepted += check_kernels(x, "random");
    }
    // both decisions were exercised
    CHECK(n_accepted > n_iter / 10 && n_accepted < n_iter - n_iter / 10,
          "random: %d of %d inputs accepted", n_accepted, n_iter);
}

// The vocabulary size of the OCR model: 969 full blocks and 36 elements after the last one.
static const int32_t n_vocab_ocr = 248100;

static void test_kernels_vocab_sized(std::mt19937 & rng) {
    const size_t n = n_vocab_ocr;
    std::vector<float> x(n);
    for (int iter = 0; iter < 16; iter++) {
        fill(x, FILL_NORMAL, rng);

        const size_t p = iter == 0 ? 0 : iter == 1 ? n - 1 : rng() % n;
        size_t q = iter == 2 ? n - 1 : iter == 3 ? 0 : rng() % (n - 1);
        if (q == p) {
            q = (p + 1) % n;
        }

        x[p] = 60.0f;
        expect(x, true, p, "vocabulary-sized, single maximum");

        const float x_q = x[q];
        x[q] = 60.0f;
        expect(x, false, 0, "vocabulary-sized, tied maximum");

        x[q] = NAN;
        expect(x, false, 0, "vocabulary-sized, NaN");

        x[q] = x_q;
        x[p] = -60.0f;
        check_kernels(x, "vocabulary-sized, as drawn");
    }
}

//
// sampler chains
//

static llama_sampler * chain_of(std::initializer_list<llama_sampler *> samplers) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    for (llama_sampler * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }
    return chain;
}

static std::string chain_names(llama_sampler * chain) {
    std::string result;
    for (int i = 0; i < llama_sampler_chain_n(chain); i++) {
        if (i > 0) {
            result += " -> ";
        }
        result += llama_sampler_name(llama_sampler_chain_get(chain, i));
    }
    return result;
}

// Chains common_sampler_init() turns the shortcut on for, built the way it builds them.
enum chain_kind {
    CHAIN_SERVER,      // the OCR server: top_k = 1, min_p = 0.05, temp = 0, the rest disabled
    CHAIN_SERVER_DIST, // the same with temp = 0.8, which ends the chain with dist
    CHAIN_TOP_K_ONLY,  // samplers = top_k
    CHAIN_ACTIVE_TAIL, // every stage after top-k enabled
    CHAIN_PENALTIES,   // a stage after top-k that keeps the token history
    CHAIN_KIND_COUNT,
};

static llama_sampler * make_chain(int kind, int32_t n_vocab) {
    switch (kind) {
        case CHAIN_SERVER:
            return chain_of({
                llama_sampler_init_penalties(n_vocab, 64, 1.0f, 0.0f, 0.0f),
                llama_sampler_init_dry(nullptr, 0.0f, 1.75f, 2, -1, nullptr, 0),
                llama_sampler_init_top_n_sigma(-1.0f),
                llama_sampler_init_top_k(1),
                llama_sampler_init_typical(1.0f, 0),
                llama_sampler_init_top_p(1.0f, 0),
                llama_sampler_init_min_p(0.05f, 0),
                llama_sampler_init_xtc(0.0f, 0.1f, 0, 42),
                llama_sampler_init_temp_ext(0.0f, 0.0f, 1.0f),
                llama_sampler_init_greedy(),
            });
        case CHAIN_SERVER_DIST:
            return chain_of({
                llama_sampler_init_penalties(n_vocab, 64, 1.0f, 0.0f, 0.0f),
                llama_sampler_init_dry(nullptr, 0.0f, 1.75f, 2, -1, nullptr, 0),
                llama_sampler_init_top_n_sigma(-1.0f),
                llama_sampler_init_top_k(1),
                llama_sampler_init_typical(1.0f, 0),
                llama_sampler_init_top_p(1.0f, 0),
                llama_sampler_init_min_p(0.05f, 0),
                llama_sampler_init_xtc(0.0f, 0.1f, 0, 42),
                llama_sampler_init_temp_ext(0.8f, 0.0f, 1.0f),
                llama_sampler_init_dist(42),
            });
        case CHAIN_TOP_K_ONLY:
            return chain_of({
                llama_sampler_init_top_k(1),
                llama_sampler_init_greedy(),
            });
        case CHAIN_ACTIVE_TAIL:
            return chain_of({
                llama_sampler_init_top_k(1),
                llama_sampler_init_typical(0.9f, 0),
                llama_sampler_init_top_p(0.9f, 0),
                llama_sampler_init_min_p(0.05f, 0),
                llama_sampler_init_xtc(0.5f, 0.1f, 0, 42),
                llama_sampler_init_temp_ext(0.7f, 0.3f, 1.0f),
                llama_sampler_init_dist(42),
            });
        default:
            return chain_of({
                llama_sampler_init_top_k(1),
                llama_sampler_init_penalties(n_vocab, 64, 1.3f, 0.1f, 0.1f),
                llama_sampler_init_temp_ext(0.8f, 0.0f, 1.0f),
                llama_sampler_init_dist(42),
            });
    }
}

static bool same_candidates(const llama_token_data_array & a, const llama_token_data_array & b) {
    static_assert(sizeof(llama_token_data) == sizeof(llama_token) + 2 * sizeof(float), "padding would break memcmp");
    return a.size == b.size && a.selected == b.selected && a.sorted == b.sorted &&
           std::memcmp(a.data, b.data, a.size * sizeof(llama_token_data)) == 0;
}

// Runs two copies of a chain side by side for n_rounds tokens. One starts each token from the
// full candidate list, the other from the single candidate of the shortcut. With mixed set the
// logits come from every fill mode, otherwise they are logit-like.
static void test_chains(std::mt19937 & rng, int32_t n_vocab, int n_rounds, bool mixed) {
    std::vector<float>            x(n_vocab);
    std::vector<llama_token_data> full(n_vocab);

    for (int kind = 0; kind < CHAIN_KIND_COUNT; kind++) {
        llama_sampler * chain_full = make_chain(kind, n_vocab);
        llama_sampler * chain_top1 = make_chain(kind, n_vocab);

        CHECK(ocr_top1_argmax::chain_starts_with_top_k(chain_full),
              "chain %d is refused: %s", kind, chain_names(chain_full).c_str());

        int n_compared = 0;
        for (int round = 0; round < n_rounds; round++) {
            if (mixed) {
                fill(x, int(rng() % FILL_MODE_COUNT), rng);
                switch (rng() % 3) {
                    case 0: plant_max(x, rng); break;
                    case 1: plant_inf(x, rng); break;
                    default:                   break;
                }
            } else {
                fill(x, FILL_NORMAL, rng);
                if (round % 2) {
                    plant_max(x, rng);
                }
            }

            size_t best = 0;
            if (!ocr_top1_argmax::unique_max(x.data(), x.size(), &best)) {
                continue; // common_sampler builds the full list for this one
            }

            // the candidates of common_sampler::set_logits()
            for (int32_t i = 0; i < n_vocab; i++) {
                full[i] = llama_token_data{i, x[i], 0.0f};
            }
            llama_token_data_array full_p = { full.data(), full.size(), -1, false };

            // the candidate of common_sampler::set_logits_top1()
            llama_token_data       top1   = llama_token_data{(llama_token) best, x[best], 0.0f};
            llama_token_data_array top1_p = { &top1, 1, -1, false };

            llama_sampler_apply(chain_full, &full_p);
            llama_sampler_apply(chain_top1, &top1_p);

            const bool same = full_p.selected >= 0 && same_candidates(full_p, top1_p);
            CHECK(same, "chain %d, n_vocab = %d, round %d: the candidates differ", kind, (int) n_vocab, round);
            if (!same) {
                break;
            }

            const llama_token id = full_p.data[full_p.selected].id;
            llama_sampler_accept(chain_full, id);
            llama_sampler_accept(chain_top1, id);

            n_compared++;
        }
        CHECK(n_compared >= n_rounds / 4,
              "chain %d, n_vocab = %d: %d of %d rounds compared", kind, (int) n_vocab, n_compared, n_rounds);

        llama_sampler_free(chain_full);
        llama_sampler_free(chain_top1);
    }
}

static void expect_gate(llama_sampler * chain, bool want) {
    CHECK(ocr_top1_argmax::chain_starts_with_top_k(chain) == want,
          "chain_starts_with_top_k is %d for: %s", int(!want), chain_names(chain).c_str());
    llama_sampler_free(chain);
}

static void test_chain_gate() {
    const int32_t n_vocab = 1000;

    llama_sampler * server = make_chain(CHAIN_SERVER, n_vocab);
    const std::string names = chain_names(server);
    CHECK(names == "?penalties -> ?dry -> ?top-n-sigma -> top-k -> ?typical -> ?top-p -> min-p -> ?xtc -> temp-ext -> greedy",
          "the server chain is: %s", names.c_str());
    expect_gate(server, true);

    // a copy keeps the names
    llama_sampler * original = make_chain(CHAIN_SERVER, n_vocab);
    expect_gate(llama_sampler_clone(original), true);
    llama_sampler_free(original);

    // the caller checks k
    expect_gate(chain_of({ llama_sampler_init_top_k(40), llama_sampler_init_dist(42) }), true);

    const llama_logit_bias bias = { 123, -100.0f };

    // a sampler that can change the logits ahead of top-k
    expect_gate(chain_of({
        llama_sampler_init_logit_bias(n_vocab, 1, &bias),
        llama_sampler_init_top_k(1),
        llama_sampler_init_greedy(),
    }), false);
    expect_gate(chain_of({
        llama_sampler_init_penalties(n_vocab, 64, 1.1f, 0.0f, 0.0f),
        llama_sampler_init_top_k(1),
        llama_sampler_init_greedy(),
    }), false);
    expect_gate(chain_of({
        llama_sampler_init_top_n_sigma(1.0f),
        llama_sampler_init_top_k(1),
        llama_sampler_init_greedy(),
    }), false);
    expect_gate(chain_of({
        llama_sampler_init_min_p(0.05f, 0),
        llama_sampler_init_top_k(1),
        llama_sampler_init_greedy(),
    }), false);

    // no active top-k
    expect_gate(chain_of({ llama_sampler_init_top_k(0), llama_sampler_init_min_p(0.05f, 0), llama_sampler_init_greedy() }), false);
    expect_gate(chain_of({ llama_sampler_init_top_k(0) }), false);
    expect_gate(chain_of({ llama_sampler_init_greedy() }), false);
    expect_gate(chain_of({}), false);

    // The comparison of test_chains() does fail for a refused chain: the bias moves the winner
    // of the full list, and the single candidate is still the largest raw logit.
    {
        std::vector<float> x(n_vocab, -1.0f);
        x[123] = 5.0f;
        x[456] = 4.0f;

        llama_sampler * chain = chain_of({
            llama_sampler_init_logit_bias(n_vocab, 1, &bias),
            llama_sampler_init_top_k(1),
            llama_sampler_init_greedy(),
        });

        std::vector<llama_token_data> full(n_vocab);
        for (int32_t i = 0; i < n_vocab; i++) {
            full[i] = llama_token_data{i, x[i], 0.0f};
        }
        llama_token_data_array full_p = { full.data(), full.size(), -1, false };

        size_t best = 0;
        CHECK(ocr_top1_argmax::unique_max(x.data(), x.size(), &best) && best == 123, "the biased token is not the maximum");
        llama_token_data       top1   = llama_token_data{(llama_token) best, x[best], 0.0f};
        llama_token_data_array top1_p = { &top1, 1, -1, false };

        llama_sampler_apply(chain, &full_p);
        llama_sampler_apply(chain, &top1_p);

        CHECK(full_p.size == 1 && full_p.data[0].id == 456, "the full list does not pick the runner-up");
        CHECK(!same_candidates(full_p, top1_p), "a logit bias ahead of top-k makes no difference");

        llama_sampler_free(chain);
    }
}

int main(void) {
    ggml_time_init();

    std::mt19937 rng(20261121);

    test_kernels_planted();
    test_kernels_random(rng);
    test_kernels_vocab_sized(rng);

    test_chains(rng, 3001, 400, true);
    test_chains(rng, n_vocab_ocr, 8, false);

    test_chain_gate();

    if (n_failed > 0) {
        printf("%d check(s) failed\n", n_failed);
        return 1;
    }

    printf("OK\n");
    return 0;
}
