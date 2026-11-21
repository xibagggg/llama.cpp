// Tests for common/ocr-draft-preload.h, the parallel draft model load of llama-server.
//
// 1. requested() reads OCR_DRAFT_PARALLEL_LOAD, and only "1" turns it on.
// 2. blocker() lets the load start for one explicit device per model without --fit, warmup and a
//    tensor split, and names the condition that is not met otherwise.
// 3. same_model_params() compares lists by content, tells a change in every field apart, and
//    does not look at the progress callback.
// 4. loader hands back nothing when it was not started or when the load failed, refuses a second
//    start, and joins a load that was never taken.
// 5. With OCR_DRAFT_PRELOAD_TEST_MODEL=<gguf> (not set by ctest): a model loaded by loader next to
//    a second load of the same file on the calling thread reports the same sizes as a model loaded
//    alone, and take() drops the model when the parameters or the path changed since start().

#include "ocr-draft-preload.h"

#include "common.h"
#include "speculative.h"

#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// blocker() and same_model_params() look at the pointer values only, so any distinct addresses do
static char fake_storage[4];

static ggml_backend_dev_t fake_dev(int i) {
    return reinterpret_cast<ggml_backend_dev_t>(&fake_storage[i]);
}

static ggml_backend_buffer_type_t fake_buft(int i) {
    return reinterpret_cast<ggml_backend_buffer_type_t>(&fake_storage[i]);
}

// target parameters the parallel load accepts: a draft model, one explicit device, no fit, no warmup
static common_params accepted_params() {
    common_params params;
    params.speculative.draft.mparams.path = "draft.gguf";
    params.fit_params = false;
    params.warmup     = false;
    params.devices    = { fake_dev(0), nullptr };
    return params;
}

//
// requested()
//

static void test_requested() {
    const char * name = "OCR_DRAFT_PARALLEL_LOAD";

    set_env(name, nullptr);
    CHECK(!ocr_draft_preload::requested(), "on although the variable is not set");

    const char * off[] = { "0", "", "11", "true", "on", " 1" };
    for (const char * value : off) {
        set_env(name, value);
        CHECK(!ocr_draft_preload::requested(), "on for '%s'", value);
    }

    set_env(name, "1");
    CHECK(ocr_draft_preload::requested(), "off for '1'");

    set_env(name, nullptr);
}

//
// blocker()
//

static void check_blocked(const char * what, const common_params & params, const char * word) {
    const char * reason = ocr_draft_preload::blocker(params);
    CHECK(reason != nullptr, "%s: not blocked", what);
    if (reason != nullptr) {
        CHECK(std::strstr(reason, word) != nullptr, "%s: blocked for another reason: %s", what, reason);
    }
}

static void check_accepted(const char * what, const common_params & params) {
    const char * reason = ocr_draft_preload::blocker(params);
    CHECK(reason == nullptr, "%s: blocked: %s", what, reason);
}

static void test_blocker() {
    {
        common_params params = accepted_params();
        check_accepted("one target device, the draft inherits it", params);

        params.speculative.draft.devices = { fake_dev(1), nullptr };
        check_accepted("one target device, one draft device", params);

        params.speculative.draft.devices = { nullptr };
        check_accepted("one target device, draft without a device", params);

        params.devices = { nullptr };
        check_accepted("no device for either model", params);

        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        check_accepted("split mode none", params);

        params.split_mode = LLAMA_SPLIT_MODE_ROW;
        check_accepted("split mode row", params);
    }

    {
        // the defaults of common_params: no draft model, fit and warmup on, no device list
        common_params params;
        check_blocked("defaults", params, "no draft model");
    }
    {
        common_params params = accepted_params();
        params.speculative.draft.mparams.path.clear();
        check_blocked("no draft model", params, "no draft model");
    }
    {
        common_params params = accepted_params();
        params.fit_params = true;
        check_blocked("fit", params, "--fit");
    }
    {
        common_params params = accepted_params();
        params.warmup = true;
        check_blocked("warmup", params, "warmup");
    }
    {
        common_params params = accepted_params();
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        check_blocked("tensor split", params, "tensor");
    }
    {
        common_params params = accepted_params();
        params.devices.clear();
        check_blocked("no device list", params, "no explicit device list");

        // an explicit draft device does not make up for it: the target would still use every device
        params.speculative.draft.devices = { fake_dev(1), nullptr };
        check_blocked("no target device list, one draft device", params, "no explicit device list");
    }
    {
        common_params params = accepted_params();
        params.devices = { fake_dev(0), fake_dev(1), nullptr };
        check_blocked("two target devices", params, "target model uses more than one");

        // the draft would be fine on its own
        params.speculative.draft.devices = { fake_dev(2), nullptr };
        check_blocked("two target devices, one draft device", params, "target model uses more than one");
    }
    {
        common_params params = accepted_params();
        params.speculative.draft.devices = { fake_dev(1), fake_dev(2), nullptr };
        check_blocked("two draft devices", params, "draft model uses more than one");
    }
}

//
// same_model_params()
//

static llama_model_kv_override kv_end() {
    llama_model_kv_override o;
    std::memset(&o, 0, sizeof(o));
    return o;
}

// fill is what the bytes the value does not use are set to
static llama_model_kv_override kv_begin(const char * key, llama_model_kv_override_type tag, unsigned char fill) {
    llama_model_kv_override o;
    std::memset(&o, fill, sizeof(o));
    o.tag = tag;
    std::snprintf(o.key, sizeof(o.key), "%s", key);
    return o;
}

static llama_model_kv_override kv_int(const char * key, int64_t value, unsigned char fill = 0) {
    llama_model_kv_override o = kv_begin(key, LLAMA_KV_OVERRIDE_TYPE_INT, fill);
    o.val_i64 = value;
    return o;
}

static llama_model_kv_override kv_float(const char * key, double value, unsigned char fill = 0) {
    llama_model_kv_override o = kv_begin(key, LLAMA_KV_OVERRIDE_TYPE_FLOAT, fill);
    o.val_f64 = value;
    return o;
}

static llama_model_kv_override kv_bool(const char * key, bool value, unsigned char fill = 0) {
    llama_model_kv_override o = kv_begin(key, LLAMA_KV_OVERRIDE_TYPE_BOOL, fill);
    o.val_bool = value;
    return o;
}

static llama_model_kv_override kv_str(const char * key, const char * value, unsigned char fill = 0) {
    llama_model_kv_override o = kv_begin(key, LLAMA_KV_OVERRIDE_TYPE_STR, fill);
    std::snprintf(o.val_str, sizeof(o.val_str), "%s", value);
    return o;
}

static bool same(common_params & a, common_params & b) {
    return ocr_draft_preload::same_model_params(common_model_params_to_llama(a), common_model_params_to_llama(b));
}

// the patterns of a and b have the same text at different addresses
static const std::string pattern_a   = "blk\\.1\\..*";
static const std::string pattern_b   = "blk\\.1\\..*";
static const std::string pattern_two = "blk\\.2\\..*";

// parameters with every list filled in; fill and pattern differ between the two sides of a comparison
static common_params full_params(unsigned char fill, const std::string & pattern) {
    common_params params = accepted_params();
    params.tensor_buft_overrides = { { pattern.c_str(), fake_buft(0) }, { nullptr, nullptr } };
    params.kv_overrides = {
        kv_int("a.int", 7, fill),
        kv_float("a.float", 0.5, fill),
        kv_bool("a.bool", true, fill),
        kv_str("a.str", "text", fill),
        kv_end(),
    };
    params.tensor_split[0] = 0.25f;
    return params;
}

static void test_same_model_params() {
    {
        common_params a = full_params(0x00, pattern_a);
        common_params b = full_params(0xaa, pattern_b);
        CHECK(pattern_a.c_str() != pattern_b.c_str(), "the two patterns share their storage");
        CHECK(same(a, b) && same(b, a), "equal content at different addresses is told apart");
        CHECK(same(a, a), "parameters differ from themselves");
    }
    {
        // nothing set: every list is null on both sides
        common_params a;
        common_params b;
        CHECK(same(a, b), "the defaults differ from the defaults");
    }

    struct change {
        const char * name;
        std::function<void(common_params &)> apply;
    };
    const change changes[] = {
        { "another device",        [](common_params & p) { p.devices = { fake_dev(1), nullptr }; } },
        { "one more device",       [](common_params & p) { p.devices = { fake_dev(0), fake_dev(1), nullptr }; } },
        { "no device",             [](common_params & p) { p.devices = { nullptr }; } },
        { "no device list",        [](common_params & p) { p.devices.clear(); } },
        { "n_gpu_layers",          [](common_params & p) { p.n_gpu_layers = 3; } },
        { "split_mode",            [](common_params & p) { p.split_mode = LLAMA_SPLIT_MODE_NONE; } },
        { "load_mode",             [](common_params & p) { p.load_mode = LLAMA_LOAD_MODE_MLOCK; } },
        { "lazy_mode",             [](common_params & p) { p.lazy_mode = LLAMA_LAZY_MODE_ON; } },
        { "main_gpu",              [](common_params & p) { p.main_gpu = 1; } },
        { "tensor_split",          [](common_params & p) { p.tensor_split[1] = 0.5f; } },
        { "last tensor_split",     [](common_params & p) { p.tensor_split[llama_max_devices() - 1] = 0.5f; } },
        { "check_tensors",         [](common_params & p) { p.check_tensors = true; } },
        { "no_extra_bufts",        [](common_params & p) { p.no_extra_bufts = true; } },
        { "no_host",               [](common_params & p) { p.no_host = true; } },
        { "no_alloc",              [](common_params & p) { p.no_alloc = true; } },
        { "load_mtp",              [](common_params & p) { p.load_mtp = true; } },
        { "kv int value",          [](common_params & p) { p.kv_overrides[0].val_i64 = 8; } },
        { "kv float value",        [](common_params & p) { p.kv_overrides[1].val_f64 = 0.25; } },
        { "kv bool value",         [](common_params & p) { p.kv_overrides[2].val_bool = false; } },
        { "kv str value",          [](common_params & p) { p.kv_overrides[3] = kv_str("a.str", "other"); } },
        { "kv key",                [](common_params & p) { p.kv_overrides[0] = kv_int("b.int", 7); } },
        { "kv tag",                [](common_params & p) { p.kv_overrides[0] = kv_bool("a.int", true); } },
        { "kv entry removed",      [](common_params & p) { p.kv_overrides.erase(p.kv_overrides.begin()); } },
        { "kv entry added",        [](common_params & p) { p.kv_overrides.insert(p.kv_overrides.begin(), kv_int("z", 1)); } },
        { "no kv overrides",       [](common_params & p) { p.kv_overrides.clear(); } },
        { "override pattern",      [](common_params & p) { p.tensor_buft_overrides[0].pattern = pattern_two.c_str(); } },
        { "override buffer type",  [](common_params & p) { p.tensor_buft_overrides[0].buft = fake_buft(1); } },
        { "override added",        [](common_params & p) {
              p.tensor_buft_overrides.insert(p.tensor_buft_overrides.begin(),
                                             llama_model_tensor_buft_override{ pattern_two.c_str(), fake_buft(0) }); } },
        { "no buffer overrides",   [](common_params & p) { p.tensor_buft_overrides.clear(); } },
    };
    for (const change & c : changes) {
        common_params a = full_params(0x00, pattern_a);
        common_params b = full_params(0xaa, pattern_b);
        c.apply(b);
        CHECK(!same(a, b) && !same(b, a), "%s: not told apart", c.name);
        c.apply(a);
        CHECK(same(a, b) && same(b, a), "%s: differs after the same change on both sides", c.name);
    }

    {
        // llama reads llama_max_devices() entries of the split
        common_params a = full_params(0x00, pattern_a);
        common_params b = full_params(0xaa, pattern_b);
        b.tensor_split[llama_max_devices()] = 0.5f;
        CHECK(same(a, b), "an entry behind the split is compared");
    }
    {
        common_params a = full_params(0x00, pattern_a);
        llama_model_params ma = common_model_params_to_llama(a);
        llama_model_params mb = ma;

        mb.progress_callback           = [](float, void *) { return true; };
        mb.progress_callback_user_data = &a;
        CHECK(ocr_draft_preload::same_model_params(ma, mb), "the progress callback is compared");

        // not reachable through common_params
        mb.vocab_only = !ma.vocab_only;
        CHECK(!ocr_draft_preload::same_model_params(ma, mb), "vocab_only: not told apart");
    }
}

//
// loader
//

// target parameters for a draft model on the CPU
static common_params cpu_params(const std::string & path_dft) {
    common_params params = accepted_params();
    params.devices = { nullptr };
    params.speculative.draft.mparams.path = path_dft;
    return params;
}

static void test_loader() {
    const std::string missing = "test-ocr-draft-preload-no-such-file.gguf";

    {
        ocr_draft_preload::loader loader;
        common_params params_dft = common_base_params_to_speculative(cpu_params(missing));
        CHECK(!loader.started(), "started before start()");
        CHECK(loader.take(params_dft) == nullptr, "a model without a load");
    }
    {
        ocr_draft_preload::loader loader;
        common_params params = cpu_params(missing);
        params.fit_params = true;
        const char * reason = loader.start(params);
        CHECK(reason != nullptr && std::strstr(reason, "--fit") != nullptr, "start() with --fit on: %s", reason ? reason : "started");
        CHECK(!loader.started(), "started although start() refused");
    }
    {
        // the load fails: there is no such file
        ocr_draft_preload::loader loader;
        common_params params = cpu_params(missing);
        const char * reason = loader.start(params);
        CHECK(reason == nullptr, "start() refused: %s", reason);
        CHECK(loader.started(), "not started after start()");

        const char * again = loader.start(params);
        CHECK(again != nullptr, "a second start() was accepted");

        common_params params_dft = common_base_params_to_speculative(params);
        CHECK(loader.take(params_dft) == nullptr, "a model from a file that does not exist");
        CHECK(!loader.started(), "still started after take()");
        CHECK(loader.take(params_dft) == nullptr, "a model from a second take()");
    }
    {
        // never taken: the destructor waits for the thread
        ocr_draft_preload::loader loader;
        common_params params = cpu_params(missing);
        const char * reason = loader.start(params);
        CHECK(reason == nullptr, "start() refused: %s", reason);
    }
}

struct model_summary {
    uint64_t    size     = 0;
    uint64_t    n_params = 0;
    int32_t     n_vocab  = 0;
    int32_t     n_meta   = 0;
    std::string desc;

    bool operator==(const model_summary & o) const {
        return size == o.size && n_params == o.n_params && n_vocab == o.n_vocab && n_meta == o.n_meta && desc == o.desc;
    }
};

static model_summary summarize(const llama_model * model) {
    model_summary s;
    char buf[256] = { 0 };
    llama_model_desc(model, buf, sizeof(buf));
    s.size     = llama_model_size(model);
    s.n_params = llama_model_n_params(model);
    s.n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    s.n_meta   = llama_model_meta_count(model);
    s.desc     = buf;
    return s;
}

static void test_loader_with_model(const std::string & path) {
    const common_params params_tgt = cpu_params(path);

    // loaded alone, the way the serial path loads it
    common_params params_ref = common_base_params_to_speculative(params_tgt);
    params_ref.load_progress_callback = [](float, void *) { return true; };
    llama_model_ptr ref = common_speculative_load_draft_model(params_ref);
    CHECK(ref != nullptr, "cannot load '%s'", path.c_str());
    if (ref == nullptr) {
        return;
    }
    const model_summary expected = summarize(ref.get());
    printf("model: %s, %llu bytes, %llu parameters, %d tokens\n", expected.desc.c_str(),
           (unsigned long long) expected.size, (unsigned long long) expected.n_params, expected.n_vocab);

    {
        // loaded by loader while this thread loads the same file, as the target load would
        ocr_draft_preload::loader loader;
        const char * reason = loader.start(params_tgt);
        CHECK(reason == nullptr, "start() refused: %s", reason);

        llama_model_ptr other = common_speculative_load_draft_model(params_ref);
        CHECK(other != nullptr && summarize(other.get()) == expected, "the load on this thread differs");

        common_params params_dft = common_base_params_to_speculative(params_tgt);
        llama_model_ptr model = loader.take(params_dft);
        CHECK(model != nullptr, "no model from the parallel load");
        CHECK(model != nullptr && summarize(model.get()) == expected, "the parallel load differs");
    }
    {
        ocr_draft_preload::loader loader;
        const char * reason = loader.start(params_tgt);
        CHECK(reason == nullptr, "start() refused: %s", reason);

        common_params params_dft = common_base_params_to_speculative(params_tgt);
        params_dft.check_tensors = !params_dft.check_tensors;
        CHECK(loader.take(params_dft) == nullptr, "a model loaded with other parameters was kept");
    }
    {
        ocr_draft_preload::loader loader;
        const char * reason = loader.start(params_tgt);
        CHECK(reason == nullptr, "start() refused: %s", reason);

        common_params params_dft = common_base_params_to_speculative(params_tgt);
        params_dft.speculative.draft.mparams.path += ".other";
        CHECK(loader.take(params_dft) == nullptr, "a model loaded from another path was kept");
    }
}

int main(void) {
    ggml_time_init();
    llama_backend_init();

    test_requested();
    test_blocker();
    test_same_model_params();
    test_loader();

    const char * model = std::getenv("OCR_DRAFT_PRELOAD_TEST_MODEL");
    if (model != nullptr && model[0] != 0) {
        test_loader_with_model(model);
    } else {
        printf("OCR_DRAFT_PRELOAD_TEST_MODEL is not set, the loads of a real model are skipped\n");
    }

    llama_backend_free();

    if (n_failed > 0) {
        printf("%d check(s) failed\n", n_failed);
        return 1;
    }

    printf("OK\n");
    return 0;
}
