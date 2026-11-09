#include "clip.h"
#include "clip-impl.h"
#include "ggml-backend.h"
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <regex>
#include <string>
#include <sys/resource.h>

static void logger(enum ggml_log_level level, const char * text, void *) {
    if (level != GGML_LOG_LEVEL_DEBUG) std::fputs(text, stderr);
}

struct snapshot_state {
    std::string directory;
    int repeat = -2;
    std::regex filter;

    snapshot_state() {
        const char * value = std::getenv("MTMD_OCR_LAYER_SNAPSHOT_FILTER");
        filter = std::regex(value ? value : "^(layer_inp_normed|attn_out|ffn_out|layer_out)-[0-9]+$");
    }
};

static bool capture_layer_outputs(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<snapshot_state *>(user_data);
    if (state->directory.empty() || state->repeat < 0 || state->repeat > 1 ||
        !std::regex_match(tensor->name, state->filter)) {
        return false;
    }
    if (ask) return true;

    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> data(nbytes);
    if (tensor->buffer != nullptr && ggml_backend_buffer_is_host(tensor->buffer)) {
        std::memcpy(data.data(), tensor->data, nbytes);
    } else {
        ggml_backend_tensor_get(tensor, data.data(), 0, nbytes);
    }
    std::string name(tensor->name);
    for (char & c : name) if (!(std::isalnum((unsigned char)c) || c == '-' || c == '_')) c = '_';
    const auto path = std::filesystem::path(state->directory) /
        ("r" + std::to_string(state->repeat) + "-" + name + ".bin");
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(data.data()), data.size());
    if (!output) return false;
    std::ofstream meta(path.string() + ".txt");
    meta << "name=" << tensor->name << "\ntype=" << ggml_type_name(tensor->type)
         << "\nbytes=" << nbytes << "\nne=" << tensor->ne[0] << ',' << tensor->ne[1] << ','
         << tensor->ne[2] << ',' << tensor->ne[3] << "\n";
    std::fprintf(stderr, "OCR_SNAPSHOT repeat=%d name=%s bytes=%zu\n", state->repeat, tensor->name, nbytes);
    return bool(meta);
}

int main(int argc, char ** argv) {
    if (argc != 7) return 1;
    const int width = std::atoi(argv[2]), height = std::atoi(argv[3]), repeats = std::atoi(argv[6]);
    if (width <= 0 || height <= 0 || repeats < 1 || repeats > 4) return 2;
    std::vector<float> pixels(size_t(width) * height * 3);
    std::ifstream input(argv[4], std::ios::binary);
    input.read((char *) pixels.data(), pixels.size() * sizeof(float));
    if (!input || input.peek() != EOF) return 3;
    ggml_log_set(logger, nullptr);
    ggml_backend_load_all();
    clip_context_params params{};
    params.use_gpu = true; params.device = ggml_backend_dev_by_name("SPACEMIT0");
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_ENABLED;
    params.image_min_tokens = -1; params.image_max_tokens = -1; params.warmup = false;
    snapshot_state snapshots;
    if (const char * dir = std::getenv("MTMD_OCR_LAYER_SNAPSHOT_DIR")) {
        snapshots.directory = dir;
        std::filesystem::create_directories(snapshots.directory);
        params.cb_eval = capture_layer_outputs;
        params.cb_eval_user_data = &snapshots;
    }
    if (!params.device) return 4;
    auto init = clip_init(argv[1], params);
    if (!init.ctx_v) return 5;
    clip_image_f32 image; image.set_size({width, height}, false, false); image.cpy_buf(pixels);
    const int tokens = clip_n_output_tokens(init.ctx_v, &image);
    const int dimensions = clip_n_mmproj_embd(init.ctx_v);
    if (tokens <= 0 || dimensions <= 0) return 10;
    std::vector<float> features(size_t(tokens) * dimensions), reference, formal_reference;
    bool formal_drift = false;
    for (int repeat = -1; repeat < repeats; ++repeat) {
        snapshots.repeat = repeat;
        const auto begin = std::chrono::steady_clock::now();
        if (!clip_image_encode(init.ctx_v, 7, &image, features)) return 6;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        for (float value : features) if (!std::isfinite(value)) return 7;
        if (repeat == -1) reference = features;
        if (repeat == 0) formal_reference = features;
        const bool warmup_exact = reference.size() == features.size() &&
            !std::memcmp(reference.data(), features.data(), features.size() * sizeof(float));
        const bool formal_exact = repeat < 1 || (formal_reference.size() == features.size() &&
            !std::memcmp(formal_reference.data(), features.data(), features.size() * sizeof(float)));
        formal_drift = formal_drift || !formal_exact;
        std::ofstream sample(std::string(argv[5]) + ".r" + std::to_string(repeat), std::ios::binary);
        sample.write((const char *) features.data(), features.size() * sizeof(float));
        if (!sample) return 11;
        struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        std::printf("{\"repeat\":%d,\"seconds\":%.9f,\"width\":%d,\"height\":%d,\"elements\":%zu,\"rss_hwm_kib\":%ld,\"warmup_exact\":%s,\"formal_repeat_exact\":%s}\n",
                    repeat, seconds, width, height, features.size(), usage.ru_maxrss,
                    warmup_exact ? "true" : "false", formal_exact ? "true" : "false");
        std::fflush(stdout);
    }
    std::ofstream output(argv[5], std::ios::binary); output.write((const char *) features.data(), features.size() * sizeof(float));
    clip_free(init.ctx_v);
    return !output ? 9 : formal_drift ? 8 : 0;
}
