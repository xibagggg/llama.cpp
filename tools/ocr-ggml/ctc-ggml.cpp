#include "ocr-float-graph.h"

#ifdef OCR_CTC_LIBRARY
namespace {
struct ctc_handle {
    ctc_graph model;
    std::mutex mutex;
};

void set_error(char * error, size_t size, const char * message) {
    if (error && size) std::snprintf(error, size, "%s", message);
}
}

extern "C" void * ctc_ggml_create(const char * path, int threads, char * error, size_t size) {
    try {
        require(path != nullptr && threads > 0 && threads <= 64, "invalid model path/threads");
        auto handle = std::make_unique<ctc_handle>();
        handle->model.load(path, threads, false);
        set_error(error, size, "");
        return handle.release();
    } catch (const std::exception & failure) {
        set_error(error, size, failure.what());
        return nullptr;
    }
}

extern "C" void ctc_ggml_destroy(void * handle) {
    delete static_cast<ctc_handle *>(handle);
}

extern "C" int ctc_ggml_describe(void * raw, char * output, size_t output_size, char * error, size_t size) {
    try {
        require(raw && output && output_size, "null description argument");
        auto & handle = *static_cast<ctc_handle *>(raw);
        std::lock_guard<std::mutex> lock(handle.mutex);
        const auto text = handle.model.describe().dump();
        require(text.size() < output_size, "description buffer too small");
        std::memcpy(output, text.c_str(), text.size() + 1);
        set_error(error, size, "");
        return 0;
    } catch (const std::exception & failure) {
        set_error(error, size, failure.what());
        return -1;
    }
}

extern "C" int ctc_ggml_shapes(void * raw, int64_t * input, int64_t * output, char * error, size_t size) {
    try {
        require(raw && input && output, "null shape argument");
        auto & handle = *static_cast<ctc_handle *>(raw);
        std::lock_guard<std::mutex> lock(handle.mutex);
        const auto in = handle.model.plan.at("input_shape").get<shape>();
        require(in.size() == 4, "input rank must be four");
        std::copy(in.begin(), in.end(), input);
        output[0] = in[0];
        output[1] = handle.model.output->ne[1];
        output[2] = handle.model.output->ne[0];
        require(output[0] * output[1] * output[2] == ggml_nelements(handle.model.output), "unexpected CTC output rank");
        set_error(error, size, "");
        return 0;
    } catch (const std::exception & failure) {
        set_error(error, size, failure.what());
        return -1;
    }
}

extern "C" int ctc_ggml_run(void * raw, const float * input, size_t input_count,
                            float * output, size_t output_count, double * seconds,
                            char * error, size_t size) {
    try {
        require(raw && input && output && seconds, "null execution argument");
        auto & handle = *static_cast<ctc_handle *>(raw);
        std::lock_guard<std::mutex> lock(handle.mutex);
        auto & model = handle.model;
        require(input_count == size_t(ggml_nelements(model.input)), "input element count mismatch");
        require(output_count == size_t(ggml_nelements(model.output)), "output element count mismatch");
        for (size_t i = 0; i < input_count; ++i) require(std::isfinite(input[i]), "nonfinite input");
        ggml_backend_tensor_set(model.input, input, 0, input_count * sizeof(float));
        const auto begin = std::chrono::steady_clock::now();
        require(model.compute() == GGML_STATUS_SUCCESS, "GGML execution failed");
        *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        ggml_backend_tensor_get(model.output, output, 0, output_count * sizeof(float));
        for (size_t i = 0; i < output_count; ++i) require(std::isfinite(output[i]), "nonfinite output");
        set_error(error, size, "");
        return 0;
    } catch (const std::exception & failure) {
        set_error(error, size, failure.what());
        return -1;
    }
}
#else
int main(int argc, char ** argv) {
    try {
        require(argc >= 4 && argc <= 7, "usage: ctc-ggml MODEL INPUT_F32 OUTPUT_F32 [THREADS] [REPEATS] [DUMP_DIR]");
        const int threads = argc > 4 ? std::stoi(argv[4]) : 4;
        const int repeats = argc > 5 ? std::stoi(argv[5]) : 1;
        require(threads > 0 && threads <= 64 && repeats > 0 && repeats <= 10, "invalid threads/repeats");
        ctc_graph model;
        const auto setup_begin = std::chrono::steady_clock::now();
        model.load(argv[1], threads, argc > 6);
        std::vector<float> input(size_t(ggml_nelements(model.input)));
        std::ifstream source(argv[2], std::ios::binary);
        source.read(reinterpret_cast<char *>(input.data()), input.size() * sizeof(float));
        require(bool(source) && source.peek() == EOF, "input file size mismatch");
        for (float v : input) require(std::isfinite(v), "nonfinite input");
        const double setup_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_begin).count();
        json times = json::array();
        std::vector<float> reference;
        for (int i = 0; i < repeats; ++i) {
            ggml_backend_tensor_set(model.input, input.data(), 0, input.size() * sizeof(float));
            const auto begin = std::chrono::steady_clock::now();
            require(model.compute() == GGML_STATUS_SUCCESS, "GGML execution failed");
            times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count());
            std::vector<float> input_check(input.size());
            ggml_backend_tensor_get(model.input, input_check.data(), 0, input_check.size() * sizeof(float));
            require(input_check == input, "input changed during execution");
            std::vector<float> current(size_t(ggml_nelements(model.output)));
            ggml_backend_tensor_get(model.output, current.data(), 0, current.size() * sizeof(float));
            if (i == 0) reference = current;
            require(!std::memcmp(current.data(), reference.data(), current.size() * sizeof(float)), "repeat output changed");
        }
        auto save = [](const std::filesystem::path & path, ggml_tensor * tensor) {
            std::vector<float> output(size_t(ggml_nelements(tensor)));
            ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
            for (float v : output) require(std::isfinite(v), "nonfinite output");
            std::ofstream sink(path, std::ios::binary);
            sink.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
            require(bool(sink), "cannot write output");
        };
        save(argv[3], model.output);
        if (argc > 6) {
            std::filesystem::create_directories(argv[6]);
            for (const auto & item : model.checkpoints) save(std::filesystem::path(argv[6]) / (item.first + ".f32"), item.second);
        }
        std::printf("%s\n", json{{"engine", "ggml-" + model.backend_mode}, {"setup_s", setup_s}, {"run_s", times},
                    {"output_elements", ggml_nelements(model.output)},
                    {"compute_bytes", model.compute_bytes()}, {"backend_info", model.describe()},
                    {"pretransposed_matmul_count", model.pretransposed_matmul_count},
                    {"pretransposed_matmul_bytes", model.pretransposed_matmul_bytes},
                    {"repeats_exact", true}, {"input_preserved", true}}.dump().c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "ctc-ggml: %s\n", error.what());
        return 1;
    }
}
#endif
