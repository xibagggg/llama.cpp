// SPDX-License-Identifier: MIT
#include "ocr-ep.h"
#include "ggml-spacemit.h"
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <memory>
#include <stdexcept>

extern "C" OrtStatus * ORT_API_CALL OrtSessionOptionsSpaceMITEnvInit(
    OrtSessionOptions *, const char * const *, const char * const *, size_t);

namespace {
using clock_type = std::chrono::steady_clock;
double elapsed(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}
std::string setting(const char * name, const char * fallback = "") {
    const char * value = std::getenv(name);
    return value && *value ? value : fallback;
}
int integer_setting(const char * name, int fallback, int minimum, int maximum) {
    const auto value = setting(name, std::to_string(fallback).c_str());
    size_t consumed = 0;
    const int result = std::stoi(value, &consumed);
    if (consumed != value.size() || result < minimum || result > maximum) {
        throw std::invalid_argument(std::string(name) + " is outside the supported range");
    }
    return result;
}
struct ai_guard {
    ai_guard() { ggml_backend_spacemit_acquire(); }
    ~ai_guard() { ggml_backend_spacemit_release(); }
};
struct session_entry {
    int width;
    int height;
    std::unique_ptr<Ort::Session> session;
};
struct runtime {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "llama-ocr-ep"};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::list<session_entry> vision;
    std::unique_ptr<Ort::Session> layout;
    unsigned sequence = 0;

    std::unique_ptr<Ort::Session> create(const std::string & path, int width, int height, const char * kind) {
        if (path.empty()) {
            throw std::runtime_error(std::string("OCR_EP_") + kind + "_MODEL must name an ONNX model");
        }
        if (setting("SPERT_SHARED_MODE") != "DDR") {
            throw std::runtime_error("EP OCR requires SPERT_SHARED_MODE=DDR");
        }
        auto start = clock_type::now();
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(8);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.DisableCpuMemArena();
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        options.AddConfigEntry("session.inter_op.allow_spinning", "0");
        const auto strict = setting("OCR_EP_STRICT", "1");
        if (strict != "0" && strict != "1") {
            throw std::invalid_argument("OCR_EP_STRICT must be 0 or 1");
        }
        options.AddConfigEntry("session.disable_cpu_ep_fallback", strict.c_str());
        const auto profile = setting("OCR_EP_PROFILE_PREFIX");
        if (!profile.empty()) {
            options.EnableProfiling((profile + "-" + kind + "-" + std::to_string(width) + "x" + std::to_string(height)).c_str());
        }
        if (width > 0 && height > 0) {
            Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options, "height", height));
            Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options, "width", width));
        }
        const std::array<const char *, 6> keys = {
            "SPERT_BACKEND", "SPACEMIT_EP_INTER_THREAD_NUM", "SPACEMIT_EP_INTRA_THREAD_NUM",
            "SPACEMIT_EP_ENABLE_DMA", "SPACEMIT_EP_DENSE_ACCURACY_LEVEL", "SPACEMIT_EP_ENABLE_BLOCKLAYOUT"};
        const auto threads = std::to_string(integer_setting("OCR_EP_THREADS", 8, 1, 8));
        const auto accuracy = std::to_string(integer_setting("OCR_EP_DENSE_ACCURACY", 0, 0, 2));
        const auto blocklayout = setting("OCR_EP_BLOCKLAYOUT", "0");
        const std::array<const char *, 6> values = {"spacemit-k3", "1", threads.c_str(), "0", accuracy.c_str(), blocklayout.c_str()};
        Ort::ThrowOnError(OrtSessionOptionsSpaceMITEnvInit(options, keys.data(), values.data(), keys.size()));
        auto result = std::make_unique<Ort::Session>(env, path.c_str(), options);
        std::fprintf(stderr, "OCR_EP_COMPILE kind=%s shape=%dx%d seconds=%.6f strict=%s scratch=DDR threads=%s\n",
                     kind, width, height, elapsed(start), strict.c_str(), threads.c_str());
        return result;
    }

    Ort::Session & vision_session(int width, int height) {
        for (auto it = vision.begin(); it != vision.end(); ++it) {
            if (it->width == width && it->height == height) {
                vision.splice(vision.begin(), vision, it);
                return *vision.front().session;
            }
        }
        const auto limit = integer_setting("OCR_EP_VISION_CACHE", 2, 1, 3);
        if (vision.size() >= static_cast<size_t>(limit)) {
            vision.pop_back();
        }
        auto session = create(setting("OCR_EP_VISION_MODEL"), width, height, "VISION");
        const auto shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape != std::vector<int64_t>({1, 3, height, width})) {
            throw std::runtime_error("Vision EP specialization did not produce the requested input shape");
        }
        vision.push_front({width, height, std::move(session)});
        return *vision.front().session;
    }
};
runtime & get_runtime() {
    static runtime value;
    return value;
}
struct float_view {
    const float * f32 = nullptr;
    const Ort::Float16_t * f16 = nullptr;
    explicit float_view(const Ort::Value & value) {
        const auto type = value.GetTensorTypeAndShapeInfo().GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            f32 = value.GetTensorData<float>();
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            f16 = value.GetTensorData<Ort::Float16_t>();
        } else {
            throw std::runtime_error("Expected float32 or float16 ONNX output");
        }
    }
    float operator[](size_t i) const { return f32 ? f32[i] : f16[i].ToFloat(); }
};
} // namespace

bool ocr_ep_vision_enabled() {
    return !setting("OCR_EP_VISION_MODEL").empty();
}

void ocr_ep_encode_hwc(const float * pixels, int width, int height, std::vector<float> & embeddings) {
    if (!pixels || width < 28 || height < 28 || width > 8192 || height > 8192 ||
        width % 28 || height % 28 || static_cast<int64_t>(width) * height > 1960000) {
        throw std::invalid_argument("Unsupported MiniCPM OCR vision input dimensions");
    }
    // Transpose only; image resizing and normalization remain exactly mtmd's.
    const size_t area = static_cast<size_t>(width) * height;
    std::vector<float> nchw(area * 3);
    for (size_t i = 0; i < area; ++i) {
        for (size_t channel = 0; channel < 3; ++channel) {
            const float value = pixels[3 * i + channel];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Non-finite vision input");
            }
            nchw[channel * area + i] = value;
        }
    }
    ai_guard guard;
    auto & owner = get_runtime();
    auto & session = owner.vision_session(width, height);
    const std::array<int64_t, 4> shape{1, 3, height, width};
    auto input = Ort::Value::CreateTensor<float>(owner.memory, nchw.data(), nchw.size(), shape.data(), shape.size());
    Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name = session.GetInputNameAllocated(0, allocator);
    const auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char * inputs[] = {input_name.get()};
    const char * outputs[] = {output_name.get()};
    auto start = clock_type::now();
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
    const double seconds = elapsed(start);
    const auto info = result[0].GetTensorTypeAndShapeInfo();
    if (info.GetShape() != std::vector<int64_t>({1, static_cast<int64_t>(area / 784), 1024})) {
        throw std::runtime_error("Unexpected MiniCPM OCR visual feature shape");
    }
    embeddings.resize(info.GetElementCount());
    const float_view features(result[0]);
    for (size_t i = 0; i < embeddings.size(); ++i) {
        embeddings[i] = features[i];
        if (!std::isfinite(embeddings[i])) {
            throw std::runtime_error("Non-finite EP visual features");
        }
    }
    const auto dump = setting("OCR_EP_DUMP_DIR");
    if (!dump.empty() && owner.sequence < static_cast<unsigned>(integer_setting("OCR_EP_DUMP_LIMIT", 10, 0, 100))) {
        const auto prefix = dump + "/vision-" + std::to_string(owner.sequence);
        std::ofstream input_file(prefix + ".input.f32", std::ios::binary);
        std::ofstream output_file(prefix + ".output.f32", std::ios::binary);
        std::ofstream metadata(prefix + ".json");
        input_file.write(reinterpret_cast<const char *>(nchw.data()), nchw.size() * sizeof(float));
        output_file.write(reinterpret_cast<const char *>(embeddings.data()), embeddings.size() * sizeof(float));
        metadata << "{\"width\":" << width << ",\"height\":" << height << "}\n";
        if (!input_file || !output_file || !metadata) {
            throw std::runtime_error("Failed to save EP visual verification data");
        }
    }
    std::fprintf(stderr, "OCR_EP_VISION sequence=%u shape=%dx%d seconds=%.6f tokens=%zu\n",
                 owner.sequence++, width, height, seconds, area / 784);
}

std::vector<ocr_ep_tensor> ocr_ep_layout(const float * pixels, size_t count, double & seconds) {
    if (!pixels || count != 3 * 800 * 800 ||
        !std::all_of(pixels, pixels + count, [](float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("Layout expects finite float32 NCHW pixels, shape 1x3x800x800");
    }
    ai_guard guard;
    auto & owner = get_runtime();
    if (!owner.layout) {
        owner.layout = owner.create(setting("OCR_EP_LAYOUT_MODEL"), 0, 0, "LAYOUT");
    }
    auto & session = *owner.layout;
    const std::array<int64_t, 4> shape{1, 3, 800, 800};
    auto input = Ort::Value::CreateTensor<float>(owner.memory, const_cast<float *>(pixels), count, shape.data(), shape.size());
    const char * inputs[] = {"pixel_values"};
    const std::array<const char *, 4> outputs{"logits", "pred_boxes", "order_logits", "out_masks"};
    auto start = clock_type::now();
    auto values = session.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs.data(), outputs.size());
    seconds = elapsed(start);
    std::vector<ocr_ep_tensor> result;
    for (size_t output = 0; output < values.size(); ++output) {
        const auto info = values[output].GetTensorTypeAndShapeInfo();
        const size_t elements = info.GetElementCount();
        const bool mask = output == 3;
        const float_view features(values[output]);
        ocr_ep_tensor tensor{outputs[output], mask ? "uint8" : "float32", info.GetShape(), {}};
        tensor.data.resize(elements * (mask ? 1 : sizeof(float)));
        for (size_t i = 0; i < elements; ++i) {
            const float value = features[i];
            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite EP layout output");
            }
            if (mask) {
                tensor.data[i] = value > 0;
            } else {
                std::memcpy(tensor.data.data() + i * sizeof(float), &value, sizeof(float));
            }
        }
        result.push_back(std::move(tensor));
    }
    std::fprintf(stderr, "OCR_EP_LAYOUT seconds=%.6f outputs=4 scratch=DDR\n", seconds);
    return result;
}
