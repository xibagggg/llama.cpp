#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "json.hpp"
#include "ctc-ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using shape = std::vector<int64_t>;

// Literal diagnostics are also used in per-element validation loops. Avoid
// constructing a temporary std::string on every successful check.
static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

static int64_t elements(const shape & dims) {
    return std::accumulate(dims.begin(), dims.end(), int64_t(1), std::multiplies<int64_t>());
}

struct value {
    ggml_tensor * storage = nullptr;
    shape dims;
    std::vector<size_t> strides;
    size_t offset = 0;
};

static value packed(ggml_tensor * tensor, const shape & dims) {
    value result{tensor, dims, std::vector<size_t>(dims.size()), 0};
    size_t stride = ggml_type_size(tensor->type);
    for (int i = int(dims.size()) - 1; i >= 0; --i) {
        result.strides[i] = stride;
        stride *= size_t(dims[i]);
    }
    require(elements(dims) == ggml_nelements(tensor), "storage/shape mismatch");
    return result;
}

static bool is_packed(const value & input) {
    size_t stride = ggml_type_size(input.storage->type);
    for (int i = int(input.dims.size()) - 1; i >= 0; --i) {
        if (input.dims[i] > 1 && input.strides[i] != stride) return false;
        stride *= size_t(input.dims[i]);
    }
    return true;
}

static ggml_tensor * tensor_view(ggml_context * ctx, const value & input) {
    require(input.dims.size() <= 4, "cannot materialize rank > 4");
    int64_t ne[4] = {1, 1, 1, 1};
    size_t nb[4] = {4, 4, 4, 4};
    const int rank = int(input.dims.size());
    size_t span = ggml_type_size(input.storage->type);
    for (int i = 0; i < rank; ++i) {
        ne[i] = input.dims[rank - 1 - i];
        nb[i] = input.strides[rank - 1 - i];
        span += size_t(ne[i] - 1) * nb[i];
    }
    for (int i = rank; i < 4; ++i) nb[i] = span;
    require(input.offset + span <= ggml_nbytes(input.storage), "view exceeds storage");
    auto * result = ggml_view_4d(ctx, input.storage, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], input.offset);
    result->nb[0] = nb[0];
    return result;
}

static ggml_tensor * contiguous(ggml_context * ctx, const value & input) {
    auto * result = tensor_view(ctx, input);
    return ggml_is_contiguous(result) ? result : ggml_cont(ctx, result);
}

#include "doclayout-backend.h"

class ctc_graph {
public:
    gguf_context * file = nullptr;
    ggml_context * weights = nullptr;
    ggml_context * context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    json plan;
    std::map<std::string, value> values;
    std::map<std::string, ggml_tensor *> checkpoints;
    std::map<std::string, ggml_tensor *> quantized_linear_weights;
    std::unique_ptr<layout_backend> scheduled_backend;
    std::string backend_mode = "cpu";
    bool pointwise_pack = std::getenv("OCR_CONV_POINTWISE_PACK") != nullptr
        && std::string(std::getenv("OCR_CONV_POINTWISE_PACK")) == "1";
    int pointwise_pack_count = 0;
    int pretransposed_matmul_count = 0;
    uint64_t pretransposed_matmul_bytes = 0;
    int quantized_linear_count = 0;
    uint64_t padded_linear_elements = 0;

    ~ctc_graph() {
        scheduled_backend.reset();
        if (allocator) ggml_gallocr_free(allocator);
        if (context) ggml_free(context);
        if (weights) ggml_free(weights);
        if (file) gguf_free(file);
        if (backend) ggml_backend_free(backend);
    }

    ggml_tensor * get(const std::string & name) { return contiguous(context, values.at(name)); }

    void load(const char * path, int threads, bool snapshots) {
        file = gguf_init_from_file(path, {false, &weights});
        require(file != nullptr, "GGUF load failed");
        const auto key = gguf_find_key(file, "ocr.ctc.graph");
        require(key >= 0, "CTC graph metadata missing");
        plan = json::parse(gguf_get_val_str(file, key));
        require(plan.at("version") == 1 || plan.at("version") == 2 || plan.at("version") == 3,
                "unsupported CTC graph version");
        backend = ggml_backend_cpu_init();
        require(backend != nullptr, "CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(backend, threads);
        if (const char * selected = std::getenv("OCR_CTC_BACKEND")) backend_mode = selected;
        require(backend_mode == "cpu" || backend_mode == "cpu-sched" || backend_mode == "spacemit",
                "invalid CTC backend");
        if (backend_mode != "cpu") {
            scheduled_backend = std::make_unique<layout_backend>(backend_mode, backend);
            scheduled_backend->import_weights(weights, plan.at("weights"));
        }
        context = ggml_init({ggml_tensor_overhead() * 12000 + ggml_graph_overhead_custom(8192, false), nullptr, true});
        require(context != nullptr, "graph metadata allocation failed");
        for (const auto & item : plan.at("weights")) {
            const std::string name = item.at("id");
            auto * t = scheduled_backend ? scheduled_backend->imported.at(name) : ggml_get_tensor(weights, name.c_str());
            require(t != nullptr, "missing CTC weight: " + name);
            const auto dims = item.at("shape").get<shape>();
            if (t->type == GGML_TYPE_F32) values.emplace(name, packed(t, dims));
            else {
                require(plan.at("version") == 3 && t->type == GGML_TYPE_Q8_0 && dims.size() == 2
                        && dims[0] > 0 && dims[1] > 0 && t->ne[0] == dims[1] && t->ne[1] == dims[0]
                        && t->ne[2] == 1 && t->ne[3] == 1 && dims[1] % 32 == 0,
                        "invalid quantized CTC linear weight: " + name);
                quantized_linear_weights.emplace(name, t);
            }
        }
        auto dims = plan.at("input_shape").get<shape>();
        require(dims.size() == 4 && dims[1] == 3 && dims[2] == 48, "invalid CTC input shape");
        input = ggml_new_tensor_4d(context, GGML_TYPE_F32, dims[3], dims[2], dims[1], dims[0]);
        ggml_set_input(input);
        ggml_set_output(input); // repeated requests must not reuse the input as scratch
        values.emplace(plan.at("input").get<std::string>(), packed(input, dims));
        for (const auto & node : plan.at("nodes")) {
            try { append(node); }
            catch (const std::exception & error) {
                throw std::runtime_error(node.at("name").get<std::string>() + ": " + error.what());
            }
            const std::string name = node.at("output");
            const auto & result = values.at(name);
            if (snapshots && result.dims.size() <= 4) {
                auto * checkpoint = contiguous(context, result);
                ggml_set_output(result.storage);
                ggml_set_name(checkpoint, name.c_str());
                ggml_set_output(checkpoint);
                checkpoints.emplace(name, checkpoint);
            }
        }
        output = get(plan.at("output"));
        ggml_set_output(output);
        graph = ggml_new_graph_custom(context, 8192, false);
        for (const auto & item : checkpoints) ggml_build_forward_expand(graph, item.second);
        ggml_build_forward_expand(graph, output);
        if (scheduled_backend) scheduled_backend->allocate(graph, input);
        else {
            allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            require(allocator && ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
        }
    }

    enum ggml_status compute() {
        return scheduled_backend ? scheduled_backend->compute(graph) : ggml_backend_graph_compute(backend, graph);
    }

    size_t compute_bytes() const {
        return scheduled_backend ? scheduled_backend->compute_bytes() : ggml_gallocr_get_buffer_size(allocator, 0);
    }

    json describe() const {
        json result = scheduled_backend ? scheduled_backend->describe() : json{{"mode", "cpu"}, {"scheduler", false}};
        result["quantized_linear_count"] = quantized_linear_count;
        result["padded_linear_elements_static"] = padded_linear_elements;
        result["weight_precision"] = quantized_linear_count ? "q8_0-linear-mixed-gguf" : "fp32-gguf";
        return result;
    }

    void append(const json & node) {
        const std::string op = node.at("op"), name = node.at("output");
        const auto ins = node.at("inputs").get<std::vector<std::string>>();
        const auto dims = node.at("shape").get<shape>();
        const auto & attrs = node.at("attrs");
        auto attr = [&](const char * key, int fallback) { return attrs.value(key, fallback); };
        auto ints = [&](size_t index) -> json { return node.at("integer_inputs").at(std::to_string(index)); };
        auto original = [&]() -> value { return values.at(ins.at(0)); };
        ggml_tensor * result = nullptr;

        if (op == "Reshape" || op == "Squeeze") {
            auto source = original();
            require(elements(source.dims) == elements(dims), "invalid reshape size");
            if (!is_packed(source)) source = packed(contiguous(context, source), source.dims);
            auto reshaped = packed(source.storage, dims);
            reshaped.offset = source.offset;
            values[name] = std::move(reshaped);
            return;
        }
        if (op == "Transpose") {
            auto source = original(), target = source;
            const auto order = attrs.at("perm").get<std::vector<int>>();
            require(order.size() == source.dims.size(), "transpose rank mismatch");
            for (size_t i = 0; i < order.size(); ++i) {
                target.dims[i] = source.dims.at(order[i]);
                target.strides[i] = source.strides.at(order[i]);
            }
            require(target.dims == dims, "transpose output mismatch");
            values[name] = std::move(target);
            return;
        }
        if (op == "Gather") {
            auto source = original();
            int axis = attr("axis", 0);
            if (axis < 0) axis += int(source.dims.size());
            const auto index_value = ints(1);
            require(index_value.is_number_integer(), "only scalar tensor Gather is implemented");
            int64_t index = index_value.get<int64_t>();
            if (index < 0) index += source.dims.at(axis);
            require(index >= 0 && index < source.dims.at(axis), "Gather index out of bounds");
            source.offset += size_t(index) * source.strides.at(axis);
            source.dims.erase(source.dims.begin() + axis);
            source.strides.erase(source.strides.begin() + axis);
            require(source.dims == dims, "Gather output mismatch");
            values[name] = std::move(source);
            return;
        }
        if (op == "Conv") {
            auto * x = get(ins.at(0));
            auto * w = get(ins.at(1));
            const auto stride = attrs.at("strides").get<std::vector<int>>();
            const auto pad = attrs.at("pads").get<std::vector<int>>();
            const auto dilation = attrs.at("dilations").get<std::vector<int>>();
            require(pad[0] == pad[2] && pad[1] == pad[3], "asymmetric Conv padding unsupported");
            const int group = attr("group", 1);
            if (group == 1) {
                // The standard conv helper stages FP16 im2col. Use F32 here
                // to establish a precision-preserving conversion reference.
                const bool pointwise = pointwise_pack && w->ne[0] == 1 && w->ne[1] == 1
                    && stride[0] == 1 && stride[1] == 1 && pad[0] == 0 && pad[1] == 0
                    && dilation[0] == 1 && dilation[1] == 1;
                ggml_tensor * patches;
                if (pointwise) {
                    // 1x1/S1/P0 convolution only changes NCHW to channel-major
                    // patches. Use native layout packing instead of generic im2col.
                    patches = ggml_cont(context, ggml_permute(context, x, 1, 2, 0, 3));
                    ++pointwise_pack_count;
                } else {
                    patches = ggml_im2col(context, w, x, stride[1], stride[0], pad[1], pad[0],
                                         dilation[1], dilation[0], true, GGML_TYPE_F32);
                }
                const int64_t k = w->ne[0] * w->ne[1] * w->ne[2];
                result = ggml_mul_mat(context, ggml_reshape_2d(context, w, k, w->ne[3]),
                    ggml_reshape_2d(context, patches, k, patches->ne[1] * patches->ne[2] * patches->ne[3]));
                require(ggml_prec_set_acc(result, GGML_PREC_F32), "FP32 accumulation unavailable");
                result = ggml_reshape_4d(context, result, w->ne[3], patches->ne[1], patches->ne[2], patches->ne[3]);
                result = ggml_cont(context, ggml_permute(context, result, 2, 0, 1, 3));
            } else {
                require(group == x->ne[2] && w->ne[2] == 1 && w->ne[3] == x->ne[2], "unsupported grouped Conv");
                result = ggml_conv_2d_dw_direct(context, w, x, stride[1], stride[0], pad[1], pad[0], dilation[1], dilation[0]);
            }
            if (ins.size() == 3) {
                auto * bias = get(ins[2]);
                result = ggml_add(context, result, ggml_reshape_4d(context, bias, 1, 1, w->ne[3], 1));
            }
        } else if (op == "Add" || op == "Mul") {
            auto * a = get(ins[0]);
            auto * b = get(ins[1]);
            if (!ggml_can_repeat(b, a) && ggml_can_repeat(a, b)) std::swap(a, b);
            require(ggml_can_repeat(b, a), "unsupported broadcast");
            result = op == "Add" ? ggml_add(context, a, b) : ggml_mul(context, a, b);
        } else if (op == "HardSwish") result = ggml_hardswish(context, get(ins[0]));
        else if (op == "Sigmoid") result = ggml_sigmoid(context, get(ins[0]));
        else if (op == "Relu") result = ggml_relu(context, get(ins[0]));
        else if (op == "HardSigmoid") {
            auto * t = ggml_scale_bias(context, get(ins[0]), attrs.value("alpha", 0.2f), attrs.value("beta", 0.5f));
            result = ggml_clamp(context, t, 0.0f, 1.0f);
        } else if (op == "GlobalAveragePool") {
            auto * t = get(ins[0]);
            result = ggml_mean(context, ggml_reshape_2d(context, t, t->ne[0] * t->ne[1], t->ne[2] * t->ne[3]));
            result = ggml_reshape_4d(context, result, 1, 1, t->ne[2], t->ne[3]);
        } else if (op == "AveragePool") {
            const auto kernel = attrs.at("kernel_shape").get<std::vector<int>>();
            const auto stride = attrs.at("strides").get<std::vector<int>>();
            const auto padding = attrs.at("pads").get<std::vector<int>>();
            require(std::all_of(padding.begin(), padding.end(), [](int p) { return p == 0; }) && !attr("ceil_mode", 0), "unsupported pool padding/ceil");
            result = ggml_pool_2d(context, get(ins[0]), GGML_OP_POOL_AVG, kernel[1], kernel[0], stride[1], stride[0], 0, 0);
        } else if (op == "LayerNormalization") {
            require(attr("axis", -1) == -1 && attr("stash_type", 1) == 1, "unsupported norm axis/precision");
            result = ggml_norm(context, get(ins[0]), attrs.value("epsilon", 1e-5f));
            result = ggml_add(context, ggml_mul(context, result, get(ins[1])), get(ins[2]));
        } else if (op == "MatMul") {
            auto * a = get(ins[0]);
            const bool quantized = node.contains("rhs_original_shape");
            ggml_tensor * b = nullptr;
            shape original_shape;
            if (quantized) {
                require(plan.at("version") == 3 && node.value("rhs_pretransposed", false),
                        "quantized CTC RHS requires pretransposed version 3 graph");
                original_shape = node.at("rhs_original_shape").get<shape>();
                require(original_shape.size() == 2, "invalid original CTC linear shape");
                b = quantized_linear_weights.at(ins[1]);
                require(original_shape[0] > 0 && original_shape[1] == a->ne[0]
                        && b->ne[0] >= original_shape[1] && b->ne[0] - original_shape[1] < 32
                        && b->ne[1] >= original_shape[0] && b->ne[1] - original_shape[0] < 32,
                        "invalid CTC linear padding");
                if (b->ne[0] != a->ne[0]) a = ggml_pad(context, a, int(b->ne[0] - a->ne[0]), 0, 0, 0);
                ++quantized_linear_count;
                padded_linear_elements += uint64_t(b->ne[0] * b->ne[1] - elements(original_shape));
            } else if (node.value("rhs_pretransposed", false)) {
                b = get(ins[1]);
                // Only the offline packer may supply this orientation. Dynamic
                // attention operands still need the original runtime transpose.
                require(plan.at("version") == 2 || plan.at("version") == 3,
                        "pretransposed RHS requires graph version 2/3");
                const auto & weight = values.at(ins[1]);
                require(weight.dims.size() == 2 && weight.offset == 0 && is_packed(weight)
                        && ggml_get_tensor(weights, ins[1].c_str()) == weight.storage,
                        "pretransposed RHS must be a constant two-dimensional weight");
                require(b->ne[0] == a->ne[0], "pretransposed RHS inner dimension mismatch");
                ++pretransposed_matmul_count;
                pretransposed_matmul_bytes += ggml_nbytes(b);
            } else {
                b = ggml_cont(context, ggml_transpose(context, get(ins[1])));
            }
            result = ggml_mul_mat(context, b, a);
            require(ggml_prec_set_acc(result, GGML_PREC_F32), "FP32 accumulation unavailable");
            if (quantized && b->ne[1] != original_shape[0]) {
                result = ggml_cont(context, ggml_view_4d(context, result, original_shape[0], result->ne[1],
                    result->ne[2], result->ne[3], result->nb[1], result->nb[2], result->nb[3], 0));
            }
        } else if (op == "Softmax") {
            const int axis = attr("axis", -1);
            require(axis == -1 || axis == int(dims.size()) - 1, "unsupported Softmax axis");
            result = ggml_soft_max(context, get(ins[0]));
        } else if (op == "Concat") {
            int axis = attr("axis", 0);
            if (axis < 0) axis += int(dims.size());
            require(dims.size() <= 4, "Concat rank > 4");
            result = get(ins[0]);
            for (size_t i = 1; i < ins.size(); ++i) result = ggml_concat(context, result, get(ins[i]), int(dims.size()) - 1 - axis);
        } else throw std::runtime_error("unsupported operation: " + op);
        require(result != nullptr && ggml_is_contiguous(result), "unexpected noncontiguous operation output");
        require(ggml_nelements(result) == elements(dims), "operation output size mismatch");
        for (size_t i = 0; i < dims.size(); ++i) require(result->ne[i] == dims[dims.size() - 1 - i], "operation output shape mismatch");
        ggml_set_name(result, name.c_str());
        values[name] = packed(result, dims);
    }
};

