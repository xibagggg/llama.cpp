#include "ocr-float-graph.h"
#include "doclayout-ops.h"
#include "doclayout-ggml.h"
#include "doclayout-backend.h"

#include <limits>
#include <cstdlib>

enum class layout_code { copy, add, mul, sub, div, greater, where, cast, clip, log, gelu,
                         concat, gather, gather_elements, reduce_sum, reduce_min, reduce_max, topk, resize };

static enum ggml_type layout_type(const std::string & name) {
    if (name == "f32") return GGML_TYPE_F32;
    if (name == "q4_0") return GGML_TYPE_Q4_0;
    if (name == "q8_0") return GGML_TYPE_Q8_0;
    if (name == "i64") return GGML_TYPE_I64;
    if (name == "bool") return GGML_TYPE_I8;
    throw std::runtime_error("unsupported layout dtype: " + name);
}

static size_t address(const value & v, int64_t index) {
    size_t offset = v.offset;
    for (int d = int(v.dims.size()) - 1; d >= 0; --d) {
        offset += size_t(index % v.dims[d]) * v.strides[d];
        index /= v.dims[d];
    }
    return offset;
}

static double read_at(const value & v, size_t offset) {
    const auto * p = static_cast<const char *>(v.storage->data) + offset;
    if (v.storage->type == GGML_TYPE_F32) return *reinterpret_cast<const float *>(p);
    if (v.storage->type == GGML_TYPE_I64) return double(*reinterpret_cast<const int64_t *>(p));
    if (v.storage->type == GGML_TYPE_I8) return *reinterpret_cast<const int8_t *>(p);
    throw std::runtime_error("unsupported native value dtype");
}

static double read_flat(const value & v, int64_t index) { return read_at(v, address(v,index)); }

static void coordinates(int64_t index, const shape & dims, shape & coord) {
    for (int d = int(dims.size()) - 1; d >= 0; --d) {
        coord[d] = index % dims[d]; index /= dims[d];
    }
}

static double broadcast(const value & v, const shape & coord) {
    require(v.dims.size() <= coord.size(), "broadcast rank mismatch");
    size_t offset = v.offset;
    const size_t shift = coord.size() - v.dims.size();
    for (size_t d = 0; d < v.dims.size(); ++d)
        offset += size_t(v.dims[d] == 1 ? 0 : coord[d + shift]) * v.strides[d];
    return read_at(v, offset);
}

static void write_at(ggml_tensor * out, int64_t index, double number) {
    if (out->type == GGML_TYPE_F32) static_cast<float *>(out->data)[index] = float(number);
    else if (out->type == GGML_TYPE_I64) static_cast<int64_t *>(out->data)[index] = int64_t(number);
    else static_cast<int8_t *>(out->data)[index] = int8_t(number != 0);
}

struct layout_operation {
    layout_code code;
    std::vector<value> inputs;
    shape dims;
    int axis = 0;
    bool keepdims = false;
    std::vector<int> axes;
    float low = -std::numeric_limits<float>::infinity(), high = std::numeric_limits<float>::infinity();
    bool linear = false;
    bool packed_concat = false;
    bool packed_masks = false;
};

static bool can_linear_masks(const layout_operation & op, enum ggml_type type) {
    if (op.inputs.empty()) return false;
    for (auto d : op.dims) if (d <= 0) return false;
    for (const auto & input : op.inputs) {
        if (!input.storage || input.strides.size() != input.dims.size() || !is_packed(input)) return false;
        for (auto d : input.dims) if (d <= 0) return false;
        if (input.dims != op.dims && (input.dims.size() > op.dims.size() || elements(input.dims) != 1)) return false;
        const size_t unit = ggml_type_size(input.storage->type);
        if (input.offset % unit || input.offset > ggml_nbytes(input.storage) ||
            size_t(elements(input.dims)) * unit > ggml_nbytes(input.storage) - input.offset) return false;
    }
    if (op.code == layout_code::cast)
        return op.inputs.size() == 1 && op.inputs[0].dims == op.dims && op.inputs[0].storage->type == GGML_TYPE_I8
            && (type == GGML_TYPE_F32 || type == GGML_TYPE_I64);
    if (op.code == layout_code::greater)
        return op.inputs.size() == 2 && type == GGML_TYPE_I8 && op.inputs[0].storage->type == GGML_TYPE_F32
            && op.inputs[1].storage->type == GGML_TYPE_F32;
    if (op.code == layout_code::where)
        return op.inputs.size() == 3 && type == GGML_TYPE_F32 && op.inputs[0].storage->type == GGML_TYPE_I8
            && op.inputs[1].storage->type == GGML_TYPE_F32 && op.inputs[2].storage->type == GGML_TYPE_F32;
    return false;
}

static void linear_masks_compute(ggml_tensor * out, int ith, int nth, const layout_operation & op) {
    const int64_t count = elements(op.dims);
    const int64_t begin = count * ith / nth, end = count * (ith + 1) / nth;
    const auto data = [&](size_t index) {
        const auto & v = op.inputs[index];
        return static_cast<const char *>(v.storage->data) + v.offset;
    };
    if (op.code == layout_code::cast) {
        const auto * src = reinterpret_cast<const int8_t *>(data(0));
        if (out->type == GGML_TYPE_F32) {
            auto * dst = static_cast<float *>(out->data);
            for (int64_t i = begin; i < end; ++i) dst[i] = float(src[i]);
        } else {
            auto * dst = static_cast<int64_t *>(out->data);
            for (int64_t i = begin; i < end; ++i) dst[i] = int64_t(src[i]);
        }
    } else if (op.code == layout_code::greater) {
        const auto * a = reinterpret_cast<const float *>(data(0));
        const auto * b = reinterpret_cast<const float *>(data(1));
        const bool a_scalar = elements(op.inputs[0].dims) == 1, b_scalar = elements(op.inputs[1].dims) == 1;
        auto * dst = static_cast<int8_t *>(out->data);
        for (int64_t i = begin; i < end; ++i) dst[i] = a[a_scalar ? 0 : i] > b[b_scalar ? 0 : i];
    } else {
        const auto * cond = reinterpret_cast<const int8_t *>(data(0));
        const auto * a = reinterpret_cast<const float *>(data(1));
        const auto * b = reinterpret_cast<const float *>(data(2));
        const bool c_scalar = elements(op.inputs[0].dims) == 1;
        const bool a_scalar = elements(op.inputs[1].dims) == 1, b_scalar = elements(op.inputs[2].dims) == 1;
        auto * dst = static_cast<float *>(out->data);
        for (int64_t i = begin; i < end; ++i) dst[i] = cond[c_scalar ? 0 : i] != 0 ? a[a_scalar ? 0 : i] : b[b_scalar ? 0 : i];
    }
}

static bool can_copy_concat(const layout_operation & op, enum ggml_type type) {
    if (op.inputs.empty() || op.axis < 0 || size_t(op.axis) >= op.dims.size()) return false;
    int64_t total = 0;
    for (const auto & part : op.inputs) {
        if (part.storage->type != type || !is_packed(part) || part.dims.size() != op.dims.size()) return false;
        for (size_t d = 0; d < op.dims.size(); ++d)
            if (part.dims[d] <= 0 || (int(d) != op.axis && part.dims[d] != op.dims[d])) return false;
        if (part.offset > ggml_nbytes(part.storage) ||
            size_t(elements(part.dims)) * ggml_type_size(type) > ggml_nbytes(part.storage) - part.offset) return false;
        total += part.dims[op.axis];
    }
    return total == op.dims[op.axis];
}

static bool concat_gather_view(const std::vector<value> & parts, const shape & concat_dims,
                               int concat_axis, int gather_axis, int64_t index,
                               const shape & result_dims, value & result) {
    if (parts.empty() || concat_axis != gather_axis || concat_axis < 0 ||
        size_t(concat_axis) >= concat_dims.size()) return false;
    auto expected = concat_dims;
    expected.erase(expected.begin() + concat_axis);
    if (expected != result_dims) return false;
    int64_t total = 0;
    for (const auto & part : parts) {
        if (part.dims.size() != concat_dims.size() || part.strides.size() != concat_dims.size() ||
            part.storage->type != parts[0].storage->type) return false;
        for (size_t d = 0; d < concat_dims.size(); ++d)
            if (part.dims[d] <= 0 || (int(d) != concat_axis && part.dims[d] != concat_dims[d])) return false;
        total += part.dims[concat_axis];
    }
    if (total != concat_dims[concat_axis]) return false;
    if (index < 0) index += total;
    if (index < 0 || index >= total) return false;
    for (const auto & part : parts) {
        if (index >= part.dims[concat_axis]) { index -= part.dims[concat_axis]; continue; }
        result = part;
        result.offset += size_t(index) * part.strides[concat_axis];
        result.dims.erase(result.dims.begin() + concat_axis);
        result.strides.erase(result.strides.begin() + concat_axis);
        return true;
    }
    return false;
}

static void layout_compute_values(ggml_tensor * out, int ith, int nth, const layout_operation & op) {
    const auto & a = op.inputs[0];
    if (op.packed_masks) {
        linear_masks_compute(out, ith, nth, op);
        return;
    }
    if (op.code == layout_code::concat && op.packed_concat) {
        const size_t unit = ggml_type_size(out->type);
        size_t outer = 1, inner = unit;
        for (int d = 0; d < op.axis; ++d) outer *= size_t(op.dims[d]);
        for (size_t d = size_t(op.axis) + 1; d < op.dims.size(); ++d) inner *= size_t(op.dims[d]);
        const size_t row_bytes = size_t(op.dims[op.axis]) * inner;
        constexpr size_t block_bytes = 64 * 1024;
        size_t prefix = 0;
        for (const auto & part : op.inputs) {
            const size_t part_bytes = size_t(part.dims[op.axis]) * inner;
            const size_t blocks = (part_bytes + block_bytes - 1) / block_bytes;
            for (size_t task = size_t(ith); task < outer * blocks; task += size_t(nth)) {
                const size_t row = task / blocks, begin = (task % blocks) * block_bytes;
                std::memcpy(static_cast<char *>(out->data) + row * row_bytes + prefix + begin,
                            static_cast<const char *>(part.storage->data) + part.offset + row * part_bytes + begin,
                            std::min(block_bytes, part_bytes - begin));
            }
            prefix += part_bytes;
        }
        return;
    }
    if (op.code == layout_code::topk) {
        require(a.dims.size() == 2 && op.axis == 1, "unsupported TopK rank/axis");
        const int64_t width = a.dims[1], k = op.dims[1];
        for (int64_t row = ith; row < a.dims[0]; row += nth) {
            std::vector<int64_t> indices(size_t(width), 0);
            std::iota(indices.begin(), indices.end(), int64_t(0));
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](int64_t i, int64_t j) {
                const float x = float(read_flat(a,row * width + i)), y = float(read_flat(a,row * width + j));
                return x > y || (x == y && i < j);
            });
            for (int64_t j = 0; j < k; ++j)
                write_at(out,row * k + j,out->type == GGML_TYPE_I64 ? double(indices[j]) : read_flat(a,row * width + indices[j]));
        }
        return;
    }
    shape coord(op.dims.size()), input_coord(a.dims.size());
    for (int64_t i = ith; i < elements(op.dims); i += nth) {
        coordinates(i,op.dims,coord);
        double number = 0;
        switch (op.code) {
            case layout_code::copy: case layout_code::cast: number = read_flat(a,i); break;
            case layout_code::add: number = float(broadcast(a,coord)) + float(broadcast(op.inputs[1],coord)); break;
            case layout_code::mul: number = float(broadcast(a,coord)) * float(broadcast(op.inputs[1],coord)); break;
            case layout_code::sub: number = float(broadcast(a,coord)) - float(broadcast(op.inputs[1],coord)); break;
            case layout_code::div: number = float(broadcast(a,coord)) / float(broadcast(op.inputs[1],coord)); break;
            case layout_code::greater: number = broadcast(a,coord) > broadcast(op.inputs[1],coord); break;
            case layout_code::where: number = broadcast(a,coord) != 0 ? broadcast(op.inputs[1],coord) : broadcast(op.inputs[2],coord); break;
            case layout_code::clip: number = std::min(op.high,std::max(op.low,float(read_flat(a,i)))); break;
            case layout_code::log: number = std::log(float(read_flat(a,i))); break;
            case layout_code::gelu: {
                const float x = float(read_flat(a,i));
                number = 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475244f));
                break;
            }
            case layout_code::concat: {
                int64_t j = coord[op.axis];
                for (const auto & part : op.inputs) {
                    if (j >= part.dims[op.axis]) { j -= part.dims[op.axis]; continue; }
                    input_coord = coord; input_coord[op.axis] = j;
                    size_t offset = part.offset;
                    for (size_t d = 0; d < input_coord.size(); ++d) offset += size_t(input_coord[d]) * part.strides[d];
                    number = read_at(part,offset); break;
                }
                break;
            }
            case layout_code::gather: {
                const auto & index = op.inputs[1];
                int64_t index_flat = 0;
                for (size_t d = 0; d < index.dims.size(); ++d)
                    index_flat = index_flat * index.dims[d] + coord[size_t(op.axis) + d];
                int64_t selected = int64_t(read_flat(index,index_flat));
                if (selected < 0) selected += a.dims[op.axis];
                require(selected >= 0 && selected < a.dims[op.axis], "Gather index out of range");
                size_t offset = a.offset;
                for (size_t d = 0; d < a.dims.size(); ++d) {
                    const int64_t c = int(d) < op.axis ? coord[d] : int(d) == op.axis ? selected : coord[d - 1 + index.dims.size()];
                    offset += size_t(c) * a.strides[d];
                }
                number = read_at(a,offset); break;
            }
            case layout_code::gather_elements: {
                input_coord = coord;
                int64_t selected = int64_t(read_flat(op.inputs[1],i));
                if (selected < 0) selected += a.dims[op.axis];
                require(selected >= 0 && selected < a.dims[op.axis], "GatherElements index out of range");
                input_coord[op.axis] = selected;
                size_t offset = a.offset;
                for (size_t d = 0; d < input_coord.size(); ++d) offset += size_t(input_coord[d]) * a.strides[d];
                number = read_at(a,offset); break;
            }
            case layout_code::reduce_sum: case layout_code::reduce_min: case layout_code::reduce_max: {
                size_t q = 0;
                int64_t count = 1;
                for (size_t d = 0; d < a.dims.size(); ++d) {
                    const bool reduced = std::find(op.axes.begin(),op.axes.end(),int(d)) != op.axes.end();
                    input_coord[d] = reduced ? 0 : coord[q];
                    if (!reduced || op.keepdims) ++q;
                    if (reduced) count *= a.dims[d];
                }
                float sum = op.code == layout_code::reduce_sum ? 0.f : op.code == layout_code::reduce_min ?
                    std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
                for (int64_t r = 0; r < count; ++r) {
                    int64_t j = r;
                    for (int d = int(op.axes.size()) - 1; d >= 0; --d) {
                        input_coord[op.axes[d]] = j % a.dims[op.axes[d]]; j /= a.dims[op.axes[d]];
                    }
                    size_t offset = a.offset;
                    for (size_t d = 0; d < input_coord.size(); ++d) offset += size_t(input_coord[d]) * a.strides[d];
                    const float x = float(read_at(a,offset));
                    sum = op.code == layout_code::reduce_sum ? sum + x : op.code == layout_code::reduce_min ? std::min(sum,x) : std::max(sum,x);
                }
                number = sum; break;
            }
            case layout_code::resize: {
                require(op.dims.size() == 4 && a.dims.size() == 4, "Resize rank mismatch");
                const float sy = float(a.dims[2]) / float(op.dims[2]), sx = float(a.dims[3]) / float(op.dims[3]);
                const float y = op.linear ? std::max(0.f,(float(coord[2]) + .5f) * sy - .5f) : float(coord[2]) * sy;
                const float x = op.linear ? std::max(0.f,(float(coord[3]) + .5f) * sx - .5f) : float(coord[3]) * sx;
                const int64_t y0 = std::min(int64_t(y),a.dims[2]-1), x0 = std::min(int64_t(x),a.dims[3]-1);
                const int64_t y1 = std::min(y0 + 1,a.dims[2]-1), x1 = std::min(x0 + 1,a.dims[3]-1);
                const auto pixel = [&](int64_t yy,int64_t xx) { return float(read_at(a,a.offset + size_t(coord[0])*a.strides[0]
                    + size_t(coord[1])*a.strides[1] + size_t(yy)*a.strides[2] + size_t(xx)*a.strides[3])); };
                if (!op.linear) number = pixel(y0,x0);
                else {
                    const float dx=x-float(x0), dy=y-float(y0);
                    number=(1.f-dy)*((1.f-dx)*pixel(y0,x0)+dx*pixel(y0,x1)) + dy*((1.f-dx)*pixel(y1,x0)+dx*pixel(y1,x1));
                }
                break;
            }
            default: throw std::runtime_error("unsupported layout custom operation");
        }
        write_at(out,i,number);
    }
}

static void layout_compute(ggml_tensor * out, int ith, int nth, void * raw) {
    const auto & operation = *static_cast<const layout_operation *>(raw);
    require(!operation.inputs.empty() && operation.inputs.size()<=GGML_MAX_SRC,"invalid custom input count");
    bool replaced=false;
    for(size_t i=0;i<operation.inputs.size();++i) {
        require(out->src[i]!=nullptr,"missing runtime custom input");
        replaced|=out->src[i]!=operation.inputs[i].storage;
    }
    if(!replaced) {
        layout_compute_values(out,ith,nth,operation);
        return;
    }
    // A scheduler may replace a source with a backend-local copy. Keep the
    // shared plan immutable; bind only this worker's metadata to actual inputs.
    auto bound=operation;
    for(size_t i=0;i<bound.inputs.size();++i) {
        const auto * original=operation.inputs[i].storage;
        auto * actual=out->src[i];
        require(actual->type==original->type,"custom input copy changed dtype");
        for(int d=0;d<GGML_MAX_DIMS;++d)
            require(actual->ne[d]==original->ne[d] && actual->nb[d]==original->nb[d],
                    "custom input copy changed storage layout");
        require(actual->data!=nullptr,"custom input copy has no data");
        bound.inputs[i].storage=actual;
    }
    layout_compute_values(out,ith,nth,bound);
}

class doclayout_graph : public ctc_graph {
public:
    std::unique_ptr<layout_backend> execution;
    ~doclayout_graph(){execution.reset();}

    enum ggml_status compute() {
        return execution?execution->compute(graph):ggml_backend_graph_compute(backend,graph);
    }
    size_t compute_bytes() const {
        return execution?execution->compute_bytes():ggml_gallocr_get_buffer_size(allocator,0);
    }
    json backend_info() const {
        auto info=execution?execution->describe():json{{"mode","cpu"},{"scheduler",false},{"hardware_execution_validated",false}};
        json weight_types=json::object(),output_types=json::object();
        for(const auto & item:plan.at("weights")) {
            const std::string name=item.at("id");
            auto * tensor=execution?execution->imported.at(name):ggml_get_tensor(weights,name.c_str());
            require(tensor,"missing weight for precision metadata: "+name);
            const std::string type=ggml_type_name(tensor->type);
            weight_types[type]=weight_types.value(type,0)+1;
        }
        for(const auto & item:outputs)output_types[item.first]=ggml_type_name(item.second->type);
        const bool q4=weight_types.value("q4_0",0)>0,q8=weight_types.value("q8_0",0)>0;
        const std::string components=quantized_linear_count?"conv-linear-mixed-gguf":"conv-mixed-gguf";
        const std::string label=q4&&q8?"q4_0-q8_0-"+components:
            q4?"q4_0-"+components:q8?"q8_0-"+components:"f32-weights-gguf";
        info["model_precision"]={{"label",label},{"weight_tensor_types",weight_types},
            {"packed_conv_matrix_count",matrix_conv_count},{"q4_conv_matrix_count",q4_matrix_conv_count},
            {"q8_conv_matrix_count",q8_matrix_conv_count},{"input_type",ggml_type_name(input->type)},
            {"quantized_linear_count",quantized_linear_count},
            {"q4_linear_count",q4_linear_count},{"q8_linear_count",q8_linear_count},
            {"output_types",output_types},
            {"scope","Loaded weight storage types and constructed convolution/linear matrices; not all-INT4/INT8 execution"}};
        return info;
    }
    // Block-quantized matrices cannot be represented by scalar byte strides.
    std::map<std::string,ggml_tensor *> quantized_weights;
    std::vector<std::vector<float>> conv_geometry_data;
    int matrix_conv_count = 0;
    int q4_matrix_conv_count = 0;
    int q8_matrix_conv_count = 0;
    int quantized_linear_count = 0, q4_linear_count = 0, q8_linear_count = 0;
    std::vector<std::unique_ptr<layout_operation>> operations;
    std::map<std::string,ggml_tensor *> outputs;
    bool native_elementwise = std::getenv("OCR_LAYOUT_NATIVE_ELEMENTWISE") != nullptr
        && std::string(std::getenv("OCR_LAYOUT_NATIVE_ELEMENTWISE")) == "1";
    int native_elementwise_count = 0;
    bool native_compact = std::getenv("OCR_LAYOUT_NATIVE_COMPACT") != nullptr
        && std::string(std::getenv("OCR_LAYOUT_NATIVE_COMPACT")) == "1";
    int native_compact_count = 0;
    int gemm_runtime_transposes = 0;
    bool packed_concat = std::getenv("OCR_LAYOUT_PACKED_CONCAT") != nullptr
        && std::string(std::getenv("OCR_LAYOUT_PACKED_CONCAT")) == "1";
    int packed_concat_count = 0;
    bool fold_concat_gather = std::getenv("OCR_LAYOUT_FOLD_CONCAT_GATHER") != nullptr
        && std::string(std::getenv("OCR_LAYOUT_FOLD_CONCAT_GATHER")) == "1";
    int folded_concat_gathers = 0;
    bool packed_masks = std::getenv("OCR_LAYOUT_PACKED_MASKS") != nullptr
        && std::string(std::getenv("OCR_LAYOUT_PACKED_MASKS")) == "1";
    int packed_masks_count = 0;
    std::map<std::string,std::vector<std::string>> concat_inputs;
    std::map<std::string,int> concat_axes;

    static value reshaped(value source,const shape & dims) {
        require(elements(source.dims)==elements(dims), "reshape size mismatch");
        source.dims=dims; source.strides.resize(dims.size());
        size_t stride=ggml_type_size(source.storage->type);
        for(int d=int(dims.size())-1;d>=0;--d) { source.strides[d]=stride; stride*=size_t(dims[d]); }
        return source;
    }

    ggml_tensor * custom(layout_operation op,enum ggml_type type) {
        require(!op.inputs.empty() && op.inputs.size()<=GGML_MAX_SRC,"custom input count unsupported");
        std::vector<ggml_tensor *> args;
        for(const auto & v:op.inputs) args.push_back(v.storage);
        int64_t ne[4]={1,1,1,1};
        if(op.dims.size()>4) ne[0]=elements(op.dims);
        else for(size_t d=0;d<op.dims.size();++d) ne[d]=op.dims[op.dims.size()-1-d];
        operations.emplace_back(std::make_unique<layout_operation>(std::move(op)));
        return ggml_custom_4d(context,type,ne[0],ne[1],ne[2],ne[3],args.data(),int(args.size()),
                              layout_compute,GGML_N_TASKS_MAX,operations.back().get());
    }

    value compact(value source) {
        if(is_packed(source)) return source;
        const auto dims=source.dims;
        bool has_broadcast=false;
        for(size_t d=0;d<dims.size();++d)
            if(dims[d]>1 && source.strides[d]==0)has_broadcast=true;
        if(native_compact && dims.size()<=4 && !has_broadcast) {
            ++native_compact_count;
            return packed(contiguous(context,source),dims);
        }
        auto * t=custom({layout_code::copy,{source},dims},source.storage->type);
        return packed(t,dims);
    }

    std::vector<int64_t> integers(const std::string & name) {
        const auto & v=values.at(name);
        require(v.storage->type==GGML_TYPE_I64 && v.storage->data,"expected constant int64 tensor");
        std::vector<int64_t> result(size_t(elements(v.dims)),0);
        for(size_t i=0;i<result.size();++i)
            result[i]=*reinterpret_cast<const int64_t *>(static_cast<const char *>(v.storage->data)+address(v,int64_t(i)));
        return result;
    }

    void append_layout(const json & node) {
        const std::string op=node.at("op");
        const auto ins=node.at("inputs").get<std::vector<std::string>>();
        const auto & attrs=node.at("attrs");
        const auto & outs=node.at("outputs");
        const auto dims=outs[0].at("shape").get<shape>();
        const std::string name=outs[0].at("id");
        auto axis=[&](int rank) { int a=attrs.value("axis",0); if(a<0)a+=rank; require(a>=0&&a<rank,"invalid axis"); return a; };
        if(op=="Concat") {
            concat_inputs[name]=ins;
            concat_axes[name]=axis(int(dims.size()));
        }
        if(fold_concat_gather && op=="Gather" && concat_inputs.count(ins[0])) {
            const auto & index=values.at(ins[1]);
            if(index.dims.empty() && index.storage->type==GGML_TYPE_I64 && index.storage->data) {
                std::vector<value> parts;
                for(const auto & id:concat_inputs.at(ins[0]))parts.push_back(values.at(id));
                value selected;
                if(concat_gather_view(parts,values.at(ins[0]).dims,concat_axes.at(ins[0]),
                                      axis(int(values.at(ins[0]).dims.size())),integers(ins[1]).at(0),dims,selected)) {
                    values[name]=selected;
                    ++folded_concat_gathers;
                    return;
                }
            }
        }
        if(op=="Reshape" || op=="Squeeze" || op=="Unsqueeze" || op=="Flatten") {
            values[name]=reshaped(compact(values.at(ins[0])),dims); return;
        }
        if(op=="Transpose") {
            auto source=values.at(ins[0]),target=source;
            const auto perm=attrs.at("perm").get<std::vector<int>>();
            for(size_t d=0;d<perm.size();++d) { target.dims[d]=source.dims.at(perm[d]); target.strides[d]=source.strides.at(perm[d]); }
            require(target.dims==dims,"transpose shape mismatch"); values[name]=target; return;
        }
        if(op=="Split") {
            auto source=values.at(ins[0]); const int a=axis(int(source.dims.size()));
            int64_t offset=0;
            for(const auto & output:outs) {
                auto part=source; part.dims=output.at("shape").get<shape>();
                part.offset+=size_t(offset)*source.strides[a]; offset+=part.dims[a];
                values[output.at("id")]=part;
            }
            require(offset==source.dims[a],"Split size mismatch"); return;
        }
        if(op=="Slice") {
            auto source=values.at(ins[0]); auto starts=integers(ins[1]);
            auto axes=ins.size()>3 && !ins[3].empty() ? integers(ins[3]) : std::vector<int64_t>();
            if(axes.empty()) for(size_t d=0;d<starts.size();++d) axes.push_back(int64_t(d));
            auto steps=ins.size()>4 && !ins[4].empty() ? integers(ins[4]) : std::vector<int64_t>(starts.size(),1);
            for(size_t d=0;d<starts.size();++d) {
                int a=int(axes[d]); if(a<0)a+=int(source.dims.size());
                require(steps[d]>0,"negative Slice step unsupported");
                int64_t start=starts[d]; if(start<0)start+=source.dims.at(a);
                start=std::max(int64_t(0),std::min(start,source.dims.at(a)));
                source.offset+=size_t(start)*source.strides.at(a); source.strides[a]*=size_t(steps[d]);
            }
            source.dims=dims; values[name]=source; return;
        }
        if(op=="Tile") {
            auto source=values.at(ins[0]); const auto repeats=integers(ins[1]);
            require(repeats.size()==source.dims.size(),"Tile rank mismatch");
            for(size_t d=0;d<repeats.size();++d) if(repeats[d]!=1) {
                require(source.dims[d]==1,"only singleton Tile supported"); source.strides[d]=0;
            }
            source.dims=dims; values[name]=source; return;
        }
        const auto type=layout_type(outs[0].at("dtype"));
        ggml_tensor * result=nullptr;
        const bool binary = op=="Add" || op=="Mul" || op=="Sub" || op=="Div";
        bool use_native = native_elementwise && binary && dims.size()<=4;
        if(use_native) for(const auto & id:ins) {
            const auto & v=values.at(id);
            if(v.storage->type!=GGML_TYPE_F32 || v.dims.size()>4)use_native=false;
        }
        if(use_native) {
            auto * a=get(ins[0]);auto * b=get(ins[1]);
            if(op=="Add" || op=="Mul") {
                if(values.at(ins[0]).dims!=dims && values.at(ins[1]).dims==dims)std::swap(a,b);
            }
            int64_t expected[4]={1,1,1,1};
            for(size_t d=0;d<dims.size();++d)expected[d]=dims[dims.size()-1-d];
            for(int d=0;d<4;++d)if(a->ne[d]!=expected[d])use_native=false;
            use_native=use_native&&ggml_can_repeat(b,a);
            if(use_native) {
                result=op=="Add"?ggml_add(context,a,b):op=="Mul"?ggml_mul(context,a,b):
                       op=="Sub"?ggml_sub(context,a,b):ggml_div(context,a,b);
                ++native_elementwise_count;
            }
        }
        if(result) {
            // Standard GGML nodes can use native vector kernels and later backend fusion.
        } else if(op=="Gemm") {
            const int trans_b=attrs.value("transB",0);
            require(attrs.value("transA",0)==0 && (trans_b==0||trans_b==1) && attrs.value("alpha",1.f)==1.f && attrs.value("beta",1.f)==1.f,"unsupported Gemm attributes");
            const bool quantized=quantized_weights.count(ins[1])!=0;
            auto * b=quantized?quantized_weights.at(ins[1]):get(ins[1]);
            if(quantized) {
                require(trans_b==1 && attrs.contains("ggml_linear_weight_shape"),
                        "quantized Gemm requires explicitly packed transB=1 weights");
                const auto original=attrs.at("ggml_linear_weight_shape").get<shape>();
                require(original.size()==2 && original[0]==b->ne[1] && original[1]==b->ne[0],
                        "quantized Gemm weight geometry mismatch");
                ++quantized_linear_count;
                q4_linear_count+=b->type==GGML_TYPE_Q4_0;
                q8_linear_count+=b->type==GGML_TYPE_Q8_0;
            }else if(trans_b==0) {
                b=ggml_cont(context,ggml_transpose(context,b));
                ++gemm_runtime_transposes;
            }
            result=ggml_mul_mat(context,b,get(ins[0]));
            require(ggml_prec_set_acc(result,GGML_PREC_F32),"FP32 Gemm unavailable");
            if(ins.size()>2 && !ins[2].empty()) result=ggml_add(context,result,get(ins[2]));
        } else if(op=="GridSample") {
            require(attrs.at("mode")=="linear" && attrs.at("padding_mode")=="zeros" && attrs.value("align_corners",0)==0,"unsupported GridSample attrs");
            result=ocr_grid_sample_2d(context,get(ins[0]),get(ins[1])); require(result,"unsupported GridSample layout");
        } else if(op=="Pad") {
            const auto p=integers(ins[1]); require(p.size()==8 && attrs.at("mode")=="constant","unsupported Pad");
            require(ins.size()<3 || ins[2].empty() || read_flat(values.at(ins[2]),0)==0,"nonzero Pad unsupported");
            result=ggml_pad_ext(context,get(ins[0]),int(p[3]),int(p[7]),int(p[2]),int(p[6]),int(p[1]),int(p[5]),int(p[0]),int(p[4]));
        } else if(op=="MaxPool") {
            const auto k=attrs.at("kernel_shape").get<std::vector<int>>(),s=attrs.at("strides").get<std::vector<int>>(),p=attrs.at("pads").get<std::vector<int>>();
            require(std::all_of(p.begin(),p.end(),[](int x){return x==0;}),"MaxPool padding unsupported");
            result=ggml_pool_2d(context,get(ins[0]),GGML_OP_POOL_MAX,k[1],k[0],s[1],s[0],0,0);
        } else if(op=="Conv" && attrs.contains("ggml_conv_weight_shape")) {
            const auto original=attrs.at("ggml_conv_weight_shape").get<shape>();
            require(original.size()==4 && std::all_of(original.begin(),original.end(),[](int64_t x){return x>0;}),"invalid matrix Conv geometry");
            const int64_t m=original[0],c=original[1],kh=original[2],kw=original[3],k=c*kh*kw;
            const auto stride=attrs.at("strides").get<std::vector<int>>();
            const auto pad=attrs.at("pads").get<std::vector<int>>();
            const auto dilation=attrs.at("dilations").get<std::vector<int>>();
            require(attrs.value("group",1)==1 && stride.size()==2 && pad.size()==4 && dilation.size()==2
                && pad[0]==pad[2] && pad[1]==pad[3],"unsupported matrix Conv attributes");
            auto * x=get(ins[0]);
            auto * w=quantized_weights.count(ins[1])?quantized_weights.at(ins[1]):get(ins[1]);
            require(x->type==GGML_TYPE_F32 && x->ne[2]==c && w->ne[0]==k && w->ne[1]==m && w->ne[2]==1 && w->ne[3]==1
                && (w->type==GGML_TYPE_F32 || w->type==GGML_TYPE_Q4_0 || w->type==GGML_TYPE_Q8_0),"matrix Conv signature mismatch");
            // IM2COL reads kernel geometry, not kernel values. Keep a small,
            // initialized [kw,kh,C,1] descriptor independent of [K,M] weights.
            auto * geometry=ggml_new_tensor_4d(context,GGML_TYPE_F32,kw,kh,c,1);
            conv_geometry_data.emplace_back(size_t(k),0.f);
            geometry->data=conv_geometry_data.back().data();
            auto * patches=ggml_im2col(context,geometry,x,stride[1],stride[0],pad[1],pad[0],
                                      dilation[1],dilation[0],true,GGML_TYPE_F32);
            result=ggml_mul_mat(context,w,ggml_reshape_2d(context,patches,k,
                patches->ne[1]*patches->ne[2]*patches->ne[3]));
            require(ggml_prec_set_acc(result,GGML_PREC_F32),"matrix Conv FP32 accumulation unavailable");
            result=ggml_reshape_4d(context,result,m,patches->ne[1],patches->ne[2],patches->ne[3]);
            result=ggml_cont(context,ggml_permute(context,result,2,0,1,3));
            if(ins.size()>2 && !ins[2].empty())result=ggml_add(context,result,ggml_reshape_4d(context,get(ins[2]),1,1,m,1));
            ++matrix_conv_count;
            q4_matrix_conv_count+=w->type==GGML_TYPE_Q4_0;
            q8_matrix_conv_count+=w->type==GGML_TYPE_Q8_0;
        } else if(op=="Conv" || op=="Relu" || op=="Sigmoid" || op=="Softmax" || op=="MatMul" || op=="LayerNormalization") {
            json shared={{"op",op},{"name",node.at("name")},{"inputs",ins},{"output",name},{"shape",dims},{"attrs",attrs}};
            ctc_graph::append(shared); return;
        } else {
            layout_operation spec{}; spec.dims=dims;
            for(const auto & input_name:ins) if(!input_name.empty()) spec.inputs.push_back(values.at(input_name));
            if(op=="Add")spec.code=layout_code::add;
            else if(op=="Mul")spec.code=layout_code::mul;
            else if(op=="Sub")spec.code=layout_code::sub;
            else if(op=="Div")spec.code=layout_code::div;
            else if(op=="Greater")spec.code=layout_code::greater;
            else if(op=="Where")spec.code=layout_code::where;
            else if(op=="Cast")spec.code=layout_code::cast;
            else if(op=="Log")spec.code=layout_code::log;
            else if(op=="Gelu") { require(attrs.value("approximate",std::string("none"))=="none","approximate Gelu unsupported"); spec.code=layout_code::gelu; }
            else if(op=="Clip") {
                spec.code=layout_code::clip;
                if(ins.size()>1&&!ins[1].empty())spec.low=float(read_flat(values.at(ins[1]),0));
                if(ins.size()>2&&!ins[2].empty())spec.high=float(read_flat(values.at(ins[2]),0));
            } else if(op=="Concat") {
                spec.code=layout_code::concat;spec.axis=axis(int(dims.size()));
                spec.packed_concat=packed_concat && can_copy_concat(spec,type);
                if(spec.packed_concat)++packed_concat_count;
            }
            else if(op=="Gather" || op=="GatherElements") {spec.code=op=="Gather"?layout_code::gather:layout_code::gather_elements;spec.axis=axis(int(spec.inputs[0].dims.size()));}
            else if(op=="ReduceSum" || op=="ReduceMin" || op=="ReduceMax") {
                spec.code=op=="ReduceSum"?layout_code::reduce_sum:op=="ReduceMin"?layout_code::reduce_min:layout_code::reduce_max;
                spec.keepdims=attrs.value("keepdims",1)!=0;
                for(auto a:integers(ins[1])) {if(a<0)a+=int64_t(spec.inputs[0].dims.size());spec.axes.push_back(int(a));}
            } else if(op=="TopK") {
                require(attrs.value("largest",1)==1&&attrs.value("sorted",1)==1,"unsupported TopK attributes");
                spec.code=layout_code::topk;spec.axis=axis(int(spec.inputs[0].dims.size()));
                for(const auto & output:outs) {
                    spec.dims=output.at("shape").get<shape>();
                    auto * tensor=custom(spec,layout_type(output.at("dtype")));
                    values[output.at("id")]=packed(tensor,spec.dims);
                }
                return;
            } else if(op=="Resize") {
                spec.code=layout_code::resize;spec.inputs.resize(1);
                spec.linear=attrs.at("mode")=="linear";
                require((spec.linear&&attrs.at("coordinate_transformation_mode")=="half_pixel") ||
                    (!spec.linear&&attrs.at("mode")=="nearest"&&attrs.at("coordinate_transformation_mode")=="asymmetric"&&attrs.at("nearest_mode")=="floor"),"unsupported Resize attrs");
            } else throw std::runtime_error("unimplemented layout op: "+op);
            spec.packed_masks=packed_masks && can_linear_masks(spec,type);
            if(spec.packed_masks)++packed_masks_count;
            result=custom(std::move(spec),type);
        }
        require(result&&ggml_nelements(result)==elements(dims),"layout output size mismatch");
        ggml_set_name(result,name.c_str()); values[name]=packed(result,dims);
    }

    void load_layout(const char * path,int threads,const std::vector<std::string> & snapshots) {
        file=gguf_init_from_file(path,{false,&weights}); require(file,"layout GGUF load failed");
        const auto key=gguf_find_key(file,"ocr.layout.graph");require(key>=0,"layout metadata absent");
        plan=json::parse(gguf_get_val_str(file,key));require(plan.at("version")==1,"unsupported layout plan");
        backend=ggml_backend_cpu_init();require(backend,"CPU unavailable");ggml_backend_cpu_set_n_threads(backend,threads);
        const char * backend_option=std::getenv("OCR_LAYOUT_BACKEND");
        const std::string backend_mode=backend_option?backend_option:"cpu";
        require(backend_mode=="cpu"||backend_mode=="cpu-sched"||backend_mode=="spacemit","invalid OCR_LAYOUT_BACKEND");
        if(backend_mode!="cpu") {
            execution=std::make_unique<layout_backend>(backend_mode,backend);
            execution->import_weights(weights,plan.at("weights"));
        }
        context=ggml_init({ggml_tensor_overhead()*40000+ggml_graph_overhead_custom(20000,false),nullptr,true});require(context,"metadata allocation failed");
        for(const auto & item:plan.at("weights")) {
            const std::string name=item.at("id");
            auto * t=execution?execution->imported.at(name):ggml_get_tensor(weights,name.c_str());
            require(t&&t->type==layout_type(item.at("dtype")),"layout weight dtype mismatch");
            const auto dims=item.at("shape").get<shape>();
            if(t->type==GGML_TYPE_Q4_0 || t->type==GGML_TYPE_Q8_0) {
                require(dims.size()==2 && dims[0]>0 && dims[1]>0 && dims[1]%32==0
                    && t->ne[0]==dims[1] && t->ne[1]==dims[0] && t->ne[2]==1 && t->ne[3]==1,"invalid quantized matrix weight");
                quantized_weights.emplace(name,t);
            } else values.emplace(name,packed(t,dims));
        }
        const auto dims=plan.at("input_shape").get<shape>();require(dims==shape({1,3,800,800}),"layout input signature mismatch");
        input=ggml_new_tensor_4d(context,GGML_TYPE_F32,800,800,3,1);ggml_set_input(input);ggml_set_output(input);
        values[plan.at("input")]=packed(input,dims);
        for(const auto & node:plan.at("nodes")) {
            try { append_layout(node); }
            catch(const std::exception & error) {throw std::runtime_error(node.at("name").get<std::string>()+": "+error.what());}
        }
        graph=ggml_new_graph_custom(context,20000,false);
        for(const auto & id:snapshots) {
            const auto source=compact(values.at(id));
            auto * t=source.dims.size()<=4 ? contiguous(context,source) : ggml_view_1d(context,source.storage,elements(source.dims),source.offset);
            ggml_set_output(source.storage);ggml_set_output(t);checkpoints[id]=t;ggml_build_forward_expand(graph,t);
        }
        for(const auto & item:plan.at("outputs")) {
            auto * t=get(item.at("id"));ggml_set_output(t);outputs[item.at("name")]=t;ggml_build_forward_expand(graph,t);
        }
        if(execution)execution->allocate(graph,input);
        else {
            allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            require(allocator&&ggml_gallocr_alloc_graph(allocator,graph),"layout allocation failed");
        }
    }
};

#ifdef OCR_LAYOUT_LIBRARY
namespace {
const char * layout_outputs[]={"logits","pred_boxes","order_logits","out_masks"};
struct layout_handle { doclayout_graph model; std::mutex mutex; };
void layout_error(char * error,size_t size,const char * text) {
    if(error&&size)std::snprintf(error,size,"%s",text);
}
}
extern "C" void * ocr_layout_create(const char * path,int threads,int native,char * error,size_t size) {
    try {
        require(path&&threads>0&&threads<=16&&native>=0&&native<=3,"invalid layout create arguments");
        auto handle=std::make_unique<layout_handle>();handle->model.native_elementwise=(native&1)!=0;
        handle->model.native_compact=(native&2)!=0;
        handle->model.load_layout(path,threads,{});
        require(handle->model.outputs.size()==4,"expected all four layout outputs");
        for(const auto * name:layout_outputs)require(handle->model.outputs.at(name)->type==GGML_TYPE_F32,"non-FP32 layout output");
        layout_error(error,size,"");return handle.release();
    } catch(const std::exception & failure) {layout_error(error,size,failure.what());return nullptr;}
}
extern "C" void ocr_layout_destroy(void * handle) {delete static_cast<layout_handle *>(handle);}
extern "C" int ocr_layout_info(void * raw,char * json_output,size_t capacity,char * error,size_t size) {
    try {
        require(raw&&json_output&&capacity,"null layout info argument");
        auto & handle=*static_cast<layout_handle *>(raw);std::lock_guard<std::mutex> lock(handle.mutex);
        const auto info=handle.model.backend_info().dump();require(info.size()<capacity,"layout info buffer too small");
        std::memcpy(json_output,info.c_str(),info.size()+1);layout_error(error,size,"");return 0;
    }catch(const std::exception & failure){layout_error(error,size,failure.what());return -1;}
}
extern "C" int ocr_layout_counts(void * raw,size_t * counts,char * error,size_t size) {
    try {
        require(raw&&counts,"null layout counts argument");auto & handle=*static_cast<layout_handle *>(raw);
        std::lock_guard<std::mutex> lock(handle.mutex);
        counts[0]=size_t(ggml_nelements(handle.model.input));
        for(size_t i=0;i<4;++i)counts[i+1]=size_t(ggml_nelements(handle.model.outputs.at(layout_outputs[i])));
        layout_error(error,size,"");return 0;
    } catch(const std::exception & failure) {layout_error(error,size,failure.what());return -1;}
}
extern "C" int ocr_layout_run(void * raw,const float * input,size_t input_count,float * output,
                              size_t output_count,double * seconds,char * error,size_t size) {
    try {
        require(raw&&input&&output&&seconds,"null layout execution argument");
        auto & handle=*static_cast<layout_handle *>(raw);std::lock_guard<std::mutex> lock(handle.mutex);
        auto & model=handle.model;
        require(input_count==size_t(ggml_nelements(model.input)),"layout input count mismatch");
        size_t expected=0;for(const auto * name:layout_outputs)expected+=size_t(ggml_nelements(model.outputs.at(name)));
        require(output_count==expected,"layout output count mismatch");
        for(size_t i=0;i<input_count;++i)require(std::isfinite(input[i]),"nonfinite layout input");
        ggml_backend_tensor_set(model.input,input,0,input_count*sizeof(float));const auto begin=std::chrono::steady_clock::now();
        require(model.compute()==GGML_STATUS_SUCCESS,"layout compute failed");
        *seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        size_t offset=0;
        for(const auto * name:layout_outputs) {
            auto * t=model.outputs.at(name);const size_t count=size_t(ggml_nelements(t));
            ggml_backend_tensor_get(t,output+offset,0,count*sizeof(float));offset+=count;
        }
        for(size_t i=0;i<output_count;++i)require(std::isfinite(output[i]),"nonfinite layout output");
        layout_error(error,size,"");return 0;
    } catch(const std::exception & failure) {layout_error(error,size,failure.what());return -1;}
}
#else
int main(int argc,char ** argv) {
    try {
        require(argc>=4&&argc<=7,"usage: doclayout-ggml MODEL INPUT_F32 OUTPUT_DIR [THREADS] [REPEATS] [SNAPSHOT_IDS_JSON]");
        const int threads=argc>4?std::stoi(argv[4]):4,repeats=argc>5?std::stoi(argv[5]):2;
        require(threads>0&&threads<=16&&repeats>0&&repeats<=5,"invalid execution limits");
        std::vector<std::string> snapshots;
        if(argc>6) {std::ifstream stream(argv[6]);json list;stream>>list;snapshots=list.get<std::vector<std::string>>();}
        doclayout_graph model;const auto begin=std::chrono::steady_clock::now();model.load_layout(argv[1],threads,snapshots);
        std::vector<float> data(size_t(ggml_nelements(model.input)));std::ifstream input(argv[2],std::ios::binary);
        input.read(reinterpret_cast<char *>(data.data()),data.size()*sizeof(float));require(bool(input)&&input.peek()==EOF,"layout input length mismatch");
        for(float x:data)require(std::isfinite(x),"nonfinite layout input");
        const double setup_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        const std::filesystem::path dest(argv[3]);std::filesystem::create_directories(dest);
        json times=json::array();std::map<std::string,std::vector<float>> previous;
        auto read=[](ggml_tensor * t){std::vector<float> result(size_t(ggml_nelements(t)));ggml_backend_tensor_get(t,result.data(),0,result.size()*sizeof(float));return result;};
        for(int r=0;r<repeats;++r) {
            ggml_backend_tensor_set(model.input,data.data(),0,data.size()*sizeof(float));const auto start=std::chrono::steady_clock::now();
            require(model.compute()==GGML_STATUS_SUCCESS,"layout compute failed");
            times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
            require(read(model.input)==data,"layout input modified");
            for(const auto & item:model.outputs) {
                auto out=read(item.second);for(float x:out)require(std::isfinite(x),"nonfinite layout output");
                if(r==0)previous[item.first]=out;
                require(!std::memcmp(out.data(),previous[item.first].data(),out.size()*sizeof(float)),"layout repeat changed");
                std::ofstream stream(dest/(item.first+".f32"),std::ios::binary);stream.write(reinterpret_cast<const char *>(out.data()),out.size()*sizeof(float));require(bool(stream),"output write failed");
            }
        }
        for(const auto & item:model.checkpoints) {
            auto * t=item.second;std::vector<char> bytes(ggml_nbytes(t));ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
            std::ofstream stream(dest/(item.first+".bin"),std::ios::binary);stream.write(bytes.data(),bytes.size());require(bool(stream),"checkpoint write failed");
        }
        std::printf("%s\n",json{{"engine",model.execution?"ggml-scheduler":"ggml-cpu"},{"backend_info",model.backend_info()},
                    {"setup_s",setup_s},{"run_s",times},{"compute_bytes",model.compute_bytes()},
                    {"native_elementwise",model.native_elementwise},{"native_elementwise_count",model.native_elementwise_count},
                    {"native_compact",model.native_compact},{"native_compact_count",model.native_compact_count},
                    {"gemm_runtime_transposes",model.gemm_runtime_transposes},{"graph_nodes",ggml_graph_n_nodes(model.graph)},
                    {"packed_concat",model.packed_concat},{"packed_concat_count",model.packed_concat_count},
                    {"fold_concat_gather",model.fold_concat_gather},{"folded_concat_gathers",model.folded_concat_gathers},
                    {"pointwise_pack",model.pointwise_pack},{"pointwise_pack_count",model.pointwise_pack_count},
                    {"packed_masks",model.packed_masks},{"packed_masks_count",model.packed_masks_count},
                    {"matrix_conv_count",model.matrix_conv_count},{"q4_matrix_conv_count",model.q4_matrix_conv_count},
                    {"q8_matrix_conv_count",model.q8_matrix_conv_count},
                    {"input_preserved",true},{"repeats_exact",true},{"output_count",model.outputs.size()}}.dump().c_str());return 0;
    } catch(const std::exception & error) {std::fprintf(stderr,"doclayout-ggml: %s\n",error.what());return 1;}
}
#endif
