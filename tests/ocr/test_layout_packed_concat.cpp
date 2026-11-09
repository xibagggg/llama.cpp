#define OCR_LAYOUT_LIBRARY 1
#include "../../tools/ocr-ggml/doclayout-ggml.cpp"
#include <thread>

static void test_concat(const shape & dims, int axis, const std::vector<int64_t> & widths, enum ggml_type type) {
    auto * ctx = ggml_init({32 * 1024 * 1024, nullptr, false});
    require(ctx != nullptr, "test context allocation failed");
    layout_operation op{};
    op.code = layout_code::concat; op.dims = dims; op.axis = axis;
    const size_t unit = ggml_type_size(type), guard = 16 * unit;
    std::vector<std::vector<char>> before;
    for (size_t p = 0; p < widths.size(); ++p) {
        auto part_dims = dims; part_dims[axis] = widths[p];
        const auto count = size_t(elements(part_dims));
        auto * t = ggml_new_tensor_1d(ctx, type, int64_t(count + 32));
        std::memset(t->data, 0x5a, ggml_nbytes(t));
        for (size_t i = 0; i < count; ++i) {
            auto * ptr = static_cast<char *>(t->data) + guard + i * unit;
            if (type == GGML_TYPE_F32) {
                const float x = float(int(i % 257) - 128) * .125f + float(p);
                std::memcpy(ptr, &x, sizeof(x));
            } else if (type == GGML_TYPE_I64) {
                const int64_t x = (int64_t(1) << 54) + int64_t(i + p * 13);
                std::memcpy(ptr, &x, sizeof(x));
            } else *ptr = char((i + p) % 2);
        }
        value part{t, part_dims, std::vector<size_t>(dims.size()), guard};
        size_t stride = unit;
        for (int d = int(dims.size()) - 1; d >= 0; --d) {
            part.strides[d] = stride; stride *= size_t(part_dims[d]);
        }
        op.inputs.push_back(part);
        before.emplace_back(static_cast<const char *>(t->data), static_cast<const char *>(t->data) + ggml_nbytes(t));
    }
    require(can_copy_concat(op, type), "valid packed concat rejected");
    auto * out = ggml_new_tensor_1d(ctx, type, elements(dims) + 32);
    auto * allocation = static_cast<char *>(out->data);
    std::memset(allocation, 0x6b, ggml_nbytes(out));
    out->data = allocation + guard;
    std::vector<char> expected(size_t(elements(dims)) * unit);
    shape coord(dims.size());
    for (int64_t i = 0; i < elements(dims); ++i) {
        coordinates(i, dims, coord);
        int64_t selected = coord[axis];
        for (const auto & part : op.inputs) {
            if (selected >= part.dims[axis]) { selected -= part.dims[axis]; continue; }
            auto c = coord; c[axis] = selected;
            size_t offset = part.offset;
            for (size_t d = 0; d < c.size(); ++d) offset += size_t(c[d]) * part.strides[d];
            std::memcpy(expected.data() + size_t(i) * unit, static_cast<const char *>(part.storage->data) + offset, unit);
            break;
        }
    }
    op.packed_concat = true;
    for(size_t i=0;i<op.inputs.size();++i)out->src[i]=op.inputs[i].storage;
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) workers.emplace_back([&, t] { layout_compute(out, t, 4, &op); });
    for (auto & worker : workers) worker.join();
    require(std::memcmp(out->data, expected.data(), expected.size()) == 0, "packed concat differs from byte reference");
    if (type != GGML_TYPE_I64) {
        std::memset(out->data, 0x2c, expected.size());
        op.packed_concat = false;
        for (int t = 0; t < 4; ++t) layout_compute(out, t, 4, &op);
        require(std::memcmp(out->data, expected.data(), expected.size()) == 0, "fallback changed packed result");
    }
    for (size_t i = 0; i < guard; ++i) {
        require(allocation[i] == char(0x6b), "output prefix overwritten");
        require(allocation[guard + expected.size() + i] == char(0x6b), "output suffix overwritten");
    }
    for (size_t p = 0; p < op.inputs.size(); ++p)
        require(std::memcmp(op.inputs[p].storage->data, before[p].data(), before[p].size()) == 0, "input modified");
    auto strided = op;
    strided.inputs[0].strides.back() *= 2;
    if (strided.inputs[0].dims.back() > 1) require(!can_copy_concat(strided, type), "strided input not rejected");
    auto bad = op;
    bad.dims[axis] += 1;
    require(!can_copy_concat(bad, type), "wrong axis sum not rejected");
    auto range = op;
    range.inputs[0].offset = ggml_nbytes(range.inputs[0].storage);
    require(!can_copy_concat(range, type), "out-of-range input not rejected");
    out->data = allocation;
    ggml_free(ctx);
}

int main() {
    try {
        test_concat({2,5,7}, 1, {3,2}, GGML_TYPE_F32);
        test_concat({3,2,3,4,5}, 0, {1,2}, GGML_TYPE_F32);
        test_concat({2,3,5}, 2, {2,1,2}, GGML_TYPE_I64);
        test_concat({2,5,7}, 1, {3,2}, GGML_TYPE_I8);
        test_concat({1,7,16385}, 1, {3,4}, GGML_TYPE_F32);
        test_concat({2,3,16385}, 2, {8193,8192}, GGML_TYPE_F32);
        std::puts("{\"passed\":true,\"cases\":6,\"threads\":4,\"guards\":true,\"offsets\":true}");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
