#define OCR_LAYOUT_LIBRARY 1
#include "../../tools/ocr-ggml/doclayout-ggml.cpp"

static void test_case(shape dims, int axis, const std::vector<int64_t> & widths, int64_t index, bool strided) {
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
    require(ctx != nullptr, "context unavailable");
    std::vector<value> parts;
    for (size_t p = 0; p < widths.size(); ++p) {
        auto part_dims = dims; part_dims[axis] = widths[p];
        auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements(part_dims) * 2 + 32);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) static_cast<float *>(t->data)[i] = float(i + 1000 * p);
        value v{t, part_dims, std::vector<size_t>(dims.size()), 16 * sizeof(float)};
        size_t stride = strided ? 2 * sizeof(float) : sizeof(float);
        for (int d = int(dims.size()) - 1; d >= 0; --d) { v.strides[d] = stride; stride *= size_t(v.dims[d]); }
        parts.push_back(v);
    }
    auto result_dims = dims; result_dims.erase(result_dims.begin() + axis);
    value result;
    require(concat_gather_view(parts, dims, axis, axis, index, result_dims, result), "legal fold rejected");
    int64_t selected = index < 0 ? index + dims[axis] : index;
    shape c(result_dims.size()), full(dims.size());
    for (int64_t i = 0; i < elements(result_dims); ++i) {
        coordinates(i, result_dims, c);
        for (size_t d = 0; d < dims.size(); ++d) full[d] = int(d) < axis ? c[d] : int(d) == axis ? selected : c[d - 1];
        int64_t local = selected;
        float expected = 0;
        for (const auto & p : parts) {
            if (local >= p.dims[axis]) { local -= p.dims[axis]; continue; }
            size_t offset = p.offset;
            for (size_t d = 0; d < dims.size(); ++d) offset += size_t(int(d) == axis ? local : full[d]) * p.strides[d];
            expected = *reinterpret_cast<const float *>(static_cast<const char *>(p.storage->data) + offset);
            break;
        }
        require(float(read_flat(result, i)) == expected, "selected view differs from independent concat index");
    }
    require(result.storage != nullptr && std::any_of(parts.begin(), parts.end(), [&](const value & p) { return p.storage == result.storage; }), "fold did not alias an input");
    require(!concat_gather_view(parts, dims, axis, axis, dims[axis], result_dims, result), "out-of-range positive accepted");
    require(!concat_gather_view(parts, dims, axis, axis, -dims[axis] - 1, result_dims, result), "out-of-range negative accepted");
    require(!concat_gather_view(parts, dims, axis, (axis + 1) % int(dims.size()), index, result_dims, result), "different axis accepted");
    require(!concat_gather_view(parts, dims, axis, axis, index, dims, result), "incorrect output shape accepted");
    auto wrong_parts = parts; wrong_parts[0].dims[axis] += 1;
    require(!concat_gather_view(wrong_parts, dims, axis, axis, index, result_dims, result), "invalid concat size accepted");
    ggml_free(ctx);
}

int main() {
    try {
        test_case({2,6,3}, 1, {1,2,3}, 0, false);
        test_case({2,6,3}, 1, {1,2,3}, 2, false);
        test_case({2,6,3}, 1, {1,2,3}, -1, false);
        test_case({1,6,3,4,5}, 1, {2,1,3}, -3, false);
        test_case({2,3,5}, 2, {2,3}, 2, true);
        test_case({3,2,4}, 0, {1,2}, -2, true);
        std::puts("{\"passed\":true,\"cases\":6,\"strided\":true,\"negative_index\":true,\"offsets\":true}");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
