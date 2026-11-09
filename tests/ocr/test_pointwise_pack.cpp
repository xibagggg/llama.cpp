#include "../../tools/ocr-ggml/ocr-float-graph.h"

static std::vector<float> run(bool enabled, int n, int c, int h, int w, int m, int kernel,
                              int stride, int pad, int dilation, bool bias, bool eligible) {
    ctc_graph model;
    model.context = ggml_init({64 * 1024 * 1024, nullptr, false});
    require(model.context != nullptr, "test context allocation failed");
    model.pointwise_pack = enabled;
    auto * x = ggml_new_tensor_4d(model.context, GGML_TYPE_F32, w, h, c, n);
    auto * weights = ggml_new_tensor_4d(model.context, GGML_TYPE_F32, kernel, kernel, c, m);
    auto * b = ggml_new_tensor_1d(model.context, GGML_TYPE_F32, m);
    for (int64_t i = 0; i < ggml_nelements(x); ++i) static_cast<float *>(x->data)[i] = float((i * 17) % 131 - 65) * .03125f;
    for (int64_t i = 0; i < ggml_nelements(weights); ++i) static_cast<float *>(weights->data)[i] = float((i * 13) % 103 - 51) * .0078125f;
    for (int i = 0; i < m; ++i) static_cast<float *>(b->data)[i] = float(i - m / 2) * .0625f;
    const auto * begin = static_cast<const float *>(x->data);
    const std::vector<float> input(begin, begin + ggml_nelements(x));
    model.values["x"] = packed(x, {n,c,h,w});
    model.values["w"] = packed(weights, {m,c,kernel,kernel});
    model.values["b"] = packed(b, {m});
    const int oh = (h + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    const int ow = (w + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
    std::vector<std::string> ins = {"x", "w"};
    if (bias) ins.push_back("b");
    model.append({{"op","Conv"}, {"name","test"}, {"inputs",ins}, {"output","y"}, {"shape",{n,m,oh,ow}},
        {"attrs",{{"group",1},{"strides",{stride,stride}},{"pads",{pad,pad,pad,pad}},{"dilations",{dilation,dilation}}}}});
    require(model.pointwise_pack_count == int(enabled && eligible), "wrong eligibility");
    auto * y = model.get("y");
    auto * graph = ggml_new_graph(model.context);
    ggml_build_forward_expand(graph, y);
    require(ggml_graph_compute_with_ctx(model.context, graph, 4) == GGML_STATUS_SUCCESS, "test compute failed");
    require(std::memcmp(x->data, input.data(), input.size() * sizeof(float)) == 0, "input changed");
    std::vector<float> result(size_t(ggml_nelements(y)));
    std::memcpy(result.data(), y->data, result.size() * sizeof(float));
    for (float f : result) require(std::isfinite(f), "nonfinite output");
    return result;
}

static void check(int n, int c, int h, int w, int m, int kernel, int stride, int pad, int dilation, bool bias) {
    const bool eligible = kernel == 1 && stride == 1 && pad == 0 && dilation == 1;
    auto a = run(false,n,c,h,w,m,kernel,stride,pad,dilation,bias,eligible);
    auto b = run(true,n,c,h,w,m,kernel,stride,pad,dilation,bias,eligible);
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, "pointwise output differs");
}

int main() {
    try {
        check(1,3,5,7,4,1,1,0,1,true);
        check(2,7,3,9,5,1,1,0,1,false);
        check(1,1,1,1,3,1,1,0,1,true);
        check(1,32,25,25,48,1,1,0,1,true);
        check(1,3,9,11,5,3,1,1,1,true);
        check(1,3,9,11,5,1,2,0,1,false);
        check(1,3,9,11,5,1,1,1,1,true);
        check(1,3,9,11,5,1,1,0,2,true);
        std::puts("{\"passed\":true,\"cases\":8,\"batch2\":true,\"fallbacks\":true,\"byte_exact\":true}");
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
