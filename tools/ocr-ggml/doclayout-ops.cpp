#include "doclayout-ops.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace {
void grid_sample(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * image = dst->src[0];
    const auto * grid = dst->src[1];
    const auto * pixels = static_cast<const float *>(image->data);
    const auto * locations = static_cast<const float *>(grid->data);
    auto * output = static_cast<float *>(dst->data);
    const int64_t width = image->ne[0], height = image->ne[1], channels = image->ne[2];
    const int64_t spatial = dst->ne[0] * dst->ne[1];
    const int64_t plane = width * height;
    // Sample coordinates and interpolation weights are shared across channels.
    // Partition independent sample points, so workers never share output bytes.
    for (int64_t i = ith; i < spatial * image->ne[3]; i += nth) {
        const int64_t batch = i / spatial, point = i % spatial;
        const float gx = locations[i * 2], gy = locations[i * 2 + 1];
        const float x = ((gx + 1.0f) * float(width) - 1.0f) * 0.5f;
        const float y = ((gy + 1.0f) * float(height) - 1.0f) * 0.5f;
        if (!std::isfinite(gx) || !std::isfinite(gy)) {
            for (int64_t c = 0; c < channels; ++c)
                output[(batch * channels + c) * spatial + point] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        // Reject before float-to-integer conversion, including huge coordinates.
        if (x <= -1.0f || y <= -1.0f || x >= float(width) || y >= float(height)) {
            for (int64_t c = 0; c < channels; ++c)
                output[(batch * channels + c) * spatial + point] = 0.0f;
            continue;
        }
        const int64_t x0 = int64_t(std::floor(x)), y0 = int64_t(std::floor(y));
        const float dx = x - float(x0), dy = y - float(y0);
        const float weights[4] = {(1.0f - dy) * (1.0f - dx), (1.0f - dy) * dx,
                                   dy * (1.0f - dx), dy * dx};
        const int64_t xs[4] = {x0, x0 + 1, x0, x0 + 1};
        const int64_t ys[4] = {y0, y0, y0 + 1, y0 + 1};
        for (int64_t c = 0; c < channels; ++c) {
            const float * source = pixels + (batch * channels + c) * plane;
            float sum = 0.0f;
            for (int corner = 0; corner < 4; ++corner) {
                if (xs[corner] >= 0 && xs[corner] < width && ys[corner] >= 0 && ys[corner] < height)
                    sum += weights[corner] * source[ys[corner] * width + xs[corner]];
            }
            output[(batch * channels + c) * spatial + point] = sum;
        }
    }
}
}

ggml_tensor * ocr_grid_sample_2d(ggml_context * ctx, ggml_tensor * image, ggml_tensor * grid) {
    if (!ctx || !image || !grid || image->type != GGML_TYPE_F32 || grid->type != GGML_TYPE_F32
        || !ggml_is_contiguous(image) || !ggml_is_contiguous(grid)
        || grid->ne[0] != 2 || image->ne[3] != grid->ne[3]) return nullptr;
    ggml_tensor * args[] = {image, grid};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, grid->ne[1], grid->ne[2], image->ne[2], image->ne[3],
                          args, 2, grid_sample, GGML_N_TASKS_MAX, nullptr);
}
