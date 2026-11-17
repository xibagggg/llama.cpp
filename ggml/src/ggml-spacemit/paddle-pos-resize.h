#pragma once

// Narrow, opt-in Paddle position interpolation. Geometry is reused only
// within this invocation and each worker owns independent ordinary scratch.
#include "ggml.h"
#include "spacemit-context.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <riscv_vector.h>

namespace paddle_pos_resize {
constexpr size_t max_taps = 27 * 27;
struct tap { uint32_t offset; float weight; };
static_assert(sizeof(tap) == 8, "tap workspace contract");
constexpr size_t lane_bytes = (max_taps * sizeof(tap) + 63) & ~size_t(63);

inline bool enabled() {
    static const bool value = [] {
        const char * p = std::getenv("GGML_SPACEMIT_PADDLE_POS_RESIZE");
        return p && !std::strcmp(p, "1");
    }();
    return value;
}

inline bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a->data || !b->data) return false;
    const auto x = reinterpret_cast<uintptr_t>(a->data);
    const auto y = reinterpret_cast<uintptr_t>(b->data);
    return x <= y ? y - x < ggml_nbytes(a) : x - y < ggml_nbytes(b);
}

inline bool supported(const ggml_tensor * op) {
    if (!enabled() || !op || op->op != GGML_OP_UPSCALE || !op->src[0]) return false;
    const auto * src = op->src[0];
    if (src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        op->op_params[0] != (GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS) ||
        src->ne[0] != 27 || src->ne[1] != 27 || src->ne[2] <= 0 || src->ne[2] > 4096 ||
        src->ne[3] != 1 || op->ne[2] != src->ne[2] || op->ne[3] != 1 ||
        op->ne[0] <= 0 || op->ne[0] > 512 || op->ne[1] <= 0 || op->ne[1] > 512 ||
        op->ne[0] * op->ne[1] > 16384 || !ggml_is_contiguous(op) ||
        src->nb[2] != sizeof(float)) return false;
    // Permit the real channel-first permutation and padded/offset views, but
    // reject overlapping spatial rows. Every tap offset is a validated u32.
    constexpr size_t limit = std::numeric_limits<uint32_t>::max();
    if (src->nb[0] % sizeof(float) || src->nb[1] % sizeof(float) ||
        src->nb[0] < size_t(src->ne[2]) * sizeof(float) || src->nb[0] > limit / 27 ||
        src->nb[1] < 27 * src->nb[0] || src->nb[1] > limit / 27 ||
        src->nb[3] < 27 * src->nb[1] || ggml_nbytes(src) > limit) return false;
    return !overlaps(src, op);
}

inline size_t work_bytes(int n_threads) {
    return n_threads > 0 && size_t(n_threads) <= SIZE_MAX / lane_bytes ? size_t(n_threads) * lane_bytes : 0;
}

inline float triangle(float x) {
    return std::max(1.0f - fabsf(x), 0.0f);
}

inline bool forward(ggml::spacemit::context & ctx, ggml_tensor * op) {
    if (!supported(op)) return false;
    const size_t needed = work_bytes(int(ctx.nth));
    if (!needed || !ctx.workspace || ctx.workspace_size < needed || ctx.ith >= ctx.nth) return false;
    const auto * src = op->src[0];
    auto * taps = reinterpret_cast<tap *>(static_cast<char *>(ctx.workspace) + size_t(ctx.ith) * lane_bytes);
    const int64_t begin = src->ne[2] * ctx.ith / ctx.nth;
    const int64_t end = src->ne[2] * (ctx.ith + 1) / ctx.nth;
    if (begin == end) return true;
    // Keep CPU formula grouping and division, including truncating bounds.
    const float sf0 = float(op->ne[0]) / src->ne[0];
    const float sf1 = float(op->ne[1]) / src->ne[1];
    const float support0 = std::max(1.0f, 1.0f / sf0);
    const float support1 = std::max(1.0f, 1.0f / sf1);
    const float inverse0 = 1.0f / support0;
    const float inverse1 = 1.0f / support1;
    for (int64_t oy = 0; oy < op->ne[1]; ++oy) {
        const float y = (float(oy) + 0.5f) / sf1;
        for (int64_t ox = 0; ox < op->ne[0]; ++ox) {
            const float x = (float(ox) + 0.5f) / sf0;
            const int64_t xmin = std::max<int64_t>(x - support0 + 0.5f, 0);
            const int64_t xmax = std::min<int64_t>(x + support0 + 0.5f, src->ne[0]);
            const int64_t ymin = std::max<int64_t>(y - support1 + 0.5f, 0);
            const int64_t ymax = std::min<int64_t>(y + support1 + 0.5f, src->ne[1]);
            size_t count = 0;
            float total = 0.0f;
            for (int64_t sy = ymin; sy < ymax; ++sy) {
                const float wy = triangle((sy - y + 0.5f) * inverse1);
                for (int64_t sx = xmin; sx < xmax; ++sx) {
                    const float wx = triangle((sx - x + 0.5f) * inverse0);
                    const float weight = wx * wy;
                    if (weight <= 0.0f) continue;
                    GGML_ASSERT(count < max_taps);
                    taps[count++] = {uint32_t(size_t(sx) * src->nb[0] + size_t(sy) * src->nb[1]), weight};
                    total += weight;
                }
            }
            for (int64_t channel = begin; channel < end;) {
                const size_t vl = __riscv_vsetvl_e32m4(size_t(end - channel));
                auto value = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                for (size_t i = 0; i < count; ++i) {
                    const auto * input = reinterpret_cast<const float *>(
                        static_cast<const char *>(src->data) + taps[i].offset + size_t(channel) * sizeof(float));
                    const auto pixels = __riscv_vle32_v_f32m4(input, vl);
                    // Existing CPU ops.o emits fmadd.s for the same recurrence.
                    value = __riscv_vfmacc_vf_f32m4(value, taps[i].weight, pixels, vl);
                }
                if (total > 0.0f) value = __riscv_vfdiv_vf_f32m4(value, total, vl);
                auto * output = reinterpret_cast<float *>(static_cast<char *>(op->data) +
                    size_t(ox) * op->nb[0] + size_t(oy) * op->nb[1] + size_t(channel) * op->nb[2]);
                __riscv_vsse32_v_f32m4(output, ptrdiff_t(op->nb[2]), value, vl);
                channel += int64_t(vl);
            }
        }
    }
    return true;
}
} // namespace paddle_pos_resize
