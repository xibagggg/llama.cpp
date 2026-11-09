// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ocr_ep_tensor {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
};

bool ocr_ep_vision_enabled();
// Pixels are the exact normalized RGBRGB... buffer produced by mtmd.
void ocr_ep_encode_hwc(const float * pixels, int width, int height, std::vector<float> & embeddings);
// Layout pixels are float32 NCHW, 1x3x800x800, normalized to [0,1].
// Masks are returned as uint8 signs; the official postprocessor uses >0.
std::vector<ocr_ep_tensor> ocr_ep_layout(const float * pixels, size_t count, double & seconds);
