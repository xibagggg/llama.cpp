#pragma once

#include "ggml.h"

// PP-DocLayoutV3: ONNX GridSample, FP32, linear, zeros, align_corners=0.
// GGML dimensions: image [W,H,C,N], grid [2,Wout,Hout,N].
// Returns nullptr for unsupported types/layouts. No ONNX runtime is required.
ggml_tensor * ocr_grid_sample_2d(ggml_context * ctx, ggml_tensor * image, ggml_tensor * grid);
