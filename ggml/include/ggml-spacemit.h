// SPDX-License-Identifier: MIT

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serializes GGML graphs with external in-process AI inference. Calls must
// be paired on the same host thread; never call from an AI worker tile.
GGML_BACKEND_API void ggml_backend_spacemit_acquire(void);
GGML_BACKEND_API void ggml_backend_spacemit_release(void);

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_spacemit_init(void);

GGML_BACKEND_API bool ggml_backend_is_spacemit(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_spacemit_reg(void);

#ifdef __cplusplus
}
#endif
