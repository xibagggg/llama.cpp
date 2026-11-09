#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Errors use the caller's buffer; there is no global error state.
void * ctc_ggml_create(const char * model, int threads, char * error, size_t error_size);
void ctc_ggml_destroy(void * handle);
int ctc_ggml_describe(void * handle, char * output, size_t output_size,
                     char * error, size_t error_size);
int ctc_ggml_shapes(void * handle, int64_t * input_shape, int64_t * output_shape,
                    char * error, size_t error_size);
int ctc_ggml_run(void * handle, const float * input, size_t input_elements,
                 float * output, size_t output_elements, double * inference_seconds,
                 char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
