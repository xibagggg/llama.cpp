#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// Native options: bit 0 uses GGML elementwise ops; bit 1 uses native compact copies.
void * ocr_layout_create(const char * model, int threads, int native_options, char * error, size_t error_size);
void ocr_layout_destroy(void * handle);
// OCR_LAYOUT_BACKEND: cpu (default), cpu-sched, or spacemit (requires board build).
// Diagnostic JSON reports assignments/buffers; assignment alone does not prove IME2 execution.
int ocr_layout_info(void * handle, char * json_output, size_t capacity, char * error, size_t error_size);
// Counts are ordered: input, logits, pred_boxes, order_logits, out_masks.
int ocr_layout_counts(void * handle, size_t * counts, char * error, size_t error_size);
// Outputs are packed in the same fixed order as counts[1..4].
int ocr_layout_run(void * handle, const float * input, size_t input_count,
                   float * output, size_t output_count, double * inference_seconds,
                   char * error, size_t error_size);
#ifdef __cplusplus
}
#endif
