# Source from the isolated candidate runtime after its config.sh.
# Changes model loading only. Keep DDR contrast and pipeline settings explicit.
export GGML_SPACEMIT_OCR_PC8_LOAD_THREADS=4
export GGML_SPACEMIT_PC8_HOST_OPTIONS=1
export GGML_SPACEMIT_SHARED_MODE=DDR
