#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build=${OCR_GRID_BUILD_DIR:-"$repo/build-ctc-x86"}
"${CXX:-g++}" -std=c++17 -O2 -ffp-contract=off -I"$repo/ggml/include" -I"$repo/tools/ocr-ggml" \
    "$repo/tests/ocr/test_doclayout_grid_sample.cpp" "$repo/tools/ocr-ggml/doclayout-ops.cpp" \
    -L"$build/bin" -Wl,-rpath,'$ORIGIN' -lggml -lggml-cpu -lggml-base -pthread \
    -o "$build/bin/test-doclayout-grid"
