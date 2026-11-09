#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build=${OCR_LAYOUT_BUILD_DIR:-"$repo/build-ctc-x86"}
"${CXX:-g++}" -std=c++17 -O2 -ffp-contract=off -I"$repo/ggml/include" \
    -I"$repo/vendor/nlohmann" "$repo/tools/ocr-ggml/doclayout-ggml.cpp" \
    "$repo/tools/ocr-ggml/doclayout-ops.cpp" -L"$build/bin" -Wl,-rpath,'$ORIGIN' \
    -lggml -lggml-cpu -lggml-base -pthread -o "$build/bin/doclayout-ggml"
"${CXX:-g++}" -std=c++17 -O2 -ffp-contract=off -fPIC -shared -DOCR_LAYOUT_LIBRARY=1 \
    -I"$repo/ggml/include" -I"$repo/vendor/nlohmann" "$repo/tools/ocr-ggml/doclayout-ggml.cpp" \
    "$repo/tools/ocr-ggml/doclayout-ops.cpp" -L"$build/bin" -Wl,-rpath,'$ORIGIN' \
    -lggml -lggml-cpu -lggml-base -pthread -o "$build/bin/libocr-layout-ggml.so"
