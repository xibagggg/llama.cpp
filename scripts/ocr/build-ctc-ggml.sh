#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build=${OCR_CTC_BUILD_DIR:-"$repo/build-ctc-x86"}
cmake -S "$repo" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF -DGGML_SPACEMIT=OFF -DGGML_CPU_RISCV64_SPACEMIT=OFF \
    -DGGML_OPENMP=OFF -DGGML_CPU_REPACK=OFF -DGGML_BACKEND_DL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$build" --target ggml -j "${OCR_BUILD_JOBS:-8}"
"${CXX:-g++}" -std=c++17 -O2 -I"$repo/ggml/include" -I"$repo/vendor/nlohmann" \
    "$repo/tools/ocr-ggml/ctc-ggml.cpp" -L"$build/bin" -Wl,-rpath,'$ORIGIN' \
    -lggml -lggml-cpu -lggml-base -pthread -o "$build/bin/ctc-ggml"
"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared -DOCR_CTC_LIBRARY=1 \
    -I"$repo/ggml/include" -I"$repo/vendor/nlohmann" \
    "$repo/tools/ocr-ggml/ctc-ggml.cpp" -L"$build/bin" -Wl,-rpath,'$ORIGIN' \
    -lggml -lggml-cpu -lggml-base -pthread -o "$build/bin/libocr-ctc-ggml.so"
