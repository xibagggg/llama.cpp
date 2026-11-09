#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build=${OCR_CPU_BUILD_DIR:-"$repo/build-ocr-matmul-cpu-ref"}
jobs=${OCR_CPU_BUILD_JOBS:-16}

cmake -S "$repo" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DGGML_SPACEMIT=OFF \
  -DGGML_CPU_RISCV64_SPACEMIT=OFF -DGGML_OPENMP=ON \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$build" --target ggml ggml-cpu -j "$jobs"

c++ -std=c++17 -O2 -I"$repo/ggml/include" "$repo/tests/ocr/test_matmul_bias_fusion.cpp" \
  -L"$build/bin" -Wl,-rpath-link,"$build/bin" -lggml-cpu -lggml -lggml-base -pthread \
  -o "$build/bin/test-matmul-cpu-reference"

sha256sum "$build/bin/test-matmul-cpu-reference" "$build/bin/libggml-base.so" "$build/bin/libggml-cpu.so"
