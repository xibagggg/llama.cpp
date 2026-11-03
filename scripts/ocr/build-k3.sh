#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RISCV_ROOT_PATH:?Set RISCV_ROOT_PATH to the SpacemiT cross-toolchain root}"
: "${SPERT_DIR:?Set SPERT_DIR to spine-runtime.riscv64.0.6.3}"
build_dir=${OCR_BUILD_DIR:-"$repo/build-ocr-k3"}
jobs=${OCR_BUILD_JOBS:-16}
cmake -S "$repo" -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$repo/cmake/riscv64-spacemit-linux-gnu-gcc.cmake" \
  -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DGGML_SPACEMIT=ON -DGGML_CPU_RISCV64_SPACEMIT=ON \
  -DSPERT_DIR="$SPERT_DIR" -DGGML_CPU_REPACK=OFF -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF -DGGML_RVV=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON \
  -DGGML_RV_ZBA=ON -DGGML_RV_ZICBOP=ON -DGGML_RV_ZIHINTPAUSE=ON \
  -DGGML_BUILD_PROFILE=OFF -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_SERVER_SMT_MTMD=OFF
cmake --build "$build_dir" --target llama-server -j "$jobs"
python3 "$repo/scripts/ocr/source-manifest.py" --repo "$repo" --build-dir "$build_dir"
