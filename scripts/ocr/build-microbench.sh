#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RISCV_ROOT_PATH:?Set RISCV_ROOT_PATH to the SpacemiT cross-toolchain root}"
: "${SPERT_DIR:?Set SPERT_DIR to spine-runtime.riscv64.0.6.3}"
build_dir=${OCR_BUILD_DIR:-"$repo/build-ocr-k3"}
"$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++" -std=c++17 -O3 -DNDEBUG \
  -march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d \
  -I"$repo/ggml/include" -I"$repo/ggml/src" \
  -I"$repo/ggml/src/ggml-spacemit" -I"$SPERT_DIR/include" \
  "$repo/tests/ocr/bench_ime2_compensated_v35.cpp" \
  -L"$build_dir/bin" -L"$SPERT_DIR/lib" \
  -Wl,-rpath-link,"$build_dir/bin" -Wl,-rpath-link,"$SPERT_DIR/lib" \
  -lggml-base -lspert -latomic -pthread \
  -o "$build_dir/bin/bench-ime2-compensated"
