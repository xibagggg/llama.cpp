#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RISCV_ROOT_PATH:?Set SpacemiT toolchain root}"
: "${SPERT_DIR:?Set spine-runtime release root}"
build=${OCR_BUILD_DIR:-"$repo/build-ocr-add-gelu"}
export OCR_BUILD_DIR="$build"
bash "$repo/scripts/ocr/build-k3.sh"
compiler="$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++"
links=(-L"$build/bin" -L"$SPERT_DIR/lib" -Wl,-rpath-link,"$build/bin" -Wl,-rpath-link,"$SPERT_DIR/lib" -lggml -lggml-spacemit -lggml-cpu -lggml-base -latomic -pthread)
"$compiler" -std=c++17 -O2 -I"$repo/ggml/include" "$repo/tests/ocr/test_add_gelu_fusion.cpp" "${links[@]}" -o "$build/bin/test-add-gelu"
"$compiler" -std=c++17 -O2 -DGGML_TEST_SPACEMIT=1 -I"$repo/ggml/include" "$repo/tests/ocr/test_matmul_bias_fusion.cpp" "${links[@]}" -o "$build/bin/test-matmul-bias-fusion"
"$compiler" -std=c++17 -O2 -I"$repo/ggml/include" -I"$repo/include" -I"$repo/tools/mtmd" "$repo/tests/ocr/bench_clip_fusion.cpp" -lmtmd -lllama "${links[@]}" -o "$build/bin/bench-clip-fusion"
sha256sum "$build/bin/libggml-spacemit.so" "$build/bin/test-add-gelu" "$build/bin/test-matmul-bias-fusion" "$build/bin/bench-clip-fusion"
