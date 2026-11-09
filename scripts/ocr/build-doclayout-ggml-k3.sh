#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RISCV_ROOT_PATH:?Set SpacemiT cross-toolchain root}"
: "${SPERT_DIR:?Set spine-runtime root}"
build=${OCR_LAYOUT_K3_BUILD_DIR:-"$repo/build-ime2-bias"}
test -f "$build/bin/libggml.so"
compiler="$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++"
flags=(-std=c++17 -O2 -ffp-contract=off -DOCR_LAYOUT_SPACEMIT=1 -I"$repo/ggml/include" -I"$repo/vendor/nlohmann")
links=(-L"$build/bin" -L"$SPERT_DIR/lib" -Wl,-rpath-link,"$build/bin" \
    -Wl,-rpath-link,"$SPERT_DIR/lib" -Wl,-rpath,'$ORIGIN:$ORIGIN/../lib' \
    -lggml -lggml-spacemit -lggml-cpu -lggml-base -latomic -pthread)
sources=("$repo/tools/ocr-ggml/doclayout-ggml.cpp" "$repo/tools/ocr-ggml/doclayout-ops.cpp")
"$compiler" "${flags[@]}" "${sources[@]}" "${links[@]}" -o "$build/bin/doclayout-ggml"
"$compiler" "${flags[@]}" -fPIC -shared -DOCR_LAYOUT_LIBRARY=1 \
    "${sources[@]}" "${links[@]}" -o "$build/bin/libocr-layout-ggml.so"
file "$build/bin/doclayout-ggml" "$build/bin/libocr-layout-ggml.so"
sha256sum "$build/bin/doclayout-ggml" "$build/bin/libocr-layout-ggml.so"
