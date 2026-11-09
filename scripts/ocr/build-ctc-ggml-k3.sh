#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RISCV_ROOT_PATH:?Set SpacemiT cross-toolchain root}"
: "${SPERT_DIR:?Set spine-runtime root}"
build=${OCR_CTC_K3_BUILD_DIR:-"$repo/build-ocr-k3"}
test -f "$build/bin/libggml.so"
compiler="$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++"
flags=(-std=c++17 -O2 -ffp-contract=off -DOCR_LAYOUT_SPACEMIT=1 -I"$repo/ggml/include" -I"$repo/vendor/nlohmann")
links=(-L"$build/bin" -L"$SPERT_DIR/lib" -Wl,-rpath-link,"$build/bin" \
    -Wl,-rpath-link,"$SPERT_DIR/lib" -Wl,-rpath,'$ORIGIN:$ORIGIN/../lib' \
    -lggml -lggml-spacemit -lggml-cpu -lggml-base -latomic -pthread)
"$compiler" "${flags[@]}" "$repo/tools/ocr-ggml/ctc-ggml.cpp" "${links[@]}" -o "$build/bin/ctc-ggml"
"$compiler" "${flags[@]}" -fPIC -shared -DOCR_CTC_LIBRARY=1 \
    "$repo/tools/ocr-ggml/ctc-ggml.cpp" "${links[@]}" -o "$build/bin/libocr-ctc-ggml.so"
sha256sum "$build/bin/ctc-ggml" "$build/bin/libocr-ctc-ggml.so"
