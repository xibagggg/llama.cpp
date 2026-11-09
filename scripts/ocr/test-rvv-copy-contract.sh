#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RISCV_ROOT_PATH:?Set SpacemiT cross-toolchain root}"
build=${OCR_COPY_TEST_DIR:-"$repo/build-rvv-copy-contract"}
source_file=${OCR_COPY_SOURCE_FILE:-"$repo/ggml/src/ggml-spacemit/rvv_kernels.cpp"}
qemu=${QEMU_RISCV64:-qemu-riscv64-static}
mkdir -p "$build"
python3 - "$source_file" "$build/rvv-copy-under-test.h" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
pieces = []
for start, end in [
    ('static inline void rvv_transposed_s32_mn_to_nm(', 'static inline void rvv_qk_dot_tile_f16_x1('),
    ('void memcpy1d(', 'void memcpy2d('),
]:
    begin = source.index(start)
    pieces.append(source[begin:source.index(end, begin)])
Path(sys.argv[2]).write_text('\n'.join(pieces))
PY
compiler="$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++"
"$compiler" -std=c++17 -O3 -static -march=rv64gcv_zfh_zba -mabi=lp64d -I"$build" \
    "$repo/tests/ocr/test_rvv_copy_contract.cpp" -o "$build/test-rvv-copy-contract"
status=0
for vlen in 128 256 512 1024; do
    echo "RVV_TEST vlen_bits=$vlen"
    timeout 30s "$qemu" -cpu "rv64,v=true,vlen=$vlen,elen=64,vext_spec=v1.0" "$build/test-rvv-copy-contract" || status=$?
done
exit "$status"
