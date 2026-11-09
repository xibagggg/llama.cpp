#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
: "${RISCV_ROOT_PATH:?Set SpacemiT cross-toolchain root}"
: "${OCR_PACK_BASE_LIB_DIR:?Set the archived RISC-V ggml-base library directory}"
build=${OCR_PACK_RVV_TEST_DIR:-"$repo/build-im2col-pack-rvv"}
qemu=${QEMU_RISCV64:-qemu-riscv64-static}
mkdir -p "$build/bin"
python3 - "$repo" "$build/rvv-quant-under-test.h" <<'PY'
from pathlib import Path
import sys
repo=Path(sys.argv[1]);header=(repo/'ggml/src/ggml-spacemit/rvv_kernels.h').read_text()
source=(repo/'ggml/src/ggml-spacemit/rvv_kernels.cpp').read_text()
definitions=header[header.index('namespace spacemit_kernels {'):header.index('using quantize_a_row_def')]
start=source.index('template <size_t MB_ROWS>\nvoid quantize_a_nrow_i8_ref(')
end=source.index('void quantize_a_row_i8k(',start)
Path(sys.argv[2]).write_text(definitions+'\nnamespace rvv {\n'+source[start:end]+'\n}\n}\n')
PY
compiler="$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++"
sysroot=$("$compiler" -print-sysroot)
ln -sfn "$OCR_PACK_BASE_LIB_DIR/libggml-base.so.0" "$build/bin/libggml-base.so.0"
ln -sfn "$OCR_PACK_BASE_LIB_DIR/libggml-base.so" "$build/bin/libggml-base.so"
"$compiler" -std=c++17 -O3 -ffp-contract=off -march=rv64gcv_zfh_zba -mabi=lp64d \
 -DOCR_PACK_RVV_TEST=1 -I"$repo/ggml/include" -I"$repo/ggml/src" -I"$build" \
 "$repo/tests/ocr/test_im2col_pack.cpp" -L"$build/bin" -Wl,-rpath,'$ORIGIN' \
 -lggml-base -pthread -o "$build/bin/test-im2col-pack-rvv"
for vlen in 128 256 512 1024; do
 echo "RVV_PACK_TEST vlen_bits=$vlen"
 LD_LIBRARY_PATH="$build/bin:$sysroot/usr/lib64:$sysroot/lib64" \
 timeout --kill-after=2s 45s "$qemu" -L "$sysroot" \
 -cpu "max,vlen=$vlen,elen=64,vext_spec=v1.0" "$build/bin/test-im2col-pack-rvv"
done
