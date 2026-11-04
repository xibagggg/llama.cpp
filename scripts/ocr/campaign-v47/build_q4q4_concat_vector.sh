set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
SRC="$ROOT/ocr-adaptation/managed-source"
EXP="$ROOT/ocr-adaptation/experiments/v49-concat-vector"
SDK="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
BUILD="$ROOT/build-ocr-managed-riscv64"
TC=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
"$TC/bin/riscv64-unknown-linux-gnu-g++" -std=c++17 -O3 -DNDEBUG -march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d \
 -I"$SRC/ggml/include" -I"$SRC/ggml/src" -I"$SRC/ggml/src/ggml-spacemit" -I"$SDK/include" \
 "$EXP/bench_concat_vector_v49.cpp" "$BUILD/ggml/src/ggml-spacemit/CMakeFiles/ggml-spacemit.dir/rvv_kernels.cpp.o" \
 -L"$BUILD/bin" -L"$SDK/lib" -Wl,-rpath-link,"$BUILD/bin" -Wl,-rpath-link,"$SDK/lib" \
 -lggml-spacemit -lggml-cpu -lggml-base -lspert -latomic -pthread -o "$EXP/bench-concat-vector-v49"
sha256sum "$EXP/bench-concat-vector-v49"
