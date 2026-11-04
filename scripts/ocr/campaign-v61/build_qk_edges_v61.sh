set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
cd "$ROOT/ocr-adaptation/managed-source"
TC=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
SPERT="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
B="$ROOT/build-ocr-managed-riscv64"
"$TC/bin/riscv64-unknown-linux-gnu-g++" -std=c++17 -O3 -DNDEBUG -march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d \
 -Iggml/include -Iggml/src -Iggml/src/ggml-spacemit -I"$SPERT/include" ../experiments/v61-qk-edges/bench_qk_edges_v61.cpp \
 "$B/ggml/src/ggml-spacemit/CMakeFiles/ggml-spacemit.dir/rvv_kernels.cpp.o" \
 -L"$B/bin" -L"$SPERT/lib" -Wl,-rpath-link,"$B/bin" -Wl,-rpath-link,"$SPERT/lib" \
 -lggml-spacemit -lggml-cpu -lggml-base -lspert -latomic -pthread -o "$ROOT/ocr-adaptation/bench-qk-edges-v61"
sha256sum "$ROOT/ocr-adaptation/bench-qk-edges-v61"
