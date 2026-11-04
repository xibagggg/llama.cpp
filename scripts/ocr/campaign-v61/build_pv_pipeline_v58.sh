set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
cd "$ROOT/ocr-adaptation/managed-source"
TC=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
SPERT="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
"$TC/bin/riscv64-unknown-linux-gnu-g++" -std=c++17 -O3 -DNDEBUG -march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d \
 -Iggml/include -Iggml/src -Iggml/src/ggml-spacemit -I"$SPERT/include" ../experiments/v58-pv-pipeline/bench_pv_pipeline_v58.cpp \
 -L"$SPERT/lib" -lspert -latomic -pthread -o "$ROOT/ocr-adaptation/bench-pv-pipeline-v58"
sha256sum "$ROOT/ocr-adaptation/bench-pv-pipeline-v58"
