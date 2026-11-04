set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
chmod +x ocr-adaptation/bench-concat-vector-v49
test "$(sha256sum ocr-adaptation/bench-concat-vector-v49 | cut -d' ' -f1)" = c000613915fbf83a3c2553729fb4e7d99cd3d4bd9e7658c51b627f6f79f58e61
cat > ocr-adaptation/check-concat-vector-v49 <<'SH'
#!/bin/bash
set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
export LD_LIBRARY_PATH="$PWD/ocr-adaptation/runtime-q4q4-concat-tcm-v47/lib"
ocr-adaptation/bench-concat-vector-v49 > results/optimization/20261104-q4q4-v49-concat-vector/concat.jsonl
SH
chmod +x ocr-adaptation/check-concat-vector-v49
taskset -c 0-7 python3 -u scripts/optimization_round.v47.py \
 --label 20261104-q4q4-v49-concat-vector --checker ocr-adaptation/check-concat-vector-v49 \
 --checker-only --checker-timeout 120 --checker-separate-stderr \
 --spacemit-core-ids 8,9,10,11,13,14,15 --min-free-gib 4
