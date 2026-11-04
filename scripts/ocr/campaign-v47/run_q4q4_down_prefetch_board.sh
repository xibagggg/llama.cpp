set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
chmod +x ocr-adaptation/bench-down-prefetch-v48
test "$(sha256sum ocr-adaptation/bench-down-prefetch-v48 | cut -d' ' -f1)" = 33118b51aacb55a1bc1939ad2c6c196f13252c4334fee493b54f7fb5b834fa25
cat > ocr-adaptation/check-down-prefetch-v48 <<'SH'
#!/bin/bash
set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
export LD_LIBRARY_PATH="$PWD/ocr-adaptation/runtime-q4q4-concat-tcm-v47/lib"
ocr-adaptation/bench-down-prefetch-v48 results/optimization/20261104-q4q4-v36-capture-off/v36-q4/down-captures > results/optimization/20261104-q4q4-v48-prefetch/down.jsonl
SH
chmod +x ocr-adaptation/check-down-prefetch-v48
taskset -c 0-7 python3 -u scripts/optimization_round.v47.py \
 --label 20261104-q4q4-v48-prefetch --checker ocr-adaptation/check-down-prefetch-v48 \
 --checker-only --checker-timeout 180 --checker-separate-stderr \
 --spacemit-core-ids 8,9,10,11,13,14,15 --min-free-gib 4
