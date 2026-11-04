set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
cat > ocr-adaptation/check-pv-pipeline-v58 <<'SH'
#!/bin/bash
set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
mkdir -p results/optimization/20261104-pv-pipeline-v58-micro
export LD_LIBRARY_PATH="$PWD/ocr-adaptation/runtime-q4-down-v50/lib"
ocr-adaptation/bench-pv-pipeline-v58 > results/optimization/20261104-pv-pipeline-v58-micro/bench.jsonl
SH
chmod +x ocr-adaptation/check-pv-pipeline-v58 ocr-adaptation/bench-pv-pipeline-v58
taskset -c 0-7 python3 -u scripts/optimization_round.v50.py --label 20261104-pv-pipeline-v58-micro \
 --checker ocr-adaptation/check-pv-pipeline-v58 --checker-only --checker-timeout 120 --checker-separate-stderr \
 --spacemit-core-ids 8,9,10,11,13,14,15 --min-free-gib 4
