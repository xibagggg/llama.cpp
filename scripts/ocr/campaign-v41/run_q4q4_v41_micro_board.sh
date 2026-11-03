set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-down-tcm-v41-riscv64.tar.gz
test ! -e ocr-adaptation/runtime-q4q4-down-tcm-v41
tar -xzf "$A" -C ocr-adaptation
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('ocr-adaptation/runtime-q4q4-down-tcm-v41');m=json.loads((r/'MANIFEST.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
p=r/'bin/check-down-v41'
p.write_text('''#!/bin/bash
set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
E=results/optimization/20261104-q4q4-v41-down-micro
mkdir "$E/phases"
export GGML_SPACEMIT_OCR_DOWN_PHASES="$PWD/$E/phases"
exec ocr-adaptation/runtime-q4q4-down-tcm-v41/bin/bench-down-tcm-v41 results/optimization/20261104-q4q4-v36-capture-off/v36-q4/down-captures
''');p.chmod(0o755)
PY
taskset -c 0-7 python3 -u scripts/optimization_round.v41.py \
 --label 20261104-q4q4-v41-down-micro \
 --checker ocr-adaptation/runtime-q4q4-down-tcm-v41/bin/check-down-v41 --checker-only \
 --checker-timeout 360 --checker-separate-stderr \
 --spacemit-core-ids 8,9,10,11,13,14,15 --min-free-gib 4
