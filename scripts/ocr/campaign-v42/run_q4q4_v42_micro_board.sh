set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-down-sched-v42-micro-riscv64.tar.gz
echo "a742dccc198bd40ae47c406ff75d77df6b10339f5fce11f19371ecc6ccb54fcb  $A" | sha256sum -c -
test ! -e ocr-adaptation/runtime-q4q4-down-sched-v42-micro
tar -xzf "$A" -C ocr-adaptation
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('ocr-adaptation/runtime-q4q4-down-sched-v42-micro');m=json.loads((r/'MANIFEST.json').read_text())
assert m['source_clean']
for n,h in m['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
p=r/'bin/check-down-v42'
p.write_text('''#!/bin/bash
set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
exec ocr-adaptation/runtime-q4q4-down-sched-v42-micro/bin/bench-down-sched-v42 results/optimization/20261104-q4q4-v36-capture-off/v36-q4/down-captures
''');p.chmod(0o755)
PY
taskset -c 0-7 python3 -u scripts/optimization_round.v41.py \
 --label 20261104-q4q4-v42-down-micro \
 --checker ocr-adaptation/runtime-q4q4-down-sched-v42-micro/bin/check-down-v42 --checker-only \
 --checker-timeout 360 --checker-separate-stderr \
 --spacemit-core-ids 8,9,10,11,13,14,15 --min-free-gib 4
