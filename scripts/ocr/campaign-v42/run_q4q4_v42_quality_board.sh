set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-down-sched-v42-riscv64.tar.gz
echo "0e12c9ccd2b1b2eb217baf44eae9d92d35d2e3f952f75f0e3b56952ba865e847  $A" | sha256sum -c -
test ! -e ocr-adaptation/runtime-q4q4-down-sched-v42
tar -xzf "$A" -C ocr-adaptation
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('ocr-adaptation/runtime-q4q4-down-sched-v42');m=json.loads((r/'MANIFEST.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
PY
set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
taskset -c 0-7 python3 -u scripts/optimization_round.v42.py \
 --label 20261104-q4q4-v42-quality-paired \
 --runtime v42=ocr-adaptation/runtime-q4q4-down-sched-v42 \
 --precision q4 --text-precision q4 --f16-m4-cache --down-tcm-tile 16 --down-schedule paired \
 --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize --ime2-down-layers "$(seq -s, 14 26)" --fa-x8 --fa-pack \
 --spacemit-core-ids 8,9,10,11,13,14,15 \
 --case-file images/optimization-v5/cases.json \
 --cases 03_table_gb_standard v5_table_merge4 chinese chinese_skew3 01_text_putnam 02_formula_putnam 04_scanned_civil_trial \
 --warmups 0 --repeats 1 --strict-quality-evidence --min-free-gib 4
