set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-concat-tcm-v47-riscv64.tar.gz
echo "fbd32444350afa9152bf0341047e9e881018ac945a8d603518ab1c33ca49a595  $A" | sha256sum -c -
test ! -e ocr-adaptation/runtime-q4q4-concat-tcm-v47
tar -xzf "$A" -C ocr-adaptation
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('ocr-adaptation/runtime-q4q4-concat-tcm-v47');m=json.loads((r/'MANIFEST.json').read_text());assert m['source_clean']
for name,h in m['files'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
PY
taskset -c 0-7 python3 -u scripts/optimization_round.v47.py --label 20261104-q4q4-v47-micro  --checker ocr-adaptation/runtime-q4q4-concat-tcm-v47/bin/bench-concat-tcm-v47 --checker-only  --checker-timeout 600 --checker-separate-stderr --spacemit-core-ids 8,9,10,11,13,14,15 --min-free-gib 4
