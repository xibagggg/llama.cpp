set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-compact-lazy-v43-profile-riscv64.tar.gz
echo "4497247480bc5749634f91e15c53ec20d9dba163acb8516f1895ad2c35c91559  $A" | sha256sum -c -
test ! -e ocr-adaptation/runtime-q4q4-compact-lazy-v43-profile
tar -xzf "$A" -C ocr-adaptation
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('ocr-adaptation/runtime-q4q4-compact-lazy-v43-profile');m=json.loads((r/'MANIFEST.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
assert m['source_commit']=='d08ae561690ebe5d24555b12e47d187bf4b7084c' and m['build_flags']['GGML_BUILD_PROFILE']
PY
taskset -c 0-7 python3 -u scripts/optimization_round.v43.py \
 --label 20261104-q4q4-v43-profile-selected \
 --runtime v43=ocr-adaptation/runtime-q4q4-compact-lazy-v43-profile \
 --precision q4 --text-precision q4 --f16-m4-cache --down-tcm-tile 16 --down-schedule paired --down-compact-tile 20 --fa-lazy-init \
 --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize \
 --ime2-down-layers "$(seq -s, 14 26)" --fa-x8 --fa-pack --trace \
 --spacemit-core-ids 8,9,10,11,13,14,15 \
 --case-file images/optimization-v5/cases.json --cases v5_table_merge4 \
 --warmups 0 --repeats 1 --strict-quality-evidence --min-free-gib 4
