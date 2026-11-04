set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-concat-tcm-v47-profile-riscv64.tar.gz
echo "a66b98bfa039bc0bc1e64a7ffa93f49928c464ec736142c90c089c36e6377715  $A" | sha256sum -c -
test ! -e ocr-adaptation/runtime-q4q4-concat-tcm-v47-profile
tar -xzf "$A" -C ocr-adaptation
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('ocr-adaptation/runtime-q4q4-concat-tcm-v47-profile');m=json.loads((r/'MANIFEST.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
assert m['source_commit']=='97e5c63240cef58b9751f08868f5cc040bfda6fc' and m['build_flags']['GGML_BUILD_PROFILE']
PY
taskset -c 0-7 python3 -u scripts/optimization_round.v47.py \
 --label 20261104-q4q4-v47-profile-selected \
 --runtime v47=ocr-adaptation/runtime-q4q4-concat-tcm-v47-profile \
 --precision q4 --text-precision q4 --f16-m4-cache --down-tcm-tile 16 --down-schedule paired --down-compact-tile 20 --fa-lazy-init --down-vector 5 --fa-direct 2 --concat-tcm 4 \
 --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize \
 --ime2-down-layers "$(seq -s, 14 26)" --fa-x8 --fa-pack --trace \
 --spacemit-core-ids 8,9,10,11,13,14,15 \
 --case-file images/optimization-v5/cases.json --cases v5_table_merge4 \
 --warmups 0 --repeats 1 --strict-quality-evidence --min-free-gib 4
