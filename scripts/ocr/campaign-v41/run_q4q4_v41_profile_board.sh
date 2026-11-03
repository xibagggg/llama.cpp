set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-down-tcm-v41-profile-riscv64.tar.gz
echo "4dd21970019fb55d346b6d7b75cc34effa7f44d03bd447f9f5acda7b0b101c45  $A" | sha256sum -c -
test ! -e ocr-adaptation/runtime-q4q4-down-tcm-v41-profile
tar -xzf "$A" -C ocr-adaptation
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('ocr-adaptation/runtime-q4q4-down-tcm-v41-profile');m=json.loads((r/'MANIFEST.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h,n
assert m['source_commit']=='91570b05aa667bed593ac715158cc95b3820dd36' and m['build_flags']['GGML_BUILD_PROFILE']
PY
taskset -c 0-7 python3 -u scripts/optimization_round.v41.py \
 --label 20261104-q4q4-v41-profile-tile16 \
 --runtime v41=ocr-adaptation/runtime-q4q4-down-tcm-v41-profile \
 --precision q4 --text-precision q4 --f16-m4-cache --down-tcm-tile 16 --down-phases \
 --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize \
 --ime2-down-layers "$(seq -s, 14 26)" --fa-x8 --fa-pack --trace \
 --spacemit-core-ids 8,9,10,11,13,14,15 \
 --case-file images/optimization-v5/cases.json --cases v5_table_merge4 \
 --warmups 0 --repeats 1 --strict-quality-evidence --min-free-gib 4
