set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-compact-lazy-v43-riscv64.tar.gz
echo "258bef671fbab47cf2bff36da8c106cde40d60a6b9cdf2c412ac2fce25177e48  $A" | sha256sum -c -
test ! -e ocr-adaptation/runtime-q4q4-compact-lazy-v43
tar -xzf "$A" -C ocr-adaptation
python3 - <<'PY'
from pathlib import Path
import json,hashlib
r=Path('ocr-adaptation/runtime-q4q4-compact-lazy-v43');m=json.loads((r/'MANIFEST.json').read_text())
assert m['source_clean']
for name,h in m['files'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
p=r/'bin/check-v43';p.write_text('#!/bin/bash\nset -euo pipefail\ncd /home/bianbu/MiniCPM-OCR-1.0\nR=ocr-adaptation/runtime-q4q4-compact-lazy-v43/bin\nE=results/optimization/20261104-q4q4-v43-micro\n"$R/bench-down-compact-v43" results/optimization/20261104-q4q4-v36-capture-off/v36-q4/down-captures > "$E/down.jsonl"\nexport GGML_SPACEMIT_OCR_FA_PACK=1 GGML_SPACEMIT_OCR_FA_X8=1\nfor N in 64 65 988 1056; do\n for layout in 0 1 2 3 4 5; do\n  for arm in off-a on-a on-b off-b; do\n   unset GGML_SPACEMIT_OCR_FA_LAZY_INIT\n   if [[ $arm == on-* ]]; then export GGML_SPACEMIT_OCR_FA_LAZY_INIT=1; fi\n   "$R/bench-fa-lazy-v43" "$N" "$layout" "$E/N$N-l$layout-$arm.bin" > "$E/N$N-l$layout-$arm.jsonl"\n  done\n  for arm in on-a on-b off-b; do cmp "$E/N$N-l$layout-off-a.bin" "$E/N$N-l$layout-$arm.bin"; done\n  echo "LAZY INIT EXACT N=$N layout=$layout"\n done\ndone\n');p.chmod(0o755)
PY
taskset -c 0-7 python3 -u scripts/optimization_round.v43.py  --label 20261104-q4q4-v43-micro  --checker ocr-adaptation/runtime-q4q4-compact-lazy-v43/bin/check-v43 --checker-only  --checker-timeout 600 --checker-separate-stderr  --spacemit-core-ids 8,9,10,11,13,14,15 --min-free-gib 4
