set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-fa-pack-v40-riscv64.tar.gz
echo "53ad0ff1c8c2f2b879f2677d5bc2f307127863c8e4711b3dd851d2dcb6a170c3  $A" | sha256sum -c -
test -d ocr-adaptation/runtime-q4q4-fa-pack-v40
python3 - <<'PY'
from pathlib import Path
import json,hashlib
r=Path('ocr-adaptation/runtime-q4q4-fa-pack-v40');m=json.loads((r/'MANIFEST.json').read_text())
for name,h in m['files'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
p=r/'bin/check-fa-v40';p.write_text('''#!/bin/bash
set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
B=ocr-adaptation/runtime-q4q4-fa-pack-v40/bin/bench-fa-x8-v40
E=results/optimization/20261104-q4q4-v40b-fa-micro
for N in 64 65 988 1008 1056; do
 for strided in 0 1 2 3; do
  for arm in off-a pack-a both-a both-b pack-b off-b; do
   unset GGML_SPACEMIT_OCR_FA_PACK GGML_SPACEMIT_OCR_FA_X8
   if [[ $arm == pack* || $arm == both* ]]; then export GGML_SPACEMIT_OCR_FA_PACK=1; fi
   if [[ $arm == both* ]]; then export GGML_SPACEMIT_OCR_FA_X8=1; fi
   "$B" "$N" "$strided" "$E/N$N-s$strided-$arm.bin" > "$E/N$N-s$strided-$arm.jsonl"
  done
  for arm in pack-a both-a both-b pack-b off-b; do
   cmp "$E/N$N-s$strided-off-a.bin" "$E/N$N-s$strided-$arm.bin"
  done
  echo "FA PACK EXACT N=$N strided=$strided"
 done
done
''');p.chmod(0o755)
PY
taskset -c 0-7 python3 -u scripts/optimization_round.v40.py \
 --label 20261104-q4q4-v40b-fa-micro \
 --checker ocr-adaptation/runtime-q4q4-fa-pack-v40/bin/check-fa-v40 --checker-only \
 --checker-timeout 360 --checker-separate-stderr \
 --spacemit-core-ids 8,9,10,11,13,14,15 --min-free-gib 4
