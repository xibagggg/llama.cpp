set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
cat > ocr-adaptation/check-qk-edges-v61 <<'SH'
#!/bin/bash
set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
export LD_LIBRARY_PATH="$PWD/ocr-adaptation/runtime-fa-compute-v61/lib"
export GGML_SPACEMIT_BACKEND=1 GGML_SPACEMIT_WORKERS=7 SPACEMIT_PERFER_CORE_ID=8,9,10,11,13,14,15
export GGML_SPACEMIT_OCR_FA_X8=1 GGML_SPACEMIT_OCR_FA_PACK=1 GGML_SPACEMIT_OCR_FA_LAZY_INIT=1 GGML_SPACEMIT_OCR_FA_DIRECT=2
D=results/optimization/20261104-qk-edges-v61
mkdir -p "$D/cases"
for spec in 988:0 989:0 1009:0 1057:0 1008:1 1008:6 1008:8 1008:16 1008:22;do
 N=${spec%:*};layout=${spec#*:}
 for mode in 0 3;do
  base="$D/cases/N${N}-L${layout}-M${mode}"
  GGML_SPACEMIT_OCR_FA_COMPUTE=$mode ocr-adaptation/bench-qk-edges-v61 "$N" "$layout" "$base.bin" > "$base.jsonl"
 done
done
python3 - <<'PY'
from pathlib import Path
import json,numpy as np
d=Path('results/optimization/20261104-qk-edges-v61/cases');rows=[]
for n,l in [(988,0),(989,0),(1009,0),(1057,0),(1008,1),(1008,6),(1008,8),(1008,16),(1008,22)]:
 a=np.fromfile(d/f'N{n}-L{l}-M0.bin',np.float32).astype(np.float64);b=np.fromfile(d/f'N{n}-L{l}-M3.bin',np.float32).astype(np.float64)
 rel=float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-30));maximum=float(np.max(np.abs(b-a)))
 assert a.shape==b.shape and np.all(np.isfinite(b)) and rel<2e-5,(n,l,rel,maximum)
 if l in [6,22]:assert np.array_equal(a,b),(n,l)
 rows.append({'N':n,'layout':l,'relative_l2':rel,'max_abs':maximum,'bitwise_equal':bool(np.array_equal(a,b))})
(d/'numerical-check.json').write_text(json.dumps(rows,indent=2))
PY
SH
chmod +x ocr-adaptation/check-qk-edges-v61 ocr-adaptation/bench-qk-edges-v61
taskset -c 0-7 python3 -u scripts/optimization_round.v50.py --label 20261104-qk-edges-v61 \
 --checker ocr-adaptation/check-qk-edges-v61 --checker-only --checker-timeout 180 --checker-separate-stderr \
 --spacemit-core-ids 8,9,10,11,13,14,15 --min-free-gib 4
