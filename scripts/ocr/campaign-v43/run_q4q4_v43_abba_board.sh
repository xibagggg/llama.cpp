set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('ocr-adaptation/runtime-q4q4-compact-lazy-v43');m=json.loads((r/'MANIFEST.json').read_text())
for name,h in m['files'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
out=Path('results/optimization/20261104-q4q4-v43-provenance');out.mkdir(exist_ok=False)
(out/'runtime-manifest.json').write_text(json.dumps(m,indent=2))
(out/'test-provenance.json').write_text(json.dumps({'runner_sha256':hashlib.sha256(Path('scripts/optimization_round.v43.py').read_bytes()).hexdigest(),'manifest_verified':True,'source_commit':m['source_commit']},indent=2))
PY
for arm in off-a on-a on-b off-b; do
 opts=()
 if [[ "$arm" == on-* ]]; then opts+=(--down-compact-tile 20 --fa-lazy-init); fi
 taskset -c 0-7 python3 -u scripts/optimization_round.v43.py  --label "20261104-q4q4-v43-abba-$arm"  --runtime v43=ocr-adaptation/runtime-q4q4-compact-lazy-v43  --precision q4 --text-precision q4 --f16-m4-cache --down-tcm-tile 16 --down-schedule paired  --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize  --ime2-down-layers "$(seq -s, 14 26)" --fa-x8 --fa-pack  --spacemit-core-ids 8,9,10,11,13,14,15  --case-file images/optimization-v5/cases.json "${opts[@]}"  --cases v5_table_merge4 03_table_gb_standard  --warmups 1 --repeats 3 --strict-quality-evidence --min-free-gib 4
done
