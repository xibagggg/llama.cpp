set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
python3 - <<'PY'
from pathlib import Path
import hashlib,json
r=Path('ocr-adaptation/runtime-q4q4-concat-tcm-v47');m=json.loads((r/'MANIFEST.json').read_text());assert m['source_clean']
assert m['source_commit']=='97e5c63240cef58b9751f08868f5cc040bfda6fc' and not m['build_flags']['GGML_BUILD_PROFILE']
for name,h in m['files'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
p=Path('results/optimization/20261104-q4q4-v47-main-stress-provenance');p.mkdir(exist_ok=False)
(p/'provenance.json').write_text(json.dumps({'manifest':m,'runner_sha256':hashlib.sha256(Path('scripts/optimization_long_stress.v47.py').read_bytes()).hexdigest(),'manifest_verified':True},indent=2))
PY
taskset -c 0-7 python3 -u scripts/optimization_long_stress.v47.py \
 --label 20261104-q4q4-v47-long-stress-main --runtime v47=ocr-adaptation/runtime-q4q4-concat-tcm-v47 \
 --precision q4 --text-precision q4 --f16-m4-cache --down-tcm-tile 16 --down-schedule paired \
 --down-compact-tile 20 --fa-lazy-init --down-vector 5 --fa-direct 2 --concat-tcm 4 \
 --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize --ime2-down-layers "$(seq -s, 14 26)" \
 --fa-x8 --fa-pack --spacemit-core-ids 8,9,10,11,13,14,15 \
 --case-file images/optimization-v5/cases.json \
 --cases 03_table_gb_standard v5_table_merge4 chinese chinese_skew3 01_text_putnam 02_formula_putnam 04_scanned_civil_trial \
 --request-deadline 2026-11-04T14:15:00+08:00 \
 --quality-reference results/optimization/20261104-q4q4-v43-quality-selected/round.json \
 --telemetry-interval 3 --warmups 1 --repeats 1000 --strict-quality-evidence --min-free-gib 4
