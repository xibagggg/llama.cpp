set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
taskset -c 0-7 python3 -u scripts/optimization_round.v47.py \
 --label 20261104-q4q4-v47-quality-selected \
 --runtime v47=ocr-adaptation/runtime-q4q4-concat-tcm-v47 \
 --precision q4 --text-precision q4 --f16-m4-cache --down-tcm-tile 16 --down-schedule paired \
 --down-compact-tile 20 --fa-lazy-init --down-vector 5 --fa-direct 2 --concat-tcm 4 \
 --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize --ime2-down-layers "$(seq -s, 14 26)" \
 --fa-x8 --fa-pack --spacemit-core-ids 8,9,10,11,13,14,15 \
 --case-file images/optimization-v5/cases.json \
 --cases 03_table_gb_standard v5_table_merge4 chinese chinese_skew3 01_text_putnam 02_formula_putnam 04_scanned_civil_trial \
 --warmups 0 --repeats 1 --strict-quality-evidence --min-free-gib 4
