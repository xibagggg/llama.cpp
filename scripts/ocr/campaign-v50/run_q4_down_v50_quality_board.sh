set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
common=(--runtime v50=ocr-adaptation/runtime-q4-down-v50 --precision q4 --text-precision q4 --f16-m4-cache
 --down-tcm-tile 16 --down-schedule paired --down-compact-tile 20 --fa-lazy-init
 --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize --ime2-down-layers 14,15,16,17,18,19,20,21,22,23,24,25,26
 --fa-x8 --fa-pack --spacemit-core-ids 8,9,10,11,13,14,15 --down-vector 5 --fa-direct 2 --concat-tcm 4
 --case-file images/optimization-v5/cases.json --warmups 0 --repeats 1 --strict-quality-evidence --min-free-gib 4)
for kind in baseline q4;do
 opts=();if [[ $kind == q4 ]];then opts+=(--padded-down --mmproj-model models/mmproj-model-q4_0-down-pad4352-all-v50.gguf);fi
 taskset -c 0-7 python3 -u scripts/optimization_round.v50.py --label "20261104-q4-down-v50-quality-$kind" "${common[@]}" "${opts[@]}" \
  --cases chinese chinese_skew3 01_text_putnam 02_formula_putnam 03_table_gb_standard v5_table_merge4 04_scanned_civil_trial v5_table_cn_clear v5_table_cn_blur v5_table_numbers5
done
