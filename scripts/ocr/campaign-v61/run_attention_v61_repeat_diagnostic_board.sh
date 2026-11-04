set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
common=(--padded-down --mmproj-model models/mmproj-model-q4_0-down-pad4352-all-v50.gguf --precision q4 --text-precision q4 --f16-m4-cache
 --down-tcm-tile 16 --down-schedule paired --down-compact-tile 20 --fa-lazy-init
 --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize --ime2-down-layers 14,15,16,17,18,19,20,21,22,23,24,25,26
 --fa-x8 --fa-pack --spacemit-core-ids 8,9,10,11,13,14,15 --down-vector 5 --fa-direct 2 --concat-tcm 4
 --case-file images/optimization-v5/cases.json --warmups 1 --repeats 3 --strict-quality-evidence --min-free-gib 4 --fa-compute 0)
for version in v50 v61;do
 runtime=ocr-adaptation/runtime-q4-down-v50
 if [[ $version == v61 ]];then runtime=ocr-adaptation/runtime-fa-compute-v61;fi
 taskset -c 0-7 python3 -u scripts/optimization_round.v60.py --label "20261104-attention-v61-repeat-diagnostic-$version" --runtime "$version=$runtime" "${common[@]}" --cases 03_table_gb_standard
done
