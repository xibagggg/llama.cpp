set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
A=ocr-adaptation/spacemit-ocr-q4q4-fa-pack-v40-profile-riscv64.tar.gz
echo "3b950ff0b964e1856cd53698ec420cfd656a4473238a7a1c333294583863d184  $A" | sha256sum -c -
test ! -e ocr-adaptation/runtime-q4q4-fa-pack-v40-profile
tar -xzf "$A" -C ocr-adaptation
taskset -c 0-7 python3 -u scripts/optimization_round.v40.py \
 --label 20261104-q4q4-v40-profile-combined \
 --runtime v40=ocr-adaptation/runtime-q4q4-fa-pack-v40-profile \
 --precision q4 --text-precision q4 --f16-m4-cache \
 --ime2-fp16-down --ime2-fp16-compensate --ime2-normalize \
 --ime2-down-layers "$(seq -s, 14 26)" --fa-x8 --fa-pack --trace \
 --spacemit-core-ids 8,9,10,11,13,14,15 \
 --case-file images/optimization-v5/cases.json --cases v5_table_merge4 \
 --warmups 0 --repeats 1 --strict-quality-evidence --min-free-gib 4
