set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
cd "$ROOT/ocr-adaptation/managed-source"
git diff --check
git add ggml/src/ggml-spacemit/ocr-fa-pack-v40.h ggml/src/ggml-spacemit/ocr-fa-x8-v39.h ggml/src/ggml-spacemit/rvv_kernels.cpp
git add -f tests/ocr/bench_down_pad_v50.cpp
git add ggml/src/ggml-spacemit/ocr-qk-ime2-v53.h
git add ggml/src/ggml-spacemit/ggml-spacemit.cpp ggml/src/ggml-spacemit/ocr-fa-phases-v50.h
if ! git diff --cached --quiet;then git commit -m 'perf(ocr): reuse IME2 key panels across sixteen attention queries';fi
export RISCV_ROOT_PATH=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
export SPERT_DIR="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
export OCR_BUILD_DIR="$ROOT/build-ocr-managed-riscv64" OCR_PROFILE=OFF OCR_BUILD_JOBS=48
bash scripts/ocr/build-k3.sh > "$ROOT/ocr-adaptation/source-management/v61-build.log" 2>&1
"$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++" -std=c++17 -O3 -DNDEBUG -march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d \
 -Iggml/include -Iggml/src -Iggml/src/ggml-spacemit -I"$SPERT_DIR/include" tests/ocr/bench_fa_group_v45.cpp \
 "$OCR_BUILD_DIR/ggml/src/ggml-spacemit/CMakeFiles/ggml-spacemit.dir/rvv_kernels.cpp.o" \
 -L"$OCR_BUILD_DIR/bin" -L"$SPERT_DIR/lib" -Wl,-rpath-link,"$OCR_BUILD_DIR/bin" -Wl,-rpath-link,"$SPERT_DIR/lib" \
 -lggml-spacemit -lggml-cpu -lggml-base -lspert -latomic -pthread -o "$ROOT/ocr-adaptation/bench-fa-compute-v51"
python3 - <<'PY'
from pathlib import Path
import shutil,hashlib,json,subprocess,os
r=Path('/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR/ocr-adaptation/runtime-fa-compute-v61');assert not r.exists()
b=Path(os.environ['OCR_BUILD_DIR']);shutil.copytree(r.parent/'runtime-q4-down-v50',r)
for p in (b/'bin').glob('*.so*'):
 if p.is_file():shutil.copy2(p,r/'lib'/p.name)
shutil.copy2(b/'bin/llama-server',r/'bin/llama-server');shutil.copy2(r.parent/'bench-fa-compute-v51',r/'bin/bench-fa-compute-v51')
shutil.copy2(b/'OCR_SOURCE_MANIFEST.json',r/'OCR_SOURCE_MANIFEST.json');(r/'MANIFEST.json').unlink()
m={'candidate':'fa-compute-v61','experimental':True,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_clean':not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'files':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file()}}
assert m['source_clean'];(r/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(m['source_commit'])
PY
cd "$ROOT/ocr-adaptation"
tar -czf spacemit-ocr-fa-compute-v61-riscv64.tar.gz runtime-fa-compute-v61
