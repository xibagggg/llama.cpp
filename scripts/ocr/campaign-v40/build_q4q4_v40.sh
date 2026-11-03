set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
cd "$ROOT/ocr-adaptation/managed-source"
git diff --check
git add ggml/src/ggml-spacemit/ocr-fa-pack-v40.h ggml/src/ggml-spacemit/rvv_kernels.cpp ggml/src/ggml-spacemit/rvv_kernels.h ggml/src/ggml-spacemit/ime.cpp
git add -f tests/ocr/bench_fa_x8_v39.cpp
if ! git diff --cached --quiet; then git commit -m 'perf(ocr): reuse transient K/V packing across OCR attention query tiles'; fi
export RISCV_ROOT_PATH=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
export SPERT_DIR="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
export OCR_BUILD_DIR="$ROOT/build-ocr-managed-riscv64" OCR_PROFILE=OFF OCR_BUILD_JOBS=48
bash scripts/ocr/build-k3.sh > "$ROOT/ocr-adaptation/source-management/v40-build.log" 2>&1
"$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++" -std=c++17 -O3 -DNDEBUG \
 -march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d \
 -Iggml/include -Iggml/src -Iggml/src/ggml-spacemit -I"$SPERT_DIR/include" \
 tests/ocr/bench_fa_x8_v39.cpp "$OCR_BUILD_DIR/ggml/src/ggml-spacemit/CMakeFiles/ggml-spacemit.dir/rvv_kernels.cpp.o" \
 -L"$OCR_BUILD_DIR/bin" -L"$SPERT_DIR/lib" \
 -Wl,-rpath-link,"$OCR_BUILD_DIR/bin" -Wl,-rpath-link,"$SPERT_DIR/lib" \
 -lggml-spacemit -lggml-cpu -lggml-base -lspert -latomic -pthread -o "$OCR_BUILD_DIR/bin/bench-fa-x8-v40" \
 > "$ROOT/ocr-adaptation/source-management/v40-micro-build.log" 2>&1
python3 - <<'PY'
from pathlib import Path
import shutil,hashlib,json,subprocess,os
root=Path('/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR');build=Path(os.environ['OCR_BUILD_DIR'])
r=root/'ocr-adaptation/runtime-q4q4-fa-pack-v40';assert not r.exists()
shutil.copytree(root/'ocr-adaptation/runtime-q4q4-boundary-v38-plain',r)
for f in (build/'bin').glob('*.so*'):
 if f.is_file():shutil.copy2(f,r/'lib'/f.name)
for name in ('llama-server','bench-fa-x8-v40'):shutil.copy2(build/'bin'/name,r/'bin'/name)
shutil.copy2(build/'OCR_SOURCE_MANIFEST.json',r/'OCR_SOURCE_MANIFEST.json')
(r/'MANIFEST.json').unlink()
m={'candidate':'q4q4-fa-pack-v40','experimental':True,'diagnostic':False,
 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'build_flags':{'GGML_BUILD_PROFILE':False},'scope':'opt-in D72 per-call K/V packing shared across query tiles, with optional eight-query reuse; prepare cost included in benchmark',
 'files':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in r.rglob('*') if f.is_file()}}
(r/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(m['source_commit'])
PY
cd "$ROOT/ocr-adaptation"
tar -czf spacemit-ocr-q4q4-fa-pack-v40-riscv64.tar.gz runtime-q4q4-fa-pack-v40
sha256sum spacemit-ocr-q4q4-fa-pack-v40-riscv64.tar.gz
