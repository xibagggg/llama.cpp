set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
cd "$ROOT/ocr-adaptation/managed-source"
git diff --check
git add ggml/src/ggml-spacemit/ocr-down-sched-v42.h
git add -f tests/ocr/bench_down_sched_v42.cpp
git commit -m 'perf(ocr): evaluate exact M4 load scheduling and 2x2 down cells'
export RISCV_ROOT_PATH=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
export SPERT_DIR="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
export OCR_BUILD_DIR="$ROOT/build-ocr-managed-riscv64"
"$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++" -std=c++17 -O3 -DNDEBUG \
 -march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d \
 -Iggml/include -Iggml/src -Iggml/src/ggml-spacemit -I"$SPERT_DIR/include" \
 tests/ocr/bench_down_sched_v42.cpp -L"$OCR_BUILD_DIR/bin" -L"$SPERT_DIR/lib" \
 -Wl,-rpath-link,"$OCR_BUILD_DIR/bin" -Wl,-rpath-link,"$SPERT_DIR/lib" \
 -lggml-spacemit -lggml-cpu -lggml-base -lspert -latomic -pthread \
 -o "$OCR_BUILD_DIR/bin/bench-down-sched-v42" > "$ROOT/ocr-adaptation/source-management/v42-micro-build.log" 2>&1
python3 - <<'PY'
from pathlib import Path
import shutil,json,hashlib,subprocess
root=Path('/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR')
r=root/'ocr-adaptation/runtime-q4q4-down-sched-v42-micro';assert not r.exists()
shutil.copytree(root/'ocr-adaptation/runtime-q4q4-down-tcm-v41',r)
shutil.copy2(root/'build-ocr-managed-riscv64/bin/bench-down-sched-v42',r/'bin/bench-down-sched-v42')
base=json.loads((r/'MANIFEST.json').read_text());(r/'MANIFEST.json').unlink()
m={'candidate':'v42-micro','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'source_clean':not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),
 'runtime_libs_source_commit':base['source_commit'],'scope':'Standalone v42 micro only; copied server/libraries remain frozen v41',
 'files':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in r.rglob('*') if f.is_file()}}
assert m['source_clean'];(r/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(m['source_commit'])
PY
cd "$ROOT/ocr-adaptation"
tar -czf spacemit-ocr-q4q4-down-sched-v42-micro-riscv64.tar.gz runtime-q4q4-down-sched-v42-micro
sha256sum spacemit-ocr-q4q4-down-sched-v42-micro-riscv64.tar.gz
