set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
cd "$ROOT/ocr-adaptation/managed-source"
test "$(git rev-parse HEAD)" = c8defb632f796f0fc9bb167af002295ead8953a2
git diff --check
git add ggml/src/ggml-spacemit/ocr-concat-tcm-v46.h
git add -f tests/ocr/bench_concat_tcm_v47.cpp
git commit -m 'perf(ocr): gate concat TCM staging to full prefill writes'
export RISCV_ROOT_PATH=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
export SPERT_DIR="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
export OCR_BUILD_DIR="$ROOT/build-ocr-managed-riscv64" OCR_PROFILE=OFF OCR_BUILD_JOBS=48
bash scripts/ocr/build-k3.sh > "$ROOT/ocr-adaptation/source-management/v47-build.log" 2>&1
for kind in concat_tcm; do
 extra=()
 if [[ "$kind" == concat_tcm ]]; then extra+=("$OCR_BUILD_DIR/ggml/src/ggml-spacemit/CMakeFiles/ggml-spacemit.dir/rvv_kernels.cpp.o"); fi
 "$RISCV_ROOT_PATH/bin/riscv64-unknown-linux-gnu-g++" -std=c++17 -O3 -DNDEBUG \
  -march=rv64gcv_zfh_zvfh_zba_zicbop -mabi=lp64d \
  -Iggml/include -Iggml/src -Iggml/src/ggml-spacemit -I"$SPERT_DIR/include" \
  "tests/ocr/bench_${kind}_v47.cpp" "${extra[@]}" \
  -L"$OCR_BUILD_DIR/bin" -L"$SPERT_DIR/lib" \
  -Wl,-rpath-link,"$OCR_BUILD_DIR/bin" -Wl,-rpath-link,"$SPERT_DIR/lib" \
  -lggml-spacemit -lggml-cpu -lggml-base -lspert -latomic -pthread -o "$OCR_BUILD_DIR/bin/bench-${kind//_/-}-v47" \
  > "$ROOT/ocr-adaptation/source-management/v47-${kind}-build.log" 2>&1
done
python3 - <<'PY'
from pathlib import Path
import shutil,hashlib,json,subprocess,os
root=Path('/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR');build=Path(os.environ['OCR_BUILD_DIR'])
r=root/'ocr-adaptation/runtime-q4q4-concat-tcm-v47';assert not r.exists()
shutil.copytree(root/'ocr-adaptation/runtime-q4q4-concat-tcm-v46',r)
for f in (build/'bin').glob('*.so*'):
 if f.is_file():shutil.copy2(f,r/'lib'/f.name)
for name in ('llama-server','bench-concat-tcm-v47',):shutil.copy2(build/'bin'/name,r/'bin'/name)
shutil.copy2(build/'OCR_SOURCE_MANIFEST.json',r/'OCR_SOURCE_MANIFEST.json')
(r/'MANIFEST.json').unlink()
m={'candidate':'q4q4-concat-tcm-v47','experimental':True,'diagnostic':False,
 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'source_clean':not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),
 'build_flags':{'GGML_BUILD_PROFILE':False},
 'scope':'bounded TCM transpose and contiguous DDR writeback for text concat',
 'files':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in r.rglob('*') if f.is_file()}}
assert m['source_clean'];(r/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(m['source_commit'])
PY
cd "$ROOT/ocr-adaptation"
tar -czf spacemit-ocr-q4q4-concat-tcm-v47-riscv64.tar.gz runtime-q4q4-concat-tcm-v47
sha256sum spacemit-ocr-q4q4-concat-tcm-v47-riscv64.tar.gz
