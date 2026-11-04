set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
cd "$ROOT/ocr-adaptation/managed-source"
git diff --check
git add tools/mtmd/clip.cpp ggml/src/ggml-spacemit/rvv_kernels.cpp ggml/src/ggml-spacemit/ocr-fa-phases-v50.h
git add -f scripts/ocr/pad-ocr-down.py
git commit -m 'feat(ocr): enable padded K4304 Q4 down on IME2 and measure attention phases'
export RISCV_ROOT_PATH=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
export SPERT_DIR="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
export OCR_BUILD_DIR="$ROOT/build-ocr-managed-riscv64" OCR_PROFILE=OFF OCR_BUILD_JOBS=48
bash scripts/ocr/build-k3.sh > "$ROOT/ocr-adaptation/source-management/v50-build.log" 2>&1
python3 - <<'PY'
from pathlib import Path
import shutil,hashlib,json,subprocess,os
r=Path('/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR/ocr-adaptation/runtime-q4-down-v50');assert not r.exists()
b=Path(os.environ['OCR_BUILD_DIR'])
shutil.copytree(r.parent/'runtime-q4q4-concat-tcm-v47',r)
for p in (b/'bin').glob('*.so*'):
 if p.is_file():shutil.copy2(p,r/'lib'/p.name)
shutil.copy2(b/'bin/llama-server',r/'bin/llama-server')
shutil.copy2(b/'OCR_SOURCE_MANIFEST.json',r/'OCR_SOURCE_MANIFEST.json')
(r/'MANIFEST.json').unlink()
m={'candidate':'q4-down-v50','experimental':True,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_clean':not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'files':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file()}}
assert m['source_clean'];(r/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(m['source_commit'])
PY
cd "$ROOT/ocr-adaptation"
tar -czf spacemit-ocr-q4-down-v50-riscv64.tar.gz runtime-q4-down-v50
sha256sum spacemit-ocr-q4-down-v50-riscv64.tar.gz
