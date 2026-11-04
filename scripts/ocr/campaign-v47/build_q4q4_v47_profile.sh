set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
cd "$ROOT/ocr-adaptation/managed-source"
test "$(git rev-parse HEAD)" = 97e5c63240cef58b9751f08868f5cc040bfda6fc
test -z "$(git status --porcelain)"
export RISCV_ROOT_PATH=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
export SPERT_DIR="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
export OCR_BUILD_DIR="$ROOT/build-ocr-managed-profile-riscv64" OCR_PROFILE=ON OCR_BUILD_JOBS=48
bash scripts/ocr/build-k3.sh > "$ROOT/ocr-adaptation/source-management/v47-profile-build.log" 2>&1
python3 - <<'PY'
from pathlib import Path
import os,shutil,subprocess,json,hashlib
root=Path('/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR');b=Path(os.environ['OCR_BUILD_DIR'])
r=root/'ocr-adaptation/runtime-q4q4-concat-tcm-v47-profile';r.mkdir();(r/'bin').mkdir()
shutil.copytree(root/'ocr-adaptation/runtime-q4q4-concat-tcm-v47/lib',r/'lib')
for f in (b/'bin').glob('*.so*'):
 if f.is_file():shutil.copy2(f,r/'lib'/f.name)
shutil.copy2(b/'bin/llama-server',r/'bin/llama-server')
shutil.copy2(b/'OCR_SOURCE_MANIFEST.json',r/'OCR_SOURCE_MANIFEST.json')
m={'candidate':'q4q4-concat-tcm-v47-profile','experimental':True,'diagnostic':True,
 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'source_clean':not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),
 'build_flags':{'GGML_BUILD_PROFILE':True},'scope':'full operator diagnostic only',
 'files':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in r.rglob('*') if f.is_file()}}
assert m['source_clean'];(r/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(m['source_commit'])
PY
cd "$ROOT/ocr-adaptation"
tar -czf spacemit-ocr-q4q4-concat-tcm-v47-profile-riscv64.tar.gz runtime-q4q4-concat-tcm-v47-profile
sha256sum spacemit-ocr-q4q4-concat-tcm-v47-profile-riscv64.tar.gz
