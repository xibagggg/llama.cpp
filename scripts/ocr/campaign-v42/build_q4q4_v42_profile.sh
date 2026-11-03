set -euo pipefail
ROOT=/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR
cd "$ROOT/ocr-adaptation/managed-source"
export RISCV_ROOT_PATH=/home/huangjingtao/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.1.2
export SPERT_DIR="$ROOT/ocr-adaptation/deps/spine-runtime.riscv64.0.6.3"
export OCR_BUILD_JOBS=48
for mode in profile; do
 if test "$mode" = plain; then
  export OCR_BUILD_DIR="$ROOT/build-ocr-managed-riscv64" OCR_PROFILE=OFF
 else
  export OCR_BUILD_DIR="$ROOT/build-ocr-managed-profile-riscv64" OCR_PROFILE=ON
 fi
 bash scripts/ocr/build-k3.sh > "$ROOT/ocr-adaptation/source-management/v42-$mode-build.log" 2>&1
 OCR_PACK_MODE=$mode python3 - <<'PY'
from pathlib import Path
import shutil,hashlib,json,subprocess,os
root=Path('/home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR');mode=os.environ['OCR_PACK_MODE']
build=Path(os.environ['OCR_BUILD_DIR'])
r=root/f'ocr-adaptation/runtime-q4q4-down-sched-v42-{mode}';assert not r.exists()
shutil.copytree(root/'ocr-adaptation/runtime-q4q4-normalize-v37',r)
for f in (build/'bin').glob('*.so*'):
 if f.is_file():shutil.copy2(f,r/'lib'/f.name)
shutil.copy2(build/'bin/llama-server',r/'bin/llama-server')
shutil.copy2(build/'OCR_SOURCE_MANIFEST.json',r/'OCR_SOURCE_MANIFEST.json')
(r/'MANIFEST.json').unlink()
m={'candidate':f'q4q4-boundary-v42-{mode}','experimental':True,'diagnostic':mode=='profile',
 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'build_flags':{'GGML_BUILD_PROFILE':mode=='profile'},
 'scope':'v42 paired down cells over v41 TCM16; operator profile diagnostic only',
 'files':{str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in r.rglob('*') if f.is_file()}}
(r/'MANIFEST.json').write_text(json.dumps(m,indent=2));print(json.dumps(m['source_commit']))
PY
 (cd "$ROOT/ocr-adaptation" && tar -czf "spacemit-ocr-q4q4-down-sched-v42-$mode-riscv64.tar.gz" "runtime-q4q4-down-sched-v42-$mode" && sha256sum "spacemit-ocr-q4q4-down-sched-v42-$mode-riscv64.tar.gz")
done
