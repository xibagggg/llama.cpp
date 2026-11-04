set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
python3 - <<'PY'
from pathlib import Path
import json,os,socket,hashlib,urllib.request,shutil,re
root=Path.cwd();base=root/'results/optimization'
ref=json.loads((base/'20261104-q4q4-v44-micro/round.json').read_text())['original']
pid=int((root/'logs/server.pid').read_text());proc=Path('/proc')/str(pid)
argv=(proc/'cmdline').read_bytes().decode().strip('\0').split('\0');cwd=os.readlink(proc/'cwd')
env=dict(v.split('=',1) for v in (proc/'environ').read_bytes().decode().strip('\0').split('\0'))
runtime_env={k:v for k,v in env.items() if k.startswith(('GGML_','SPACEMIT_','MTMD_'))}
assert argv==ref['argv'] and cwd==ref['cwd'] and runtime_env==ref['runtime_environment']
with socket.socket() as sock:idle=sock.connect_ex(('127.0.0.1',18582))!=0
assert idle
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
health=json.load(opener.open('http://127.0.0.1:18581/health',timeout=10));assert health['status']=='ok'
tests=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().decode(errors='replace').strip('\0').split('\0');exe=Path(args[0]).name
  is_runtime=exe=='llama-server' and bool(re.search(r'/runtime-q4q4-(vector-direct-v44|query-group-v45|concat-tcm-v4[67])',args[0]))
  is_runner=exe.startswith('python') and any(re.search(r'optimization_(round|long_stress|cache_test)\.v4[4567]\.py$',a) for a in args[1:])
  if is_runtime or is_runner:tests.append({'pid':int(p.parent.name),'cmd':args})
 except OSError:pass
assert not tests,tests
models={}
want_models={'MiniCPM-OCR-1_0-Q4_0.gguf':'9c193d8e51755a52d759f6c6c8ec4793243226b2c9e7ee0650cd990627aae87c','mmproj-model-q4_0-4x.gguf':'c0e426a7a029a238ee0b9c30d4fa2c5fca8aa10ae146de3cc760bac550525be1'}
for flag in ['--model','--mmproj']:
 path=Path(argv[argv.index(flag)+1]);h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==want_models[path.name]
 models[path.name]={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
archives={}
for runtime,want,commit in [
 ('vector-direct-v44','dbe2aeb328bd643ea2bf9eab184a395cf13cd010d4305a969790760bcd3d6eeb','6caaa5ff3cc1a51fc28f8e537eae13c5ae1728c0'),
 ('query-group-v45','b3cb8a7b8fd0d8f0c9d93715ad67f83c662c69960fc2f498c9bbf50e38690e2a','c4b230a57e3d6a2befd5eba358a3776770052b0e'),
 ('concat-tcm-v46','93e7d340030cfafdc8ef9fcec6315a9921a72a3b9a5e379f3e14cbc096fd226e','c8defb632f796f0fc9bb167af002295ead8953a2'),
 ('concat-tcm-v47','fbd32444350afa9152bf0341047e9e881018ac945a8d603518ab1c33ca49a595','97e5c63240cef58b9751f08868f5cc040bfda6fc'),
 ('concat-tcm-v47-profile','a66b98bfa039bc0bc1e64a7ffa93f49928c464ec736142c90c089c36e6377715','97e5c63240cef58b9751f08868f5cc040bfda6fc')]:
 path=root/f'ocr-adaptation/spacemit-ocr-q4q4-{runtime}-riscv64.tar.gz'
 got=hashlib.sha256(path.read_bytes()).hexdigest();assert got==want
 archives[path.name]=got
 r=root/f'ocr-adaptation/runtime-q4q4-{runtime}'
 manifest=json.loads((r/'OCR_SOURCE_MANIFEST.json').read_text())
 assert manifest['source_clean'] and manifest['commit']==commit
 for name,h in json.loads((r/'MANIFEST.json').read_text())['files'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,(runtime,name)
state={'pid':pid,'health':health,'original_argv_restored':True,'original_cwd_restored':True,'original_runtime_environment_restored':True,'test_port_18582_idle':idle,'test_processes':tests,'free_gib':shutil.disk_usage(root).free/1024**3,'deployment_changed':False,'models':models,'runtime_archives_sha256':archives,'source_clean_manifest_verified':True}
out=base/'20261104-q4q4-long-1430/final-board-state.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
PY
