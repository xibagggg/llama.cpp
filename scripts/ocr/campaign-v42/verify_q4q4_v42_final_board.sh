set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
python3 - <<'PY'
from pathlib import Path
import json,os,socket,hashlib,urllib.request,shutil
root=Path.cwd();base=root/'results/optimization'
ref=json.loads((base/'20261104-q4q4-v42-down-micro/round.json').read_text())['original']
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
  if (exe=='llama-server' and '/runtime-q4q4-down-sched-v42' in args[0]) or (exe.startswith('python') and any(a.endswith('optimization_round.v42.py') for a in args[1:])):tests.append({'pid':int(p.parent.name),'cmd':args})
 except OSError:pass
assert not tests,tests
models={}
for flag in ['--model','--mmproj']:
 path=Path(argv[argv.index(flag)+1]);h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 models[path.name]={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
archives={}
for mode,want in [('', '0e12c9ccd2b1b2eb217baf44eae9d92d35d2e3f952f75f0e3b56952ba865e847'),('-profile','fb4786ff3ffbc4462c282a64d9d7ce7b8c7019198a42764f5552c1a9f9d4ad49')]:
 path=root/f'ocr-adaptation/spacemit-ocr-q4q4-down-sched-v42{mode}-riscv64.tar.gz'
 got=hashlib.sha256(path.read_bytes()).hexdigest();assert got==want
 archives[path.name]=got
 manifest=json.loads((root/f'ocr-adaptation/runtime-q4q4-down-sched-v42{mode}/OCR_SOURCE_MANIFEST.json').read_text())
 assert manifest['source_clean'] and manifest['commit']=='63ad486a6973eeb345fc1a0fdae43b094ed2dc44'
state={'pid':pid,'health':health,'original_argv_restored':True,'original_cwd_restored':True,'original_runtime_environment_restored':True,'test_port_18582_idle':idle,'test_processes':tests,'free_gib':shutil.disk_usage(root).free/1024**3,'deployment_changed':False,'models':models,'runtime_archives_sha256':archives,'source_clean_manifest_verified':True}
out=base/'20261104-q4q4-v42-provenance/final-board-state.json';out.write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
PY
