set -euo pipefail
cd /home/bianbu/MiniCPM-OCR-1.0
python3 - <<'PY'
from pathlib import Path
import json,os,socket,hashlib,urllib.request,shutil
root=Path.cwd();base=root/'results/optimization';out=base/'20261104-attention-v61';out.mkdir(exist_ok=True)
ref=json.loads((base/'20261104-q4-down-v50-abba-off-a/round.json').read_text())['original']
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
  is_runtime=exe=='llama-server' and any(s in args[0] for s in ['/runtime-q4-down-v50/','/runtime-fa-compute-v'])
  is_runner=exe.startswith('python') and any(a.endswith(('optimization_round.v50.py','optimization_round.v54.py','optimization_round.v60.py')) for a in args[1:])
  if is_runtime or is_runner:tests.append({'pid':int(p.parent.name),'cmd':args})
 except OSError:pass
assert not tests,tests
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
models={}
for name,want in {
 'MiniCPM-OCR-1_0-Q4_0.gguf':'9c193d8e51755a52d759f6c6c8ec4793243226b2c9e7ee0650cd990627aae87c',
 'mmproj-model-q4_0-4x.gguf':'c0e426a7a029a238ee0b9c30d4fa2c5fca8aa10ae146de3cc760bac550525be1'}.items():
 p=root/'models'/name;got=sha(p);assert got==want;models[name]={'bytes':p.stat().st_size,'sha256':got}
for kind in ['all','none']:
 p=root/f'models/mmproj-model-q4_0-down-pad4352-{kind}-v50.gguf';m=json.loads(p.with_suffix('.manifest.json').read_text())
 got=sha(p);assert got==m['output_sha256'] and m['all_unchanged_tensors_verified']
 assert len(m['changed'])==27 and len(m['q4_layers'])==(27 if kind=='all' else 0)
 models[p.name]={'bytes':p.stat().st_size,'sha256':got,'conversion':m,'manifest_sha256':sha(p.with_suffix('.manifest.json'))}
runtimes={}
for name,commit in [('runtime-q4-down-v50','54afbdeaa7709cacdf061171bc107d7bd1257347'),('runtime-fa-compute-v61','67de3f567ddef0deea4d37591d8656c0e6cb6c7c')]:
 r=root/'ocr-adaptation'/name;m=json.loads((r/'MANIFEST.json').read_text())
 assert m['source_clean'] and m['source_commit']==commit
 for file,want in m['files'].items():assert sha(r/file)==want,(name,file)
 runtimes[name]={'manifest':m,'manifest_sha256':sha(r/'MANIFEST.json'),'all_files_verified':True}
state={'pid':pid,'health':health,'original_argv_restored':True,'original_cwd_restored':True,'original_runtime_environment_restored':True,
 'test_port_18582_idle':idle,'test_processes':tests,'free_gib':shutil.disk_usage(root).free/1024**3,'deployment_changed':False,
 'models':models,'runtimes':runtimes}
(out/'final-board-state.json').write_text(json.dumps(state,indent=2))
print(json.dumps({k:v for k,v in state.items() if k not in ['models','runtimes']},indent=2))
PY
