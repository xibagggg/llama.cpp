#!/usr/bin/env python3
"""Parent-controlled candidate test. Never modifies an engine or deployment."""
import argparse, datetime, fcntl, hashlib, importlib.util, json, os, pathlib, shlex, shutil, subprocess, threading, time

parser=argparse.ArgumentParser()
parser.add_argument('--reps',type=int,default=3)
parser.add_argument('--lock-held',action='store_true',help='Parent already holds SDK benchmark lock')
parser.add_argument('--timeout',type=int,default=240)
parser.add_argument('--smi-hold-ms',type=int,default=1500)
args=parser.parse_args()
base=pathlib.Path(__file__).resolve().parent
sdk=pathlib.Path('/home/bianbu/minicpm/model-zoo-ocr')
engine=pathlib.Path('/home/bianbu/minicpm/minicpm\u4f18\u5316/src/llama.cpp')
os.sched_setaffinity(0,set(range(6)))
lock=None
if not args.lock_held:
    lock=(sdk/'output/.minicpm-benchmark.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
stamp=datetime.datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')
result=sdk/'output/minicpm_ocr/candidates/gdn-snapshot-direct-20261118/runs'/stamp
result.mkdir(parents=True,exist_ok=False)
env=os.environ.copy()
env.update({'GGML_SPACEMIT_SHARED_MODE':'PINNED_TCM','GGML_SPACEMIT_TCM_OWNERSHIP':'1',
    'GGML_SPACEMIT_WORKERS':'6','SPACEMIT_PERFER_CORE_ID':'8,9,10,11,12,13',
    'GGML_SPACEMIT_STRICT_AI_CORES':'1','SPERT_SHARED_MODE':'TCM','SPERT_BACKEND':'spacemit-k3',
    'SPERT_HOST_PREPARE_STACKS':'1','SPERT_CTX_RVV':'0','SPERT_HOT_SPIN':'0','OMP_NUM_THREADS':'6'})
env.pop('GGML_SPACEMIT_N_THREADS',None)
env.pop('GGML_SPACEMIT_GDN_DEBUG',None)
flags_file=engine/'build/ggml/src/ggml-spacemit/CMakeFiles/ggml-spacemit.dir/flags.make'
flags_raw=flags_file.read_bytes()
flags_line=next(line.partition('=')[2].strip() for line in flags_raw.decode().splitlines() if line.startswith('CXX_FLAGS ='))
machine_flags=[flag for flag in shlex.split(flags_line) if flag.startswith(('-march=','-mabi='))]
assert any(flag.startswith('-march=') for flag in machine_flags)
command=['/usr/bin/g++','-std=gnu++17','-O3','-UNDEBUG',*machine_flags,str(base/'snapshot-bench.cpp'),
    '-I'+str(base/'fixtures'),'-I'+str(base),'-I/usr/include','-L/usr/lib','-Wl,-rpath,/usr/lib',
    '-lspert','-latomic','-ldl','-pthread','-o',str(base/'snapshot-bench')]
metadata={'status':'BUILDING','created':datetime.datetime.now().astimezone().isoformat(),
    'host_cpus':sorted(os.sched_getaffinity(0)),'ai_cpus':[8,9,10,11,12,13],
    'compile_command':command,'flags_file':str(flags_file),'flags_sha256':hashlib.sha256(flags_raw).hexdigest(),
    'source_sha256':{str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in base.rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.patch','.py') and
        'original' not in p.parts and 'modified' not in p.parts and p.name!='fetch_readonly.py'},
    'environment':{k:v for k,v in env.items() if k.startswith(('GGML_SPACEMIT','SPERT_','SPACEMIT_','OMP_NUM','LD_LIBRARY_PATH'))},
    'contract':'Frozen existing RVV K16 arithmetic + exact current row-CPY vs direct destination; full cache, attention, padding, inputs, guards. 48 cases.',
    'tcm_note':'Test claims physical private TCM with production lease; scalar T16 GDN scratch remains DDR under existing policy. No new TCM speed claim.'}
def save(): (result/'result.json').write_text(json.dumps(metadata,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
save()
sources=result/'sources';sources.mkdir()
for relative in metadata['source_sha256']:
    target=sources/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/relative,target)
(result/'flags.make').write_bytes(flags_raw)
build=subprocess.run(command,cwd=base,env=env,capture_output=True,timeout=120)
(result/'build.stdout.log').write_bytes(build.stdout);(result/'build.stderr.log').write_bytes(build.stderr)
metadata['build_returncode']=build.returncode
if build.returncode:
    metadata['status']='BUILD_FAILED';save();print(str(result),flush=True);raise SystemExit(build.returncode)
stop=threading.Event();samples=[]
def sample_tcm():
    with (result/'tcm-smi.log').open('w',encoding='utf-8') as log:
        while not stop.is_set():
            now=datetime.datetime.now().astimezone().isoformat()
            try:
                probe=subprocess.run(['spacemit-tcm-smi','-i'],capture_output=True,text=True,timeout=2)
                sample={'time':now,'returncode':probe.returncode,'stdout':probe.stdout,'stderr':probe.stderr}
            except subprocess.TimeoutExpired:
                sample={'time':now,'returncode':None,'stdout':'','stderr':'TIMEOUT'}
            samples.append(sample);log.write(now+'\n'+sample['stdout']+sample['stderr']+'\n');log.flush()
            stop.wait(.5)
thread=threading.Thread(target=sample_tcm,daemon=True);thread.start()
metadata['status']='RUNNING';save();began=time.monotonic();rows=[]
try:
    command=[str(base/'snapshot-bench'),str(max(1,args.reps)),str(max(0,min(3000,args.smi_hold_ms)))]
    metadata['run_command']=command
    run=subprocess.run(command,cwd=base,env=env,capture_output=True,timeout=args.timeout)
    (result/'stdout.jsonl').write_bytes(run.stdout);(result/'stderr.log').write_bytes(run.stderr)
    metadata['run_returncode']=run.returncode
    for line in run.stdout.decode('utf-8',errors='replace').splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue
        if 'request' in row: rows.append(row)
    metadata['cases']=len(rows)
    metadata['failed_cases']=sum(row.get('status')!='PASS' for row in rows)
    metadata['numerical_pass']=run.returncode==0 and len(rows)==48 and metadata['failed_cases']==0
    metadata['status']='PENDING_TCM_VALIDATION'
except subprocess.TimeoutExpired as exc:
    metadata['status']='TIMEOUT'
    for name,data in [('stdout.partial.log',exc.stdout),('stderr.partial.log',exc.stderr)]:
        (result/name).write_bytes(data or b'')
finally:
    stop.set();thread.join(timeout=3)
    (result/'tcm-smi-samples.json').write_text(json.dumps(samples,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    spec=importlib.util.spec_from_file_location('standalone_tcm_smi',base/'standalone-tcm-smi.py')
    smi=importlib.util.module_from_spec(spec);spec.loader.exec_module(smi)
    metadata['tcm_validation']=smi.validate(samples,rows)
    if metadata['status']=='PENDING_TCM_VALIDATION':
        metadata['status']='PASS' if metadata['numerical_pass'] and metadata['tcm_validation']['physical_tcm_verified'] else 'FAIL'
    metadata['elapsed_s']=time.monotonic()-began;save()
print(json.dumps({'result':str(result),'status':metadata['status'],'cases':len(rows),
    'failed_cases':metadata.get('failed_cases'),'tcm_validation':metadata['tcm_validation']},ensure_ascii=True,indent=2),flush=True)
raise SystemExit(0 if metadata['status']=='PASS' else 1)
