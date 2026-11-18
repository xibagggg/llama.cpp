#!/usr/bin/env python3
"""Derived publication runner. Its device path changes have NOT been device-tested.

Original tests/references retain their measured bytes. No production file is
written. Host-only mode performs no AI launch or SMI. Native modes are serial,
take the SDK benchmark lock and refuse a changed production/reference binding.
"""
from pathlib import Path
import argparse,contextlib,datetime,hashlib,importlib.util,json,os,platform,re,shlex,shutil,subprocess,threading,time

BASE=Path(__file__).resolve().parent

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def execute(command,directory,name,env=None,timeout=120):
    write(directory/(name+'.command.json'),command)
    try:
        result=subprocess.run(command,capture_output=True,env=env,timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        (directory/(name+'.stdout')).write_bytes(exc.stdout or b'');(directory/(name+'.stderr')).write_bytes(exc.stderr or b'');raise
    (directory/(name+'.stdout')).write_bytes(result.stdout);(directory/(name+'.stderr')).write_bytes(result.stderr)
    if result.returncode:raise RuntimeError(name+' returned '+str(result.returncode))
    return result.stdout.decode('utf8',errors='replace'),result.stderr.decode('utf8',errors='replace')

def host_cases(args,out):
    reports={}
    cases=[('mapped-contract124',BASE/'mapped/test_mapped_contract.cpp',[BASE/'mapped/test_mapped_contract_stubs',BASE/'mapped/contract-headers']),
           ('gdn-admission44-copy8',BASE/'gdn/frozen48/fixtures/admission-and-store.cpp',[BASE/'gdn/frozen48/fixtures'])]
    for name,source,includes in cases:
        directory=out/name;directory.mkdir();binary=directory/('test.exe' if os.name=='nt' else 'test')
        command=[args.cxx,'-std=c++17','-O2','-UNDEBUG',*['-I'+str(p) for p in includes],str(source),'-o',str(binary)]
        execute(command,directory,'build');stdout,stderr=execute([str(binary)],directory,'run')
        if name.startswith('mapped'):
            if not re.search(r'^PASS mapped callback metadata/refusal cases=124;',stdout,re.M):raise RuntimeError('mapped host did not report all124 cases')
            reports[name]={'status':'PASS','metadata_cases':124}
        else:
            report=json.loads(stdout)
            if report!={'status':'PASS','admission_cases':44,'copy_cases':8,'failures':0}:raise RuntimeError('GDN host case counts differ')
            reports[name]=report
        binary.unlink() # Host publication checks never leave a binary in the package.
    return reports

def machine_flags(args):
    path=args.flags_file or args.engine_root/'build/ggml/src/ggml-spacemit/CMakeFiles/ggml-spacemit.dir/flags.make'
    raw=path.read_bytes();line=next(line.partition('=')[2].strip() for line in raw.decode().splitlines() if line.startswith('CXX_FLAGS ='))
    flags=[v for v in shlex.split(line) if v.startswith(('-march=','-mabi='))]
    if not any(v.startswith('-march=') for v in flags):raise RuntimeError('production machine flags missing')
    return flags,{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}

def native_environment(gdn):
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('GGML_SPACEMIT_','SPERT_','SPACEMIT_','OCR_','MTMD_OCR_')):env.pop(key)
    env.update({'GGML_SPACEMIT_WORKERS':'6','SPACEMIT_PERFER_CORE_ID':'8,9,10,11,12,13','OMP_NUM_THREADS':'6','SPERT_BACKEND':'spacemit-k3','SPERT_CTX_RVV':'0','SPERT_HOST_PREPARE_STACKS':'1'})
    if gdn:env.update({'GGML_SPACEMIT_SHARED_MODE':'PINNED_TCM','GGML_SPACEMIT_TCM_OWNERSHIP':'1','GGML_SPACEMIT_STRICT_AI_CORES':'1','SPERT_SHARED_MODE':'TCM','SPERT_HOT_SPIN':'0'})
    return env

def mapped_case(args,out,flags):
    directory=out/'mapped1326';directory.mkdir();binary=directory/'test_mapped_argmax';env=native_environment(False)
    command=[args.cxx,'-std=c++17','-O3','-UNDEBUG','-fno-fast-math','-fno-tree-vectorize','-fno-tree-slp-vectorize',*flags,str(BASE/'mapped/test_mapped_argmax.cpp'),'-I'+str(BASE/'mapped/reference'),'-I/usr/include','-L/usr/lib','-Wl,-rpath,/usr/lib','-lspert','-latomic','-ldl','-pthread','-o',str(binary)]
    execute(command,directory,'build',env=env);stdout,stderr=execute([str(binary)],directory,'run',env=env,timeout=args.timeout)
    if not re.search(r'SPERT status=0 workers=6 cpus=\s*8 9 10 11 12 13 vlenb=\d+ executor_lane=0 fast=948 fallback=378',stdout) or 'PASS 1326 independently materialized full-vocab cases' not in stdout:raise RuntimeError('mapped1326 counts/actual AI6 placement not confirmed')
    return {'status':'PASS','cases':1326,'fast':948,'fallback':378,'source_sha256':sha(BASE/'mapped/test_mapped_argmax.cpp'),'header_sha256':sha(BASE/'mapped/reference/ocr-mapped-argmax-rvv.h'),'environment':{k:v for k,v in env.items() if k.startswith(('GGML_SPACEMIT_','SPACEMIT_','SPERT_','OMP_NUM'))},'scope':'Frozen standalone exact helper numerical cases only; no model latency or TCM benefit claim.'}

def gdn_case(args,out,flags):
    directory=out/'gdn48';directory.mkdir();source=BASE/'gdn/frozen48';binary=directory/'snapshot-bench';env=native_environment(True)
    command=[args.cxx,'-std=gnu++17','-O3','-UNDEBUG',*flags,str(source/'snapshot-bench.cpp'),'-I'+str(source/'fixtures'),'-I'+str(source),'-I/usr/include','-L/usr/lib','-Wl,-rpath,/usr/lib','-lspert','-latomic','-ldl','-pthread','-o',str(binary)]
    execute(command,directory,'build',env=env)
    samples=[];stop=threading.Event();monitor_errors=[]
    def monitor():
        with (directory/'tcm-smi.log').open('x',encoding='utf8') as output:
            while not stop.is_set():
                stamp=datetime.datetime.now().astimezone().isoformat()
                try:
                    result=subprocess.run(['spacemit-tcm-smi','-i'],capture_output=True,text=True,timeout=3)
                    sample={'time':stamp,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
                except Exception as exc:
                    sample={'time':stamp,'returncode':None,'stdout':'','stderr':repr(exc)};monitor_errors.append(repr(exc))
                samples.append(sample);output.write(stamp+'\n'+sample['stdout']+sample['stderr']+'\n');output.flush();stop.wait(.5)
    thread=threading.Thread(target=monitor);thread.start()
    try:stdout,stderr=execute([str(binary),str(args.reps),str(args.smi_hold_ms)],directory,'run',env=env,timeout=args.timeout)
    finally:
        stop.set();thread.join(timeout=5);write(directory/'tcm-smi-samples.json',samples)
    rows=[json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    zeros=['attention_byte_differences','cache_byte_differences','temporary_snapshot_byte_changes','nonfinite','guard_errors','core_errors','tcm_ownership_errors']
    numerical=len(rows)==48 and all(row.get('status')=='PASS' and row.get('input_changed') is False and all(row.get(key)==0 for key in zeros) and row.get('worker_cpus')==list(range(8,14)) and row.get('tcm_block_ids')==list(range(6)) for row in rows)
    validator=load_module('gdn_tcm_validation',source/'standalone-tcm-smi.py');tcm=validator.validate(samples,rows)
    if not numerical or monitor_errors or not tcm.get('physical_tcm_verified'):raise RuntimeError('GDN48 numerical/guards/AI6/physical TCM checks failed')
    return {'status':'PASS','cases':48,'source_sha256':sha(source/'snapshot-bench.cpp'),'tcm_validation':tcm,'environment':{k:v for k,v in env.items() if k.startswith(('GGML_SPACEMIT_','SPACEMIT_','SPERT_','OMP_NUM'))},'scope':'Frozen original standalone48; integrated trace guard has separate binding. Scalar T16 scratch remains DDR. No model/full-page/E2E benefit claimed.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['host','mapped','gdn','all'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--engine-root',type=Path,default=BASE.parents[2]);p.add_argument('--sdk-root',type=Path);p.add_argument('--cxx',default='g++');p.add_argument('--flags-file',type=Path);p.add_argument('--reps',type=int,default=3);p.add_argument('--smi-hold-ms',type=int,default=1500);p.add_argument('--timeout',type=int,default=240);a=p.parse_args()
    if a.output.resolve().is_relative_to(BASE):raise RuntimeError('output must be outside tracked test package')
    if a.reps<1 or not 0<=a.smi_hold_ms<=3000:raise RuntimeError('invalid repetition/hold arguments')
    package=load_module('package_verifier',BASE/'verify-package.py').verify()
    if package['status']!='PASS':raise RuntimeError(str(package))
    a.output.mkdir(parents=True,exist_ok=False);result={'status':'RUNNING','runner_kind':'derived publication runner, not the historical measured runner','runner_sha256':sha(__file__),'case':a.case,'package':package,'results':{}}
    lock=None
    try:
        if a.case in ('host','all'):result['results']['host']=host_cases(a,a.output)
        if a.case!='host':
            if platform.system()!='Linux' or not platform.machine().startswith('riscv'):raise RuntimeError('native tests require the actual Linux RISC-V device')
            if a.sdk_root is None:raise RuntimeError('native tests require --sdk-root for the shared device lock')
            import fcntl
            os.sched_setaffinity(0,set(range(6)));lock=(a.sdk_root/'output/.minicpm-benchmark.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            binding=load_module('production_binding',BASE/'check-production-binding.py').check(a.engine_root);result['production_binding_before']=binding
            if binding['status']!='PASS':raise RuntimeError('current production differs from frozen tests; review/revalidate explicitly before updating references')
            flags,identity=machine_flags(a);result['production_flags']=identity;result['host_cpus']=sorted(os.sched_getaffinity(0));result['ai_cpus']=list(range(8,14))
            if a.case in ('mapped','all'):result['results']['mapped']=mapped_case(a,a.output,flags)
            if a.case in ('gdn','all'):result['results']['gdn']=gdn_case(a,a.output,flags)
            after=load_module('production_binding_after',BASE/'check-production-binding.py').check(a.engine_root);result['production_binding_after']=after
            if after!=binding:raise RuntimeError('production source changed during tests')
        result['status']='PASS'
    except BaseException as exc:
        result['status']='FAIL';result['error']=type(exc).__name__+': '+str(exc);raise
    finally:
        if lock:lock.close()
        write(a.output/'result.json',result);print(json.dumps({'status':result['status'],'output':str(a.output),'case':a.case,'error':result.get('error')},indent=2))

if __name__=='__main__':main()
