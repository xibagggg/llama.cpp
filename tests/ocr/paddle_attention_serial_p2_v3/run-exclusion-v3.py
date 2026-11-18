"""Board-only driver; root invokes explicitly after independent compilation."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import re
import subprocess

GATES=('mask','softcap','sinks','compact0','direct0','softmax0','softmax2','fullkv1','k_inf','v_inf','q_nan')
p=argparse.ArgumentParser()
p.add_argument('--binary',required=True,type=Path)
p.add_argument('--output',required=True,type=Path,help='New directory; refuses reuse')
p.add_argument('--library-dir',required=True,type=Path)
p.add_argument('--gates',default=','.join(GATES))
args=p.parse_args()
binary=args.binary.resolve();library=args.library_dir.resolve()
selected=tuple(args.gates.split(','))
assert selected and len(set(selected))==len(selected) and all(g in GATES for g in selected)
assert binary.is_file() and library.is_dir()
args.output.mkdir(parents=True,exist_ok=False)
out=args.output.resolve()
env=os.environ.copy()
env.update({'LD_LIBRARY_PATH':str(library)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
    'GGML_BACKEND_PATH':str(library),'GGML_SPACEMIT_WORKERS':'6','GGML_SPACEMIT_ALLOW_EIGHT_WORKERS':'0',
    'SPACEMIT_PERFER_CORE_ID':'8,9,10,11,12,13'})
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report={'fixture_version':3,'binary':str(binary),'binary_sha256':sha(binary),
    'driver_sha256':sha(Path(__file__)),'gates':selected,'runs':[],'status':'RUNNING',
    'not_a_benchmark':True,'expected_serial_successes':0}
target=out/'serial-p2-v3-exclusion-result.json'
target.write_text(json.dumps(report,indent=2)+'\n')
ok=True
for gate in selected:
    ref=out/(gate+'.reference.bin')
    for mode in (0,1,2):
        name=f'{gate}-serial{mode}';phase=out/(name+'.phases');phase.mkdir()
        run_env=env.copy();run_env['GGML_SPACEMIT_OCR_FA_PHASES']=str(phase)
        cmd=[str(binary),'--write-reference' if mode==0 else '--compare-reference',str(ref),
            '--gate',gate,'--serial',str(mode)]
        stdout=out/(name+'.stdout.log');stderr=out/(name+'.stderr.log')
        try:
            result=subprocess.run(cmd,env=run_env,capture_output=True,text=True,timeout=1200)
        except (subprocess.TimeoutExpired,OSError) as error:
            def partial(value):
                return value.decode('utf-8',errors='replace') if isinstance(value,bytes) else (value or '')
            stdout.write_text(partial(getattr(error,'stdout','')),encoding='utf-8')
            stderr.write_text(partial(getattr(error,'stderr',''))+'\n'+str(error)+'\n',encoding='utf-8')
            report['runs'].append({'gate':gate,'serial_mode':mode,'command':cmd,'status':'FAIL',
                'launch_or_timeout_error':str(error),'evidence_sha256':{stdout.name:sha(stdout),stderr.name:sha(stderr)}})
            report['status']='FAIL';target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({'gate':gate,'serial_mode':mode,'status':'FAIL','error':str(error)}),flush=True)
            raise SystemExit(1)
        stdout.write_text(result.stdout,encoding='utf-8');stderr.write_text(result.stderr,encoding='utf-8')
        trace=phase/'attention-phases.jsonl';trace_error=None;stages=collections.Counter();names=set();lanes=[];phase_workers=set()
        try:
            assert trace.is_file(),'missing phase trace'
            for line in trace.read_text().splitlines():
                item=json.loads(line)
                if item['kind']=='lane':
                    lanes.append(item)
                    assert item['dropped']==0,'phase record drop'
                    assert item['worker']<6 or item['used']==0,'used unauthorized worker'
                elif item['kind']=='phase':
                    assert item['worker']<6,'unauthorized phase worker'
                    assert item['N'] in (848,1500),'unexpected N'
                    assert item['start_ns']>0 and item['end_ns']>=item['start_ns'],'invalid phase interval'
                    stages[item['stage']]+=1;names.add(item['name'])
                    phase_workers.add(item['worker'])
            assert len(lanes)==8,'missing lane summary'
            assert {lane['worker'] for lane in lanes}==set(range(8)),'duplicate/missing lane IDs'
            assert phase_workers==set(range(6)),'missing authorized phase worker'
            expected={f'serial_excl_{gate}_n{n}_l{layout}' for n in (848,1500) for layout in (0,1)}
            assert names==expected,{'expected_names':sorted(expected),'actual_names':sorted(names)}
            assert stages and stages['fullkv-s3']==0,'serial route succeeded in excluded gate'
        except (AssertionError,KeyError,ValueError) as e:trace_error=str(e)
        rows=[line for line in result.stdout.splitlines() if line.startswith('exclusion gate=') and ' N=' in line]
        footer=f'exclusion gate={gate} serial={mode} records=8 bitwise_complete=1 PASS'
        numerical=len(rows)==8 and all(line.endswith(' PASS') for line in rows) and footer in result.stdout
        passed=result.returncode==0 and numerical and trace_error is None
        row={'gate':gate,'serial_mode':mode,'command':cmd,'returncode':result.returncode,
            'executions':len(rows),'numerical_and_bitwise':numerical,'phase_exclusion':trace_error is None,
            'phase_error':trace_error,'stage_counts':dict(stages),'status':'PASS' if passed else 'FAIL',
            'evidence_sha256':{stdout.name:sha(stdout),stderr.name:sha(stderr)}}
        if trace.exists():row['evidence_sha256'][str(trace.relative_to(out))]=sha(trace)
        if ref.exists():row['reference_sha256']=sha(ref)
        report['runs'].append(row);ok=ok and passed
        target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:row[k] for k in ('gate','serial_mode','returncode','executions','stage_counts','status')},sort_keys=True),flush=True)
        if not passed:
            report['status']='FAIL';target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            raise SystemExit(1)
report['status']='PASS' if ok else 'FAIL'
report['total_executions']=sum(r['executions'] for r in report['runs'])
target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'runs':len(report['runs']),'total_executions':report['total_executions'],
    'report_sha256':sha(target)},sort_keys=True))
