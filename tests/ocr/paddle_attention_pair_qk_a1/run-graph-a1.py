#!/usr/bin/env python3
"""Root-invoked fresh-process graph equality plus exact pair-hit proof."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,os,re,subprocess
p=argparse.ArgumentParser()
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--library-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True,help='New directory; refuses reuse')
a=p.parse_args();binary=a.binary.resolve();library=a.library_dir.resolve()
assert binary.is_file() and library.is_dir()
a.output.mkdir(parents=True,exist_ok=False);out=a.output.resolve()
env=os.environ.copy();env.update({'GGML_BACKEND_PATH':str(library),
    'LD_LIBRARY_PATH':str(library)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
    'GGML_SPACEMIT_WORKERS':'6','GGML_SPACEMIT_ALLOW_EIGHT_WORKERS':'0',
    'SPACEMIT_PERFER_CORE_ID':'8,9,10,11,12,13'})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Regular shape cases:2layouts x2steps. At N1500, additionally3rejection
# kinds x2layouts x2validity orders x1validstep, with invalid whole-op fallback.
expected=Counter()
for n in (1121,1496,1500,1504,1536):expected[16*((n+15)//16)*((n+63)//64)]+=4
expected[16*((1500+15)//16)*((1500+63)//64)]+=12
report={'status':'RUNNING','binary':str(binary),'binary_sha256':sha(binary),
    'driver_sha256':sha(Path(__file__)),'graph_executions_per_side':92,
    'candidate_valid_serial_graphs':sum(expected.values()),
    'expected_pair_hit_multiset':dict(expected),'expected_pair_calls':sum(k*v for k,v in expected.items()),
    'all_F32_bits_equal_required':True,'runs':[],'not_a_benchmark':True}
target=out/'graph-a1-result.json';target.write_text(json.dumps(report,indent=2)+'\n')
ref=out/'v3-a1.reference.bin'
for mode,name in ((0,'reference'),(1,'candidate')):
    cmd=[str(binary),'--write-reference' if mode==0 else '--compare-reference',str(ref)]
    result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=1200)
    stdout=out/(name+'.stdout.log');stderr=out/(name+'.stderr.log')
    stdout.write_text(result.stdout,encoding='utf-8');stderr.write_text(result.stderr,encoding='utf-8')
    hits=[int(v) for v in re.findall(r'ocr-fa-serial-v3-a1: paired_tiles=(\d+)',result.stdout+'\n'+result.stderr)]
    multiset=Counter(hits)
    hit_pass=not hits if mode==0 else multiset==expected
    rows=[line for line in result.stdout.splitlines() if line.startswith('case=')]
    numerical=len(rows)==92 and all(line.endswith(' PASS') for line in rows) and \
        bool(re.search(r'suite cases=46 executions=92 output_f32=\d+ fixture_io=1 PASS',result.stdout))
    paths=len([line for line in result.stdout.splitlines() if line.startswith('serial_path ') and line.endswith(' PASS')])==92
    passed=result.returncode==0 and numerical and paths and hit_pass
    row={'mode':mode,'name':name,'command':cmd,'returncode':result.returncode,
        'graph_numerical_all_bits':numerical,'original_v3_path_counts':paths,
        'pair_hit_proof':hit_pass,'actual_pair_hit_multiset':dict(multiset),
        'actual_pair_calls':sum(hits),'status':'PASS' if passed else 'FAIL',
        'evidence_sha256':{stdout.name:sha(stdout),stderr.name:sha(stderr)}}
    if ref.exists():row['reference_sha256']=sha(ref)
    report['runs'].append(row);target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row,sort_keys=True),flush=True)
    if not passed:
        report['status']='FAIL';target.write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(1)
report['status']='PASS';report['graph_executions']=184
target.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','graph_executions':184,'paired_calls':report['expected_pair_calls'],
    'result_sha256':sha(target)},sort_keys=True))
