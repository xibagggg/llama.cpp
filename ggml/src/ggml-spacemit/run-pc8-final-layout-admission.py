#!/usr/bin/env python3
"""Sequential numerical/layout admission only; never a performance benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=int,default=300)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update({'SPERT_BACKEND':'spacemit-k3','SPERT_RT_PRIO':'0',
        'SPERT_SHARED_MODE':'TCM','SPACEMIT_PERFER_CORE_ID':'8,9,10,11,12,13',
        'GGML_SPACEMIT_WORKERS':'6','GGML_SPACEMIT_ALLOW_EIGHT_WORKERS':'0',
        'GGML_SPACEMIT_SHARED_MODE':'PINNED_TCM','GGML_SPACEMIT_STRICT_AI_CORES':'1'})
    report={'purpose':'native numerical/layout admission; no performance claim',
        'workers':6,'allow_eight':0,'tests':[],'passed':False}
    def run(label,binary,arguments=(),options=None):
        path=args.binary_dir/binary
        call_env=env.copy();call_env.update(options or {})
        command=[str(path),*map(str,arguments)]
        item={'label':label,'command':command,'binary_sha256':digest(path)}
        try:
            r=subprocess.run(command,env=call_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                timeout=args.timeout,check=False)
            data=r.stdout;item['exit_code']=r.returncode
        except subprocess.TimeoutExpired as error:
            data=error.stdout or b'';item['timeout']=args.timeout
        log=args.output/(label+'.log');log.write_bytes(data)
        item['log']=str(log);item['sha256']=digest(log);report['tests'].append(item)
        if item.get('exit_code')!=0:raise RuntimeError(label+' did not finish successfully')
        return data.decode('utf8',errors='replace')
    try:
        output=run('host','test-pc8-final-layout-host')
        assert 'policy=2800 load_cases=288' in output and output.rstrip().endswith('PASS')
        output=run('probe','test-pc8-final-layout-raw',('--probe',))
        line=re.search(r'^PC8_FINAL_MAPPING (\[.*\])$',output,re.M)
        assert line and 'bytes=1024 bijection=1 native_vlen128=1 PASS' in output
        mapping=json.loads(line.group(1));assert len(mapping)==1024
        expected=[]
        for dest in range(1024):
            group=dest//256;within=dest%256;half=within//128;tail=within%128
            expected.append(group*256+(tail//16)*32+half*16+tail%16)
        assert mapping==expected,'native mapping differs from portable candidate'
        report['native_mapping']=mapping
        output=run('raw','test-pc8-final-layout-raw')
        cases=re.findall(r'^PC8_FINAL_RAW K=.* PASS$',output,re.M)
        assert len(cases)==90 and 'cases=90 workers=6 ai_mask=0x3f00 five_frm=1 native_byte_oracle=1 PASS' in output
        assert len(re.findall(r'^PC8_FINAL_TCM lane=',output,re.M))==6
        reference=args.output/'graph-reference.bin'
        output=run('graph-legacy','test-pc8-final-layout-graph',('--write-reference',reference),
            {'GGML_SPACEMIT_OCR_PC8_LOAD_PREPACK':'0'})
        assert 'PC8_FINAL_GRAPH cases=53 passes=159' in output and output.rstrip().endswith('PASS')
        report['graph_reference_sha256']=digest(reference)
        output=run('graph-final','test-pc8-final-layout-graph',('--compare-reference',reference),
            {'GGML_SPACEMIT_OCR_PC8_LOAD_PREPACK':'1'})
        assert 'PC8_FINAL_GRAPH cases=53 passes=159' in output and output.rstrip().endswith('PASS')
        assert digest(reference)==report['graph_reference_sha256']
        report['passed']=True
    except Exception as error:
        report['error']=repr(error)
    finally:
        (args.output/'admission.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'passed':report['passed'],'report':str(args.output/'admission.json')},indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':
    sys.exit(main())
