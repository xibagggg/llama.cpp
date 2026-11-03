"""Audit v42 exact arithmetic, real OCR quality, and same-binary incremental ABBA."""
from pathlib import Path
import hashlib,json,re,statistics
import ast,math

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/optimization'
OUT=BASE/'20261104-q4q4-v42'
OUT.mkdir(exist_ok=True)
CORES=[8,9,10,11,13,14,15]

def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(name,d):
    (OUT/name).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
    return d
def load(name):
    p=BASE/name;d=read(p/'round.json')
    assert d['restored']['health']['status']=='ok'
    if d['runs']:assert len(d['runs'])==1 and d['runs'][0]['status']=='completed'
    return p,d,d['runs'][0] if d['runs'] else None
def recovered_timing(p,q):
    if q['vision_timing_events']['valid']:return q['vision_timing_events']
    source=(ROOT/'scripts/optimization_round.v42.py').read_text(encoding='utf8')
    parsed=ast.parse(source)
    nodes=[x for x in parsed.body if (isinstance(x,ast.FunctionDef) and x.name=='parse_vision_encode_timing') or
           (isinstance(x,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='LOG_TIMESTAMP_RE' for v in x.targets))]
    assert len(nodes)==2
    ns={'re':re,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),'v42_timing_parser','exec'),ns)
    r=read(p/'round.json')['runs'][0]
    raw=(p/r['name']/'server.log').read_bytes()[q['server_log_start_byte']:q['server_log_end_byte']]
    timing=ns['parse_vision_encode_timing'](raw.decode('utf8',errors='replace'));assert timing['valid']
    save('quality-timing-recovery.json',{'case':q['case_id'],'raw_round_preserved':True,'original':q['vision_timing_events'],
         'recovered':timing,'log_segment_sha256':hashlib.sha256(raw).hexdigest(),'parser_runner_sha256':hashlib.sha256(source.encode()).hexdigest(),
         'reason':'Observed timestamp on adjacent truncated mtmd_batch_encode begin line with matching media-worker stage prefix; explicit pairs only'})
    return timing
def request(p,q,allow_timing_recovery=False):
    assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop' and q['token_ids']
    assert (recovered_timing(p,q) if allow_timing_recovery else q['vision_timing_events'])['valid']
    assert hashlib.sha256((p/q['directory']/'vision-embeddings.bin').read_bytes()).hexdigest()==q['embedding_sha256']
    assert hashlib.sha256(q['output'].encode()).hexdigest()==q['output_sha256']

def micro():
    p,d,_=load('20261104-q4q4-v42-down-micro');assert d['checker_returncode']==0
    a=[json.loads(l) for l in (p/'checker.log').read_text(encoding='utf8').splitlines() if l.startswith('{')]
    assert a[-1]=={'kind':'final','input_immutable':True,'all_exact':True}
    assert all(x['bitwise_exact'] and x['guards'] for x in a if x['kind']=='timing')
    refs=[x for x in a if x['kind']=='reference'];assert all(x['saved_bit_mismatch']==0 for x in refs)
    rows=[]
    for n in [7,65,988,989,1008,1009,1056,1057,1060]:
        row={'N':n,'real_capture_N':1056,'input_source':'Real layer-0 capture; truncate or cyclic extension when N differs','mode_ms':{}}
        for mode in [0,1,2]:
            q=[x['ms'] for x in a if x['kind']=='timing' and x['N']==n and x['mode']==mode];assert len(q)==6
            row['mode_ms'][str(mode)]={'mean':statistics.mean(q),'sd':statistics.stdev(q),'min':min(q),'max':max(q)}
        row['paired_reduction_pct']=100*(1-row['mode_ms']['2']['mean']/row['mode_ms']['0']['mean'])
        row['relaxed_reduction_pct']=100*(1-row['mode_ms']['1']['mean']/row['mode_ms']['0']['mean']);rows.append(row)
    return save('micro-audit.json',{'passed':True,'formal_runs':162,'all_outputs_bitwise_exact':True,'input_immutable':True,'guards':True,
        'standalone_micro_source_commit':'764d38e877ef8e59fc1f98f82c52ecbd0415ed93',
        'runtime_libraries_source_commit':'91570b05aa667bed593ac715158cc95b3820dd36',
        'production_source_commit':'63ad486a6973eeb345fc1a0fdae43b094ed2dc44',
        'modes':{'0':'frozen v41 M4xN1 TCM16','1':'relaxed M4xN1 TCM16','2':'paired M2xN2 TCM16'},'rows':rows})

def quality():
    p,d,r=load('20261104-q4q4-v42-quality-paired');bp,bd,br=load('20261104-q4q4-v41-quality-tile16')
    b={q['case_id']:q for q in br['requests']};rows=[]
    for q in r['requests']:
        request(p,q,allow_timing_recovery=True);z=b[q['case_id']];timing=recovered_timing(p,q)
        for k in ['output','token_ids','embedding_sha256']:assert q[k]==z[k],(q['case_id'],k)
        rows.append({'case':q['case_id'],'output_exact':True,'tokens_exact':True,'last_slice_embedding_exact':True,
                     'e2e_s':q['e2e_s'],'vision_s':timing['vision_encode_s'],'prefill_s':q['timings']['prompt_ms']/1000-timing['vision_encode_s'],
                     'generation_s':q['timings']['predicted_ms']/1000,'generation_token_s':q['timings']['predicted_per_second'],
                     'rss_after_mib':q['system_after']['process_memory_kib']['VmRSS']/1024})
    assert len(rows)==7
    log=(p/r['name']/'server.log').read_text(encoding='utf8')
    assert 'ocr-down-sched-v42: active mode=paired tile=16' in log
    return save('quality-audit.json',{'passed':True,'scope':'Seven single-run cases; saved embedding covers the final slice only on multi-slice pages','rows':rows})

def abba():
    provenance=read(BASE/'20261104-q4q4-v42-provenance/test-provenance.json')
    manifest=read(BASE/'20261104-q4q4-v42-provenance/runtime-manifest.json')
    assert provenance['manifest_verified'] and manifest['source_commit']=='63ad486a6973eeb345fc1a0fdae43b094ed2dc44'
    assert not manifest['build_flags']['GGML_BUILD_PROFILE']
    _,_,quality_run=load('20261104-q4q4-v42-quality-paired')
    quality_refs={q['case_id']:(q['output'],q['token_ids'],q['embedding_sha256']) for q in quality_run['requests']}
    refs={};groups={};bins=set();models=set();starts=[];ends=[];count=0
    arms=['off-a','on-a','on-b','off-b']
    for arm in arms:
        p,d,r=load('20261104-q4q4-v42-abba-'+arm);on=arm.startswith('on');m=d['measurement'];env=r['environment']
        assert d['case_file_sha256']==hashlib.sha256((ROOT/'images/optimization-v5/cases.json').read_bytes()).hexdigest()
        assert m['warmups']==1 and m['repeats']==3 and m['text_precision']=='q4'
        assert m['spacemit_core_ids']==CORES and m['f16_m4_cache'] and not m['diagnostic_trace'] and not m['down_phases']
        assert m['down_tcm_tile']==16 and m['down_schedule']==('paired' if on else None)
        for flag in ['GGML_SPACEMIT_F16_M4_CACHE','GGML_SPACEMIT_IME2_FP16_DOWN','GGML_SPACEMIT_IME2_FP16_COMPENSATE','GGML_SPACEMIT_IME2_FP16_NORMALIZE','GGML_SPACEMIT_OCR_FA_X8','GGML_SPACEMIT_OCR_FA_PACK']:assert env[flag]=='1'
        assert env['GGML_SPACEMIT_OCR_DOWN_TCM_TILE']=='16'
        assert env.get('GGML_SPACEMIT_OCR_DOWN_SCHEDULE')==('paired' if on else None)
        assert env['GGML_SPACEMIT_IME2_FP16_DOWN_LAYERS']==','.join(map(str,range(14,27)))
        assert env['GGML_SPACEMIT_WORKERS']=='7' and env['GGML_SPACEMIT_BACKEND']=='1'
        cmd=r['command'];bins.add(cmd[0]);models.add((cmd[cmd.index('--model')+1],cmd[cmd.index('--mmproj')+1]))
        starts.append(d['started']);ends.append(d['finished'])
        log=(p/r['name']/'server.log').read_text(encoding='utf8')
        assert ('ocr-down-sched-v42: active mode=paired tile=16' in log)==on
        for marker in ['ocr-fa-pack-v40: active','ocr-fa-x8-v39: active']:assert marker in log
        pools=re.findall(r'ggml-spacemit: num_cores=(\d+), core_ids=([0-9,]+),?\s*arch_id=',log)
        assert pools and all(int(n)==7 and list(map(int,c.strip(',').split(',')))==CORES for n,c in pools)
        assert {(q['case_id'],q['repeat']) for q in r['requests']}=={(c,n) for c in ['v5_table_merge4','03_table_gb_standard'] for n in [-1,0,1,2]}
        for q in r['requests']:
            request(p,q);c=q['case_id'];sig=(q['output'],q['token_ids'],q['embedding_sha256']);refs.setdefault(c,sig);assert refs[c]==sig==quality_refs[c],(arm,c)
            for phase in ['system_before','system_after']:
                obs=q[phase];singleton={a['allowed_cpu_ids'][0] for a in obs['server_thread_affinities'] if len(a['allowed_cpu_ids'])==1}
                assert set(CORES)<=singleton
                assert obs['cpufreq']['policy8']['scaling_cur_freq_khz']==1800000 and obs['cpufreq']['policy0']['scaling_cur_freq_khz']==2000000
            count+=1
            if not q['warmup']:groups.setdefault((on,c),[]).append(q)
    assert len(bins)==len(models)==1 and all(ends[i]<=starts[i+1] for i in range(3))
    text_model,vision_model=next(iter(models))
    assert Path(text_model).name=='MiniCPM-OCR-1_0-Q4_0.gguf' and Path(vision_model).name=='mmproj-model-q4_0-4x.gguf'
    rows=[]
    for c in ['v5_table_merge4','03_table_gb_standard']:
        pair={}
        for on in [False,True]:
            q=groups[on,c];assert len(q)==6
            avg=lambda f:statistics.mean(f(x) for x in q)
            pair['on' if on else 'off']={'formal_requests':6,'vision_s':avg(lambda x:x['vision_encode_s']),'prefill_s':avg(lambda x:x['text_prefill_s']),
                'generation_s':avg(lambda x:x['timings']['predicted_ms']/1000),'generation_token_s':avg(lambda x:x['timings']['predicted_per_second']),
                'e2e_s':avg(lambda x:x['e2e_s']),'e2e_sd_s':statistics.stdev(x['e2e_s'] for x in q),
                'e2e_min_s':min(x['e2e_s'] for x in q),'e2e_max_s':max(x['e2e_s'] for x in q),
                'rss_after_mib':avg(lambda x:x['system_after']['process_memory_kib']['VmRSS']/1024),
                'process_cumulative_hwm_max_mib':max(x['system_after']['process_memory_kib']['VmHWM']/1024 for x in q)}
        rows.append({'case':c,**pair,'e2e_reduction_pct':100*(1-pair['on']['e2e_s']/pair['off']['e2e_s']),
                     'vision_reduction_pct':100*(1-pair['on']['vision_s']/pair['off']['vision_s'])})
    return save('abba-audit.json',{'passed':True,'requests_checked_including_warmups':count,'arm_order':arms,
        'baseline':'v41 combined flags with TCM16 on the same v42 binary; only paired scheduling differs','same_binary':next(iter(bins)),
        'models':list(next(iter(models))),'output_tokens_and_last_slice_embeddings_exact_across_states':True,
        'rss_scope':'RSS after requests; HWM process cumulative, not per-request peak','rows':rows})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['micro','quality','abba','all'],default='all');args=p.parse_args()
    for name in ['micro','quality','abba']:
        if args.stage in [name,'all']:print(json.dumps(globals()[name](),ensure_ascii=False,indent=2))
