"""Audit v43 exactness, guarded configuration and incremental formal measurements."""
from pathlib import Path
import hashlib,json,re,statistics
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-q4q4-v43'
CORES=[8,9,10,11,13,14,15]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8');return x
def load(name):
 p=BASE/name;d=read(p/'round.json');assert d['restored']['health']['status']=='ok'
 assert not d['runs'] or len(d['runs'])==1 and d['runs'][0]['status']=='completed'
 return p,d,d['runs'][0] if d['runs'] else None
def request(p,q):
 assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop' and q['token_ids']
 assert q['vision_timing_events']['valid']
 assert hashlib.sha256(q['output'].encode()).hexdigest()==q['output_sha256']
 assert hashlib.sha256((p/q['directory']/'vision-embeddings.bin').read_bytes()).hexdigest()==q['embedding_sha256']
def micro():
 p,d,_=load('20261104-q4q4-v43-micro');assert d['checker_returncode']==0
 a=[json.loads(l) for l in (p/'down.jsonl').read_text().splitlines() if l.startswith('{')]
 assert a[-1]=={'kind':'final','input_immutable':True,'all_exact':True}
 assert all(x['bitwise_exact'] and x['guards'] for x in a if x['kind']=='timing')
 assert all(x['saved_bit_mismatch']==0 for x in a if x['kind']=='reference')
 down=[]
 for n in [7,65,988,989,1008,1009,1056,1057,1060]:
  row={'N':n,'modes':{}}
  for mode in [0,1,2]:
   t=[x['ms'] for x in a if x['kind']=='timing' and x['N']==n and x['mode']==mode];assert len(t)==6
   row['modes'][str(mode)]={'mean_ms':statistics.mean(t),'sd_ms':statistics.stdev(t)}
  for mode in [1,2]:row[f'mode{mode}_reduction_pct']=100*(1-row['modes'][str(mode)]['mean_ms']/row['modes']['0']['mean_ms'])
  down.append(row)
 log=(p/'checker.log').read_text(encoding='utf8');fa=[]
 for n in [64,65,988,1056]:
  for layout in range(6):
   assert f'LAZY INIT EXACT N={n} layout={layout}' in log
   states={}
   for state in ['off','on']:
    rows=[]
    for suffix in ['a','b']:
     arm=[json.loads(l) for l in (p/f'N{n}-l{layout}-{state}-{suffix}.jsonl').read_text().splitlines() if l.startswith('{')]
     assert len(arm)==3 and all(x['finite'] and x['guards'] and x['input_immutable'] and x['repeat_exact'] and x['N']==n and x['layout']==layout for x in arm)
     rows+=arm
    t=[x['ms'] for x in rows];states[state]={'mean_ms':statistics.mean(t),'sd_ms':statistics.stdev(t),'formal_runs':len(t)}
   fa.append({'N':n,'layout':layout,**states,'reduction_pct':100*(1-states['on']['mean_ms']/states['off']['mean_ms'])})
 return save('micro-audit.json',{'passed':True,'down_formal_runs':162,'attention_formal_runs':288,'exact':True,
  'down_modes':{'0':'unchanged v42 paired16, reserved4 weight rows','1':'compact paired16, reserved2','2':'compact paired20, reserved2'},
  'down_input_scope':'Real layer0 N1056; other sizes truncated/cyclically extended','down':down,'attention':fa,
  'attention_input_scope':'Deterministic synthetic D72 H16, contiguous/interleaved, unmasked/causal/some fully masked query rows; scratch poisoned before each call, preparation included'})
def quality():
 p,d,r=load('20261104-q4q4-v43-quality-selected');_,_,br=load('20261104-q4q4-v42-quality-paired')
 refs={q['case_id']:q for q in br['requests']};rows=[]
 for q in r['requests']:
  request(p,q);b=refs[q['case_id']]
  assert all(q[k]==b[k] for k in ['output','token_ids','embedding_sha256'])
  rows.append({'case':q['case_id'],'output_exact':True,'tokens_exact':True,'last_slice_embedding_exact':True,
   'e2e_s':q['e2e_s'],'vision_s':q['vision_encode_s'],'prefill_s':q['text_prefill_s'],
   'generation_s':q['timings']['predicted_ms']/1000,'generation_token_s':q['timings']['predicted_per_second'],
   'rss_after_mib':q['system_after']['process_memory_kib']['VmRSS']/1024})
 assert len(rows)==7
 selected=read(OUT/'selection.json');log=(p/r['name']/'server.log').read_text(encoding='utf8')
 assert ('ocr-fa-lazy-v43: active' in log)==selected['fa_lazy_init']
 assert ('ocr-down-compact-v43: active' in log)==bool(selected['down_compact_tile'])
 return save('quality-audit.json',{'passed':True,'scope':'Seven single requests; multi-slice embedding dump covers final slice only','rows':rows})
def abba():
 selected=read(OUT/'selection.json');identity=read(OUT/'build-identity.json')
 manifest=read(BASE/'20261104-q4q4-v43-provenance/runtime-manifest.json')
 provenance=read(BASE/'20261104-q4q4-v43-provenance/test-provenance.json')
 assert provenance['manifest_verified'] and manifest==identity['manifest'] and not manifest['build_flags']['GGML_BUILD_PROFILE']
 assert provenance['runner_sha256']==hashlib.sha256((ROOT/'scripts/optimization_round.v43.py').read_bytes()).hexdigest()
 _,_,qr=load('20261104-q4q4-v43-quality-selected');refs={q['case_id']:(q['output'],q['token_ids'],q['embedding_sha256']) for q in qr['requests']}
 groups={};bins=set();models=set();starts=[];ends=[];count=0
 arms=['off-a','on-a','on-b','off-b']
 for arm in arms:
  p,d,r=load('20261104-q4q4-v43-abba-'+arm);on=arm.startswith('on');m=d['measurement'];env=r['environment']
  assert d['case_file_sha256']==hashlib.sha256((ROOT/'images/optimization-v5/cases.json').read_bytes()).hexdigest()
  assert m['warmups']==1 and m['repeats']==3 and m['text_precision']=='q4' and m['spacemit_core_ids']==CORES
  assert m['f16_m4_cache'] and not m['diagnostic_trace'] and not m['down_phases'] and m['down_tcm_tile']==16 and m['down_schedule']=='paired'
  assert m['down_compact_tile']==(selected['down_compact_tile'] if on else None) and m['fa_lazy_init']==(selected['fa_lazy_init'] if on else False)
  assert env.get('GGML_SPACEMIT_OCR_DOWN_COMPACT_TILE')==(str(selected['down_compact_tile']) if on and selected['down_compact_tile'] else None)
  assert env.get('GGML_SPACEMIT_OCR_FA_LAZY_INIT')==('1' if on and selected['fa_lazy_init'] else None)
  for flag in ['GGML_SPACEMIT_F16_M4_CACHE','GGML_SPACEMIT_IME2_FP16_DOWN','GGML_SPACEMIT_IME2_FP16_COMPENSATE','GGML_SPACEMIT_IME2_FP16_NORMALIZE','GGML_SPACEMIT_OCR_FA_X8','GGML_SPACEMIT_OCR_FA_PACK']:assert env[flag]=='1'
  assert env['GGML_SPACEMIT_OCR_DOWN_SCHEDULE']=='paired' and env['GGML_SPACEMIT_OCR_DOWN_TCM_TILE']=='16'
  assert env['GGML_SPACEMIT_IME2_FP16_DOWN_LAYERS']==','.join(map(str,range(14,27)))
  assert env['GGML_SPACEMIT_WORKERS']=='7' and env['GGML_SPACEMIT_BACKEND']=='1'
  cmd=r['command'];bins.add(cmd[0]);models.add((cmd[cmd.index('--model')+1],cmd[cmd.index('--mmproj')+1]));starts.append(d['started']);ends.append(d['finished'])
  log=(p/r['name']/'server.log').read_text(encoding='utf8')
  assert ('ocr-fa-lazy-v43: active' in log)==(on and selected['fa_lazy_init'])
  assert ('ocr-down-compact-v43: active' in log)==(on and bool(selected['down_compact_tile']))
  for marker in ['ocr-fa-pack-v40: active','ocr-fa-x8-v39: active']:assert marker in log
  pools=re.findall(r'ggml-spacemit: num_cores=(\d+), core_ids=([0-9,]+),?\s*arch_id=',log)
  assert pools and all(int(n)==7 and list(map(int,c.strip(',').split(',')))==CORES for n,c in pools)
  assert {(q['case_id'],q['repeat']) for q in r['requests']}=={(c,n) for c in ['v5_table_merge4','03_table_gb_standard'] for n in [-1,0,1,2]}
  for q in r['requests']:
   request(p,q);assert (q['output'],q['token_ids'],q['embedding_sha256'])==refs[q['case_id']]
   for phase in ['system_before','system_after']:
    obs=q[phase];singleton={x['allowed_cpu_ids'][0] for x in obs['server_thread_affinities'] if len(x['allowed_cpu_ids'])==1};assert set(CORES)<=singleton
    assert obs['cpufreq']['policy8']['scaling_cur_freq_khz']==1800000 and obs['cpufreq']['policy0']['scaling_cur_freq_khz']==2000000
   count+=1
   if not q['warmup']:groups.setdefault((on,q['case_id']),[]).append(q)
 assert len(bins)==len(models)==1 and all(ends[i]<=starts[i+1] for i in range(3))
 text,vision=next(iter(models));assert Path(text).name=='MiniCPM-OCR-1_0-Q4_0.gguf' and Path(vision).name=='mmproj-model-q4_0-4x.gguf'
 rows=[]
 for case in ['v5_table_merge4','03_table_gb_standard']:
  pair={}
  for on in [False,True]:
   q=groups[on,case];assert len(q)==6;avg=lambda f:statistics.mean(f(x) for x in q)
   pair['on' if on else 'off']={'formal_requests':6,'vision_s':avg(lambda x:x['vision_encode_s']),'prefill_s':avg(lambda x:x['text_prefill_s']),
    'generation_s':avg(lambda x:x['timings']['predicted_ms']/1000),'generation_token_s':avg(lambda x:x['timings']['predicted_per_second']),
    'e2e_s':avg(lambda x:x['e2e_s']),'e2e_sd_s':statistics.stdev(x['e2e_s'] for x in q),
    'e2e_min_s':min(x['e2e_s'] for x in q),'e2e_max_s':max(x['e2e_s'] for x in q),
    'rss_after_mib':avg(lambda x:x['system_after']['process_memory_kib']['VmRSS']/1024),
    'process_cumulative_hwm_max_mib':max(x['system_after']['process_memory_kib']['VmHWM']/1024 for x in q)}
  rows.append({'case':case,**pair,'e2e_reduction_pct':100*(1-pair['on']['e2e_s']/pair['off']['e2e_s']),
   'vision_reduction_pct':100*(1-pair['on']['vision_s']/pair['off']['vision_s'])})
 return save('abba-audit.json',{'passed':True,'requests_checked_including_warmups':count,'arm_order':arms,
  'baseline':'complete v42 selected flags on the same v43 binary','same_binary':next(iter(bins)),'models':list(next(iter(models))),
  'output_tokens_and_last_slice_embeddings_exact_across_states':True,'rows':rows,
  'rss_scope':'RSS after requests; HWM process cumulative, not per-request peak'})
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['micro','quality','abba','all'],default='all');a=p.parse_args()
 for name in ['micro','quality','abba']:
  if a.stage in [name,'all']:print(json.dumps(globals()[name](),ensure_ascii=False,indent=2))
