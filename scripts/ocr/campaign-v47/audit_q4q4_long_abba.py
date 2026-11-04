"""Strict same-binary Q4/Q4 ABBA audit with frozen v43 quality reference."""
from pathlib import Path
import argparse,hashlib,json,re,statistics
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization'
p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--on-concat',type=int);a=p.parse_args()
OUT=BASE/('20261104-q4q4-'+a.version);load=lambda p:json.loads(p.read_text(encoding='utf8'))
refs={q['case_id']:q for q in load(BASE/'20261104-q4q4-v43-quality-selected/round.json')['runs'][0]['requests']}
identity=load(OUT/'build-identity.json');provenance=load(BASE/('20261104-q4q4-'+a.version+'-provenance/provenance.json'))
assert identity['manifest']==provenance['manifest'] and provenance['manifest_verified']
assert not identity['manifest']['build_flags']['GGML_BUILD_PROFILE']
assert provenance['runner_sha256']==hashlib.sha256((ROOT/f'scripts/optimization_round.{a.version}.py').read_bytes()).hexdigest()
arms=['off-a','on-a','on-b','off-b'];CORES=[8,9,10,11,13,14,15]
cases=['v5_table_merge4','03_table_gb_standard','04_scanned_civil_trial']
groups={};starts=[];ends=[];commands=[];count=0
for arm in arms:
 folder=BASE/('20261104-q4q4-'+a.version+'-abba-'+arm);d=load(folder/'round.json')
 assert d['restored']['health']['status']=='ok' and len(d['runs'])==1
 r=d['runs'][0];assert r['status']=='completed';m=d['measurement'];env=r['environment'];on=arm.startswith('on')
 assert d['case_file_sha256']==hashlib.sha256((ROOT/'images/optimization-v5/cases.json').read_bytes()).hexdigest()
 starts.append(d['started']);ends.append(d['finished']);commands.append(r['command'])
 assert m['warmups']==1 and m['repeats']==3 and m['text_precision']=='q4' and m['spacemit_core_ids']==CORES
 assert m['down_compact_tile']==20 and m['fa_lazy_init'] and m['down_tcm_tile']==16 and m['down_schedule']=='paired'
 assert m['f16_m4_cache']
 assert not m['diagnostic_trace'] and not m['down_phases'] and m.get('fa_group') is None
 assert m['down_vector']==(5 if on else None) and m['fa_direct']==(2 if on else None)
 assert m.get('concat_tcm')==(a.on_concat if on else None)
 assert env['SPACEMIT_PERFER_CORE_ID']==','.join(map(str,CORES)) and env['GGML_SPACEMIT_WORKERS']=='7'
 for key in ['GGML_SPACEMIT_F16_M4_CACHE','GGML_SPACEMIT_IME2_FP16_DOWN','GGML_SPACEMIT_IME2_FP16_COMPENSATE','GGML_SPACEMIT_IME2_FP16_NORMALIZE','GGML_SPACEMIT_OCR_FA_X8','GGML_SPACEMIT_OCR_FA_PACK','GGML_SPACEMIT_OCR_FA_LAZY_INIT']:assert env[key]=='1'
 for key,value in {'GGML_SPACEMIT_OCR_DOWN_SCHEDULE':'paired','GGML_SPACEMIT_OCR_DOWN_TCM_TILE':'16','GGML_SPACEMIT_OCR_DOWN_COMPACT_TILE':'20'}.items():assert env[key]==value
 for key,value in {'GGML_SPACEMIT_OCR_DOWN_VECTOR':'5','GGML_SPACEMIT_OCR_FA_DIRECT':'2','GGML_SPACEMIT_OCR_CONCAT_TCM':str(a.on_concat) if a.on_concat else None}.items():assert env.get(key)==(value if on else None)
 assert env.get('GGML_SPACEMIT_OCR_FA_GROUP') is None
 assert env['GGML_SPACEMIT_IME2_FP16_DOWN_LAYERS']==','.join(map(str,range(14,27)))
 assert {(q['case_id'],q['repeat']) for q in r['requests']}=={(c,n) for c in cases for n in [-1,0,1,2]}
 for q in r['requests']:
  b=refs[q['case_id']]
  assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop' and q['token_ids']
  assert q['vision_timing_events']['valid'] and q['text_prefill_s']>=0
  assert all(q[k]==b[k] for k in ['output','token_ids','embedding_sha256'])
  assert hashlib.sha256((folder/q['directory']/'output.txt').read_bytes()).hexdigest()==q['output_sha256']
  assert hashlib.sha256((folder/q['directory']/'vision-embeddings.bin').read_bytes()).hexdigest()==q['embedding_sha256']
  for phase in ['system_before','system_after']:
   obs=q[phase];single={t['allowed_cpu_ids'][0] for t in obs['server_thread_affinities'] if len(t['allowed_cpu_ids'])==1}
   assert set(CORES)<=single
   assert obs['cpufreq']['policy8']['scaling_cur_freq_khz']==1800000 and obs['cpufreq']['policy0']['scaling_cur_freq_khz']==2000000
  count+=1
  if not q['warmup']:groups.setdefault((on,q['case_id']),[]).append(q)
 assert '--trace' not in r['command']
 assert Path(r['command'][r['command'].index('--model')+1]).name=='MiniCPM-OCR-1_0-Q4_0.gguf'
 assert Path(r['command'][r['command'].index('--mmproj')+1]).name=='mmproj-model-q4_0-4x.gguf'
 log=(folder/r['name']/'server.log').read_text(encoding='utf8')
 pools=re.findall(r'ggml-spacemit: num_cores=(\d+), core_ids=([0-9,]+),?\s*arch_id=',log)
 assert pools and all(int(n)==7 and list(map(int,c.strip(',').split(',')))==CORES for n,c in pools)
 for marker in ['ocr-fa-pack-v40: active','ocr-fa-x8-v39: active','ocr-fa-lazy-v43: active']:assert marker in log
 assert ('ocr-down-compact-v43: active' in log)==(not on)
 assert ('ocr-down-vector-v44: active mode=5' in log)==on
 assert ('ocr-fa-direct-v44: active mode=2' in log)==on
 if a.on_concat:assert ('ocr-concat-tcm-v46: active' in log)==on
assert all(ends[i]<=starts[i+1] for i in range(3)) and all(c==commands[0] for c in commands)
rows=[]
for case in cases:
 pair={}
 for on in [False,True]:
  q=groups[on,case];assert len(q)==6
  avg=lambda f:statistics.mean(f(x) for x in q)
  pair['on' if on else 'off']={'formal_requests':6,'vision_s':avg(lambda x:x['vision_encode_s']),
   'prefill_s':avg(lambda x:x['text_prefill_s']),'generation_s':avg(lambda x:x['timings']['predicted_ms']/1000),
   'generation_token_s':avg(lambda x:x['timings']['predicted_per_second']),'e2e_s':avg(lambda x:x['e2e_s']),
   'e2e_sd_s':statistics.stdev(x['e2e_s'] for x in q),'e2e_min_s':min(x['e2e_s'] for x in q),'e2e_max_s':max(x['e2e_s'] for x in q),
   'rss_after_mib':avg(lambda x:x['system_after']['process_memory_kib']['VmRSS']/1024),
   'cumulative_hwm_max_mib':max(x['system_after']['process_memory_kib']['VmHWM']/1024 for x in q)}
 rows.append({'case':case,**pair,'e2e_reduction_pct':100*(1-pair['on']['e2e_s']/pair['off']['e2e_s']),
  'vision_reduction_pct':100*(1-pair['on']['vision_s']/pair['off']['vision_s']),
  'prefill_reduction_pct':100*(1-pair['on']['prefill_s']/pair['off']['prefill_s'])})
result={'passed':True,'version':a.version,'formal_requests':36,'warmups':12,'quality_checked':count,
 'baseline':'complete v43 flags on the same binary','selected':{'down_vector':5,'fa_direct':2,'concat_tcm':a.on_concat},
 'same_command_and_binary':True,'arithmetic_precision_changed':False,'rows':rows,
 'rss_scope':'sampled after request, not peak; HWM process cumulative','embedding_scope':'last saved vision slice only'}
(OUT/'abba-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for row in rows:print(row['case'],row['off']['e2e_s'],'->',row['on']['e2e_s'],row['e2e_reduction_pct'])
