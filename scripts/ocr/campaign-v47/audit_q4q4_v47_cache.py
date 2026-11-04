"""Two-arm cache configuration supplement, separate from final kernel ABBA/soak."""
from pathlib import Path
import hashlib,json,re,statistics
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-q4q4-long-1430'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
ref={q['case_id']:q for q in read(BASE/'20261104-q4q4-v43-quality-selected/round.json')['runs'][0]['requests']}
provenance=read(BASE/'20261104-q4q4-v47-cache-provenance/provenance.json')
identity=read(BASE/'20261104-q4q4-v47/build-identity.json')
assert provenance['manifest_verified'] and identity['manifest']==provenance['manifest']
assert not identity['manifest']['build_flags']['GGML_BUILD_PROFILE']
assert provenance['runner_sha256']==hashlib.sha256((ROOT/'scripts/optimization_cache_test.v47.py').read_bytes()).hexdigest()
groups={};stats={};commands=[];starts=[];ends=[];count=0;missing=[]
for arm in ['off','on']:
 folder=BASE/f'20261104-q4q4-v47-cache-{arm}';d=read(folder/'round.json');r=d['runs'][0];m=d['measurement']
 assert r['status']=='completed' and d['restored']['health']['status']=='ok'
 assert m['cache_ram_mib']==(0 if arm=='on' else None) and m['cache_prompt'] is False
 assert m['down_vector']==5 and m['fa_direct']==2 and m['concat_tcm']==4 and m['fa_group'] is None
 assert m['warmups']==1 and m['repeats']==3 and m['spacemit_core_ids']==[8,9,10,11,13,14,15]
 assert d['case_file_sha256']==hashlib.sha256((ROOT/'images/optimization-v5/cases.json').read_bytes()).hexdigest()
 env=r['environment']
 for key,value in {'GGML_SPACEMIT_OCR_DOWN_VECTOR':'5','GGML_SPACEMIT_OCR_FA_DIRECT':'2','GGML_SPACEMIT_OCR_CONCAT_TCM':'4','GGML_SPACEMIT_OCR_DOWN_COMPACT_TILE':'20','GGML_SPACEMIT_OCR_DOWN_TCM_TILE':'16','GGML_SPACEMIT_OCR_DOWN_SCHEDULE':'paired','GGML_SPACEMIT_OCR_FA_LAZY_INIT':'1','GGML_SPACEMIT_WORKERS':'7','SPACEMIT_PERFER_CORE_ID':'8,9,10,11,13,14,15'}.items():assert env[key]==value
 for key in ['GGML_SPACEMIT_F16_M4_CACHE','GGML_SPACEMIT_IME2_FP16_DOWN','GGML_SPACEMIT_IME2_FP16_COMPENSATE','GGML_SPACEMIT_IME2_FP16_NORMALIZE','GGML_SPACEMIT_OCR_FA_X8','GGML_SPACEMIT_OCR_FA_PACK']:assert env[key]=='1'
 assert env.get('GGML_SPACEMIT_OCR_FA_GROUP') is None
 assert env['GGML_SPACEMIT_IME2_FP16_DOWN_LAYERS']==','.join(map(str,range(14,27)))
 cmd=r['command'];assert ('--cache-ram' in cmd)==(arm=='on')
 assert '--trace' not in cmd and m['text_precision']=='q4' and not m['diagnostic_trace'] and not m['down_phases']
 assert Path(cmd[cmd.index('--model')+1]).name=='MiniCPM-OCR-1_0-Q4_0.gguf'
 assert Path(cmd[cmd.index('--mmproj')+1]).name=='mmproj-model-q4_0-4x.gguf'
 if arm=='on':assert cmd[-2:]==['--cache-ram','0'];cmd=cmd[:-2]
 commands.append(cmd);starts.append(d['started']);ends.append(d['finished'])
 assert {(q['case_id'],q['repeat']) for q in r['requests']}=={(c,n) for c in ['v5_table_merge4','03_table_gb_standard'] for n in [-1,0,1,2]}
 for q in r['requests']:
  assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop'
  assert all(q[k]==ref[q['case_id']][k] for k in ['output','token_ids','embedding_sha256'])
  assert hashlib.sha256((folder/q['directory']/'vision-embeddings.bin').read_bytes()).hexdigest()==q['embedding_sha256']
  assert hashlib.sha256((folder/q['directory']/'output.txt').read_bytes()).hexdigest()==q['output_sha256']
  if not q['vision_timing_events']['valid']:missing.append({'arm':arm,'case':q['case_id'],'repeat':q['repeat']})
  for phase in ['system_before','system_after']:
   assert q[phase]['cpufreq']['policy8']['scaling_cur_freq_khz']==1800000 and q[phase]['cpufreq']['policy0']['scaling_cur_freq_khz']==2000000
   single={t['allowed_cpu_ids'][0] for t in q[phase]['server_thread_affinities'] if len(t['allowed_cpu_ids'])==1}
   assert {8,9,10,11,13,14,15}<=single
  count+=1
  if not q['warmup']:groups.setdefault((arm,q['case_id']),[]).append(q)
 log=(folder/r['name']/'server.log').read_text(encoding='utf8')
 pools=re.findall(r'ggml-spacemit: num_cores=(\d+), core_ids=([0-9,]+),?\s*arch_id=',log)
 assert pools and all(int(n)==7 and list(map(int,c.strip(',').split(',')))==[8,9,10,11,13,14,15] for n,c in pools)
 for marker in ['ocr-fa-pack-v40: active','ocr-fa-x8-v39: active','ocr-fa-lazy-v43: active','ocr-down-vector-v44: active mode=5','ocr-fa-direct-v44: active mode=2','ocr-concat-tcm-v46: active']:assert marker in log
 assert ('prompt cache is enabled, size limit: 8192 MiB' in log)==(arm=='off')
 assert ('prompt cache is disabled' in log)==(arm=='on')
 cache=re.findall(r'cache state: (\d+) prompts, ([0-9.]+) MiB',log)
 stats[arm]={'last_cache_state':{'prompts':int(cache[-1][0]),'mib':float(cache[-1][1])} if cache else None,
  'max_cache_mib':max([float(z[1]) for z in cache],default=0),'hwm_max_mib':max(q['system_after']['process_memory_kib']['VmHWM']/1024 for q in r['requests'])}
assert commands[0]==commands[1] and ends[0]<=starts[1]
rows=[]
for case in ['v5_table_merge4','03_table_gb_standard']:
 pair={}
 for arm in ['off','on']:
  qs=groups[arm,case];assert len(qs)==3
  pair[arm]={'requests':3,'e2e_s':statistics.mean(q['e2e_s'] for q in qs),'e2e_sd_s':statistics.stdev(q['e2e_s'] for q in qs),
   'e2e_min_s':min(q['e2e_s'] for q in qs),'e2e_max_s':max(q['e2e_s'] for q in qs),
   'rss_after_mib':statistics.mean(q['system_after']['process_memory_kib']['VmRSS']/1024 for q in qs)}
 rows.append({'case':case,**pair,'e2e_reduction_pct':100*(1-pair['on']['e2e_s']/pair['off']['e2e_s']),
  'rss_after_reduction_mib':pair['off']['rss_after_mib']-pair['on']['rss_after_mib']})
result={'passed':True,'requests_checked':count,'formal_requests':12,'warmups':4,'same_binary':True,'quality_exact':True,
 'rows':rows,'cache_states':stats,'missing_phase_timings':missing,
 'scope':'Two-arm OFF(default cache)/ON(cache-ram0) exploratory supplement, 3 formal per state per case. Less drift control than four-arm ABBA. Same final v47 kernels and cache_prompt=false; all E2E retained. No claim of disabling every KV/media cache, no long-soak validation of cache-ram0.'}
(OUT/'cache-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for r in rows:print(r['case'],r['off']['e2e_s'],'->',r['on']['e2e_s'],'RSS saved',r['rss_after_reduction_mib'])
