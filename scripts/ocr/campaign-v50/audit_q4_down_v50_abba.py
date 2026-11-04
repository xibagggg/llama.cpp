from pathlib import Path
from statistics import mean,stdev
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-q4-down-attention'
groups={'off':{},'on':{}};warmups=0;identities={}
for arm in ['off-a','on-a','on-b','off-b']:
 d=BASE/f'20261104-q4-down-v50-abba-{arm}';s=json.loads((d/'round.json').read_text(encoding='utf8'))
 assert s['restored']['health']['status']=='ok';assert len(s['runs'])==1
 r=s['runs'][0];assert r['status']=='completed' and len(r['requests'])==6
 state=arm.split('-')[0];identities[arm]={'command':r['command'],'environment':r['environment']}
 assert s['measurement']['repeats']==1 and s['measurement']['warmups']==1
 for q in r['requests']:
  assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop'
  for name,key in [('output.txt','output_sha256'),('vision-embeddings.bin','embedding_sha256')]:
   assert hashlib.sha256((d/q['directory']/name).read_bytes()).hexdigest()==q[key]
  if q['warmup']:warmups+=1;continue
  groups[state].setdefault(q['case_id'],[]).append(q)
assert identities['off-a']['command']==identities['off-b']['command'] and identities['on-a']['command']==identities['on-b']['command']
rows=[]
for case in ['03_table_gb_standard','v5_table_merge4','04_scanned_civil_trial']:
 z={'case':case};ref={}
 for state in ['off','on']:
  qs=groups[state][case];assert len(qs)==2
  assert len({q['output_sha256'] for q in qs})==1 and len({q['embedding_sha256'] for q in qs})==1
  assert all(q['token_ids']==qs[0]['token_ids'] for q in qs)
  for q in qs:
   assert q['vision_timing_events']['valid'] and q['text_prefill_s']>=0
   assert all(q['system_after']['cpufreq'][p]['scaling_cur_freq_khz']==v for p,v in [('policy0',2000000),('policy8',1800000)])
  def avg(key):return mean(q[key] for q in qs)
  z[state]={'n':2,'e2e_s':avg('e2e_s'),'e2e_sd_s':stdev(q['e2e_s'] for q in qs),'vision_s':avg('vision_encode_s'),
            'prefill_s':avg('text_prefill_s'),'decode_s':mean(q['timings']['predicted_ms']/1000 for q in qs),
            'tps':mean(q['timings']['predicted_per_second'] for q in qs),
            'rss_after_mib':mean(q['system_after']['process_memory_kib']['VmRSS']/1024 for q in qs)}
  ref[state]=qs[0]
 z['text_equal']=ref['off']['output']==ref['on']['output'];z['tokens_equal']=ref['off']['token_ids']==ref['on']['token_ids']
 z['e2e_reduction_pct']=100*(1-z['on']['e2e_s']/z['off']['e2e_s'])
 z['vision_reduction_pct']=100*(1-z['on']['vision_s']/z['off']['vision_s'])
 rows.append(z)
result={'passed':True,'same_binary':True,'formal_requests':12,'warmup_requests':warmups,
        'repeat_scope':'OFF-A/ON-A/ON-B/OFF-B, each arm each case warmup1 + formal1; n=2 per state/case',
        'quantization_changes_numerics':True,'within_state_text_tokens_last_slice_embeddings_exact':True,'rows':rows,'identities':identities}
(OUT/'formal-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(rows,ensure_ascii=False,indent=2))
