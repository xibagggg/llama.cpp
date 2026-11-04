"""Publish descriptive timing despite failed feature acceptance, never mark it passed."""
from pathlib import Path
from statistics import mean,stdev
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-attention-v61'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
failure=read(OUT/'feature-repeat-failure.json');assert failure['acceptance_passed'] is False
frozen=read(BASE/'20261104-q4-down-v50-quality-q4/round.json')
reference={q['case_id']:q for q in frozen['runs'][0]['requests']}
groups={'off':{},'on':{}};identities={};warm=0;all_requests=[]
for arm in ['off-a','on-a','on-b','off-b']:
 label='20261104-attention-v61-abba-'+arm;d=BASE/label;s=read(d/'round.json')
 assert s['restored']['health']['status']=='ok'
 r=s['runs'][0];assert r['status']=='completed' and len(r['requests'])==6
 state=arm.split('-')[0];identities[label]={'command':r['command'],'environment':r['environment']}
 assert r['environment']['MTMD_OCR_PADDED_DOWN']=='1'
 assert r['environment']['GGML_SPACEMIT_OCR_FA_COMPUTE']==('0' if state=='off' else '3')
 for q in r['requests']:
  assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop'
  for name,key in [('output.txt','output_sha256'),('vision-embeddings.bin','embedding_sha256')]:
   assert hashlib.sha256((d/q['directory']/name).read_bytes()).hexdigest()==q[key]
  assert q['output_sha256']==reference[q['case_id']]['output_sha256'] and q['token_ids']==reference[q['case_id']]['token_ids']
  assert q['vision_timing_events']['valid'] and q['text_prefill_s']>=0
  assert all(q['system_after']['cpufreq'][p]['scaling_cur_freq_khz']==v for p,v in [('policy0',2000000),('policy8',1800000)])
  all_requests.append(q)
  if q['warmup']:warm+=1;continue
  groups[state].setdefault(q['case_id'],[]).append(q)
assert warm==12 and len(all_requests)==24
ref=next(iter(identities.values()))
for item in identities.values():
 assert item['command']==ref['command']
 skip={'MTMD_DEBUG_EMBEDDINGS','GGML_SPACEMIT_OCR_FA_COMPUTE'}
 assert {k:v for k,v in item['environment'].items() if k not in skip}=={k:v for k,v in ref['environment'].items() if k not in skip}
rows=[]
for case in ['03_table_gb_standard','v5_table_merge4','04_scanned_civil_trial']:
 row={'case':case}
 for state in ['off','on']:
  qs=groups[state][case];assert len(qs)==2
  row[state]={'n':2,'vision_s':mean(q['vision_encode_s'] for q in qs),'prefill_s':mean(q['text_prefill_s'] for q in qs),
   'decode_s':mean(q['timings']['predicted_ms']/1000 for q in qs),'tps':mean(q['timings']['predicted_per_second'] for q in qs),
   'e2e_s':mean(q['e2e_s'] for q in qs),'e2e_sd_s':stdev(q['e2e_s'] for q in qs),
   'rss_after_mib':mean(q['system_after']['process_memory_kib']['VmRSS']/1024 for q in qs),
   'samples':[{'e2e_s':q['e2e_s'],'vision_s':q['vision_encode_s']} for q in qs],
   'feature_repeat_exact':len({q['embedding_sha256'] for q in qs})==1,
   'frozen_v50_feature_matches':[q['embedding_sha256']==reference[case]['embedding_sha256'] for q in qs]}
 row.update(text_equal=True,tokens_equal=True)
 row['e2e_reduction_pct']=100*(1-row['on']['e2e_s']/row['off']['e2e_s'])
 row['vision_reduction_pct']=100*(1-row['on']['vision_s']/row['off']['vision_s'])
 rows.append(row)
assert any(not row[state]['feature_repeat_exact'] for row in rows for state in ['off','on'])
formal={'passed':False,'timing_evidence_valid':True,'acceptance_failure':'Saved last-slice features differ across repeated OFF and ON requests. Original strict audit retained and still fails.',
 'formal_requests':12,'warmup_requests':12,'same_binary':True,'rows':rows,'all_24_text_tokens_equal_frozen_v50':True,
 'scope':'Descriptive observations only, OFF-A/ON-A/ON-B/OFF-B; n=2 per state/case. Not an accepted speed candidate.',
 'within_state_repeat_exact':False,'off_matches_frozen_v50_text_tokens_last_slice_embedding':False,'identities':identities}
(OUT/'formal-audit.json').write_text(json.dumps(formal,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in formal.items() if k!='identities'},indent=2))
