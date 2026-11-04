"""Separate optimization parity from actual OCR errors on four explicit table GTs."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-q4q4-long-1430'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
groups={};commands=[];unavailable_timings=[]
cases={'v5_table_cn_clear','v5_table_cn_blur','v5_table_merge4','v5_table_numbers5'}
for arm in ['off','on']:
 folder=BASE/f'20261104-q4q4-v47-extra-tables-{arm}';d=read(folder/'round.json')
 assert d['restored']['health']['status']=='ok' and len(d['runs'])==1
 r=d['runs'][0];assert r['status']=='completed';commands.append(r['command'])
 assert d['measurement']['repeats']==1 and d['measurement']['warmups']==0
 assert d['measurement']['down_vector']==(5 if arm=='on' else None)
 assert d['measurement']['fa_direct']==(2 if arm=='on' else None)
 assert d['measurement']['concat_tcm']==(4 if arm=='on' else None)
 assert {q['case_id'] for q in r['requests']}==cases
 groups[arm]={}
 for q in r['requests']:
  assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop'
  # This round is for quality only; retain unavailable interleaved-log timing.
  if not q['vision_timing_events']['valid'] or q.get('text_prefill_s') is None:
   unavailable_timings.append({'arm':arm,'case':q['case_id'],'vision_timing_events':q['vision_timing_events']})
  for name,key in [('vision-embeddings.bin','embedding_sha256'),('output.txt','output_sha256')]:assert hashlib.sha256((folder/q['directory']/name).read_bytes()).hexdigest()==q[key]
  groups[arm][q['case_id']]=q
assert commands[0]==commands[1]
for case in cases:assert all(groups['off'][case][k]==groups['on'][case][k] for k in ['output','token_ids','embedding_sha256']),case
gt=read(OUT/'extra-tables-quality/quality.json')
assert len(gt['summary'])==2 and all(r['requests']==4 and r['unique_cases']==4 for r in gt['summary'])
result={'passed':True,'requests':8,'same_binary':True,'all_four_cases_exact_text_tokens_last_slice_embeddings':True,
 'gt_summary':gt['summary'],'gt_details_sha256':hashlib.sha256((OUT/'extra-tables-quality/quality.json').read_bytes()).hexdigest(),
 'unavailable_timings':unavailable_timings,
 'scope':'four GT cases, one request per state; reports OCR/structure errors as measured, never corrects output'}
(OUT/'extra-tables-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('Extra tables: eight requests, four cases exact across states; GT results preserved')
