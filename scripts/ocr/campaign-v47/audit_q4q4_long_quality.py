"""Check complete OCR evidence against frozen v43, including fresh saved embeddings."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization'
p=argparse.ArgumentParser();p.add_argument('round');p.add_argument('--out',required=True);a=p.parse_args()
load=lambda p:json.loads(p.read_text(encoding='utf8'))
reference=load(BASE/'20261104-q4q4-v43-quality-selected/round.json')['runs'][0]
refs={q['case_id']:q for q in reference['requests']}
folder=BASE/a.round;d=load(folder/'round.json');assert d['restored']['health']['status']=='ok'
rows=[]
for r in d['runs']:
 assert r['status']=='completed'
 for q in r['requests']:
  b=refs[q['case_id']]
  assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop'
  assert q['vision_timing_events']['valid'] and q['text_prefill_s']>=0
  assert all(q[k]==b[k] for k in ['output','token_ids','embedding_sha256']),(q['case_id'],q.get('repeat'))
  assert hashlib.sha256((folder/q['directory']/'vision-embeddings.bin').read_bytes()).hexdigest()==q['embedding_sha256']
  rows.append({'case':q['case_id'],'repeat':q['repeat'],'warmup':q['warmup'],'exact_output':True,
   'exact_token_ids':True,'exact_last_slice_embedding':True,'e2e_s':q['e2e_s'],'vision_s':q['vision_encode_s'],
   'prefill_s':q['text_prefill_s'],'generation_s':q['timings']['predicted_ms']/1000,
   'generation_token_s':q['timings']['predicted_per_second'],
   'rss_after_mib':q['system_after']['process_memory_kib']['VmRSS']/1024})
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'passed':True,'round':a.round,'requests':len(rows),
 'scope':'saved final-slice embeddings only; exact vs v43 not a Q4 vs Q8/FP16 accuracy study','rows':rows},indent=2),encoding='utf8')
print(a.round,'exact',len(rows),'requests')
