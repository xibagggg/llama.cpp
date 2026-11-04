"""Offline forensics for failed bitwise-repeat acceptance; preserve the failure."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-attention-v61'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
baseline=BASE/'20261104-q4-down-v50-quality-q4'
refs={q['case_id']:(baseline/q['directory']/'vision-embeddings.bin') for q in read(baseline/'round.json')['runs'][0]['requests']}
records={}
for arm in ['off-a','on-a','on-b','off-b']:
 d=BASE/('20261104-attention-v61-abba-'+arm)
 for q in read(d/'round.json')['runs'][0]['requests']:
  p=d/q['directory']/'vision-embeddings.bin';assert hashlib.sha256(p.read_bytes()).hexdigest()==q['embedding_sha256']
  records.setdefault(q['case_id'],[]).append((arm,q,p))
rows=[]
for case,rs in records.items():
 ref_blob=refs[case].read_bytes();ref=np.frombuffer(ref_blob,dtype='<f4',offset=8).astype(np.float64)
 first={}
 for arm,q,p in rs:
  blob=p.read_bytes();assert blob[:8]==ref_blob[:8]
  state=arm.split('-')[0];y=np.frombuffer(blob,dtype='<f4',offset=8).astype(np.float64);assert y.shape==ref.shape and np.all(np.isfinite(y))
  first.setdefault(state,y)
  def metric(x):
   diff=y-x
   return {'relative_l2':float(np.linalg.norm(diff)/np.linalg.norm(x)),'max_abs':float(np.max(np.abs(diff))),
    'different_elements':int(np.count_nonzero(diff)),'total_elements':len(y),'rms_diff':float(np.sqrt(np.mean(diff**2)))}
  rows.append({'case':case,'arm':arm,'warmup':q['warmup'],'directory':q['directory'],'sha256':q['embedding_sha256'],
   'vs_frozen_v50':metric(ref),'vs_first_same_state':metric(first[state])})
out={'acceptance_passed':False,'reason':'Formal bitwise repeat failed for saved last-slice vision features; text/tokens equal.',
 'scope':'All 24 ABBA requests including warmups. Float32 saved last-slice only. No new board inference.', 'rows':rows}
(OUT/'feature-repeat-failure.json').write_text(json.dumps(out,indent=2),encoding='utf8')
for r in rows:print(json.dumps({k:v for k,v in r.items() if k not in ['directory','sha256']},ensure_ascii=False))
