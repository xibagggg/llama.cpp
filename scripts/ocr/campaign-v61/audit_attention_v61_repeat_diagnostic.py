from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-attention-v61'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
b=BASE/'20261104-q4-down-v50-quality-q4';ref=next(q for q in read(b/'round.json')['runs'][0]['requests'] if q['case_id']=='03_table_gb_standard')
rb=(b/ref['directory']/'vision-embeddings.bin').read_bytes();x=np.frombuffer(rb,dtype='<f4',offset=8).astype(np.float64)
rows=[]
for version in ['v50','v61']:
 d=BASE/f'20261104-attention-v61-repeat-diagnostic-{version}';s=read(d/'round.json');assert s['restored']['health']['status']=='ok'
 qs=s['runs'][0]['requests'];assert len(qs)==4
 for q in qs:
  assert q['output']==ref['output'] and q['token_ids']==ref['token_ids']
  blob=(d/q['directory']/'vision-embeddings.bin').read_bytes()
  assert hashlib.sha256(blob).hexdigest()==q['embedding_sha256'] and blob[:8]==rb[:8]
  y=np.frombuffer(blob,dtype='<f4',offset=8).astype(np.float64);assert np.all(np.isfinite(y))
  rows.append({'version':version,'repeat':q['repeat'],'text_tokens_equal':True,'feature_equal_frozen_v50':blob==rb,
   'sha256':q['embedding_sha256'],'last_slice_relative_l2':float(np.linalg.norm(y-x)/np.linalg.norm(x))})
result={'scope':'One scan table; one warmup plus three formal per runtime, both compute0. Diagnostic only; not proof v50 generally stable.',
 'v50_four_exact':all(r['feature_equal_frozen_v50'] for r in rows if r['version']=='v50'),
 'v61_mismatches':sum(not r['feature_equal_frozen_v50'] for r in rows if r['version']=='v61'),'rows':rows}
assert result['v50_four_exact'] and result['v61_mismatches']==1
(OUT/'repeat-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
