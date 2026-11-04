"""Summarize the bounded D72 QK experiment; no whole-model extrapolation."""
from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/optimization'
OUT=BASE/'20261104-q4-down-attention'
d=BASE/'20261104-qk-ime2-v55-micro'
s=json.loads((d/'round.json').read_text(encoding='utf8'))
assert s['restored']['health']['status']=='ok'
checks=json.loads((d/'cases/numerical-check.json').read_text())
assert len(checks)==6 and all(r['relative_l2']<2e-5 for r in checks)
rows=[]
for n in [65,1008]:
 for layout in [0,1,6]:
  times={}
  for mode in [0,3]:
   rs=[json.loads(l) for l in (d/f'cases/N{n}-L{layout}-M{mode}.jsonl').read_text().splitlines()]
   assert len(rs)==6 and all(r['repeat_exact'] and r['finite'] and r['guards'] and r['input_immutable'] for r in rs)
   times[mode]=statistics.mean(r['ms'] for r in rs)
  rows.append({'N':n,'layout':layout,'mean_ms':times,'reduction_pct':100*(1-times[3]/times[0])})
z={'passed':True,'formal_runs':72,'scope':'full attention microbenchmark, synthetic input; includes shared packing, copies and barriers; not end-to-end OCR',
   'rows':rows,'numerical':checks,'candidate_faster_on_all_N1008_layouts':all(r['reduction_pct']>0 for r in rows if r['N']==1008)}
(OUT/'qk-v55-micro-audit.json').write_text(json.dumps(z,indent=2),encoding='utf8')
print(json.dumps(z,indent=2))
