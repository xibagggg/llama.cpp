from pathlib import Path
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1];DIR=ROOT/'results/optimization/20261104-fa-compute-v52-micro';OUT=ROOT/'results/optimization/20261104-q4-down-attention'
s=json.loads((DIR/'round.json').read_text(encoding='utf8'));assert s['restored']['health']['status']=='ok'
assert s['checker']['exit_code']==0 if 'checker' in s else True
rows=[];output_hashes=json.loads((DIR/'cases/output-hashes.json').read_text())
for N in [65,1008]:
 for layout in [0,1,6]:
  times={};hashes={}
  for mode in [0,1,2]:
   p=DIR/f'cases/N{N}-L{layout}-M{mode}';z=[json.loads(l) for l in p.with_suffix('.jsonl').read_text().splitlines()]
   assert len(z)==6 and all(r['repeat_exact'] and r['finite'] and r['guards'] and r['input_immutable'] for r in z)
   times[mode]=statistics.mean(r['ms'] for r in z);hashes[mode]=output_hashes[p.with_suffix('.bin').name]
  assert len(set(hashes.values()))==1,(N,layout,hashes)
  rows.append({'N':N,'layout':layout,'mean_ms':times,'reduction_pct':{m:100*(times[0]-times[m])/times[0] for m in [1,2]},'all_modes_bitwise_equal':True})
down=[json.loads(l) for l in (DIR/'down.jsonl').read_text().splitlines() if l.startswith('{')]
assert len(down)==15 and all(r['repeat_exact'] and r['finite'] and r['padding_zero'] for r in down)
z={'passed':True,'attention_formal_runs':108,'down_formal_runs':15,'rows':rows,'down':down}
(OUT/'micro-audit.json').write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2))
