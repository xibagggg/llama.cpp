from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization'
P=BASE/'20261104-q4q4-v45-micro';OUT=BASE/'20261104-q4q4-v45'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
d=load(P/'round.json');assert d['checker_returncode']==0 and d['restored']['health']['status']=='ok'
log=(P/'checker.log').read_text(encoding='utf8');rows=[]
for direct in [0,2]:
 for n in [64,65,988,1056]:
  for layout in range(6):
   assert f'GROUP EXACT N={n} layout={layout} direct={direct}' in log
   modes={}
   for mode in range(7):
    r=[]
    for arm in ['a','b']:
     z=[json.loads(l) for l in (P/f'N{n}-l{layout}-d{direct}-{mode}{arm}.jsonl').read_text().splitlines() if l.startswith('{')]
     assert len(z)==3 and all(x['finite'] and x['guards'] and x['input_immutable'] and x['repeat_exact'] for x in z)
     r+=z
    t=[x['ms'] for x in r];modes[str(mode)]={'mean_ms':statistics.mean(t),'sd_ms':statistics.stdev(t)}
   for mode in modes:modes[mode]['reduction_pct']=100*(1-modes[mode]['mean_ms']/modes['0']['mean_ms'])
   rows.append({'N':n,'layout':layout,'direct':direct,'modes':modes})
result={'passed':True,'attention_formal_runs':2016,'scope':'synthetic D72/H16; poison scratch and preparation included',
 'modes':{'0':'QK8 PV8','1':'QK10 PV8','2':'QK12 PV8','3':'QK8 PV10','4':'QK8 PV12','5':'QK10 PV10','6':'QK12 PV12'},'rows':rows}
(OUT/'micro-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for r in rows:
 if r['N']>=988 and r['layout'] in [0,1]:
  print('GROUP',r['N'],r['layout'],'direct',r['direct'],[(k,round(v['mean_ms'],3),round(v['reduction_pct'],2)) for k,v in r['modes'].items()])
