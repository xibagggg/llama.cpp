from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization'
P=BASE/'20261104-q4q4-v46-micro';OUT=BASE/'20261104-q4q4-v46'
d=json.loads((P/'round.json').read_text(encoding='utf8'));assert d['checker_returncode']==0 and d['restored']['health']['status']=='ok'
raw=[json.loads(l) for l in (P/'checker.log').read_text(encoding='utf8').splitlines() if l.startswith('{')]
assert len(raw)==1080 and all(x['bitwise_exact'] and x['guards'] and x['input_immutable'] for x in raw)
keys=sorted({(x['tokens'],x['channels'],x['batches'],x['col0']) for x in raw});assert len(keys)==36
rows=[]
for tokens,channels,batches,col0 in keys:
 modes={}
 for m in range(5):
  t=[r['ms'] for r in raw if (r['tokens'],r['channels'],r['batches'],r['col0'],r['mode'])==(tokens,channels,batches,col0,m)];assert len(t)==6
  modes[str(m)]={'mean_ms':statistics.mean(t),'sd_ms':statistics.stdev(t)}
 for m in modes:modes[m]['reduction_pct']=100*(1-modes[m]['mean_ms']/modes['0']['mean_ms'])
 row={'tokens':tokens,'channels':channels,'batches':batches,'col0':col0,'modes':modes};rows.append(row)
 if channels==6144 and tokens>=63 and col0==0:print(tokens,[(m,round(z['mean_ms'],3),round(z['reduction_pct'],2)) for m,z in modes.items()])
result={'passed':True,'formal_runs':1080,'modes':{'0':'original strided DDR stores','1':'C16 T32','2':'C32 T64','3':'C32 T128','4':'C64 T64'},
 'scope':'synthetic arbitrary float bit patterns including NaN/infinity; actual CONCAT shapes, token/channel tails, two batches, full/trailing-only writes; TCM poisoning included','rows':rows}
(OUT/'micro-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
