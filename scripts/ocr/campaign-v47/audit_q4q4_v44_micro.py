from pathlib import Path
import json,statistics
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization'
P=BASE/'20261104-q4q4-v44-micro';OUT=BASE/'20261104-q4q4-v44'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
d=load(P/'round.json');assert d['checker_returncode']==0 and d['restored']['health']['status']=='ok'
a=[json.loads(l) for l in (P/'down.jsonl').read_text().splitlines() if l.startswith('{')]
assert a[-1]=={'kind':'final','input_immutable':True,'all_exact':True}
assert all(x['bitwise_exact'] and x['guards'] for x in a if x['kind']=='timing')
assert all(x['saved_bit_mismatch']==0 for x in a if x['kind']=='reference')
down=[]
for n in [7,65,988,989,1008,1009,1056,1057,1060]:
 modes={}
 for m in range(8):
  times=[x['ms'] for x in a if x['kind']=='timing' and x['N']==n and x['mode']==m];assert len(times)==6
  modes[str(m)]={'mean_ms':statistics.mean(times),'sd_ms':statistics.stdev(times)}
 for m in modes:modes[m]['reduction_pct']=100*(1-modes[m]['mean_ms']/modes['0']['mean_ms'])
 down.append({'N':n,'modes':modes})
fa=[];log=(P/'checker.log').read_text(encoding='utf8')
for n in [64,65,988,1056]:
 for layout in range(6):
  assert f'DIRECT EXACT N={n} layout={layout}' in log
  modes={}
  for m in range(4):
   rows=[]
   for suffix in ['a','b']:
    r=[json.loads(l) for l in (P/f'N{n}-l{layout}-{m}{suffix}.jsonl').read_text().splitlines() if l.startswith('{')]
    assert len(r)==3 and all(x['finite'] and x['guards'] and x['input_immutable'] and x['repeat_exact'] for x in r)
    rows+=r
   t=[x['ms'] for x in rows];modes[str(m)]={'mean_ms':statistics.mean(t),'sd_ms':statistics.stdev(t)}
  for m in modes:modes[m]['reduction_pct']=100*(1-modes[m]['mean_ms']/modes['0']['mean_ms'])
  fa.append({'N':n,'layout':layout,'modes':modes})
result={'passed':True,'down_formal_runs':432,'attention_formal_runs':576,
 'down_scope':'real layer0 N1056; other N truncated/cyclic extended; v43 compact20 baseline',
 'attention_scope':'synthetic D72 H16; six layouts, poison scratch included; v43 lazy initialization baseline',
 'down_modes':{'0':'v43 compact20','1':'e32m4 dynamic VL','2':'fixed VL main plus tail',
  '3':'two-row interleaved widening + e32','4':'two-row widening + fixed VL','5':'aligned4352 + e32',
  '6':'aligned4352 + fixed VL','7':'aligned4352 + two-row widening'},
 'attention_modes':{'0':'copy K/V into TCM','1':'K direct DDR','2':'V direct DDR','3':'K/V direct DDR'},
 'down':down,'attention':fa}
(OUT/'micro-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for r in down:
 print('DOWN',r['N'],[(k,round(v['mean_ms'],3),round(v['reduction_pct'],2)) for k,v in r['modes'].items()])
for r in fa:
 if r['N']>=988:print('FA',r['N'],r['layout'],[(k,round(v['mean_ms'],3),round(v['reduction_pct'],2)) for k,v in r['modes'].items()])
