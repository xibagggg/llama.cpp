"""Nominal matrix work per observed profile interval; no bus-counter/utilization claim."""
from pathlib import Path
import collections,csv,json,re
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results/optimization/20261104-q4q4-v47-profile-selected/analysis'
groups=collections.defaultdict(lambda:{'calls':0,'wall_ms':0.0,'nominal_operations':0})
for r in csv.DictReader((P/'operators.csv').open(encoding='utf-8-sig')):
 if r['operator']!='MUL_MAT':continue
 w=json.loads(r['src0_shape']);x=json.loads(r['src1_shape']);y=json.loads(r['out_shape'])
 assert w[0]==x[0] and w[1]==y[0] and x[1]==y[1]
 mode='all'
 if r['phase']=='vision' and r['category']=='FFN_DOWN_F16':
  layer=int(re.search(r'v\.blk\.(\d+)\.',r['src0'])[1]);mode='RVV_first14' if layer<14 else 'IME2_last13'
 key=(r['phase'],r['category'],mode,tuple(w),tuple(x),tuple(y))
 z=groups[key];z['calls']+=1;z['wall_ms']+=float(r['wall_ms']);z['nominal_operations']+=2*w[0]*y[0]*y[1]*y[2]*y[3]
rows=[]
for (phase,category,mode,w,x,y),z in groups.items():
 rows.append({'phase':phase,'category':category,'path':mode,'weight_shape':w,'input_shape':x,'output_shape':y,**z,
  'mean_ms':z['wall_ms']/z['calls'],'nominal_gop_per_s':z['nominal_operations']/(z['wall_ms']/1000)/1e9})
rows.sort(key=lambda r:-r['wall_ms'])
result={'scope':'Derived from one PROFILE ON complete request. Two nominal operations per multiply-add; includes interval preparation/packing/synchronization. Not hardware instruction counters, utilization, physical DDR bandwidth, or PROFILE OFF performance.',
 'rows':rows}
(P/'matrix-shapes.json').write_text(json.dumps(result,indent=2),encoding='utf8')
for r in rows:
 if r['phase']=='vision' and r['category']=='FFN_DOWN_F16' or r['phase']=='decode' and r['category']=='MUL_MAT_q6_K':print(json.dumps(r))
