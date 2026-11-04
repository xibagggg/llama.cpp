from pathlib import Path
from statistics import mean
from collections import defaultdict
import argparse,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/optimization/20261104-q4-down-attention'
p=argparse.ArgumentParser();p.add_argument('round');p.add_argument('--output',default='attention-phases-audit.json');a=p.parse_args()
d=ROOT/'results/optimization'/a.round;s=json.loads((d/'round.json').read_text(encoding='utf8'))
assert s['restored']['health']['status']=='ok' and len(s['runs'])==1
r=s['runs'][0];q=r['requests'];assert len(q)==1 and q[0]['status']=='completed'
records=[json.loads(l) for l in (d/r['name']/'fa-phases/attention-phases.jsonl').read_text().splitlines()]
lanes=[r for r in records if r['kind']=='lane'];assert len(lanes)==8 and all(r['dropped']==0 for r in lanes)
records=[r for r in records if r['kind']=='phase'];by=defaultdict(list)
for r in records:
 assert r['N']==1008 and 0<=r['worker']<7 and r['end_ns']>=r['start_ns']
 by[r['worker'],r['stage']].append(r)
assert all(len(by[w,'prepare'])==135 and len(by[w,'compute'])==135 for w in range(7))
keys=sorted({k for r in records for k in r if k.endswith('_ns')}-{'start_ns','end_ns'})
ns={k:sum(r.get(k,0) for r in records) for k in keys}
total=sum(ns.values());envelopes=[]
for stage in ['prepare','compute','final_sync']:
 for i in range(len(by[0,stage])):
  group=[by[w,stage][i] for w in range(7)];assert len({z['name'] for z in group})==1
  envelopes.append((min(z['start_ns'] for z in group),max(z['end_ns'] for z in group)))
merged=[]
for start,end in sorted(envelopes):
 if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
 else:merged.append((start,end))
wall=sum(end-start for start,end in merged)
result={'diagnostic_only':True,'scope':'five vision slices x27 layers, D72/H16/N1008; phase numbers are sum of worker elapsed times divided by7, NOT additive critical-path wall time or hardware counters',
        'vision_attention_calls':135,'workers':7,'wall_interval_union_over_stages_s':wall/1e9,
        'stage_envelope_sum_s':sum(end-start for start,end in envelopes)/1e9,
        'stage_mean_worker_s':{k:v/7e9 for k,v in ns.items()},'stage_share_pct':{k:100*v/total for k,v in ns.items()},
        'phase_coverage':{'records':len(records),'final_sync_records':sum(len(by[w,'final_sync']) for w in range(7))},
        'output_sha256':q[0]['output_sha256'],'token_ids':q[0]['token_ids']}
(OUT/a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result['stage_mean_worker_s'],indent=2))
