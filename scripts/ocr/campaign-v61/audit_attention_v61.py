"""Bounded attention micro, quality and optional same-binary formal audit."""
from pathlib import Path
from statistics import mean,stdev
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-attention-v61';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def exact_record(d,q):
 assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop'
 for f,key in [('output.txt','output_sha256'),('vision-embeddings.bin','embedding_sha256')]:
  assert hashlib.sha256((d/q['directory']/f).read_bytes()).hexdigest()==q[key]
def round_records(label):
 d=BASE/label;s=read(d/'round.json');assert s['restored']['health']['status']=='ok'
 return d,s,s['runs'][0]['requests']
micro=[]
for version in [57,59,60,61]:
 d=BASE/f'20261104-qk-ime2-v{version}-micro';s=read(d/'round.json');assert s['restored']['health']['status']=='ok'
 mode=4 if version==60 else 3
 for n in [65,1008]:
  for layout in [0,6]:
   ts={}
   for m in [0,mode]:
    rs=[json.loads(l) for l in (d/f'cases/N{n}-L{layout}-M{m}.jsonl').read_text().splitlines()]
    assert len(rs)==6 and all(r['finite'] and r['guards'] and r['repeat_exact'] and r['input_immutable'] for r in rs)
    ts[m]=mean(r['ms'] for r in rs)
   micro.append({'version':version,'N':n,'layout':layout,'mean_ms':ts,'reduction_pct':100*(1-ts[mode]/ts[0])})
edges=read(BASE/'20261104-qk-edges-v61/cases/numerical-check.json');assert len(edges)==9
for p in (BASE/'20261104-qk-edges-v61/cases').glob('*.jsonl'):
 rs=[json.loads(l) for l in p.read_text().splitlines()];assert len(rs)==3 and all(r['finite'] and r['guards'] and r['repeat_exact'] and r['input_immutable'] for r in rs)
(OUT/'micro-audit.json').write_text(json.dumps({'passed':True,'rows':micro,'edge_cases':edges,'edge_formal_runs':54},indent=2),encoding='utf8')
label='20261104-q4-down-v61-quality'
if not (BASE/label/'round.json').exists():raise SystemExit('Micro audit complete; quality round not fetched yet.')
d,s,qs=round_records(label);assert len(qs)==10
bd,bs,bqs=round_records('20261104-q4-down-v50-quality-q4');bmap={q['case_id']:q for q in bqs}
quality=[]
for q in qs:
 exact_record(d,q);b=bmap[q['case_id']];exact_record(bd,b)
 xb=(bd/b['directory']/'vision-embeddings.bin').read_bytes();yb=(d/q['directory']/'vision-embeddings.bin').read_bytes()
 assert xb[:8]==yb[:8]
 x=np.frombuffer(xb,dtype='<f4',offset=8).astype(np.float64)
 y=np.frombuffer(yb,dtype='<f4',offset=8).astype(np.float64)
 assert x.shape==y.shape and np.all(np.isfinite(y))
 quality.append({'case':q['case_id'],'text_equal':q['output']==b['output'],'tokens_equal':q['token_ids']==b['token_ids'],
 'embedding_exact':q['embedding_sha256']==b['embedding_sha256'],'last_slice_embedding_rel_l2':float(np.linalg.norm(y-x)/np.linalg.norm(x)),
 'last_slice_embedding_max_abs':float(np.max(np.abs(y-x))),'e2e_s':q['e2e_s'],'vision_s':q['vision_encode_s'],
 'output_sha256':q['output_sha256']})
(OUT/'quality-audit.json').write_text(json.dumps({'all_ten_text_tokens_equal':all(r['text_equal'] and r['tokens_equal'] for r in quality),
 'scope':'ten selected documents, one request each; feature check covers last saved slice only; reference v50 padded Q4 model', 'rows':quality},indent=2),encoding='utf8')
labels=[f'20261104-attention-v61-abba-{arm}' for arm in ['off-a','on-a','on-b','off-b']]
if not all((BASE/p/'round.json').exists() for p in labels):raise SystemExit('Quality audit complete; formal rounds not fetched yet.')
groups={'off':{},'on':{}};identities={};warm=0
for label in labels:
 d,s,qs=round_records(label);r=s['runs'][0];assert r['status']=='completed' and len(qs)==6
 state=label.split('-')[-2];identities[label]={'command':r['command'],'environment':r['environment']}
 assert r['environment']['MTMD_OCR_PADDED_DOWN']=='1'
 assert r['environment']['GGML_SPACEMIT_OCR_FA_COMPUTE']==('0' if state=='off' else '3')
 for q in qs:
  exact_record(d,q)
  if q['warmup']:warm+=1;continue
  groups[state].setdefault(q['case_id'],[]).append((d,q))
rows=[]
for case in ['03_table_gb_standard','v5_table_merge4','04_scanned_civil_trial']:
 z={'case':case}
 for state in ['off','on']:
  qs=[q for d,q in groups[state][case]];assert len(qs)==2
  assert len({q['output_sha256'] for q in qs})==len({q['embedding_sha256'] for q in qs})==1
  assert all(q['token_ids']==qs[0]['token_ids'] for q in qs)
  assert all(q['vision_timing_events']['valid'] and q['text_prefill_s']>=0 for q in qs)
  assert all(q['system_after']['cpufreq'][p]['scaling_cur_freq_khz']==v for q in qs for p,v in [('policy0',2000000),('policy8',1800000)])
  z[state]={'n':2,'vision_s':mean(q['vision_encode_s'] for q in qs),'prefill_s':mean(q['text_prefill_s'] for q in qs),
   'decode_s':mean(q['timings']['predicted_ms']/1000 for q in qs),'tps':mean(q['timings']['predicted_per_second'] for q in qs),
   'e2e_s':mean(q['e2e_s'] for q in qs),'e2e_sd_s':stdev(q['e2e_s'] for q in qs),
   'rss_after_mib':mean(q['system_after']['process_memory_kib']['VmRSS']/1024 for q in qs),
   'samples':[{'e2e_s':q['e2e_s'],'vision_s':q['vision_encode_s']} for q in qs]}
  if state=='off':
   b=bmap[case];assert all(q['output_sha256']==b['output_sha256'] and q['embedding_sha256']==b['embedding_sha256'] and q['token_ids']==b['token_ids'] for q in qs)
 a,b=[groups[state][case][0][1] for state in ['off','on']]
 z['text_equal']=a['output']==b['output'];z['tokens_equal']=a['token_ids']==b['token_ids']
 assert z['text_equal'] and z['tokens_equal'],case
 z['e2e_reduction_pct']=100*(1-z['on']['e2e_s']/z['off']['e2e_s']);z['vision_reduction_pct']=100*(1-z['on']['vision_s']/z['off']['vision_s'])
 rows.append(z)
assert warm==12
reference=next(iter(identities.values()))
for record in identities.values():
 assert record['command']==reference['command']
 skip={'MTMD_DEBUG_EMBEDDINGS','GGML_SPACEMIT_OCR_FA_COMPUTE'}
 assert {k:v for k,v in record['environment'].items() if k not in skip}=={k:v for k,v in reference['environment'].items() if k not in skip}
formal={'passed':True,'formal_requests':12,'warmup_requests':warm,'same_binary':True,'rows':rows,
 'scope':'OFF-A/ON-A/ON-B/OFF-B, one warmup+one formal per case/arm; n=2 per state/case',
 'off_matches_frozen_v50_text_tokens_last_slice_embedding':True,'within_state_repeat_exact':True,'identities':identities}
(OUT/'formal-audit.json').write_text(json.dumps(formal,indent=2),encoding='utf8');print(json.dumps(rows,indent=2))
