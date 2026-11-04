"""Split a one-request v33 trace into vision, prefill and decode operator unions."""
import argparse,collections,csv,hashlib,json
from pathlib import Path
from review_profile_by_vision_chunk import union_us,classify

def analyze(round_dir):
    state=json.loads((round_dir/'round.json').read_text(encoding='utf8'))
    assert len(state['runs'])==1
    run=state['runs'][0];assert run['status']=='completed' and len(run['requests'])==1
    request=run['requests'][0];trace=round_dir/run['name']/'ggml_trace.json'
    pending={};lanes=collections.defaultdict(list);events=collections.Counter()
    for line in trace.open(encoding='utf8'):
        line=line.strip().rstrip(',')
        if not line.startswith('{'):continue
        event=json.loads(line);ph=event.get('ph');pid=event.get('pid');tid=event.get('tid')
        events[(pid,ph)]+=1
        if pid!=3:continue
        if ph=='B':
            assert tid not in pending
            pending[tid]=event
        elif ph=='E':
            begin=pending.pop(tid);assert begin['name']==event['name'] and begin['ts']<=event['ts']
            lanes[tid].append({'start':begin['ts'],'end':event['ts'],'name':begin['name'],'args':begin['args']})
    assert not pending and len(lanes)==7
    assert len({len(v) for v in lanes.values()})==1
    groups=[]
    def signature(x):return (x['name'],*(x['args'].get(k) for k in ('src0_name','src1_name','out_shape','src0_shape','src1_shape')))
    for peers in zip(*(sorted(lanes[k],key=lambda x:x['start']) for k in sorted(lanes))):
        ref=peers[0];assert all(signature(x)==signature(ref) for x in peers)
        intervals=[(x['start'],x['end']) for x in peers]
        groups.append(dict(ref,start=min(x['start'] for x in peers),end=max(x['end'] for x in peers),
                           wall_ms=union_us(intervals)/1000,intervals=intervals))
    vision_windows=[];start=None
    for i,g in enumerate(groups):
        a=g['args']
        if a['op_name']=='MUL_MAT' and 'v.patch_embd' in a.get('src1_name',''):
            assert start is None;start=i
        if a.get('out_name')=='merger_out' and a.get('src1_name')=='mm.down.bias':
            assert start is not None;vision_windows.append((start,i));start=None
    assert start is None and len(vision_windows)==5
    before=[i for i,g in enumerate(groups[:vision_windows[0][0]]) if g['args'].get('out_name')=='result_output']
    assert len(before)==2, 'Expected two initialization text warmups before single request'
    request_lo=before[-1]+1
    prefill_hi=next(i for i in range(vision_windows[-1][1]+1,len(groups)) if groups[i]['args'].get('out_name')=='result_output')
    predicted=request['timings']['predicted_n']
    decode_outputs=sum(g['args'].get('out_name')=='result_output' for g in groups[prefill_hi+1:])
    assert decode_outputs==predicted-1,(decode_outputs,predicted)
    vision_set={i for lo,hi in vision_windows for i in range(lo,hi+1)}
    agg=collections.defaultdict(lambda:{'count':0,'wall_ms':0.})
    rows=[];phase_intervals=collections.defaultdict(list)
    for i,g in enumerate(groups):
        if i<request_lo:continue
        phase='vision' if i in vision_set else 'prefill' if i<=prefill_hi else 'decode'
        a=g['args'];kind=classify(g) if phase=='vision' else a['op_name']+('_'+a.get('src0_type','') if a['op_name']=='MUL_MAT' else '')
        entry=agg[(phase,kind)];entry['count']+=1;entry['wall_ms']+=g['wall_ms']
        phase_intervals[phase].extend(g['intervals'])
        rows.append({'phase':phase,'index':i,'category':kind,'operator':a['op_name'],'name':g['name'],
                     'src0':a.get('src0_name',''),'src0_type':a.get('src0_type',''),
                     'src0_shape':a.get('src0_shape',''),'src1_shape':a.get('src1_shape',''),
                     'out_shape':a.get('out_shape',''),'wall_ms':g['wall_ms']})
    phases={}
    for phase in ('vision','prefill','decode'):
        chosen=[r for r in rows if r['phase']==phase]
        active=union_us(phase_intervals[phase])/1000
        assert abs(active-sum(r['wall_ms'] for r in chosen))<1e-5
        phases[phase]={'operator_count':len(chosen),'active_ms':active,
                       'aggregate':[dict(category=k,count=v['count'],wall_ms=v['wall_ms'],mean_ms=v['wall_ms']/v['count'],share_active_pct=v['wall_ms']/active*100)
                                    for (p,k),v in sorted(agg.items(),key=lambda kv:-kv[1]['wall_ms']) if p==phase]}
    result={'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'workers':7,'balanced':True,
        'begin_count':events[(3,'B')],'end_count':events[(3,'E')],
        'excluded_startup_output_graphs':2,'request_operator_count':len(rows),'decoded_forward_passes':decode_outputs,
        'precision':{'text':state['measurement']['text_precision'],'vision':run['name'].split('-')[-1]},
        'vision_graph_envelope_sum_ms':sum((groups[hi]['end']-groups[lo]['start'])/1000 for lo,hi in vision_windows),
        'vision_chunks':len(vision_windows),'phases':phases,'request_timings':request['timings'],
        'e2e_s':request['e2e_s'],'output_sha256':request['output_sha256'],'embedding_sha256':request['embedding_sha256'],
        'limits':['PROFILE ON single complete request, no explicit fixture warmup, not a production speed benchmark.',
                  'Only ggml-spacemit operator intervals are captured. CPU fallback/copies, host time, graph gaps and sampling are excluded.',
                  'Per-operator worker intervals are unioned, never summed across seven workers.',
                  'Initialization graphs excluded by two result_output boundaries; decode validated against predicted_n-1.',
                  'Packing/compute/barriers inside each operator are not individually separated.']}
    out=round_dir/'analysis';out.mkdir(exist_ok=True)
    (out/'full_profile.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'operators.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ('phases','limits')},ensure_ascii=True))
    for p,v in phases.items():print(p,v['active_ms'],[(a['category'],round(a['wall_ms'],3)) for a in v['aggregate'][:5]])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('round_dirs',type=Path,nargs='+');a=p.parse_args()
    for d in a.round_dirs:analyze(d)
