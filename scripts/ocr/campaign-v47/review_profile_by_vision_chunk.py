"""Separate interleaved vision graphs before aggregating worker wall time."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def union_us(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def classify(g):
    a = g['args']
    op = a['op_name']
    w = a.get('src0_name', '')
    if op != 'MUL_MAT':
        return op
    if 'v.patch_embd' in a.get('src1_name', ''):
        return 'PATCH_EMBED_F16'
    if w.startswith('v.') and '.ffn_down.weight' in w:
        return 'FFN_DOWN_' + a.get('src0_type', '').upper()
    if w.startswith('v.') and '.ffn_up.weight' in w:
        return 'FFN_UP_' + a.get('src0_type', '').upper()
    if w.startswith('v.') and any('.attn_' + x + '.weight' in w for x in ('q', 'k', 'v')):
        return 'QKV_' + a.get('src0_type', '').upper()
    if w.startswith('v.') and '.attn_out.weight' in w:
        return 'ATTN_OUT_' + a.get('src0_type', '').upper()
    if w.startswith('mm.'):
        return 'MERGER_' + a.get('src0_type', '').upper()
    return 'OTHER_MUL_MAT_' + a.get('src0_type', '').upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    args = ap.parse_args()
    pending = {}
    lanes = collections.defaultdict(list)
    errors = []
    with args.trace.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip().rstrip(',')
            if not s.startswith('{'):
                continue
            e = json.loads(s)
            key = (e.get('pid'), e.get('tid'))
            if e.get('ph') == 'B':
                if key in pending:
                    errors.append(['nested', line_no, key])
                pending[key] = e
            elif e.get('ph') == 'E':
                b = pending.pop(key, None)
                if b is None or b['name'] != e['name'] or e['ts'] < b['ts']:
                    errors.append(['unmatched', line_no, key])
                    continue
                lanes[key].append({'start': b['ts'], 'end': e['ts'],
                                   'name': b['name'], 'args': b.get('args', {})})
    workers = {tid: sorted(v, key=lambda x: x['start'])
               for (pid, tid), v in lanes.items() if pid == 3}
    assert workers and not errors, errors
    assert len({len(v) for v in workers.values()}) == 1
    groups = []
    for index, peers in enumerate(zip(*(workers[i] for i in sorted(workers)))):
        ref = peers[0]
        def signature(x):
            a = x['args']
            return (x['name'], a.get('src0_name'), a.get('src1_name'),
                    a.get('src0_shape'), a.get('src1_shape'), a.get('out_shape'))
        assert all(signature(x) == signature(ref) for x in peers)
        groups.append(dict(ref, index=index, start=min(x['start'] for x in peers),
                           end=max(x['end'] for x in peers),
                           intervals=[(x['start'], x['end']) for x in peers],
                           wall_ms=union_us((x['start'], x['end']) for x in peers)/1000))
    windows = []
    current = None
    for g in groups:
        a = g['args']
        if 'v.patch_embd' in a.get('src1_name', '') and a.get('op_name') == 'MUL_MAT':
            assert current is None, 'nested vision start'
            current = g['index']
        if a.get('out_name') == 'merger_out' and a.get('src1_name') == 'mm.down.bias':
            assert current is not None, 'unmatched vision end'
            windows.append((current, g['index']))
            current = None
    assert current is None and windows
    agg = collections.defaultdict(lambda: {'count': 0, 'wall_ms': 0.0})
    all_rows = []
    chunks = []
    selected = set()
    for chunk, (lo, hi) in enumerate(windows):
        gs = groups[lo:hi+1]
        start, end = gs[0]['start'], gs[-1]['end']
        assert not any(start <= e['ts'] <= end for e in pending.values())
        intervals = [it for g in gs for it in g['intervals']]
        active = union_us(intervals)/1000
        assert abs(active-sum(g['wall_ms'] for g in gs)) < 1e-5, 'overlapping operators'
        chunks.append({'chunk': chunk, 'lo': lo, 'hi': hi, 'operators': len(gs),
                       'envelope_ms': (end-start)/1000, 'active_ms': active,
                       'input_shape': gs[0]['args'].get('src0_shape')})
        for g in gs:
            selected.add(g['index'])
            a = g['args']
            kind = classify(g)
            agg[kind]['count'] += 1
            agg[kind]['wall_ms'] += g['wall_ms']
            all_rows.append({'chunk': chunk, 'index': g['index'], 'category': kind,
                             'operator': a['op_name'], 'name': g['name'],
                             'weight_or_src0': a.get('src0_name', ''),
                             'src0_type': a.get('src0_type', ''),
                             'src0_shape': a.get('src0_shape', ''),
                             'src1_shape': a.get('src1_shape', ''),
                             'wall_ms': g['wall_ms']})
    total_envelope = sum(c['envelope_ms'] for c in chunks)
    total_active = sum(c['active_ms'] for c in chunks)
    aggregate = [{'category': k, **v, 'share_visual_pct': v['wall_ms']/total_envelope*100,
                  'mean_operator_ms': v['wall_ms']/v['count']}
                 for k, v in sorted(agg.items(), key=lambda kv: -kv[1]['wall_ms'])]
    interleaved = [g for g in groups[windows[0][0]:windows[-1][1]+1] if g['index'] not in selected]
    text_agg = collections.defaultdict(lambda: {'count': 0, 'wall_ms': 0.0})
    for g in interleaved:
        a=g['args']
        kind = a['op_name'] + ('_'+a.get('src0_type','') if a['op_name']=='MUL_MAT' else '')
        text_agg[kind]['count'] += 1
        text_agg[kind]['wall_ms'] += g['wall_ms']
    outer=(groups[windows[-1][1]]['end']-groups[windows[0][0]]['start'])/1000
    result={'trace':str(args.trace.resolve()), 'trace_sha256':hashlib.sha256(args.trace.read_bytes()).hexdigest(),
            'workers':len(workers),'chunk_count':len(chunks),'chunks':chunks,
            'vision_envelope_sum_ms':total_envelope,'vision_active_sum_ms':total_active,
            'vision_internal_gaps_ms':total_envelope-total_active,'aggregate':aggregate,
            'mixed_outer_envelope_ms':outer,'interleaved_text_and_gaps_ms':outer-total_envelope,
            'interleaved_text_operator_sum_ms':sum(g['wall_ms'] for g in interleaved),
            'interleaved_text_aggregate':[{'category':k,**v} for k,v in sorted(text_agg.items(),key=lambda kv:-kv[1]['wall_ms'])],
            'pending_tail':[{'lane':list(k),'name':v['name'],'ts':v['ts']} for k,v in pending.items()],
            'limitations':['Diagnostic PROFILE ON v30, Q4 text/Q8 vision, 7 workers, v5_table_merge4, one request/max_tokens=1, no warmup.',
                          'Each vision window is complete; the full trace has a partial later text tail.',
                          'Interleaved text table excludes text before first vision and after last merger; it is not full prefill or generation.',
                          'Worker intervals are unioned; internal copy/packing/compute/barriers are not separately measured.',
                          'This is not an operator profile of the 18-row v8 benchmark.']}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'profile_by_vision_chunk.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,rows in [('vision_operators.csv',all_rows),('vision_categories.csv',aggregate)]:
        with (args.output_dir/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='pending_tail'},ensure_ascii=True))


if __name__=='__main__':
    main()
