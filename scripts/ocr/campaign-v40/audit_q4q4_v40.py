"""Audit frozen quality and ABBA artifacts, including warmups and stage timings."""
from pathlib import Path
import hashlib
import json
import re
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/optimization'
OUT = RESULTS / '20261104-q4q4-v36'
CORES = [8, 9, 10, 11, 13, 14, 15]


def load(name):
    path = RESULTS / name
    data = json.loads((path / 'round.json').read_text(encoding='utf8'))
    assert data['restored']['health']['status'] == 'ok'
    assert len(data['runs']) == 1 and data['runs'][0]['status'] == 'completed'
    return path, data, data['runs'][0]


def verify_request(path, q):
    assert q['status'] == 'completed' and q['quality_valid']
    blob = (path / q['directory'] / 'vision-embeddings.bin').read_bytes()
    assert hashlib.sha256(blob).hexdigest() == q['embedding_sha256']
    assert hashlib.sha256(q['output'].encode()).hexdigest() == q['output_sha256']
    assert q['finish_reason'] == 'stop' and q['token_ids']
    assert q['vision_timing_events']['valid']
    return blob


def quality():
    p, d, r = load('20261104-q4q4-v40-quality-combined')
    bp, bd, br = load('20261104-q4q4-v37-quality-off')
    cp, cd, cr = load('20261104-q4q4-v39-quality-combined')
    baseline = {q['case_id']: q for q in br['requests']}
    combined = {q['case_id']: q for q in cr['requests']}
    rows = []
    for q in r['requests']:
        blob = verify_request(p, q)
        b, c = baseline[q['case_id']], combined[q['case_id']]
        assert q['output'] == b['output'] and q['token_ids'] == b['token_ids']
        assert q['embedding_sha256'] == c['embedding_sha256']
        x = np.frombuffer((bp/b['directory']/'vision-embeddings.bin').read_bytes(), dtype='<f4').astype('f8')
        y = np.frombuffer(blob, dtype='<f4').astype('f8')
        rows.append({'case': q['case_id'], 'output_exact_to_baseline': True,
                     'tokens_exact_to_baseline': True, 'embedding_exact_to_v39_combined': True,
                     'embedding_relative_l2_to_baseline': float(np.linalg.norm(y-x)/np.linalg.norm(x)),
                     'baseline_e2e_s': b['e2e_s'], 'candidate_e2e_s': q['e2e_s'],
                     'baseline_vision_s': b['vision_encode_s'], 'candidate_vision_s': q['vision_encode_s'],
                     'text_prefill_s': q['text_prefill_s'],
                     'generation_s': q['timings']['predicted_ms']/1000,
                     'generation_token_s': q['timings']['predicted_per_second']})
    log = (p/r['name']/'server.log').read_text(encoding='utf8')
    for marker in ['ocr-fa-x8-v39: active', 'ocr-fa-pack-v40: active']:
        assert marker in log, marker
    assert len(rows) == 7
    result = {'passed': True, 'scope': 'Seven single-run quality cases; last-slice embedding only for multi-slice pages', 'rows': rows}
    (OUT/'v40-quality-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    return result


def abba():
    manifest=json.loads((RESULTS/'20261104-q4q4-v40-provenance/runtime-manifest.json').read_text(encoding='utf8'))
    provenance=json.loads((RESULTS/'20261104-q4q4-v40-provenance/test-provenance.json').read_text(encoding='utf8'))
    assert manifest['source_commit']=='d84277ecaaa87cfbc22dcfb5d34109ae4bc37d8a'
    assert manifest['build_flags']['GGML_BUILD_PROFILE'] is False
    assert provenance['manifest_verified'] is True
    arms = ['off-a', 'on-a', 'on-b', 'off-b']
    refs, embeddings, groups, rows = {}, {}, {}, []
    starts, finishes, binaries, models = [], [], set(), set()
    requests_checked = 0
    for arm in arms:
        p, d, r = load(f'20261104-q4q4-v40-abba-{arm}')
        m = d['measurement']; on = arm.startswith('on')
        assert d['case_file_sha256'] == hashlib.sha256((ROOT/'images/optimization-v5/cases.json').read_bytes()).hexdigest()
        assert m['warmups'] == 1 and m['repeats'] == 3
        assert m['text_precision'] == 'q4' and m['spacemit_core_ids'] == CORES
        assert m['f16_m4_cache'] and not m['diagnostic_trace']
        assert m['ime2_fp16_down'] == on and m['fa_pack'] == on and m['fa_x8'] == on
        env = r['environment']
        for flag in ['GGML_SPACEMIT_IME2_FP16_DOWN', 'GGML_SPACEMIT_IME2_FP16_COMPENSATE',
                     'GGML_SPACEMIT_IME2_FP16_NORMALIZE', 'GGML_SPACEMIT_OCR_FA_X8', 'GGML_SPACEMIT_OCR_FA_PACK']:
            assert env.get(flag) == ('1' if on else None)
        assert env.get('GGML_SPACEMIT_IME2_FP16_DOWN_LAYERS') == (','.join(map(str, range(14, 27))) if on else None)
        assert env['GGML_SPACEMIT_WORKERS'] == '7' and env['GGML_SPACEMIT_BACKEND'] == '1'
        cmd = r['command']; binaries.add(cmd[0])
        model, vision = cmd[cmd.index('--model')+1], cmd[cmd.index('--mmproj')+1]
        models.add((model, vision))
        assert Path(model).name == 'MiniCPM-OCR-1_0-Q4_0.gguf'
        assert Path(vision).name == 'mmproj-model-q4_0-4x.gguf'
        starts.append(d['started']); finishes.append(d['finished'])
        log = (p/r['name']/'server.log').read_text(encoding='utf8')
        pools = re.findall(r'ggml-spacemit: num_cores=(\d+), core_ids=([0-9,]+),?\s*arch_id=', log)
        assert pools and all(int(n) == 7 and [int(v) for v in cores.strip(',').split(',')] == CORES for n, cores in pools)
        assert ('ocr-fa-pack-v40: active' in log) == on
        assert ('ocr-fa-x8-v39: active' in log) == on
        expected = {(c, repeat) for c in ['v5_table_merge4', '03_table_gb_standard'] for repeat in [-1, 0, 1, 2]}
        assert {(q['case_id'], q['repeat']) for q in r['requests']} == expected
        for q in r['requests']:
            blob = verify_request(p, q); case = q['case_id']
            refs.setdefault(case, (q['output'], q['token_ids']))
            assert (q['output'], q['token_ids']) == refs[case]
            embeddings.setdefault((on, case), q['embedding_sha256'])
            assert embeddings[on, case] == q['embedding_sha256']
            for phase in ['system_before', 'system_after']:
                singleton = {a['allowed_cpu_ids'][0] for a in q[phase]['server_thread_affinities'] if len(a['allowed_cpu_ids']) == 1}
                assert set(CORES).issubset(singleton)
                assert q[phase]['cpufreq']['policy8']['scaling_cur_freq_khz'] == 1800000
                assert q[phase]['cpufreq']['policy0']['scaling_cur_freq_khz'] == 2000000
            requests_checked += 1
            if q['warmup']: continue
            groups.setdefault((on, case), []).append(q)
    assert len(binaries) == len(models) == 1
    assert all(finishes[i] <= starts[i+1] for i in range(3)), 'ABBA arms overlap'
    for case in ['v5_table_merge4', '03_table_gb_standard']:
        pair = {}
        for on in [False, True]:
            qs = groups[on, case]; assert len(qs) == 6
            avg = lambda f: statistics.mean(f(q) for q in qs)
            pair['on' if on else 'off'] = {
                'formal_requests': 6, 'vision_s': avg(lambda q:q['vision_encode_s']),
                'prefill_s': avg(lambda q:q['text_prefill_s']),
                'generation_s': avg(lambda q:q['timings']['predicted_ms']/1000),
                'generation_token_s': avg(lambda q:q['timings']['predicted_per_second']),
                'e2e_s': avg(lambda q:q['e2e_s']),
                'e2e_min_s': min(q['e2e_s'] for q in qs), 'e2e_max_s': max(q['e2e_s'] for q in qs),
                'e2e_sd_s': statistics.stdev(q['e2e_s'] for q in qs),
                'rss_after_mib': avg(lambda q:q['system_after']['process_memory_kib']['VmRSS']/1024),
                'process_cumulative_hwm_max_mib': max(q['system_after']['process_memory_kib']['VmHWM']/1024 for q in qs),
                'output_sha256': qs[0]['output_sha256'], 'embedding_sha256': qs[0]['embedding_sha256']}
        rows.append({'case':case, **pair,
                     'e2e_reduction_pct':100*(1-pair['on']['e2e_s']/pair['off']['e2e_s']),
                     'vision_reduction_pct':100*(1-pair['on']['vision_s']/pair['off']['vision_s'])})
    result = {'passed':True, 'requests_checked_including_warmups':requests_checked,
              'same_binary':next(iter(binaries)), 'models':list(next(iter(models))),
              'arm_order':arms, 'starts':starts, 'finishes':finishes,
              'output_and_token_exact_across_states':True, 'embedding_stable_within_each_state':True,
              'rss_scope':'RSS sampled after each request; HWM is process cumulative, not per-request peak',
              'rows':rows}
    (OUT/'v40-abba-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    return result


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--quality-only',action='store_true');args=p.parse_args()
    print(json.dumps(quality(),ensure_ascii=False,indent=2))
    if not args.quality_only:print(json.dumps(abba(),ensure_ascii=False,indent=2))
