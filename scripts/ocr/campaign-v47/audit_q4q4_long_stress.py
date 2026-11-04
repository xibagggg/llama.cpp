"""Summarize completed soak: exact per-request evidence and sampled stability."""
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,statistics,math
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization'
p=argparse.ArgumentParser();p.add_argument('round');p.add_argument('--out',required=True);p.add_argument('--provenance-round');a=p.parse_args()
folder=BASE/a.round;load=lambda p:json.loads(p.read_text(encoding='utf8'));d=load(folder/'round.json')
assert d['restored']['health']['status']=='ok' and d.get('finished')
assert len(d['runs'])==1;r=d['runs'][0];assert r['status']=='completed' and r.get('end_reason')=='request_deadline'
refs={q['case_id']:q for run in load(BASE/'20261104-q4q4-v43-quality-selected/round.json')['runs'] for q in run['requests']}
requests=r['requests'];assert requests and all(q['status']=='completed' for q in requests)
version=r['name'].split('-')[0]
identity=load(BASE/f'20261104-q4q4-{version}/build-identity.json')
provenance=load(BASE/(a.provenance_round or f'20261104-q4q4-{version}-stress-provenance')/'provenance.json')
assert identity['manifest']==provenance['manifest'] and provenance['manifest_verified']
assert not identity['manifest']['build_flags']['GGML_BUILD_PROFILE']
assert provenance['runner_sha256']==hashlib.sha256((ROOT/f'scripts/optimization_long_stress.{version}.py').read_bytes()).hexdigest()
assert d['case_file_sha256']==hashlib.sha256((ROOT/'images/optimization-v5/cases.json').read_bytes()).hexdigest()
assert d['measurement']['text_precision']=='q4' and '--trace' not in r['command']
assert d['measurement']['spacemit_core_ids']==[8,9,10,11,13,14,15]
assert d['measurement']['down_vector']==5 and d['measurement']['fa_direct']==2 and d['measurement']['concat_tcm']==4 and d['measurement']['fa_group'] is None
env=r['environment']
for key,value in {'GGML_SPACEMIT_OCR_DOWN_VECTOR':'5','GGML_SPACEMIT_OCR_FA_DIRECT':'2','GGML_SPACEMIT_OCR_CONCAT_TCM':'4','GGML_SPACEMIT_OCR_DOWN_COMPACT_TILE':'20','GGML_SPACEMIT_OCR_DOWN_TCM_TILE':'16','GGML_SPACEMIT_OCR_DOWN_SCHEDULE':'paired','GGML_SPACEMIT_OCR_FA_LAZY_INIT':'1','GGML_SPACEMIT_WORKERS':'7'}.items():assert env[key]==value
assert env.get('GGML_SPACEMIT_OCR_FA_GROUP') is None
for key in ['GGML_SPACEMIT_F16_M4_CACHE','GGML_SPACEMIT_IME2_FP16_DOWN','GGML_SPACEMIT_IME2_FP16_COMPENSATE','GGML_SPACEMIT_IME2_FP16_NORMALIZE','GGML_SPACEMIT_OCR_FA_X8','GGML_SPACEMIT_OCR_FA_PACK']:assert env[key]=='1'
assert set(q['case_id'] for q in requests)==set(refs)
for q in requests:
 assert q['reference_exact'] and q['finish_reason_valid'] and q['timing_valid'] and q['quality_valid']
 assert all(q[k]==refs[q['case_id']][k] for k in ['output','token_ids','embedding_sha256'])
 assert hashlib.sha256((folder/q['directory']/'vision-embeddings.bin').read_bytes()).hexdigest()==q['embedding_sha256']
 assert hashlib.sha256((folder/q['directory']/'output.txt').read_bytes()).hexdigest()==q['output_sha256']
 assert q['embedding_preexisting_removed']
 for phase in ['system_before','system_after']:
  single={t['allowed_cpu_ids'][0] for t in q[phase]['server_thread_affinities'] if len(t['allowed_cpu_ids'])==1}
  assert {8,9,10,11,13,14,15}<=single
telemetry=[json.loads(l) for l in (folder/r['name']/'telemetry.jsonl').read_text().splitlines()];assert len(telemetry)>10
def describe(v):
 v=sorted(v);return {'n':len(v),'mean':statistics.mean(v),'min':v[0],'max':v[-1],
  'p50':statistics.median(v),'p95':v[math.ceil(len(v)*.95)-1]}

def memory_trend(qs):
 # Compare the same case across cycles. A positive RSS slope can reflect
 # caches/allocator retention as well as leaks; it is not a leak diagnosis.
 steady=[q for q in qs if q['repeat']>=3]
 n=max(1,len(steady)//4)
 first=statistics.mean(q['system_after']['process_memory_kib']['VmRSS']/1024 for q in steady[:n])
 last=statistics.mean(q['system_after']['process_memory_kib']['VmRSS']/1024 for q in steady[-n:])
 late=steady[len(steady)//2:]
 ts=[datetime.fromisoformat(q['system_after']['recorded_at']) for q in late]
 xs=[(t-ts[0]).total_seconds()/3600 for t in ts]
 ys=[q['system_after']['process_memory_kib']['VmRSS']/1024 for q in late]
 denominator=sum((x-statistics.mean(xs))**2 for x in xs)
 slope=sum((x-statistics.mean(xs))*(y-statistics.mean(ys)) for x,y in zip(xs,ys))/denominator if denominator else None
 return {'steady_requests':len(steady),'quarter_requests':n,
  'first_quarter_mean_mib':first,'last_quarter_mean_mib':last,'delta_mib':last-first,
  'late_half_requests':len(late),'late_half_duration_s':(ts[-1]-ts[0]).total_seconds(),
  'late_half_ols_slope_mib_per_hour':slope,
  'interpretation':'Descriptive same-case RSS trend, not proof of a leak or leak freedom; slot caches and allocator retention are not isolated.'}
rows=[]
for case in sorted({q['case_id'] for q in requests}):
 qs=[q for q in requests if q['case_id']==case and not q['warmup']];assert len(qs)>=3
 steady=[q for q in qs if q['repeat']>=3];n=max(1,len(steady)//4)
 early_latency=statistics.mean(q['e2e_s'] for q in steady[:n])
 late_latency=statistics.mean(q['e2e_s'] for q in steady[-n:])
 rows.append({'case':case,'requests':len(qs),'first_e2e_s':qs[0]['e2e_s'],'last_e2e_s':qs[-1]['e2e_s'],
  'steady_latency_first_quarter_s':early_latency,'steady_latency_last_quarter_s':late_latency,
  'steady_latency_change_pct':100*(late_latency/early_latency-1),
  'e2e_s':describe([q['e2e_s'] for q in qs]),'vision_s':describe([q['vision_encode_s'] for q in qs if q.get('vision_timing_valid',True)]),
  'prefill_s':describe([q['text_prefill_s'] for q in qs if q.get('vision_timing_valid',True)]),'generation_s':describe([q['timings']['predicted_ms']/1000 for q in qs]),
  'unavailable_phase_timings':sum(not q.get('vision_timing_valid',True) for q in qs),
  'generation_token_s':describe([q['timings']['predicted_per_second'] for q in qs]),
  'memory_trend':memory_trend(qs),
  'rss_after_mib':describe([q['system_after']['process_memory_kib']['VmRSS']/1024 for q in qs]),
  'steady_rss_first_quarter_mean_mib':statistics.mean(q['system_after']['process_memory_kib']['VmRSS']/1024 for q in steady[:n]) if steady else None,
  'steady_rss_last_quarter_mean_mib':statistics.mean(q['system_after']['process_memory_kib']['VmRSS']/1024 for q in steady[-n:]) if steady else None})
thermal={}
for name in {k for t in telemetry for k in t['thermal']}:
 thermal[name]=describe([t['thermal'][name]['millidegrees_c']/1000 for t in telemetry if name in t['thermal']])
freq={name:sorted({t['cpufreq'][name]['scaling_cur_freq_khz'] for t in telemetry if name in t['cpufreq']}) for name in {k for t in telemetry for k in t['cpufreq']}}
assert freq=={'policy8':[1800000],'policy0':[2000000]},freq
sampled_rss=[t['process_memory_kib']['VmRSS']/1024 for t in telemetry if t.get('process_memory_kib')]
sampled_hwm=[t['process_memory_kib']['VmHWM']/1024 for t in telemetry if t.get('process_memory_kib')]
start=datetime.fromisoformat(telemetry[0]['recorded_at']);end=datetime.fromisoformat(telemetry[-1]['recorded_at'])
deadline=datetime.fromisoformat(d['measurement']['request_deadline'])
assert datetime.fromisoformat(d['deadline_reached'])>=deadline
assert end>=deadline.replace(microsecond=0) or (deadline-end).total_seconds()<=5
timestamps=[datetime.fromisoformat(t['recorded_at']) for t in telemetry]
max_gap=max((b-a).total_seconds() for a,b in zip(timestamps,timestamps[1:]))
assert max_gap<15,max_gap
result={'passed':True,'round':a.round,'started':d['started'],'finished':d['finished'],
 'telemetry_started':start.isoformat(),'telemetry_ended':end.isoformat(),'sampled_duration_s':(end-start).total_seconds(),
 'request_deadline':d['measurement']['request_deadline'],'end_reason':r['end_reason'],
 'total_requests':len(requests),'formal_requests':sum(not q['warmup'] for q in requests),'warmups':sum(q['warmup'] for q in requests),
 'all_ocr_and_tokens_exact':True,'all_saved_last_slice_embeddings_exact':True,'request_failures':0,
 'unavailable_phase_timings':[{k:q.get(k) for k in ['case_id','repeat','directory','stage_timing_unavailable']} for q in requests if not q.get('vision_timing_valid',True)],
 'telemetry_samples':len(telemetry),'thermal_c':thermal,'frequency_values_khz':freq,
 'telemetry_max_gap_s':max_gap,'source_commit':identity['manifest']['source_commit'],
 'sampled_rss_mib':describe(sampled_rss),'process_cumulative_hwm_max_mib':max(sampled_hwm),'rows':rows,
 'scope':'serial mixed seven-case soak; CPU/memory sampled every 3 seconds, not a concurrent-client throughput test; embeddings cover saved last slice only; no general Q4/Q8 accuracy equivalence',
 'telemetry_sha256':hashlib.sha256((folder/r['name']/'telemetry.jsonl').read_bytes()).hexdigest(),
 'round_sha256':hashlib.sha256((folder/'round.json').read_bytes()).hexdigest()}
out=Path(a.out);out.write_text(json.dumps(result,indent=2),encoding='utf8')
print('Stress audit:',len(requests),'requests,',(end-start).total_seconds()/3600,'hours, thermal max',max(x['max'] for x in thermal.values()))
