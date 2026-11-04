"""K3-only, serial OCR diagnostic round with restoration of the project service."""
import argparse
import base64
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import time
import urllib.request
import math

from benchmark import cases, edit_distance, normalize

ROOT = Path(__file__).resolve().parents[1]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ROW_CACHE_ABBA_CORES = [8, 9, 10, 11, 13, 14, 15]
LOG_TIMESTAMP_RE = re.compile(r'(?<![\d.])(\d+)\.(\d{2})\.(\d{3})\.(\d{3})(?![\d.])')

def parse_vision_encode_timing(log_text):
    """Recover mtmd start/end events even when another logger prefixes the line."""
    # A worker's decimal thread ID can be concatenated directly with a slot
    # timestamp. Separate only IDs independently observed with a delimiter;
    # otherwise the timestamp regex would interpret that ID as huge minutes.
    thread_ids=set(re.findall(r'on thread (\d+)(?=\s|$)',log_text,re.M))
    for thread_id in thread_ids:
        log_text=re.sub(r'(on thread '+re.escape(thread_id)+r')(?=\d{1,6}\.\d{2}\.\d{3}\.\d{3}(?![\d.]))',r'\1 ',log_text)
    starts=[]
    ends=[]
    lines=log_text.splitlines()

    def seconds(match):
        a,b,c,d=map(int,match.groups())
        return a*60+b+c/1000+d/1000000

    # New llama.cpp versions can place the media-worker timestamp after the
    # stage text on the same line (because another logger prefixes the output).
    # Identify events per line first, instead of assuming timestamps precede
    # their event text. Prefer this explicit marker over the legacy slot line.
    timestamped_stage_starts=[]
    for line in lines:
        matches=list(LOG_TIMESTAMP_RE.finditer(line))
        if "stage 'mtmd_batch_encode' begin" in line and matches:
            marker=line.index("stage 'mtmd_batch_encode' begin")
            following=[m for m in matches if m.start()>=marker]
            timestamped_stage_starts.append(seconds(following[0] if following else matches[-1]))
        if 'decoding image batch 1/' in line and matches:
            marker=line.index('decoding image batch 1/')
            preceding=[m for m in matches if m.start()<marker]
            ends.append(seconds(preceding[-1] if preceding else matches[0]))
    # Media-worker stage markers are usually untimestamped; only a subset
    # may inherit another logger's prefix. Never select that partial subset
    # over the complete timestamped slot sequence.
    slot_starts=[]
    for line_index,line in enumerate(lines):
        if 'encoding mtmd batch' not in line:
            continue
        marker=line.index('encoding mtmd batch')
        matches=list(LOG_TIMESTAMP_RE.finditer(line))
        preceding=[m for m in matches if m.start()<marker]
        if matches:
            slot_starts.append(seconds(preceding[-1] if preceding else matches[0]))
        elif line_index:
            # Interleaved loggers may insert a newline between a timestamp-only
            # prefix and the slot payload. Accept only that adjacent bare prefix.
            previous=lines[line_index-1]
            prefixes=list(LOG_TIMESTAMP_RE.finditer(previous))
            remainder=LOG_TIMESTAMP_RE.sub('',previous).strip()
            stage_only=re.fullmatch(r"(?:[IWDET]\s+)?\[media-worker\] backend 'mtmd' stage 'mtmd_batch_encode' begin on thread \d+",remainder)
            # Another observed interleaving splits the media-worker message at
            # "'mtmd" around the preceding slot record. The encoding prefix
            # then contains only the exact stage-message suffix. Require its
            # matching prefix immediately before it; do not infer timestamps.
            stage_tail=(line_index>=2 and
                "[media-worker] backend 'mtmd" in lines[line_index-2] and
                re.fullmatch(r"(?:[IWDET]\s+)?' stage 'mtmd_batch_encode' begin on thread \d+",remainder))
            encode_name_tail=(line_index>=2 and
                "[media-worker] backend 'mtmd' stage '" in lines[line_index-2] and
                re.fullmatch(r"(?:[IWDET]\s+)?mtmd_batch_encode' begin on thread \d+",remainder))
            if len(prefixes)==1 and (remainder in ('I','W','D','E','T','') or stage_only or stage_tail or encode_name_tail):
                slot_starts.append(seconds(prefixes[0]))
    if slot_starts and len(slot_starts)==len(ends):
        starts=slot_starts
    elif timestamped_stage_starts and len(timestamped_stage_starts)==len(ends):
        starts=timestamped_stage_starts
    else:
        starts=slot_starts or timestamped_stage_starts
    starts.sort()
    ends.sort()
    intervals=[]
    if len(starts)==len(ends):
        intervals=[end-start for start,end in zip(starts,ends)]
    valid=(bool(intervals) and all(math.isfinite(v) and v>0 for v in intervals))
    return {'starts':starts,'ends':ends,'intervals_s':intervals,
            'vision_encode_s':sum(intervals) if valid else None,
            'valid':valid and len(starts)==len(ends)}

def affinity_evidence(observation, expected_core_ids):
    """Check that every selected SpacemiT worker CPU is visible as a singleton mask."""
    affinities=observation.get('server_thread_affinities') or []
    singletons=sorted({row['allowed_cpu_ids'][0] for row in affinities
                       if len(row.get('allowed_cpu_ids',[]))==1})
    missing=sorted(set(expected_core_ids)-set(singletons))
    return {'single_thread_affinity_cpu_ids':singletons,
            'expected_worker_core_ids':list(expected_core_ids),
            'unexpected_singleton_cpu_ids':sorted(set(singletons)-set(expected_core_ids)),
            'missing_expected_worker_core_ids':missing,
            'expected_worker_affinities_present':not missing}

def verify_q128_runtime(runtime_dir):
    """Validate the dedicated Q128 runtime identity and installed manifest hashes when present."""
    runtime_dir=runtime_dir.resolve()
    if runtime_dir.name != 'runtime-fa-q128':
        raise ValueError('row-cache ABBA runtime directory must be runtime-fa-q128')
    binary=runtime_dir/'bin/llama-server'
    if not binary.is_file():
        raise ValueError('Q128 runtime is missing bin/llama-server')
    manifest_path=runtime_dir/'MANIFEST.json'
    if not manifest_path.is_file():
        raise ValueError('Q128 runtime is missing MANIFEST.json')
    result={'path':str(runtime_dir),'manifest':str(manifest_path) if manifest_path.is_file() else None,
            'manifest_candidate':None,'manifest_hash_checks':{}}
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding='utf8'))
        if manifest.get('candidate') != 'fa-q128':
            raise ValueError('runtime manifest candidate is not fa-q128')
        files=manifest.get('files',{})
        if not files or 'bin/llama-server' not in files:
            raise ValueError('Q128 runtime manifest has no file hash inventory')
        for relative,expected in files.items():
            path=runtime_dir/relative
            actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            result['manifest_hash_checks'][relative]={'expected':expected,'actual':actual,
                                                       'matches':bool(expected and actual==expected)}
            if not result['manifest_hash_checks'][relative]['matches']:
                raise ValueError(f'Q128 runtime manifest hash mismatch: {relative}')
        result['manifest_candidate']=manifest['candidate']
    return result

def verify_abba_startup(log_path, expected_core_ids):
    text=log_path.read_text(errors='replace')
    pattern=re.compile(r'ggml-spacemit: num_cores=(\d+), core_ids=([0-9]+(?:,[0-9]+)*),?\s*arch_id=')
    found=[(int(n),list(map(int,ids.split(',')))) for n,ids in pattern.findall(text)]
    if not found or any(n!=len(expected_core_ids) or ids!=expected_core_ids for n,ids in found):
        raise RuntimeError(f'Q128 SpacemiT worker pool mismatch in startup log: {found!r}')
    return {'worker_pool_records':found,'workers':len(expected_core_ids),
            'core_ids':list(expected_core_ids),'matches':True}

def http(port, path, data=None, timeout=180):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type':'application/json'})
    with OPENER.open(req, timeout=timeout) as res:
        return json.load(res)

def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

def wait_ready(port, proc=None):
    for _ in range(240):
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f'server exit {proc.returncode}')
        try:
            if http(port, '/health', timeout=1).get('status') == 'ok':
                return
        except Exception:
            pass
        time.sleep(.5)
    raise TimeoutError('server readiness')

def runtime_observation(server_pid=None):
    """Instantaneous observations outside HTTP E2E; no clock/governor changes."""
    observation={'recorded_at':datetime.now().astimezone().isoformat(), 'cpufreq':{}, 'thermal':{}}
    for path in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq'):
        try:
            observation['cpufreq'][path.parent.name]={'scaling_cur_freq_khz':int(path.read_text()),
                'governor':path.with_name('scaling_governor').read_text().strip()}
        except (OSError,ValueError):
            pass
    for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            observation['thermal'][path.parent.name]={'millidegrees_c':int(path.read_text()),
                'type':path.with_name('type').read_text().strip()}
        except (OSError,ValueError):
            pass
    if server_pid is not None:
        try:
            status=(Path('/proc')/str(server_pid)/'status').read_text()
            observation['process_memory_kib']={k:int(v) for k,v in re.findall(r'^(VmRSS|VmHWM|RssAnon|RssFile|RssShmem):\s+(\d+) kB$',status,re.M)}
        except (OSError,ValueError):
            pass
        observation['server_thread_affinities']=[]
        for task in sorted((Path('/proc')/str(server_pid)/'task').glob('*')):
            try:
                observation['server_thread_affinities'].append({
                    'tid':int(task.name),'comm':(task/'comm').read_text().strip(),
                    'allowed_cpu_ids':sorted(os.sched_getaffinity(int(task.name)))})
            except (OSError,ValueError):
                pass
    return observation

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--runtime', action='append', default=[],
                        help='Optional NAME=directory; run the supplied runtime on Q4/Q8 visual cases')
    parser.add_argument('--precision', choices=['q8','q4'], nargs='+', default=['q8','q4'])
    parser.add_argument('--cpu-reference', action='store_true')
    parser.add_argument('--no-flash-attn', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--checker', type=Path, action='append', default=[])
    parser.add_argument('--checker-only', action='store_true')
    parser.add_argument('--checker-timeout', type=int, default=300)
    parser.add_argument('--checker-separate-stderr', action='store_true',
                        help='Keep checker stdout and stderr in separate files to avoid torn case records')
    parser.add_argument('--cases', nargs='+', default=['03_table_gb_standard'])
    parser.add_argument('--case-file', type=Path,
                        help='Additional cases JSON with a cases list; image paths are relative to images/')
    parser.add_argument('--max-tokens', type=int,
                        help='Override max_tokens, for bounded diagnostic profiling')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--warmups', type=int, default=0,
                        help='Unmeasured complete passes over selected cases')
    parser.add_argument('--text-precision', choices=['q4','q8'], default='q4')
    parser.add_argument('--q4-force-k32', action='store_true',
                        help='Diagnostic runtime only: select native Q4 K32 before weight repack')
    parser.add_argument('--q6-full-tile', action='store_true',
                        help='Diagnostic runtime only: copy complete Q6_K repacked physical tiles')
    parser.add_argument('--f16-row-cache', action='store_true',
                        help='Diagnostic runtime only: cache one widened F16 weight row in TCM')
    parser.add_argument('--f16-row-cache-abba', action='store_true',
                        help='Strict Q128/Q4+Q8/7-core/M4-off ABBA on the three canonical cases; requires >=1 warmup and >=3 repeats')
    parser.add_argument('--f16-m4-cache', action='store_true',
                        help='Diagnostic runtime only: cache four widened F16 rows and reuse activations')
    parser.add_argument('--ime2-fp16-down', action='store_true',
                        help='Isolated IME2 candidate: F16 FFN-down with explicit activation conversion')
    parser.add_argument('--ime2-fp16-compensate', action='store_true',
                        help='Isolated v35 candidate: split F32 activation into FP16 high and residual parts')
    parser.add_argument('--ime2-down-layers', help='Comma-separated vision down layer IDs, or none')
    parser.add_argument('--mmproj-model',type=Path)
    parser.add_argument('--padded-down',action='store_true')
    parser.add_argument('--fa-phases',action='store_true')
    parser.add_argument('--down-capture', action='store_true', help='Diagnostic tensor I/O; timings invalid')
    parser.add_argument('--quant-capture', action='store_true', help='Capture actual Q8 packed activations for first two layers')
    parser.add_argument('--ime2-normalize', action='store_true', help='v37: normalize weights and recover tiny high activations')
    parser.add_argument('--down-tcm-tile', type=int, choices=(8,16), help='v41 exact M4 activation TCM tile')
    parser.add_argument('--down-compact-tile',type=int,choices=(16,20))
    parser.add_argument('--fa-lazy-init',action='store_true')
    parser.add_argument('--down-vector',type=int,choices=range(1,8))
    parser.add_argument('--fa-direct',type=int,choices=range(1,4))
    parser.add_argument('--fa-group',type=int,choices=range(1,7))
    parser.add_argument('--concat-tcm',type=int,choices=range(1,5))
    parser.add_argument('--down-schedule', choices=('relaxed','paired'), help='v42 exact down load scheduling')
    parser.add_argument('--down-phases', action='store_true', help='v41 coarse timers; diagnostic only')
    parser.add_argument('--fa-pack', action='store_true', help='v40: transient per-call D72 K/V packing')
    parser.add_argument('--fa-x8', action='store_true', help='v39: share unmasked D72 K/V across eight queries')
    parser.add_argument('--replay-embeddings', type=Path,
                        help='Diagnostic runtime only: inject this saved embedding at the mtmd encoder boundary')
    parser.add_argument('--vision-layer-select',
                        help='Diagnostic clip only: save graph inputs and selected named nodes; none keeps the graph whole')
    parser.add_argument('--expected-embedding-sha256',
                        help='Single runtime/precision/case diagnostic: preserve and stop at the first mismatching embedding')
    parser.add_argument('--strict-quality-evidence', action='store_true',
                        help='Require a fresh non-empty visual embedding, token IDs and OCR text for every request')
    parser.add_argument('--worker-observer', action='store_true',
                        help='Diagnostic backend: record worker/view/status metadata without splitting the graph')
    parser.add_argument('--worker-observer-match', default='v.blk.',
                        help='Observer graph metadata substring; empty selects every graph')
    parser.add_argument('--worker-tail', action='store_true',
                        help='Diagnostic backend: save selected split last tensor after the original launch future')
    parser.add_argument('--cpu-adapter-observer', action='store_true',
                        help='Diagnostic CPU adapter: record dispatch lanes, TLS, CPU and FCSR without reading tensor data')
    parser.add_argument('--tail-sampler', action='store_true',
                        help='Diagnostic backend: sparse samples of the last committed output after the original launch future')
    parser.add_argument('--worker-owned-sampler', action='store_true',
                        help='Diagnostic backend: pre-barrier samples only from each worker-owned output range')
    parser.add_argument('--flash-input-audit', action='store_true',
                        help='Diagnostic backend: exact full Flash Q/K/V input equality before original launch')
    parser.add_argument('--norm-input-audit', action='store_true',
                        help='Diagnostic backend: full worker-owned real NORM input bytes before compute')
    parser.add_argument('--full-boundary-audit', action='store_true',
                        help='Diagnostic backend: full NORM/Flash/attention projection boundaries')
    parser.add_argument('--qkv-boundary-audit', action='store_true',
                        help='Diagnostic backend: extend full boundaries to QKV and norm affine operations')
    parser.add_argument('--kv-bias-audit', action='store_true',
                        help='Diagnostic v20: extend full boundaries only to K/V bias outputs')
    parser.add_argument('--kv-projection-audit', action='store_true',
                        help='Diagnostic v21: add K/V activation/projection and bias full boundaries')
    parser.add_argument('--stop-on-boundary-change', action='store_true',
                        help='Preserve completed request and stop at first full internal boundary mismatch')
    parser.add_argument('--min-free-gib', type=float, default=0,
                        help='Stop before another request if project filesystem has less free space')
    parser.add_argument('--spacemit-core-ids',
                        help='Comma-separated distinct K3 AI cores (8-15); caps the requested 8 workers')
    args=parser.parse_args()
    core_ids=None
    if args.spacemit_core_ids is not None:
        if not re.fullmatch(r'\d+(,\d+)*',args.spacemit_core_ids):
            parser.error('spacemit-core-ids must be comma-separated CPU IDs')
        core_ids=list(map(int,args.spacemit_core_ids.split(',')))
        if len(set(core_ids))!=len(core_ids) or not set(core_ids)<=set(range(8,16)):
            parser.error('spacemit-core-ids requires distinct K3 AI cores in 8-15')
    # The backend creates one worker per GGML_SPACEMIT_WORKERS value.  Keep
    # the worker count equal to an explicitly supplied core list; passing a
    # seven-core list with the old hard-coded value 8 silently reintroduced
    # the omitted CPU through the backend's automatic affinity fallback.
    worker_count = len(core_ids) if core_ids is not None else 8
    if args.checker_timeout < 1 or (args.checker_only and not args.checker):
        parser.error('checker-only requires a checker; timeout must be positive')
    if args.f16_row_cache_abba and (
            len(args.runtime) != 1 or args.precision != ['q8'] or args.text_precision != 'q4'
            or args.f16_row_cache or args.f16_m4_cache or args.checker or args.checker_only
            or args.cpu_reference or core_ids != ROW_CACHE_ABBA_CORES
            or args.warmups < 1 or args.repeats < 3 or args.no_flash_attn or args.trace
            or args.cases != ['01_text_putnam','02_formula_putnam','03_table_gb_standard']
            or args.case_file is not None or args.max_tokens is not None
            or args.q4_force_k32 or args.q6_full_tile or args.replay_embeddings
            or args.vision_layer_select is not None or args.expected_embedding_sha256
            or args.worker_observer or args.worker_tail or args.cpu_adapter_observer
            or args.tail_sampler or args.worker_owned_sampler or args.flash_input_audit
            or args.norm_input_audit or args.full_boundary_audit or args.qkv_boundary_audit
            or args.kv_bias_audit or args.kv_projection_audit or args.stop_on_boundary_change):
        parser.error('f16-row-cache-abba requires one Q128 runtime, the three canonical cases, Q4 text/Q8 vision, exactly 7 workers on 8,9,10,11,13,14,15, M4-off, >=1 warmup, >=3 repeats, flash attention on, and no replay/diagnostic/checker options')
    row_cache_runtime_evidence=None
    if args.f16_row_cache_abba:
        try:
            _name, _runtime = args.runtime[0].split('=',1)
            row_cache_runtime_evidence=verify_q128_runtime(ROOT/_runtime)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            parser.error(f'f16-row-cache-abba requires an intact runtime-fa-q128 package: {exc}')
    if args.worker_tail and not args.worker_observer:
        parser.error('worker-tail requires worker-observer')
    if args.tail_sampler and not args.worker_observer:
        parser.error('tail-sampler requires worker-observer')
    if args.worker_owned_sampler and not (args.worker_observer and args.tail_sampler):
        parser.error('worker-owned-sampler requires worker-observer and tail-sampler')
    if args.flash_input_audit and not args.worker_owned_sampler:
        parser.error('flash-input-audit requires worker-owned-sampler')
    if args.norm_input_audit and not args.worker_owned_sampler:
        parser.error('norm-input-audit requires worker-owned-sampler')
    if args.full_boundary_audit and not (args.norm_input_audit and args.flash_input_audit):
        parser.error('full-boundary-audit requires norm-input-audit and flash-input-audit')
    if args.qkv_boundary_audit and not args.full_boundary_audit:
        parser.error('qkv-boundary-audit requires full-boundary-audit')
    if args.kv_bias_audit and (not args.full_boundary_audit or args.qkv_boundary_audit):
        parser.error('kv-bias-audit requires full-boundary-audit and excludes qkv-boundary-audit')
    if args.kv_projection_audit and (not args.full_boundary_audit or args.kv_bias_audit or args.qkv_boundary_audit):
        parser.error('kv-projection-audit requires full-boundary-audit and excludes other extensions')
    if args.stop_on_boundary_change and not args.full_boundary_audit:
        parser.error('stop-on-boundary-change requires full-boundary-audit')
    available={case['id']:case for case in cases()}
    if args.case_file:
        extra=json.loads(args.case_file.read_text(encoding='utf8'))['cases']
        for case in extra:
            if case['id'] in available:
                parser.error('duplicate additional case id: '+case['id'])
            available[case['id']]=case
    if args.max_tokens is not None and args.max_tokens<1:
        parser.error('max_tokens must be positive')
    if set(args.cases)-available.keys() or args.repeats<1 or args.warmups<0:
        parser.error('unknown case or invalid repeat/warmup count')
    for caseid in args.cases:
        image_path=ROOT/'images'/available[caseid]['image']
        if not image_path.is_file():
            parser.error('missing image for case: '+caseid)
        expected=available[caseid].get('input_sha256')
        if expected and hashlib.sha256(image_path.read_bytes()).hexdigest()!=expected:
            parser.error('input hash mismatch for case: '+caseid)
    if args.replay_embeddings and not args.replay_embeddings.is_file():
        parser.error('replay embedding does not exist')
    if args.expected_embedding_sha256:
        if (not re.fullmatch('[0-9a-f]{64}', args.expected_embedding_sha256)
                or len(args.runtime)!=1 or len(args.precision)!=1 or len(args.cases)!=1
                or args.checker_only):
            parser.error('expected embedding requires a lowercase SHA256 and one runtime, precision and case')
    batch=len(args.cases)>1 or args.repeats>1 or args.warmups>0
    out=ROOT/'results/optimization'/args.label
    out.mkdir(parents=True, exist_ok=False)
    lock=(ROOT/'logs/adapted-spacemit/.test.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    port=18582
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port)) == 0:
            raise RuntimeError('test port occupied')
    state={'label':args.label,'started':datetime.now().astimezone().isoformat(),'runs':[],
           'measurement':{'cases':args.cases,'repeats':args.repeats,'warmups':args.warmups,
                          'text_precision':args.text_precision,'cache_prompt':False,
                          'diagnostic_trace':args.trace,'max_tokens_override':args.max_tokens,
                          'vision_layer_select':args.vision_layer_select,
                          'expected_embedding_sha256':args.expected_embedding_sha256,
                          'strict_quality_evidence':args.strict_quality_evidence,
                          'worker_observer':args.worker_observer,
                          'worker_observer_match':args.worker_observer_match if args.worker_observer else None,
                          'worker_tail':args.worker_tail,
                          'cpu_adapter_observer':args.cpu_adapter_observer,
                          'tail_sampler':args.tail_sampler,
                          'worker_owned_sampler':args.worker_owned_sampler,
                          'flash_input_audit':args.flash_input_audit,
                          'norm_input_audit':args.norm_input_audit,
                          'full_boundary_audit':args.full_boundary_audit,
                          'kv_bias_audit':args.kv_bias_audit,
                          'kv_projection_audit':args.kv_projection_audit,
                          'qkv_boundary_audit':args.qkv_boundary_audit,
                          'stop_on_boundary_change':args.stop_on_boundary_change,
                          'min_free_gib':args.min_free_gib,
                          'spacemit_core_ids':core_ids,
                          'requested_spacemit_workers':worker_count,
                          'f16_row_cache':args.f16_row_cache,
                          'f16_row_cache_abba':args.f16_row_cache_abba,
                          'f16_m4_cache':args.f16_m4_cache,
                          'ime2_fp16_down':args.ime2_fp16_down,
                          'ime2_fp16_compensate':args.ime2_fp16_compensate,
                          'ime2_down_layers':args.ime2_down_layers,
                          'ime2_normalize':args.ime2_normalize,
                          'fa_x8':args.fa_x8,
                          'fa_pack':args.fa_pack,
                          'down_tcm_tile':args.down_tcm_tile,
                          'down_phases':args.down_phases,
                          'down_schedule':args.down_schedule,
                          'down_compact_tile':args.down_compact_tile,'fa_lazy_init':args.fa_lazy_init,
                          'down_vector':args.down_vector,'fa_direct':args.fa_direct,'fa_group':args.fa_group,'concat_tcm':args.concat_tcm,
                          'q128_runtime_evidence':row_cache_runtime_evidence,
                          'checker_only':args.checker_only,'checker_timeout_s':args.checker_timeout,
                          'checker_separate_stderr':args.checker_separate_stderr,
                          'case_file':str(args.case_file) if args.case_file else None}}
    if args.case_file:
        state['additional_cases']={caseid:available[caseid] for caseid in args.cases}
        state['case_file_sha256']=hashlib.sha256(args.case_file.read_bytes()).hexdigest()
    def save():
        temp=out/'round.json.tmp'
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        temp.replace(out/'round.json')
    original=None
    pidfile=ROOT/'logs/server.pid'
    if pidfile.exists():
        pid=int(pidfile.read_text())
        procpath=Path('/proc')/str(pid)
        argv=procpath.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        if argv[0] != str(ROOT/'src/llama-b11901/build/bin/llama-server'):
            raise RuntimeError('unexpected project server executable')
        env=dict(item.split('=',1) for item in procpath.joinpath('environ').read_bytes().decode().strip('\0').split('\0'))
        original=(argv, env, os.readlink(procpath/'cwd'))
        state['original']={'pid':pid,'argv':argv,'cwd':original[2],
            'runtime_environment':{k:v for k,v in env.items() if k.startswith(('GGML_','SPACEMIT_','MTMD_'))}}
        os.kill(pid,signal.SIGTERM)
        for _ in range(40):
            if not procpath.exists() or procpath.joinpath('stat').read_text().split()[2]=='Z':
                break
            time.sleep(.25)
        else:
            raise RuntimeError('original server did not stop')
    save()
    proc=None
    abba_quality_reference={}
    abba_invalid=False
    try:
        for index, checker_arg in enumerate(args.checker):
            checker=checker_arg.resolve()
            checkenv=dict(os.environ, LD_LIBRARY_PATH=str(checker.parent.parent/'lib'),
                          GGML_SPACEMIT_BACKEND='1', GGML_SPACEMIT_WORKERS=str(worker_count))
            if core_ids is not None:
                checkenv['SPACEMIT_PERFER_CORE_ID']=','.join(map(str,core_ids))
            for k in ('SPACEMIT_DISABLE_TCM','GGML_TRACE'):
                checkenv.pop(k,None)
            if args.f16_row_cache:
                checkenv['GGML_SPACEMIT_F16_ROW_CACHE']='1'
            else:
                checkenv.pop('GGML_SPACEMIT_F16_ROW_CACHE',None)
            if args.f16_m4_cache:
                checkenv['GGML_SPACEMIT_F16_M4_CACHE']='1'
            else:
                checkenv.pop('GGML_SPACEMIT_F16_M4_CACHE',None)
            checker_log='checker.log' if len(args.checker)==1 else checker.name+'.log'
            with ExitStack() as stack:
                log=stack.enter_context((out/checker_log).open('w'))
                stderr_log=Path(checker_log).stem+'.stderr.log' if args.checker_separate_stderr else None
                checker_stderr=stack.enter_context((out/stderr_log).open('w')) if stderr_log else subprocess.STDOUT
                try:
                    result=subprocess.run([str(checker)], env=checkenv, cwd=out,
                        stdout=log, stderr=checker_stderr,timeout=args.checker_timeout)
                except subprocess.TimeoutExpired:
                    state.setdefault('checkers',[]).append({'path':str(checker),'log':checker_log,
                        'returncode':None,'timed_out':True,'timeout_s':args.checker_timeout})
                    save()
                    raise
            state['checker_returncode']=result.returncode
            state.setdefault('checkers',[]).append({'path':str(checker),'log':checker_log,
                'stderr_log':stderr_log,'returncode':result.returncode})
            save()
            if result.returncode:
                raise RuntimeError('kernel checker failed; skipping model tests')
        if args.checker_only:
            return
        if args.runtime:
            specs=[]
            if args.f16_row_cache_abba:
                name, runtime = args.runtime[0].split('=', 1)
                for suffix, enabled in [('off-a', False), ('on-a', True), ('on-b', True), ('off-b', False)]:
                    specs.append((f'{name}-row-cache-{suffix}-q8', runtime, 'q8', True, args.trace, False, enabled))
            else:
                for spec in args.runtime:
                    name,runtime=spec.split('=',1)
                    for precision in args.precision:
                        specs.append((name+'-'+precision,runtime,precision,True,args.trace,False,args.f16_row_cache))
        else:
            specs=[
                ('fixed-default','runtimes/spacemit-ocr-855a53b-ci-fixed','q8',False,False,False,args.f16_row_cache),
                ('fixed-special','runtimes/spacemit-ocr-855a53b-ci-fixed','q8',True,False,False,args.f16_row_cache),
                ('profile-special','runtimes/spacemit-ocr-855a53b-profile','q8',True,True,False,args.f16_row_cache),
                ('old-same-q8','src/llama-b11901/build','q8',True,False,True,args.f16_row_cache),
            ]
        for name,rt,precision,special,trace,old,run_f16_row_cache in specs:
            dest=out/name
            dest.mkdir()
            runtime=ROOT/rt
            env=dict(os.environ)
            for k in list(env):
                if k.startswith(('GGML_','SPACEMIT_','SPERT_','SPINE_','LLAMA_ARG_','MTMD_')) or k in ('LD_LIBRARY_PATH','LD_PRELOAD'):
                    env.pop(k)
            if args.f16_row_cache_abba:
                # Keep this explicit: inherited shell state must never silently enable M4.
                env.pop('GGML_SPACEMIT_F16_M4_CACHE',None)
            cpu=old or args.cpu_reference
            env.update(GGML_SPACEMIT_BACKEND='0' if cpu else '1',GGML_SPACEMIT_WORKERS=str(worker_count),
                       MTMD_DEBUG_EMBEDDINGS=str(dest/'vision-embeddings.bin'))
            if core_ids is not None:
                env['SPACEMIT_PERFER_CORE_ID']=','.join(map(str,core_ids))
            if args.replay_embeddings:
                env['MTMD_DEBUG_REPLAY_EMBEDDINGS']=str(args.replay_embeddings.resolve())
            if args.q4_force_k32:
                env['GGML_SPACEMIT_Q4_FORCE_K32']='1'
            if args.q6_full_tile:
                env['GGML_SPACEMIT_Q6_FULL_TILE']='1'
            if run_f16_row_cache:
                env['GGML_SPACEMIT_F16_ROW_CACHE']='1'
            if args.f16_m4_cache:
                env['GGML_SPACEMIT_F16_M4_CACHE']='1'
            if args.ime2_fp16_down:
                env['GGML_SPACEMIT_IME2_FP16_DOWN']='1'
            if args.ime2_fp16_compensate:
                if not args.ime2_fp16_down:
                    raise ValueError('--ime2-fp16-compensate requires --ime2-fp16-down')
                env['GGML_SPACEMIT_IME2_FP16_COMPENSATE']='1'
            if args.ime2_down_layers is not None:
                if not args.ime2_fp16_down:
                    raise ValueError('--ime2-down-layers requires --ime2-fp16-down')
                if args.ime2_down_layers != 'none':
                    layers=[int(v) for v in args.ime2_down_layers.split(',')]
                    if not layers or len(set(layers))!=len(layers) or any(v<0 or v>=27 for v in layers):
                        raise ValueError('Invalid or duplicate layer IDs')
                env['GGML_SPACEMIT_IME2_FP16_DOWN_LAYERS']=args.ime2_down_layers
            if args.ime2_normalize:
                if not args.ime2_fp16_down or not args.ime2_fp16_compensate:
                    raise ValueError('--ime2-normalize requires down and compensation switches')
                env['GGML_SPACEMIT_IME2_FP16_NORMALIZE']='1'
            if args.down_tcm_tile:
                env['GGML_SPACEMIT_OCR_DOWN_TCM_TILE']=str(args.down_tcm_tile)
            env.pop('GGML_SPACEMIT_OCR_DOWN_COMPACT_TILE',None)
            env.pop('GGML_SPACEMIT_OCR_FA_LAZY_INIT',None)
            if args.down_compact_tile:env['GGML_SPACEMIT_OCR_DOWN_COMPACT_TILE']=str(args.down_compact_tile)
            if args.fa_lazy_init:env['GGML_SPACEMIT_OCR_FA_LAZY_INIT']='1'
            if args.down_vector:env['GGML_SPACEMIT_OCR_DOWN_VECTOR']=str(args.down_vector)
            if args.fa_direct:env['GGML_SPACEMIT_OCR_FA_DIRECT']=str(args.fa_direct)
            if args.fa_group:env['GGML_SPACEMIT_OCR_FA_GROUP']=str(args.fa_group)
            if args.concat_tcm:env['GGML_SPACEMIT_OCR_CONCAT_TCM']=str(args.concat_tcm)
            if args.down_schedule:
                if args.down_tcm_tile != 16 or args.down_phases:
                    raise ValueError('v42 scheduling needs tile16 with internal phases off')
                env['GGML_SPACEMIT_OCR_DOWN_SCHEDULE']=args.down_schedule
            if args.down_phases:
                phase_dir=dest/'down-phases'
                phase_dir.mkdir()
                env['GGML_SPACEMIT_OCR_DOWN_PHASES']=str(phase_dir)
            if args.padded_down:env['MTMD_OCR_PADDED_DOWN']='1'
            if args.fa_phases:
                phase_dir=dest/'fa-phases';phase_dir.mkdir()
                env['GGML_SPACEMIT_OCR_FA_PHASES']=str(phase_dir)
            if args.fa_pack:
                env['GGML_SPACEMIT_OCR_FA_PACK']='1'
            if args.fa_x8:
                env['GGML_SPACEMIT_OCR_FA_X8']='1'
            if args.down_capture:
                capture_dir=dest/'down-captures'
                capture_dir.mkdir()
                env['GGML_SPACEMIT_OCR_DOWN_CAPTURE']=str(capture_dir)
            if args.quant_capture:
                capture_dir=dest/'quant-captures'
                capture_dir.mkdir()
                env['GGML_SPACEMIT_OCR_QUANT_CAPTURE']=str(capture_dir)
            if args.vision_layer_select is not None:
                env['MTMD_DEBUG_LAYER_HASHES']=str(dest/'layer-snapshots')
                env['MTMD_DEBUG_LAYER_HASHES_SELECT']=args.vision_layer_select
            if args.worker_observer:
                observer_dir=dest/'worker-observer'
                observer_dir.mkdir()
                env['GGML_SPACEMIT_OBSERVER_V9_DIR']=str(observer_dir)
                env['GGML_SPACEMIT_OBSERVER_V9_MATCH']=args.worker_observer_match
                if args.worker_tail:
                    env['GGML_SPACEMIT_OBSERVER_V10_TAIL']='1'
                    env['GGML_SPACEMIT_OBSERVER_V10_TAIL_MATCH']='v.blk.,merger_out'
            if args.cpu_adapter_observer:
                cpu_observer_dir=dest/'cpu-observer'
                cpu_observer_dir.mkdir()
                env['GGML_CPU_ADAPTER_OBSERVER_V11_DIR']=str(cpu_observer_dir)
            if args.tail_sampler:
                env['GGML_SPACEMIT_OBSERVER_V12_SAMPLES']='1'
            if args.worker_owned_sampler:
                env['GGML_SPACEMIT_OBSERVER_V13_OWNED']='1'
            if args.flash_input_audit:
                env['GGML_SPACEMIT_OBSERVER_V14_INPUTS']='1'
            if args.norm_input_audit:
                env['GGML_SPACEMIT_OBSERVER_V16_NORM_INPUTS']='1'
            if args.full_boundary_audit:
                env['GGML_SPACEMIT_OBSERVER_V18_FULL']='1'
            if args.qkv_boundary_audit:
                env['GGML_SPACEMIT_OBSERVER_V19_QKV']='1'
            if args.kv_bias_audit:
                env['GGML_SPACEMIT_OBSERVER_V20_KV_BIAS']='1'
            if args.kv_projection_audit:
                env['GGML_SPACEMIT_OBSERVER_V21_KV_PROJECTION']='1'
            if old:
                env.update(SPACEMIT_DISABLE_TCM='1',MTMD_SPACEMIT_REPACK='1',GGML_CPU_DISABLE_GENERIC_REPACK='1')
                binary=runtime/'bin/llama-server'
            else:
                env['LD_LIBRARY_PATH']=str(runtime/'lib')
                binary=runtime/'bin/llama-server'
            if trace:
                env['GGML_TRACE']='1'
            command=[str(binary),'--model',str(ROOT/f'models/MiniCPM-OCR-1_0-{args.text_precision.upper()}_0.gguf'),
                '--mmproj',str(args.mmproj_model.resolve() if args.mmproj_model else ROOT/f'models/mmproj-model-{precision}_0-4x.gguf'),
                '--n-gpu-layers','0' if cpu else '999','--threads','8','--threads-batch','8',
                '--ctx-size','8192','--batch-size','512','--ubatch-size','512','--parallel','1',
                '--flash-attn','off' if args.no_flash_attn else 'on','--jinja','--reasoning','off','--verbosity','4',
                '--host','127.0.0.1','--port',str(port),'--alias','minicpm-ocr']
            command+=['--device','none','--no-mmproj-offload'] if cpu else ['--device','SPACEMIT0','--mmproj-offload','--mmproj-device','SPACEMIT0']
            if special:
                command+=['--special']
            if args.f16_row_cache_abba:
                expected_row_switch='1' if run_f16_row_cache else None
                if (env.get('GGML_SPACEMIT_BACKEND')!='1'
                        or env.get('GGML_SPACEMIT_WORKERS')!='7'
                        or env.get('SPACEMIT_PERFER_CORE_ID')!=','.join(map(str,ROW_CACHE_ABBA_CORES))
                        or env.get('GGML_SPACEMIT_F16_ROW_CACHE')!=expected_row_switch
                        or 'GGML_SPACEMIT_F16_M4_CACHE' in env
                        or Path(command[command.index('--model')+1]).name!='MiniCPM-OCR-1_0-Q4_0.gguf'
                        or Path(command[command.index('--mmproj')+1]).name!='mmproj-model-q8_0-4x.gguf'
                        or '--flash-attn' not in command
                        or command[command.index('--flash-attn')+1]!='on'):
                    raise RuntimeError('constructed row-cache ABBA arm does not match fixed Q128/Q4/Q8/7-worker/M4-off configuration')
            rec={'name':name,'command':command,'environment':{k:v for k,v in env.items() if k.startswith(('GGML_','MTMD_','SPACEMIT_'))},'status':'starting'}
            state['runs'].append(rec)
            save()
            start=time.monotonic()
            try:
                with (dest/'server.log').open('w') as log:
                    proc=subprocess.Popen(command,env=env,cwd=dest,stdout=log,stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL,start_new_session=True)
                rec['pid']=proc.pid
                wait_ready(port,proc)
                rec['startup_s']=time.monotonic()-start
                rec['requests']=[]
                if args.f16_row_cache_abba:
                    rec['worker_pool_startup']=verify_abba_startup(dest/'server.log',ROW_CACHE_ABBA_CORES)
                    rec['row_cache_switch_enabled']=run_f16_row_cache
                    rec['m4_cache_switch_enabled']=False
                    if env.get('GGML_SPACEMIT_F16_M4_CACHE') is not None:
                        raise RuntimeError('M4 cache environment variable unexpectedly present')
                    save()
                for repeat in range(-args.warmups,args.repeats):
                    for caseid in args.cases:
                        if shutil.disk_usage(ROOT).free < args.min_free_gib*1024**3:
                            raise RuntimeError('free space below requested floor; previous complete requests preserved')
                        case=available[caseid]
                        requestdir=dest/(f'{caseid}-'+('warmup' if repeat<0 else 'r')+str(abs(repeat))) if batch else dest
                        requestdir.mkdir(exist_ok=True)
                        sample={'case_id':caseid,'repeat':repeat,'warmup':repeat<0,
                                'status':'started',
                                'directory':str(requestdir.relative_to(out)),
                                'server_log_start_byte':(dest/'server.log').stat().st_size}
                        rec['requests'].append(sample)
                        data=base64.b64encode((ROOT/'images'/case['image']).read_bytes()).decode()
                        body={'model':'minicpm-ocr','messages':[{'role':'user','content':[
                            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+data}},
                            {'type':'text','text':case['prompt']}]}],
                            'temperature':0,'top_p':1,'seed':42,'repeat_penalty':1,'max_tokens':args.max_tokens or case.get('max_tokens',768),
                            'stream':False,'cache_prompt':False,'verbose':True,'return_tokens':True}
                        rec['status']='requesting'
                        # Never let a failed encoder write masquerade as a valid
                        # request by reusing the previous request's embedding.
                        # This is especially important for long drift checks.
                        embedding_path = dest/'vision-embeddings.bin'
                        embedding_path.unlink(missing_ok=True)
                        sample['embedding_preexisting_removed'] = True
                        sample['system_before']=runtime_observation(proc.pid)
                        if args.f16_row_cache_abba:
                            sample['worker_affinity_before']=affinity_evidence(
                                sample['system_before'],ROW_CACHE_ABBA_CORES)
                        boundary_version=21 if args.kv_projection_audit else 20 if args.kv_bias_audit else 19 if args.qkv_boundary_audit else 18
                        boundary_pattern=f'full-v{boundary_version}-p*-g*.jsonl'
                        boundary_before=set((dest/'worker-observer').glob(boundary_pattern)) if args.stop_on_boundary_change else set()
                        save()
                        start=time.monotonic()
                        response=http(port,'/v1/chat/completions',body,timeout=900)
                        sample['e2e_s']=time.monotonic()-start
                        sample['system_after']=runtime_observation(proc.pid)
                        if args.f16_row_cache_abba:
                            sample['worker_affinity_after']=affinity_evidence(
                                sample['system_after'],ROW_CACHE_ABBA_CORES)
                        sample['server_log_end_byte']=(dest/'server.log').stat().st_size
                        (requestdir/'response.json').write_text(json.dumps(response,ensure_ascii=False,indent=2))
                        text=response['choices'][0]['message']['content']
                        (requestdir/'output.txt').write_text(text)
                        sample.update(output=text,usage=response.get('usage'),timings=response.get('timings'),
                                      token_ids=response.get('__verbose',{}).get('tokens',[]),
                                      finish_reason=response['choices'][0].get('finish_reason'),
                                      output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                      tags={t:text.count('<'+t+'>') for t in ('nl','ucel','ecel','fcel')})
                        if case.get('reference') and case.get('metric')=='normalized_CER':
                            truth=normalize(case['reference'])
                            sample['normalized_CER']=edit_distance(normalize(text.replace('<|im_end|>','')),truth)/len(truth)
                        if case.get('expected_fields'):
                            sample['field_matches_raw']={field:field in text for field in case['expected_fields']}
                            sample['field_matches']={field:normalize(field) in normalize(text) for field in case['expected_fields']}
                        emb=dest/'vision-embeddings.bin'
                        embedding_blob=None
                        if emb.exists():
                            embedding_blob=emb.read_bytes()
                            sample['embedding_sha256']=hashlib.sha256(embedding_blob).hexdigest()
                            if batch:
                                (requestdir/'vision-embeddings.bin').write_bytes(embedding_blob)
                        with (dest/'server.log').open('rb') as log:
                            log.seek(sample['server_log_start_byte'])
                            snippet=log.read(sample['server_log_end_byte']-sample['server_log_start_byte']).decode(errors='replace')
                        (requestdir/'request-server.log').write_text(snippet)
                        vision_timing=parse_vision_encode_timing(snippet)
                        sample['vision_encode_blocks']=len(vision_timing['starts'])
                        sample['vision_timing_events']={k:v for k,v in vision_timing.items() if k!='intervals_s'}
                        if vision_timing['valid']:
                            sample['vision_encode_s']=vision_timing['vision_encode_s']
                            if sample['timings']:
                                sample['text_prefill_s']=sample['timings']['prompt_ms']/1000-sample['vision_encode_s']
                        if args.f16_row_cache_abba:
                            affinity_ok=sample['worker_affinity_after']['expected_worker_affinities_present']
                            timing_fields=sample.get('timings') or {}
                            timing_values=[sample.get('e2e_s'),timing_fields.get('prompt_ms'),
                                           timing_fields.get('predicted_ms'),sample.get('vision_encode_s'),
                                           sample.get('text_prefill_s')]
                            timing_ok=(vision_timing['valid']
                                and all(isinstance(v,(int,float)) and math.isfinite(v) for v in timing_values)
                                and all(v>0 for v in timing_values[:4]) and timing_values[4]>=0)
                            quality_ok=(isinstance(text,str) and isinstance(sample['token_ids'],list)
                                        and bool(sample['token_ids']) and embedding_blob is not None)
                            sample['affinity_valid']=bool(affinity_ok)
                            sample['timing_valid']=bool(timing_ok)
                            sample['quality_valid']=bool(quality_ok)
                            if not affinity_ok:
                                sample['status']='affinity_mismatch_after_request'
                            elif not timing_ok:
                                sample['status']='timing_incomplete'
                            elif not quality_ok:
                                sample['status']='quality_evidence_missing'
                        if args.strict_quality_evidence:
                            if not isinstance(text, str) or not text or not sample['token_ids'] or embedding_blob is None:
                                sample['status']='quality_evidence_missing'
                                sample['quality_valid']=False
                                save()
                                raise RuntimeError('strict quality evidence missing (fresh embedding, token IDs and OCR text required)')
                            sample['quality_valid']=True
                        if not batch:
                            rec.update(sample)
                            raw=sample['token_ids']
                            if raw:
                                for allow in (False,True):
                                    decoded=http(port,'/detokenize',{'tokens':raw,'special':allow})
                                    (dest/f'detokenize-{allow}.json').write_text(json.dumps(decoded,ensure_ascii=False,indent=2))
                        if not args.f16_row_cache_abba:
                            sample['status']='completed'
                        elif sample.get('affinity_valid') and sample.get('timing_valid') and sample.get('quality_valid'):
                            sample['status']='completed'
                        save()
                        if args.f16_row_cache_abba:
                            key=(caseid,repeat)
                            evidence={'output':text,'token_ids':sample['token_ids'],
                                      'embedding_sha256':sample.get('embedding_sha256'),
                                      'embedding_bytes':embedding_blob}
                            previous=abba_quality_reference.get(key)
                            if previous is None:
                                if not name.endswith('-row-cache-off-a-q8'):
                                    sample['status']='quality_reference_missing'
                                    abba_invalid=True
                                    state['row_cache_abba_invalid_reason']='OFF-A reference request is missing'
                                    save()
                                    raise RuntimeError(state['row_cache_abba_invalid_reason'])
                                abba_quality_reference[key]=evidence
                                comparison={'case_id':caseid,'repeat':repeat,'arm':'off-a','reference':True,
                                            'embedding_sha256':evidence['embedding_sha256']}
                            else:
                                comparison={'case_id':caseid,'repeat':repeat,'arm':name,
                                    'embedding_bytes_equal':evidence['embedding_bytes']==previous['embedding_bytes'],
                                    'embedding_sha256_equal':evidence['embedding_sha256']==previous['embedding_sha256'],
                                    'token_ids_equal':evidence['token_ids']==previous['token_ids'],
                                    'ocr_output_equal':evidence['output']==previous['output']}
                                comparison['exact_all_modal_outputs']=all(comparison[k] for k in
                                    ('embedding_bytes_equal','embedding_sha256_equal','token_ids_equal','ocr_output_equal'))
                            state.setdefault('row_cache_abba_quality_checks',[]).append(comparison)
                            if not comparison.get('reference',False) and not comparison['exact_all_modal_outputs']:
                                sample['status']='quality_mismatch'
                                abba_invalid=True
                                state['row_cache_abba_invalid_reason']=f"quality mismatch for {caseid} repeat {repeat} against OFF-A"
                                save()
                                raise RuntimeError(state['row_cache_abba_invalid_reason'])
                            if not sample.get('affinity_valid') or not sample.get('timing_valid') or not sample.get('quality_valid'):
                                abba_invalid=True
                                state['row_cache_abba_invalid_reason']=f"invalid affinity/timing/quality evidence for {caseid} repeat {repeat}"
                                save()
                                raise RuntimeError(state['row_cache_abba_invalid_reason'])
                            state['row_cache_abba_valid_so_far']=True
                            save()
                        print(json.dumps({'name':name,'case':caseid,'repeat':repeat,'e2e_s':sample['e2e_s'],
                                          'vision_encode_s':sample.get('vision_encode_s'),'tags':sample['tags']},ensure_ascii=False),flush=True)
                        if args.expected_embedding_sha256:
                            sample['expected_embedding_sha256']=args.expected_embedding_sha256
                            sample['embedding_matches_expected']=(sample.get('embedding_sha256')==args.expected_embedding_sha256)
                            save()
                            if not sample['embedding_matches_expected']:
                                sample['status']='embedding_mismatch'
                                state['first_embedding_mismatch']={
                                    'directory':sample['directory'],
                                    'expected_sha256':args.expected_embedding_sha256,
                                    'actual_sha256':sample.get('embedding_sha256')}
                                save()
                                raise RuntimeError('embedding mismatch; complete request evidence preserved')
                        if args.stop_on_boundary_change:
                            changed=[]
                            paths=set((dest/'worker-observer').glob(boundary_pattern))-boundary_before
                            assert len(paths)==28, 'Expected 28 complete visual full-boundary records'
                            for path in sorted(paths,key=lambda p:int(p.name.split('-g')[1].split('.')[0])):
                                rows=[json.loads(line) for line in path.read_text().splitlines()]
                                assert rows[0]['complete'] and len(rows)==rows[0]['captures']+1
                                changed.extend({'invocation':rows[0]['invocation'],'ordinal':r['ordinal'],'role':r['role'],'changed_words':r['changed_words']}
                                               for r in rows[1:] if not r['equal_to_baseline'])
                            sample['changed_full_boundaries']=changed
                            save()
                            if changed:
                                state['first_boundary_mismatch']={'directory':sample['directory'],'boundaries':changed}
                                save()
                                raise RuntimeError('internal boundary mismatch; complete request evidence preserved')
                rec['status']='completed'
            except Exception as exc:
                rec.update(status='failed',error=str(exc))
                if args.f16_row_cache_abba:
                    abba_invalid=True
                    state['row_cache_abba_valid_so_far']=False
                    state['row_cache_abba_invalid_reason']=str(exc)
                print(name,repr(exc),flush=True)
            finally:
                if proc:
                    stop(proc)
                    rec['returncode']=proc.returncode
                    proc=None
                emb=dest/'vision-embeddings.bin'
                if emb.exists():
                    rec['embedding_sha256']=hashlib.sha256(emb.read_bytes()).hexdigest()
                rec['trace_bytes']=(dest/'ggml_trace.json').stat().st_size if (dest/'ggml_trace.json').exists() else 0
                save()
            if args.f16_row_cache_abba and abba_invalid:
                break
    finally:
        if proc:
            stop(proc)
        if original:
            argv,env,cwd=original
            with (out/'original-restored.log').open('w') as stream:
                restored=subprocess.Popen(argv,env=env,cwd=cwd,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            pidfile.write_text(str(restored.pid)+'\n')
            wait_ready(18581,restored)
            state['restored']={'pid':restored.pid,'health':http(18581,'/health')}
        state['finished']=datetime.now().astimezone().isoformat()
        if args.f16_row_cache_abba:
            complete=(not abba_invalid and len(state['runs'])==4 and
                      all(run.get('status')=='completed' for run in state['runs']) and
                      bool(state.get('row_cache_abba_quality_checks')) and
                      all(row.get('reference') or row.get('exact_all_modal_outputs')
                          for row in state['row_cache_abba_quality_checks']))
            state['row_cache_abba_valid']=complete
            if not complete and not state.get('row_cache_abba_invalid_reason'):
                state['row_cache_abba_invalid_reason']='ABBA did not complete all four arms and exact quality gates'
        save()
        print('ROUND FINISHED',json.dumps(state.get('restored')),flush=True)
    if any(run.get('status')!='completed' for run in state['runs']):
        raise RuntimeError('one or more requests failed; inspect the preserved round.json')
    if args.f16_row_cache_abba and not state.get('row_cache_abba_valid'):
        raise RuntimeError('row-cache ABBA is invalid; inspect the preserved round.json')

if __name__=='__main__':
    main()
