from pathlib import Path
import json,statistics,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-q4q4-long-1430'
F=BASE/'20261104-q4q4-v48-prefetch'
d=json.loads((F/'round.json').read_text(encoding='utf8'))
assert d['restored']['health']['status']=='ok' and d['finished']
lines=[json.loads(l) for l in (F/'down.jsonl').read_text().splitlines()]
assert lines[-1]['all_exact'] and lines[-1]['input_immutable']
groups={};formal=[r for r in lines if r['kind']=='timing']
for r in formal:
 assert r['bitwise_exact'] and r['guards'] and r['ms']>0
 groups.setdefault((r['N'],r['mode']),[]).append(r['ms'])
rows=[]
for N in sorted({r['N'] for r in formal}):
 modes={m:{'mean_ms':statistics.mean(groups[N,m]),'sd_ms':statistics.stdev(groups[N,m]),'n':len(groups[N,m])} for m in [5,8,9,10]}
 assert all(z['n']==6 for z in modes.values())
 rows.append({'N':N,'modes':modes,'reduction_pct':{m:100*(1-modes[m]['mean_ms']/modes[5]['mean_ms']) for m in [8,9,10]}})
src=Path('Y:/llama.cpp-tool-spacemit-mini-CPM-OCR/ocr-adaptation/experiments/v48-prefetch')
manifest=json.loads((src/'source-manifest.json').read_text(encoding='utf8'))
for name,want in manifest['files'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==want
result={'passed':True,'formal_runs':len(formal),'rows':rows,'source':manifest,
 'binary_sha256':hashlib.sha256((src/'bench-down-prefetch-v48').read_bytes()).hexdigest(),
 'scope':'Standalone exact real-input replay only; reversed candidate order, 6 formal per mode/shape. Not built into OCR runtime; no full OCR or end-to-end gain claim. Prefetch instructions confirmed in disassembly.'}
(OUT/'prefetch-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
dst=OUT/'prefetch-source';dst.mkdir(exist_ok=True)
for name in [*manifest['files'],'source-manifest.json']:shutil.copy2(src/name,dst/name)
print(json.dumps({'formal':len(formal),'rows':[{k:r[k] for k in ['N','reduction_pct']} for r in rows]},indent=2))
