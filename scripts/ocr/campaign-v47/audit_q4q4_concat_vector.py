from pathlib import Path
import json,statistics,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-q4q4-long-1430';F=BASE/'20261104-q4q4-v49-concat-vector'
d=json.loads((F/'round.json').read_text(encoding='utf8'));assert d['restored']['health']['status']=='ok' and d['finished']
lines=[json.loads(l) for l in (F/'concat.jsonl').read_text().splitlines()];groups={}
for r in lines:
 assert r['bitwise_exact'] and r['guards'] and r['input_immutable']
 groups.setdefault((r['tokens'],r['channels'],r['batches'],r['col0'],r['mode']),[]).append(r['ms'])
rows=[]
for key in sorted({k[:-1] for k in groups}):
 a,b=groups[key+(4,)],groups[key+(5,)];assert len(a)==len(b)==6
 rows.append(dict(zip(['tokens','channels','batches','col0'],key),baseline_ms=statistics.mean(a),candidate_ms=statistics.mean(b),reduction_pct=100*(1-statistics.mean(b)/statistics.mean(a))))
src=Path('Y:/llama.cpp-tool-spacemit-mini-CPM-OCR/ocr-adaptation/experiments/v49-concat-vector');m=json.loads((src/'source-manifest.json').read_text(encoding='utf8'))
for name,want in m['files'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==want
result={'passed':True,'formal_runs':len(lines),'rows':rows,'source':m,'binary_sha256':hashlib.sha256((src/'bench-concat-vector-v49').read_bytes()).hexdigest(),'scope':'Standalone operator replay only; reversed order, six formal per variant/shape; arbitrary bit patterns and guards. No complete OCR or end-to-end claim.'}
(OUT/'concat-vector-audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
dst=OUT/'concat-vector-source';dst.mkdir(exist_ok=True)
for name in [*m['files'],'source-manifest.json']:shutil.copy2(src/name,dst/name)
print(json.dumps({'formal':len(lines),'full_6144':[r for r in rows if r['channels']==6144 and r['col0']==0]},indent=2))
