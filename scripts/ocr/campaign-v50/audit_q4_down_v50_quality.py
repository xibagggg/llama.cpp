from pathlib import Path
import json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/optimization';OUT=BASE/'20261104-q4-down-attention'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
groups={};dirs={}
for kind in ['baseline','q4']:
 d=BASE/f'20261104-q4-down-v50-quality-{kind}';s=read(d/'round.json')
 assert s['restored']['health']['status']=='ok' and len(s['runs'])==1
 r=s['runs'][0];assert len(r['requests'])==10 and r['status']=='completed'
 assert r['environment']['MTMD_OCR_PADDED_DOWN']=='1' if kind=='q4' else 'MTMD_OCR_PADDED_DOWN' not in r['environment']
 groups[kind]={q['case_id']:q for q in r['requests']};dirs[kind]=d
 for q in r['requests']:
  assert q['status']=='completed' and q['quality_valid'] and q['finish_reason']=='stop'
  for f,key in [('output.txt','output_sha256'),('vision-embeddings.bin','embedding_sha256')]:
   assert hashlib.sha256((d/q['directory']/f).read_bytes()).hexdigest()==q[key]
rows=[]
for case,a in groups['baseline'].items():
 b=groups['q4'][case];x=np.fromfile(dirs['baseline']/a['directory']/'vision-embeddings.bin',np.float32).astype(np.float64)
 y=np.fromfile(dirs['q4']/b['directory']/'vision-embeddings.bin',np.float32).astype(np.float64)
 assert x.shape==y.shape and np.all(np.isfinite(y))
 row={'case':case,'text_equal':a['output']==b['output'],'tokens_equal':a['token_ids']==b['token_ids'],
      'last_slice_embedding_cosine':float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),
      'last_slice_embedding_rel_l2':float(np.linalg.norm(y-x)/np.linalg.norm(x)),
      'baseline_e2e_s':a['e2e_s'],'q4_e2e_s':b['e2e_s'],
      'baseline_vision_s':a.get('vision_encode_s'),'q4_vision_s':b.get('vision_encode_s'),
      'baseline_CER':a.get('normalized_CER'),'q4_CER':b.get('normalized_CER')}
 rows.append(row)
z={'all_ten_output_tokens_equal':all(r['text_equal'] and r['tokens_equal'] for r in rows),
   'scope':'ten selected documents, one request per model; embeddings cover last saved slice only; not general Q4/Q8 accuracy or confidence equivalence','rows':rows}
(OUT/'quality-audit.json').write_text(json.dumps(z,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(z,ensure_ascii=False,indent=2))
