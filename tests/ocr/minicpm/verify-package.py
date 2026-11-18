"""Verify uploaded bytes and C/C++ local include closure; read-only."""
from pathlib import Path
import hashlib,json,re

def verify():
    base=Path(__file__).resolve().parent;manifest=json.loads((base/'PACKAGE-MANIFEST.json').read_text(encoding='utf8'));issues=[]
    for relative,item in manifest['files'].items():
        path=base/relative
        if not path.is_file() or path.stat().st_size!=item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:issues.append('missing/changed package file: '+relative)
    for path in base.rglob('*'):
        if not path.is_file():continue
        if '__pycache__' in path.parts:continue # interpreter may generate cache after upload
        if path.suffix in {'.exe','.so','.o','.a','.gguf','.onnx','.bin'}:issues.append('forbidden binary/model payload: '+str(path.relative_to(base)))
        if path.suffix not in {'.cpp','.h'}:continue
        for name in re.findall(r'^\s*#\s*include\s*"([^"]+)"',path.read_text(encoding='utf8'),re.M):
            roots=[path.parent,base/'mapped/reference',base/'mapped/contract-headers',base/'mapped/test_mapped_contract_stubs',base/'gdn/frozen48',base/'gdn/frozen48/fixtures']
            if not any((root/name).is_file() for root in roots):issues.append('unresolved local include: '+str(path.relative_to(base))+' -> '+name)
    return {'status':'PASS' if not issues else 'FAIL','files_checked':len(manifest['files']),'issues':issues,'scope':'Package byte identities and include closure only; no execution.'}

if __name__=='__main__':
    result=verify();print(json.dumps(result,indent=2));raise SystemExit(result['status']!='PASS')
