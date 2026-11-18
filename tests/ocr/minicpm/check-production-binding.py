"""Read-only current production/reference binding; never modifies engine files."""
from pathlib import Path
import argparse,hashlib,json

def check(engine_root):
    base=Path(__file__).resolve().parent;engine_root=Path(engine_root)
    bindings=json.loads((base/'PRODUCTION-BINDINGS.json').read_text(encoding='utf8'))
    sha=lambda b:hashlib.sha256(b).hexdigest()
    files={};issues=[]
    for relative,item in bindings['expected_production_headers'].items():
        path=engine_root/relative
        if not path.is_file():issues.append('production header missing: '+relative);continue
        raw=path.read_bytes();same=sha(raw.replace(b'\r\n',b'\n'))==item['canonical_lf_sha256']
        files[relative]={'actual_sha256':sha(raw),'canonical_lf_sha256':sha(raw.replace(b'\r\n',b'\n')),'reference':item['reference'],'matches_reference_allowing_EOL_only':same}
        if not same:issues.append('production header text differs: '+relative)
    relative='ggml/src/ggml-spacemit/rvv_kernels.cpp';path=engine_root/relative
    if not path.is_file():issues.append('production GDN source missing')
    else:
        raw=path.read_bytes();text=raw.decode('utf8').replace('\r\n','\n')
        try:
            begin=text.index('static inline float rvv_gdn_fused_row_f32(');end=text.index('\nvoid forward_scale_f32(',begin)
            candidate=('namespace gdn_candidate {\n'+text[begin:end]+'\n}\n').encode()
            reference=(base/'gdn/frozen48/gdn-candidate.h').read_bytes().replace(b'\r\n',b'\n')
            same=candidate==reference
            files[relative]={'actual_sha256':sha(raw),'extracted_candidate_sha256':sha(candidate),'frozen_candidate_canonical_lf_sha256':sha(reference),'candidate_region_exact_allowing_EOL_only':same}
            if not same:issues.append('current production GDN region differs from frozen48 candidate')
            begin=text.index('void memcpy1d(');end=text.index('\nvoid memcpy2d(',begin);copy=text[begin:end]
            begin=text.index('void forward_cpy_f32_row_strided(');end=text.index('\n// Permuting K/V changes',begin);copy+=text[begin:end]
            actual=('namespace gdn_copy {\n'+copy+'\n}\n').encode();reference=(base/'gdn/frozen48/gdn-copy-reference.h').read_bytes().replace(b'\r\n',b'\n')
            files[relative]['original_copy_region_matches']=actual==reference
            if actual!=reference:issues.append('current production CPY region differs from frozen reference')
        except (ValueError,UnicodeError) as exc:issues.append('cannot bind production function regions: '+str(exc))
    return {'status':'PASS' if not issues else 'FAIL','issues':issues,'files':files,'scope':'Read-only exact text binding with only CRLF/LF normalization. Historical raw source identities remain separate. No compiler/device/full-graph execution.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine-root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--report',type=Path);a=p.parse_args()
    result=check(a.engine_root);text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if a.report:a.report.write_text(text,encoding='utf8')
    raise SystemExit(result['status']!='PASS')
