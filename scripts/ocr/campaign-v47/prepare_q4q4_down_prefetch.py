"""Isolated real-input down-kernel candidates; does not modify the tested runtime."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SRC=Path('Y:/llama.cpp-tool-spacemit-mini-CPM-OCR/ocr-adaptation/managed-source')
DST=SRC.parent/'experiments/v48-prefetch';DST.mkdir(parents=True,exist_ok=False)
h=(SRC/'ggml/src/ggml-spacemit/ocr-down-vector-v44.h').read_text(encoding='utf8')
h=h.replace('namespace ocr_down_vector_v44 {','namespace ocr_down_prefetch_v48 {')
old='''      const vfloat16m2_t a=__riscv_vle16_v_f16m2(wp+k,vl);'''
new='''      if constexpr(MODE>=8){
       constexpr int DIST=MODE==8?2:(MODE==9?4:8);
       if(m+row+DIST<r1){
        const auto *next=(const _Float16*)((const char*)w->data+(m+row+DIST)*w->nb[1]);
        __builtin_prefetch(next+k,0,3);
       }
      }
      const vfloat16m2_t a=__riscv_vle16_v_f16m2(wp+k,vl);'''
assert h.count(old)==1;h=h.replace(old,new)
(DST/'ocr-down-prefetch-v48.h').write_text(h,encoding='utf8',newline='\n')
b=(SRC/'tests/ocr/bench_down_vector_v44.cpp').read_text(encoding='utf8')
b=b.replace('#include "ocr-down-vector-v44.h"','#include "ocr-down-prefetch-v48.h"')
b=b.replace('std::vector<int> sizes={7,65,988,989,1008,1009,1056,1057,1060}','std::vector<int> sizes={7,65,988,1008,1056,1057}')
b=b.replace('default:std::abort();}', '''case 8:ocr_down_prefetch_v48::kernel<8>(ctx,yt,c);break;
    case 9:ocr_down_prefetch_v48::kernel<9>(ctx,yt,c);break;
    case 10:ocr_down_prefetch_v48::kernel<10>(ctx,yt,c);break;
    default:std::abort();}''')
b=b.replace('ocr_down_vector_v44::kernel<','ocr_down_prefetch_v48::kernel<')
b=b.replace('const int order[]={0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0};','const int order[]={5,8,9,10,10,9,8,5};')
b=b.replace('arm<16','arm<8')
(DST/'bench_down_prefetch_v48.cpp').write_text(b,encoding='utf8',newline='\n')
manifest={'base_commit':'97e5c63240cef58b9751f08868f5cc040bfda6fc','kind':'standalone replay only',
 'candidates':{'5':'v47 baseline','8':'prefetch two weight rows ahead','9':'prefetch four rows ahead','10':'prefetch eight rows ahead'},
 'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DST.iterdir()}}
(DST/'source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(DST)
