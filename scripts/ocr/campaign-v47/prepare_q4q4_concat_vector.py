from pathlib import Path
import json,hashlib
SRC=Path('Y:/llama.cpp-tool-spacemit-mini-CPM-OCR/ocr-adaptation/managed-source');DST=SRC.parent/'experiments/v49-concat-vector';DST.mkdir(parents=True,exist_ok=False)
h=(SRC/'ggml/src/ggml-spacemit/ocr-concat-tcm-v46.h').read_text(encoding='utf8')
h=h.replace('namespace ocr_concat_tcm_v46 {','namespace ocr_concat_vector_v49 {').replace('template<int C,int T>','template<int C,int T,bool VECTOR=false>')
old='''    for(int64_t off=0;off<cols;++off)
     std::memcpy(d+(c+off)*op->nb[1]+(s0->ne[0]+t0)*4,scratch+off*T,size_t(tokens)*4);'''
new='''    for(int64_t off=0;off<cols;++off){
     auto*dst=(float*)(d+(c+off)*op->nb[1]+(s0->ne[0]+t0)*4);
     if constexpr(VECTOR){
      for(int64_t i=0;i<tokens;){const size_t vl=__riscv_vsetvl_e32m8(tokens-i);
       const auto v=__riscv_vle32_v_f32m8(scratch+off*T+i,vl);
       __riscv_vse32_v_f32m8(dst+i,v,vl);i+=vl;}
     }else std::memcpy(dst,scratch+off*T,size_t(tokens)*4);
    }'''
assert h.count(old)==1;h=h.replace(old,new).replace('case 4:ok=kernel<64,64>(ctx,op,col0);break;default:','case 4:ok=kernel<64,64>(ctx,op,col0);break;case 5:ok=kernel<64,64,true>(ctx,op,col0);break;default:')
(DST/'ocr-concat-vector-v49.h').write_text(h,encoding='utf8',newline='\n')
b=(SRC/'tests/ocr/bench_concat_tcm_v47.cpp').read_text(encoding='utf8').replace('ocr-concat-tcm-v46.h','ocr-concat-vector-v49.h').replace('ocr_concat_tcm_v46::','ocr_concat_vector_v49::')
b=b.replace('const int order[]={0,1,2,3,4,4,3,2,1,0};','const int order[]={4,5,5,4};').replace('arm<10','arm<4')
(DST/'bench_concat_vector_v49.cpp').write_text(b,encoding='utf8',newline='\n')
(DST/'source-manifest.json').write_text(json.dumps({'base_commit':'97e5c63240cef58b9751f08868f5cc040bfda6fc','scope':'standalone replay; mode4 libc copy versus mode5 RVV m8 contiguous writeback', 'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DST.iterdir()}},indent=2),encoding='utf8')
print(DST)
