from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1];D=Path('Y:/llama.cpp-tool-spacemit-mini-CPM-OCR/ocr-adaptation/experiments/v58-pv-pipeline');D.mkdir(exist_ok=True)
header=subprocess.check_output(['git','-C',str(ROOT/'source-management/llama-publish'),'show','54afbde:ggml/src/ggml-spacemit/ocr-fa-x8-v39.h'],text=True)
header='#include <cstdlib>\n#include <cstring>\n#include <riscv_vector.h>\n'+header
start=header.index('static inline void ocr_pv_f16_x8')
ref=header[start:]
functions=[]
for block in [2,4]:
 f=f'''__attribute__((noinline)) static void pv_u{block}(float *__restrict__ dst,const float *__restrict__ p,const _Float16 *__restrict__ v,int dv,int stride,int count){{
 for(int off=0;off<dv;){{
  const size_t vl=__riscv_vsetvl_e16m1(dv-off);
'''
 for i in range(8):f+=f'  auto a{i}=__riscv_vle32_v_f32m2(dst+{i}*dv+off,vl);\n'
 f+=f'  int t=0;for(;t+{block-1}<count;t+={block}){{\n'
 for j in range(block):f+=f'   const auto v{j}=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+(t+{j})*dv+off,vl),vl);\n'
 for j in range(block):
  for i in range(8):f+=f'   a{i}=__riscv_vfmacc_vf_f32m2(a{i},p[{i}*stride+t+{j}],v{j},vl);\n'
 f+='  }\n  for(;t<count;++t){const auto vv=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+t*dv+off,vl),vl);\n'
 for i in range(8):f+=f'   a{i}=__riscv_vfmacc_vf_f32m2(a{i},p[{i}*stride+t],vv,vl);\n'
 f+='  }\n'
 for i in range(8):f+=f'  __riscv_vse32_v_f32m2(dst+{i}*dv+off,a{i},vl);\n'
 f+='  off+=vl;\n }\n}\n';functions.append(f)
(D/'pv-pipeline-v58.h').write_text(header+'\n'+'\n'.join(functions),encoding='utf8',newline='\n')
bench=r'''#include "spacemit-context.h"
#include "pv-pipeline-v58.h"
#include <cstdio>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
using Clock=std::chrono::steady_clock;
__attribute__((noinline)) static void pv_ref(float*d,const float*p,const _Float16*v,int dv,int stride,int count){ocr_pv_f16_x8(d,p,v,dv,stride,count);}
using Fn=void(*)(float*,const float*,const _Float16*,int,int,int);
int main(){
 spert::StreamConfig cfg;cfg.n_cores=7;cfg.core_ids={8,9,10,11,13,14,15};spert::Stream stream(cfg);if(!stream.valid())return 1;
 Fn fns[]={pv_ref,pv_u2,pv_u4};constexpr int D=72,K=64,L=256;
 for(int count:{1,7,31,63,64}){
  std::array<int,7>valid{};std::array<double,7>elapsed{};
  for(int mode:{0,1,2,2,1,0})for(int rep=0;rep<3;++rep){
   auto future=stream.launch(spert::Grid{7},[&](spert::Context*rt){
    int w=rt->program_id(0);auto sh=rt->shared_buffer();if(sh.size<65536)return;
    auto*p=(float*)sh.data;auto*v=(_Float16*)(p+8*K+16);auto*d=(float*)(v+K*D+64);auto*ref=d+8*D+16;
    for(int i=0;i<8*K;++i)p[i]=float((i*17+3)%101)*.001f;
    for(int i=0;i<K*D;++i)v[i]=(_Float16)(float((i*13+7)%211-105)*.003f);
    std::fill(d,d+8*D+16,-123456.f);std::fill(ref,ref+8*D,0.f);std::fill(d,d+8*D,0.f);
    pv_ref(ref,p,v,D,K,count);fns[mode](d,p,v,D,K,count);
    if(std::memcmp(ref,d,8*D*4))return;
    for(int i=0;i<16;++i)if(d[8*D+i]!=-123456.f)return;
    auto t=Clock::now();for(int i=0;i<L;++i)fns[mode](d,p,v,D,K,count);
    elapsed[w]=std::chrono::duration<double,std::micro>(Clock::now()-t).count()/L;
    valid[w]=1;
   });if(future.sync()!=spert::Status::Ok)return 2;
   if(std::find(valid.begin(),valid.end(),0)!=valid.end())return 3;
   double average=0;for(auto t:elapsed)average+=t/7;
   std::printf("{\"count\":%d,\"mode\":%d,\"repeat\":%d,\"mean_worker_us\":%.6f,\"exact_vs_original\":true,\"guards\":true,\"VL72\":%zu}\n",count,mode,rep,average,__riscv_vsetvl_e16m1(72));
  }
 }
}
'''
(D/'bench_pv_pipeline_v58.cpp').write_text(bench,encoding='utf8',newline='\n')
