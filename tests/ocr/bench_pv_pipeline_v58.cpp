#include "spacemit-context.h"
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
