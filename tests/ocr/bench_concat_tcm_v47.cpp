#include "ggml.h"
#include "ggml-impl.h"
#include "rvv_kernels.h"
#include "ocr-concat-tcm-v46.h"
#include <vector>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstring>
static uint32_t pattern(size_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;return uint32_t(x*0x846ca68b);}
int main(){
 constexpr int THREADS=7;
 spert::StreamConfig cfg;cfg.n_cores=THREADS;cfg.core_ids={8,9,10,11,13,14,15};spert::Stream stream(cfg);GGML_ASSERT(stream.valid());
 ggml_init_params gp{};gp.mem_size=1024*1024;gp.no_alloc=true;auto*gc=ggml_init(gp);
 for(int tokens:{1,3,4,63,64,65,252,256,257})for(int variant:{0,1}){
  const int channels=variant?65:6144,batches=variant?2:1,total=tokens+3;
  auto*a=ggml_new_tensor_4d(gc,GGML_TYPE_F32,3,channels,batches,1);
  auto*b=ggml_new_tensor_4d(gc,GGML_TYPE_F32,tokens,channels,batches,1);
  b->nb[0]=channels*4;b->nb[1]=4;b->nb[2]=tokens*channels*4;b->nb[3]=b->nb[2]*batches;
  std::vector<uint32_t>x(size_t(3)*channels*batches),y(size_t(tokens)*channels*batches);
  for(size_t i=0;i<x.size();++i)x[i]=pattern(i+1);for(size_t i=0;i<y.size();++i)y[i]=pattern(i+99);
  // Include NaN/infinity/sign-bit patterns: CONCAT must copy arbitrary bits.
  x[0]=0x7fc01234;y[0]=0xff800000;const auto xc=x,yc=y;a->data=x.data();b->data=y.data();
  auto*op=ggml_concat(gc,a,b,0);std::vector<uint32_t>out(size_t(total)*channels*batches+128,0xA7B8C9D0);op->data=out.data()+64;
  for(int part:{0,1}){
   const int col0=part?total-3:0;
   std::vector<uint32_t>ref=out;std::fill(ref.begin(),ref.end(),0xA7B8C9D0);
   for(int batch=0;batch<batches;++batch)for(int c=0;c<channels;++c)for(int t=col0;t<total;++t)
    ref[64+size_t(batch)*channels*total+size_t(c)*total+t]=t<3?x[size_t(batch)*channels*3+c*3+t]:y[size_t(batch)*channels*tokens+size_t(t-3)*channels+c];
   auto run=[&](int mode){
    std::fill(out.begin(),out.end(),0xA7B8C9D0);auto start=std::chrono::steady_clock::now();
    auto f=stream.launch(spert::Grid{THREADS},[&](spert::Context*rt){
     auto sh=rt->shared_buffer();ggml::spacemit::context ctx;ctx.reset(*rt,rt->program_id(0),THREADS,nullptr,0,sh);
     std::memset(sh.data,0xFF,sh.size);
     bool ok=mode&&ocr_concat_tcm_v46::forward(ctx,op,col0,mode);
     if(!ok)ok=spacemit_kernels::rvv::forward_concat_dim0_transposed_f32(ctx,op,col0);
     GGML_ASSERT(ok);
    });GGML_ASSERT(f.sync()==spert::Status::Ok);
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    GGML_ASSERT(out==ref&&x==xc&&y==yc);return ms;
   };
   const int order[]={0,1,2,3,4,4,3,2,1,0};
   for(int arm=0;arm<10;++arm){const int mode=order[arm];run(mode);
    for(int repeat=0;repeat<3;++repeat){const double ms=run(mode);
     std::printf("{\"tokens\":%d,\"channels\":%d,\"batches\":%d,\"col0\":%d,\"mode\":%d,\"arm\":%d,\"repeat\":%d,\"ms\":%.6f,\"bitwise_exact\":true,\"guards\":true,\"input_immutable\":true}\n",tokens,channels,batches,col0,mode,arm,repeat,ms);std::fflush(stdout);}
   }
  }
 }
 ggml_free(gc);return 0;
}
