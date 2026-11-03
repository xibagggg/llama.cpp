#include "ggml.h"
#include "ggml-impl.h"
#include "spacemit-context.h"
#include "rvv_kernels.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;return x;}
int main(int argc,char **argv){
 if(argc!=4)return 1;const int N=std::atoi(argv[1]);const int layout=std::atoi(argv[2]);const bool strided=layout&1,masked=layout&2;constexpr int D=72,H=16,T=7;
 if(N<64)return 2;
 std::vector<float>q(size_t(N)*D*H),out(size_t(N)*D*H+64,-987654.f);
 std::vector<_Float16>k(q.size()),v(q.size());
 for(size_t i=0;i<q.size();++i){q[i]=(int(mix(i+31)%20001)-10000)*.0001f;k[i]=(_Float16)((int(mix(i+51)%20001)-10000)*.00005f);v[i]=(_Float16)((int(mix(i+79)%20001)-10000)*.00007f);}
 ggml_init_params gp{};gp.mem_size=1024*1024;gp.no_alloc=true;auto *gc=ggml_init(gp);
 auto *qt=ggml_new_tensor_4d(gc,GGML_TYPE_F32,D,N,H,1),*kt=ggml_new_tensor_4d(gc,GGML_TYPE_F16,D,N,H,1),*vt=ggml_new_tensor_4d(gc,GGML_TYPE_F16,D,N,H,1);
 qt->data=q.data();kt->data=k.data();vt->data=v.data();
 if(strided)for(auto *t:{qt,kt,vt}){t->nb[1]=D*H*ggml_type_size(t->type);t->nb[2]=D*ggml_type_size(t->type);}
 std::vector<_Float16>maskdata;ggml_tensor *mask=nullptr;
 if(masked){
  const int padded=(N+63)/64*64;mask=ggml_new_tensor_2d(gc,GGML_TYPE_F16,N,padded);maskdata.resize(size_t(N)*padded);
  for(int i=0;i<padded;++i)for(int j=0;j<N;++j)maskdata[size_t(i)*N+j]=(_Float16)(j>i?-INFINITY:-.001f*(j%11));
  mask->data=maskdata.data();
 }
 auto *yt=ggml_flash_attn_ext(gc,qt,kt,vt,mask,1.f/std::sqrt(float(D)),0,0);yt->data=out.data();
 spert::StreamConfig cfg;cfg.n_cores=T;cfg.core_ids={8,9,10,11,13,14,15};spert::Stream stream(cfg);if(!stream.valid())return 3;
 std::vector<float>workspace(T*1024*1024);
 auto run=[&]{auto start=std::chrono::steady_clock::now();
  auto f=stream.launch(spert::Grid{T},[&](spert::Context *rt){
   int rank=rt->program_id(0);auto sh=rt->shared_buffer();ggml::spacemit::context ctx;ctx.reset(*rt,rank,T,workspace.data(),workspace.size()*4,sh);
   const int dr=(N*H+T-1)/T;
   spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(ctx,yt,dr*rank,std::min(dr*(rank+1),N*H));
  });if(f.sync()!=spert::Status::Ok)std::exit(4);
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
 run();for(int r=0;r<3;++r)std::printf("{\"N\":%d,\"layout\":%d,\"repeat\":%d,\"ms\":%.6f}\n",N,layout,r,run());
 for(size_t i=0;i<size_t(N)*D*H;++i)if(!std::isfinite(out[i]))return 5;
 for(size_t i=size_t(N)*D*H;i<out.size();++i)if(out[i]!=-987654.f)return 6;
 std::ofstream file(argv[3],std::ios::binary);file.write((const char*)out.data(),q.size()*4);if(!file)return 7;
 ggml_free(gc);return 0;
}
