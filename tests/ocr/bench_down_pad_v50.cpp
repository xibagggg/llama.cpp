// Real K4304 activations -> PAD48 -> stock Q4_0/IME2 K4352 matmul.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-quants.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
template<class T>std::vector<T> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);GGML_ASSERT(f);size_t n=f.tellg();GGML_ASSERT(n%sizeof(T)==0);std::vector<T>v(n/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);GGML_ASSERT(f);return v;}
int main(int argc,char**argv){
 if(argc!=2)return 1;const std::string dir=argv[1];constexpr int K=4304,P=4352,M=1152;
 auto wf=read<ggml_fp16_t>(dir+"/weight-0.f16");auto orig=read<float>(dir+"/0000-layer-00-x.f32");
 GGML_ASSERT(wf.size()==size_t(K)*M&&orig.size()%K==0);int realN=orig.size()/K;
 std::vector<float>w(size_t(P)*M,0),deq(w.size());
 for(int m=0;m<M;++m)ggml_fp16_to_fp32_row(wf.data()+m*K,w.data()+m*P,K);
 std::vector<unsigned char>qw(size_t(P/32)*18*M);
 GGML_ASSERT(ggml_quantize_chunk(GGML_TYPE_Q4_0,w.data(),qw.data(),0,M,P,nullptr)==qw.size());
 for(int m=0;m<M;++m)dequantize_row_q4_0((const block_q4_0*)(qw.data()+m*(P/32)*18),deq.data()+m*P,P);
 for(int m=0;m<M;++m)for(int k=K;k<P;++k)GGML_ASSERT(deq[m*P+k]==0);
 ggml_backend_load_all();auto backend=ggml_backend_init_by_name("SPACEMIT0",nullptr);GGML_ASSERT(backend);
 for(int N:{7,65,988,1008,1057}){
  ggml_init_params params{};params.mem_size=2*1024*1024;params.no_alloc=true;auto*ctx=ggml_init(params);
  auto*wt=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_0,P,M);ggml_set_name(wt,"v.blk.0.ffn_down.weight");
  auto*xt=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,N);auto*xp=ggml_pad(ctx,xt,P-K,0,0,0);auto*yt=ggml_mul_mat(ctx,wt,xp);
  auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,yt);
  GGML_ASSERT(ggml_backend_supports_op(backend,xp)&&ggml_backend_supports_op(backend,yt));
  auto buf=ggml_backend_alloc_ctx_tensors(ctx,backend);GGML_ASSERT(buf);
  std::vector<float>x(size_t(K)*N),y(size_t(M)*N),reference;
  for(int n=0;n<N;++n)std::memcpy(x.data()+n*K,orig.data()+(n%realN)*K,K*4);
  ggml_backend_tensor_set(wt,qw.data(),0,qw.size());ggml_backend_tensor_set(xt,x.data(),0,x.size()*4);
  for(int r=-1;r<3;++r){
   auto start=std::chrono::steady_clock::now();GGML_ASSERT(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS);ggml_backend_synchronize(backend);
   double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   ggml_backend_tensor_get(yt,y.data(),0,y.size()*4);for(float a:y)GGML_ASSERT(std::isfinite(a));
   if(r<0)reference=y;else GGML_ASSERT(std::memcmp(reference.data(),y.data(),y.size()*4)==0);
   double err=0,base=0;for(int j=0;j<96;++j){int n=(j*37)%N,m=(j*101)%M;double expected=0;
    for(int k=0;k<K;++k)expected+=double(deq[m*P+k])*double(x[n*K+k]);
    const double diff=y[n*M+m]-expected;err+=diff*diff;base+=expected*expected;}
   double relative=std::sqrt(err/base);GGML_ASSERT(relative<.05);
   if(r>=0)std::printf("{\"N\":%d,\"repeat\":%d,\"pad_and_q4_ms\":%.6f,\"sampled_w4a8_rel_l2_vs_w4_f32\":%.9f,\"samples\":96,\"repeat_exact\":true,\"finite\":true,\"padding_zero\":true}\n",N,r,ms,relative);
  }
  std::vector<float>back(x.size());ggml_backend_tensor_get(xt,back.data(),0,back.size()*4);GGML_ASSERT(back==x);
  ggml_backend_buffer_free(buf);ggml_free(ctx);
 }
 ggml_backend_free(backend);
}
