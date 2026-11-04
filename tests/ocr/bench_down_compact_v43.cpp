// Replay real FFN-down activations against unchanged v42 paired TCM16.
#include "ocr-down-compact-v43.h"
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
constexpr int K=4304,M=1152,T=7;
template<class V> V read(const std::string &path){
 std::ifstream f(path,std::ios::binary|std::ios::ate);GGML_ASSERT(f.good());
 size_t bytes=size_t(f.tellg());GGML_ASSERT(bytes%sizeof(typename V::value_type)==0);
 V v(bytes/sizeof(typename V::value_type));f.seekg(0);f.read((char*)v.data(),bytes);GGML_ASSERT(f.good());return v;
}
template<class V> uint64_t hash(const V &v){uint64_t h=1469598103934665603ULL;const auto*p=(const unsigned char*)v.data();
 for(size_t i=0;i<v.size()*sizeof(typename V::value_type);++i){h^=p[i];h*=1099511628211ULL;}return h;}
int main(int argc,char **argv){
 if(argc!=2)return 1;const std::string dir=argv[1];
 auto w=read<std::vector<_Float16>>(dir+"/weight-0.f16");
 auto original=read<std::vector<float>>(dir+"/0000-layer-00-x.f32");
 auto saved=read<std::vector<float>>(dir+"/0000-layer-00-y.f32");
 GGML_ASSERT(w.size()==size_t(M)*K&&original.size()%K==0);
 const int realN=int(original.size()/K);const uint64_t wh=hash(w),xh=hash(original);
 spert::StreamConfig cfg;cfg.n_cores=T;cfg.core_ids={8,9,10,11,13,14,15};spert::Stream stream(cfg);GGML_ASSERT(stream.valid());
 ggml_init_params gp{};gp.mem_size=1024*1024;gp.no_alloc=true;auto*gc=ggml_init(gp);
 auto*wt=ggml_new_tensor_2d(gc,GGML_TYPE_F16,K,M);wt->data=w.data();ggml_set_name(wt,"v.blk.0.ffn_down.weight");
 std::vector<int> sizes={7,65,988,989,1008,1009,1056,1057,1060};if(std::find(sizes.begin(),sizes.end(),realN)==sizes.end())sizes.push_back(realN);
 for(int N:sizes){
  std::vector<float>x(size_t(N)*K);for(int n=0;n<N;++n)std::memcpy(x.data()+size_t(n)*K,original.data()+size_t(n%realN)*K,K*4);
  const uint64_t before=hash(x);std::vector<float>ref(size_t(N)*M),out(ref.size()+128,-987654.f);
  auto*xt=ggml_new_tensor_2d(gc,GGML_TYPE_F32,K,N);xt->data=x.data();auto*yt=ggml_mul_mat(gc,wt,xt);yt->data=out.data()+64;
  auto run=[&](int mode,bool profile){
   std::fill(out.begin(),out.end(),-987654.f);auto start=std::chrono::steady_clock::now();
   auto f=stream.launch(spert::Grid{T},[&](spert::Context*rt){
    auto sh=rt->shared_buffer();GGML_ASSERT(sh.size>=size_t(22)*K*4+64);ggml::spacemit::context ctx;ctx.reset(*rt,rt->program_id(0),T,nullptr,0,sh);
    auto*c=(float*)sh.data;
    if(mode==0)ocr_down_sched_v42::paired_kernel<16>(ctx,yt,c);
    else if(mode==1)ocr_down_compact_v43::compact_kernel<16>(ctx,yt,c);
    else ocr_down_compact_v43::compact_kernel<20>(ctx,yt,c);
   });GGML_ASSERT(f.sync()==spert::Status::Ok);
   double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   for(size_t i=0;i<64;++i)GGML_ASSERT(out[i]==-987654.f&&out[out.size()-1-i]==-987654.f);
   for(size_t i=0;i<ref.size();++i)GGML_ASSERT(std::isfinite(out[i+64]));
   return ms;
  };
  run(0,false);std::memcpy(ref.data(),out.data()+64,ref.size()*4);
  size_t saved_mismatch=0;if(N==realN){GGML_ASSERT(saved.size()==ref.size());for(size_t i=0;i<ref.size();++i)saved_mismatch+=std::memcmp(&saved[i],&ref[i],4)!=0;}
  GGML_ASSERT(saved_mismatch==0);
  std::printf("{\"kind\":\"reference\",\"N\":%d,\"real_capture_N\":%d,\"saved_bit_mismatch\":%zu}\n",N,realN,saved_mismatch);
  const int order[]={0,1,2,2,1,0};
  for(int arm=0;arm<6;++arm){int mode=order[arm];run(mode,false);
   for(int r=0;r<3;++r){double ms=run(mode,false);bool exact=std::memcmp(ref.data(),out.data()+64,ref.size()*4)==0;
    std::printf("{\"kind\":\"timing\",\"N\":%d,\"mode\":%d,\"arm\":%d,\"repeat\":%d,\"ms\":%.6f,\"bitwise_exact\":%s,\"guards\":true}\n",N,mode,arm,r,ms,exact?"true":"false");std::fflush(stdout);GGML_ASSERT(exact);
   }
  }
  GGML_ASSERT(hash(x)==before);
 }
 GGML_ASSERT(hash(w)==wh&&hash(original)==xh);
 std::printf("{\"kind\":\"final\",\"input_immutable\":true,\"all_exact\":true}\n");ggml_free(gc);return 0;
}
