// Exercise the actual candidate forward(), including N-tail writes and scratch.
#include "ggml.h"
#include "ggml-impl.h"
#include "spacemit-context.h"
#include "ocr-ime2-fp16-v35.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace baseline {
#include "candidate_function.inc"
}
constexpr int K=4304,M=1152,T=7;
static uint32_t mix(uint32_t x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);}
int main(int argc,char **argv){
    const int N=argc>1?std::atoi(argv[1]):1020;if(N<1)return 2;
    setenv("GGML_SPACEMIT_IME2_FP16_DOWN","1",1);
    setenv("GGML_SPACEMIT_IME2_FP16_COMPENSATE","1",1);
    const auto info=spert::backend_info();if(info.core_arch_id!=0xa064||info.shared_mem_size<346624)return 3;
    spert::StreamConfig cfg;cfg.n_cores=T;cfg.core_ids={8,9,10,11,13,14,15};spert::Stream stream(cfg);
    if(!stream.valid())return 4;
    std::vector<_Float16>w(size_t(M)*K);std::vector<float>x(size_t(N)*K),ref(size_t(N)*M),out(size_t(N)*M+64,-987654.f);
    for(size_t i=0;i<w.size();++i)w[i]=(_Float16)((int(mix(i+5)%20001)-10000)*0.00005f);
    for(size_t i=0;i<x.size();++i)x[i]=(int(mix(i+77)%20001)-10000)*0.0003f;
    ggml_init_params gp{};gp.mem_size=1024*1024;gp.no_alloc=true;auto *gc=ggml_init(gp);
    auto *wt=ggml_new_tensor_2d(gc,GGML_TYPE_F16,K,M),*xt=ggml_new_tensor_2d(gc,GGML_TYPE_F32,K,N);
    wt->data=w.data();xt->data=x.data();ggml_set_name(wt,"v.blk.0.ffn_down.weight");auto *yt=ggml_mul_mat(gc,wt,xt);
    if(!ocr_ime2_fp16_v35::eligible(yt))return 5;
    const size_t bytes=ocr_ime2_fp16_v35::work_bytes(yt);std::vector<_Float16>workspace(bytes/2);
    auto run=[&](bool ime){
        yt->data=ime?out.data():ref.data();auto start=std::chrono::steady_clock::now();
        auto f=stream.launch(spert::Grid{T},[&](spert::Context *rt){
            int rank=rt->program_id(0);auto sh=rt->shared_buffer();
            ggml::spacemit::context ctx;ctx.reset(*rt,rank,T,workspace.data(),bytes,sh);
            if(ime){GGML_ASSERT(ocr_ime2_fp16_v35::forward(ctx,yt));}
            else baseline::forward_mul_mat_f16_f32_m4_cache(ctx,yt,(float*)sh.data);
        });
        if(f.sync()!=spert::Status::Ok)std::exit(6);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    };
    run(false);run(true);
    for(int cycle=0;cycle<3;++cycle)for(int arm=0;arm<4;++arm){
        bool ime=arm==1||arm==2;double elapsed=run(ime);
        std::printf("{\"N\":%d,\"cycle\":%d,\"arm\":%d,\"mode\":\"%s\",\"all_ms\":%.6f}\n",N,cycle,arm,ime?"ime_compensated":"rvv_m4",elapsed);
    }
    double e2=0,r2=0,maxerr=0,fp64max=0;int bad=0;bool guard=true;
    for(size_t i=0;i<ref.size();++i){double e=double(out[i])-ref[i];e2+=e*e;r2+=double(ref[i])*ref[i];maxerr=std::max(maxerr,std::abs(e));if(!std::isfinite(out[i]))++bad;}
    for(size_t i=ref.size();i<out.size();++i)guard=guard&&out[i]==-987654.f;
    for(int j=0;j<256;++j){int n=mix(j+21)%N,m=mix(j+409)%M;double r=0;
        for(int k=0;k<K;++k)r+=double(w[m*K+k])*x[n*K+k];
        fp64max=std::max(fp64max,std::abs(double(out[n*M+m])-r));
    }
    double relative=std::sqrt(e2/r2);
    std::printf("{\"quality\":true,\"N\":%d,\"nonfinite\":%d,\"output_tail_guard\":%s,\"scratch_bytes\":346624,\"relative_l2_vs_m4\":%.12g,\"max_abs_vs_m4\":%.12g,\"sample_max_vs_fp64_f32_input\":%.12g}\n",N,bad,guard?"true":"false",relative,maxerr,fp64max);
    ggml_free(gc);return bad||!guard||relative>0.000002||fp64max>0.001?7:0;
}
