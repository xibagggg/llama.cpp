// Real OCR inputs and independent FP64 samples, plus exact tiny-value fixtures.
#include "ggml.h"
#include "ggml-impl.h"
#include "spacemit-context.h"
#include "ocr-ime2-fp16-v35.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
namespace baseline {
#include "candidate_function.inc"
}
constexpr int K=4304,M=1152,T=7;
template<class V>static V read(const std::string &path){
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)std::exit(2);
    const size_t bytes=size_t(f.tellg());if(bytes%sizeof(typename V::value_type))std::exit(3);
    V v(bytes/sizeof(typename V::value_type));f.seekg(0);f.read((char*)v.data(),bytes);if(!f)std::exit(4);return v;
}
int main(int argc,char **argv){
    if(argc<3)return 1;
    setenv("GGML_SPACEMIT_IME2_FP16_DOWN","1",1);setenv("GGML_SPACEMIT_IME2_FP16_COMPENSATE","1",1);
    const int layer=std::atoi(argv[2]);const std::string dir=argv[1];
    std::vector<_Float16>w;std::vector<float>x,saved;int N=8;
    const bool fixture=dir=="weight-subnormal"||dir=="activation-subnormal";
    if(fixture){
        w.assign(size_t(M)*K,0);x.assign(size_t(N)*K,0);
        for(int m=0;m<M;++m)w[size_t(m)*K]=(_Float16)(dir=="weight-subnormal"?0x1p-24f:1.f);
        for(int n=0;n<N;++n)x[size_t(n)*K]=dir=="weight-subnormal"?1.f:0x1p-20f;
    }else{
        char stem[96];std::snprintf(stem,sizeof(stem),"/%04d-layer-%02d",layer,layer);
        w=read<std::vector<_Float16>>(dir+"/weight-"+std::to_string(layer)+".f16");
        x=read<std::vector<float>>(dir+stem+"-x.f32");saved=read<std::vector<float>>(dir+stem+"-y.f32");
        N=int(x.size()/K);if(w.size()!=size_t(K)*M||x.size()!=size_t(K)*N||saved.size()!=size_t(M)*N)return 5;
    }
    spert::StreamConfig cfg;cfg.n_cores=T;cfg.core_ids={8,9,10,11,13,14,15};spert::Stream stream(cfg);if(!stream.valid())return 6;
    ggml_init_params gp{};gp.mem_size=1024*1024;gp.no_alloc=true;auto *gc=ggml_init(gp);
    auto *wt=ggml_new_tensor_2d(gc,GGML_TYPE_F16,K,M),*xt=ggml_new_tensor_2d(gc,GGML_TYPE_F32,K,N);
    wt->data=w.data();xt->data=x.data();ggml_set_name(wt,(("v.blk."+std::to_string(layer))+".ffn_down.weight").c_str());
    auto *yt=ggml_mul_mat(gc,wt,xt);const size_t bytes=ocr_ime2_fp16_v35::work_bytes(yt);
    std::vector<_Float16>workspace(bytes/2);std::vector<float>ref(size_t(N)*M),out(size_t(N)*M+64,-987654.f);
    auto run=[&](bool ime){
        yt->data=ime?out.data():ref.data();auto start=std::chrono::steady_clock::now();
        auto f=stream.launch(spert::Grid{T},[&](spert::Context *rt){
            auto sh=rt->shared_buffer();ggml::spacemit::context ctx;ctx.reset(*rt,rt->program_id(0),T,workspace.data(),bytes,sh);
            if(ime){GGML_ASSERT(ocr_ime2_fp16_v35::forward(ctx,yt));}
            else baseline::forward_mul_mat_f16_f32_m4_cache(ctx,yt,(float*)sh.data);
        });
        if(f.sync()!=spert::Status::Ok)std::exit(7);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    };
    run(false);run(true);double offms=run(false),onms=run(true);
    double e2=0,r2=0,maxerr=0,savedmax=0,fp64e2=0,fp64r2=0,ref64e2=0;int bad=0;bool guard=true;
    for(size_t i=0;i<ref.size();++i){double e=double(out[i])-ref[i];e2+=e*e;r2+=double(ref[i])*ref[i];maxerr=std::max(maxerr,std::abs(e));if(!std::isfinite(out[i]))++bad;
        if(!saved.empty())savedmax=std::max(savedmax,std::abs(double(ref[i])-saved[i]));}
    for(size_t i=ref.size();i<out.size();++i)guard=guard&&out[i]==-987654.f;
    for(int j=0;j<512;++j){int n=(j*661+21)%N,m=(j*313+17)%M;double r=0;
        for(int k=0;k<K;++k)r+=double(w[size_t(m)*K+k])*x[size_t(n)*K+k];
        double e=double(out[size_t(n)*M+m])-r;e2+=0;fp64e2+=e*e;fp64r2+=r*r;
        e=double(ref[size_t(n)*M+m])-r;ref64e2+=e*e;
    }
    double relative=std::sqrt(e2/r2),fp64rel=std::sqrt(fp64e2/fp64r2);
    std::printf("{\"layer\":%d,\"N\":%d,\"fixture\":\"%s\",\"normalized\":%s,\"off_ms\":%.6f,\"on_ms\":%.6f,\"relative_l2_vs_m4\":%.12g,\"max_abs_vs_m4\":%.12g,\"m4_max_abs_vs_saved\":%.12g,\"fp64_sample_relative_l2\":%.12g,\"m4_fp64_sample_relative_l2\":%.12g,\"nonfinite\":%d,\"output_guard\":%s,\"first_output\":%.12g}\n",layer,N,fixture?dir.c_str():"real_ocr",ocr_ime2_fp16_v35::normalized()?"true":"false",offms,onms,relative,maxerr,savedmax,fp64rel,std::sqrt(ref64e2/fp64r2),bad,guard?"true":"false",out[0]);
    ggml_free(gc);return bad||!guard||(ocr_ime2_fp16_v35::normalized()&&(relative>2e-6||fp64rel>2e-6))?8:0;
}
