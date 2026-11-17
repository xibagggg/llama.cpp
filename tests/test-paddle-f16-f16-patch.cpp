#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static bool selector(ggml_backend_t ai, bool enabled) {
    auto ctx=ggml_init({1024*1024,nullptr,true});
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,588,17);
    auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,588,1152);
    auto y=ggml_mul_mat(ctx,x,w);
    bool ok=ggml_backend_supports_op(ai,y)==enabled;
    if(enabled) {
        int rejected=0;
        const auto reject=[&]{const bool pass=!ggml_backend_supports_op(ai,y);ok=ok&&pass;rejected+=pass;};
        w->ne[1]=1151;reject();w->ne[1]=1152;
        x->type=GGML_TYPE_F32;reject();x->type=GGML_TYPE_F16;
        y->ne[0]=16;reject();y->ne[0]=17;
        y->ne[1]=1151;reject();y->ne[1]=1152;
        y->nb[1]+=4;reject();y->nb[1]-=4;
        ok=ok&&ggml_backend_supports_op(ai,y);
        std::printf("selector rejected=%d/5 %s\n",rejected,ok?"PASS":"FAIL");
    } else {
        std::printf("disabled selector %s\n",ok?"PASS":"FAIL");
    }
    ggml_free(ctx);return ok;
}

static bool compute(ggml_backend_t backend, int m,
                    const std::vector<ggml_fp16_t> & xval,
                    const std::vector<ggml_fp16_t> & wval,
                    std::vector<float> & output) {
    constexpr int k=588, n=1152;
    ggml_init_params p{64*1024*1024,nullptr,true};
    auto ctx=ggml_init(p);
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,k,m);
    auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,k,n);
    auto y=ggml_mul_mat(ctx,x,w); // output ne=[M,N], storage is N rows of M values
    ggml_set_input(x); ggml_set_input(w); ggml_set_output(y);
    auto graph=ggml_new_graph(ctx); ggml_build_forward_expand(graph,y);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer){ggml_free(ctx);return false;}
    auto values=xval;
    ggml_backend_tensor_set(x,values.data(),0,values.size()*sizeof(values[0]));
    ggml_backend_tensor_set(w,wval.data(),0,wval.size()*sizeof(wval[0]));
    bool ok=true;
    output.resize(size_t(m)*n*2);
    for(int pass=0;pass<2&&ok;++pass) {
        if(pass) {
            for(size_t i=0;i<values.size();++i)
                values[i]=ggml_fp32_to_fp16(-0.5f*ggml_fp16_to_fp32(values[i])+float(int(i%23)-11)/64.f);
            ggml_backend_tensor_set(x,values.data(),0,values.size()*sizeof(values[0]));
        }
        ok=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
        if(ok)ggml_backend_tensor_get(y,output.data()+size_t(pass)*m*n,0,size_t(m)*n*sizeof(float));
    }
    ggml_backend_buffer_free(buffer); ggml_free(ctx); return ok;
}

int main() {
    const char * option=std::getenv("GGML_SPACEMIT_PADDLE_F16_F16_PATCH");
    const bool enabled=option&&!std::strcmp(option,"1");
    ggml_backend_load_all();
    const auto ad=ggml_backend_dev_by_name("SPACEMIT0");
    const auto cd=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if(!ad||!cd)return 2;
    auto ai=ggml_backend_dev_init(ad,nullptr),cpu=ggml_backend_dev_init(cd,nullptr);
    if(!ai||!cpu)return 3;
    constexpr int k=588,n=1152;
    std::mt19937 rng(588); std::uniform_real_distribution<float> d(-1.f,1.f);
    bool all=selector(ai,enabled);
    if(!enabled) {
        ggml_backend_free(ai);ggml_backend_free(cpu);return all?0:1;
    }
    for(int m:{1,7,8,9,17,848}) {
        std::vector<ggml_fp16_t> xv(size_t(k)*m),wv(size_t(k)*n);
        for(auto &v:xv)v=ggml_fp32_to_fp16(d(rng));
        for(auto &v:wv)v=ggml_fp32_to_fp16(0.1f*d(rng));
        for(size_t i=0;i<xv.size();i+=97)xv[i]=ggml_fp32_to_fp16(i%194?-0x1p-23f:0x1p-23f);
        for(size_t i=0;i<wv.size();i+=101)wv[i]=ggml_fp32_to_fp16(i%202?-0x1p-23f:0x1p-23f);
        if(m==8)std::fill(xv.begin(),xv.end(),ggml_fp32_to_fp16(0));
        if(m==9)std::fill_n(xv.begin(),k,ggml_fp32_to_fp16(0));
        std::vector<float> got,ref;
        bool ok=compute(ai,m,xv,wv,got)&&compute(cpu,m,xv,wv,ref)&&got.size()==ref.size();
        double ss=0,sr=0,maxabs=0;
        for(size_t i=0;i<got.size()&&ok;++i) {
            const double e=double(got[i])-ref[i]; ss+=e*e; sr+=double(ref[i])*ref[i];
            maxabs=std::max(maxabs,std::abs(e)); ok=ok&&std::isfinite(got[i]);
        }
        const double rel=std::sqrt(ss/std::max(sr,1e-30));
        ok=ok&&rel<2e-5&&maxabs<0.002; all=all&&ok;
        std::printf("M=%d passes=2 relative_l2=%.9g max_abs=%.9g %s\n",m,rel,maxabs,ok?"PASS":"FAIL");
    }
    ggml_backend_free(ai); ggml_backend_free(cpu); return all?0:1;
}
