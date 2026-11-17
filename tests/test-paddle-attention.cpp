#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Check Paddle vision FA against sampled FP64 attention rows, including small
// crops below the old 988-patch cutoff, partial Q/KV tiles and large images.
// The sub-tile case also checks that the original RVV fallback remains valid.
int main() {
    setenv("GGML_SPACEMIT_OCR_FA_PACK","1",1);
    setenv("GGML_SPACEMIT_OCR_FA_COMPUTE","3",1);
    setenv("GGML_SPACEMIT_OCR_PV_IME2","2",1);
    setenv("GGML_SPACEMIT_OCR_FA_DYNAMIC_ROWS","512",1);
    ggml_backend_load_all();
    const auto dev=ggml_backend_dev_by_name("SPACEMIT0");
    if(!dev)return 2;
    const auto backend=ggml_backend_dev_init(dev,nullptr);
    if(!backend)return 3;
    std::mt19937 rng(176);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    constexpr int D=72,H=16;
    bool ok=true;
    for(int n:{63,64,65,752,804,848,896,987,988,1001,4097,10320}) {
        const auto ctx=ggml_init({32*1024*1024,nullptr,true});
        auto q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,n,H);
        auto k=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,n,H);
        auto v=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,n,H);
        const float scale=1.f/std::sqrt(float(D));
        auto y=ggml_flash_attn_ext(ctx,q,k,v,nullptr,scale,0,0);
        ggml_set_input(q);ggml_set_input(k);ggml_set_input(v);ggml_set_output(y);
        const auto graph=ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,y);
        const auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        if(!buffer)return 4;
        std::vector<float> qs(size_t(D)*n*H),out(qs.size());
        std::vector<ggml_fp16_t> ks(qs.size()),vs(qs.size());
        for(size_t i=0;i<qs.size();++i) {
            qs[i]=dist(rng)*2;
            ks[i]=ggml_fp32_to_fp16(dist(rng)*2);
            vs[i]=ggml_fp32_to_fp16(i%13?dist(rng):0x1p-23f);
        }
        ggml_backend_tensor_set(q,qs.data(),0,qs.size()*4);
        ggml_backend_tensor_set(k,ks.data(),0,ks.size()*2);
        ggml_backend_tensor_set(v,vs.data(),0,vs.size()*2);
        auto start=std::chrono::steady_clock::now();
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 5;
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
        for(auto a:out)if(!std::isfinite(a))ok=false;
        double sqerr=0,sqref=0,maxabs=0;
        for(int h=0;h<H;++h)for(int row:{0,n/2,n-1}) {
            std::vector<double> scores(n),ref(D);
            for(int j=0;j<n;++j) {
                double dot=0;
                for(int d=0;d<D;++d)dot+=double(qs[(size_t(h)*n+row)*D+d])*ggml_fp16_to_fp32(ks[(size_t(h)*n+j)*D+d]);
                scores[j]=dot*scale;
            }
            const auto maximum=*std::max_element(scores.begin(),scores.end());
            double sum=0;
            for(auto &a:scores){a=std::exp(a-maximum);sum+=a;}
            for(int j=0;j<n;++j)for(int d=0;d<D;++d)
                ref[d]+=scores[j]*ggml_fp16_to_fp32(vs[(size_t(h)*n+j)*D+d])/sum;
            // FA output is [D, heads, query_tokens].
            for(int d=0;d<D;++d) {
                const double err=out[(size_t(row)*H+h)*D+d]-ref[d];
                sqerr+=err*err;sqref+=ref[d]*ref[d];maxabs=std::max(maxabs,std::abs(err));
            }
        }
        const double relative=std::sqrt(sqerr/std::max(sqref,1e-30));
        const bool pass=relative<0.002 && maxabs<0.002;
        ok=ok&&pass;
        std::printf("N=%d H=%d D=%d relative_l2=%.9g max_abs=%.9g seconds=%.6f %s\n",n,H,D,relative,maxabs,seconds,pass?"PASS":"FAIL");
        std::fflush(stdout);
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return ok?0:1;
}
