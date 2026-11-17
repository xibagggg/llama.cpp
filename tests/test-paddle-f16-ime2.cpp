#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
    setenv("GGML_SPACEMIT_PADDLE_F16_IME2", "1", 1);
    ggml_backend_load_all();
    auto dev=ggml_backend_dev_by_name("SPACEMIT0");
    if (!dev) return 2;
    auto backend=ggml_backend_dev_init(dev,nullptr);
    if (!backend) return 3;
    std::mt19937 rng(176);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    bool ok=true;
    for (auto shape : {std::pair<int,int>{1152,1152},{1152,4304},{4304,1152}}) {
        for (int m : {1,7,8,9,17,216}) {
            const int k=shape.first,n=shape.second;
            ggml_init_params params{32*1024*1024,nullptr,true};
            auto ctx=ggml_init(params);
            auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,k,n);
            ggml_set_name(w,"v.blk.0.attn_q.weight");
            ggml_set_param(w);
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);
            ggml_set_input(x);
            auto y=ggml_mul_mat(ctx,w,x);
            ggml_set_output(y);
            auto graph=ggml_new_graph(ctx);
            ggml_build_forward_expand(graph,y);
            auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
            if (!buffer) return 4;
            if (!w->extra) { std::fprintf(stderr,"Paddle F16 IME2 trait is inactive\n"); return 6; }
            std::vector<ggml_fp16_t> weights(size_t(k)*n);
            std::vector<float> values(size_t(k)*m),output(size_t(n)*m);
            for (auto &a:weights) a=ggml_fp32_to_fp16(dist(rng)*0.1f);
            // Include subnormals, signs, zeros and tiny activations.
            for (size_t i=0;i<weights.size();i+=31) weights[i]=ggml_fp32_to_fp16(0x1p-23f);
            for (size_t i=0;i<values.size();++i) values[i]=i%11 ? dist(rng)*2.f : dist(rng)*1e-5f;
            if (m==17) for (auto &a:values) a*=1e-10f;
            if (m==9) std::fill_n(values.begin(),k,0.f);
            ggml_backend_tensor_set(w,weights.data(),0,weights.size()*2);
            ggml_backend_tensor_set(x,values.data(),0,values.size()*4);
            const auto status=ggml_backend_graph_compute(backend,graph);
            if (status!=GGML_STATUS_SUCCESS) return 5;
            ggml_backend_tensor_get(y,output.data(),0,output.size()*4);
            double sqerr=0,sqref=0,maxabs=0;
            for (float a:output) if (!std::isfinite(a)) ok=false;
            for (int row=0;row<m;++row) for(int j=0;j<66;++j) {
                const int col=j<64?(j*71)%n:n-(j==64?1:16);
                double ref=0;
                for(int z=0;z<k;++z) ref+=double(ggml_fp16_to_fp32(weights[size_t(col)*k+z]))*values[size_t(row)*k+z];
                const double err=output[size_t(row)*n+col]-ref;
                sqerr+=err*err;sqref+=ref*ref;maxabs=std::max(maxabs,std::abs(err));
            }
            const double relative=std::sqrt(sqerr/std::max(sqref,1e-30));
            const bool pass=relative<2e-5 && maxabs<0.002;
            ok=ok&&pass;
            std::printf("K=%d N=%d M=%d relative_l2=%.9g max_abs=%.9g %s\n",k,n,m,relative,maxabs,pass?"PASS":"FAIL");
            ggml_backend_buffer_free(buffer);ggml_free(ctx);
        }
    }
    ggml_backend_free(backend);
    return ok?0:1;
}
