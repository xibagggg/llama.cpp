#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Compare the actual CPU and SpacemiT implementations over every output.
// A view has padding in all row strides and a nonzero buffer offset; inplace
// cases catch writes that accidentally restore the second half of the head.
struct config { ggml_type type; int d, n; bool view, inplace, yarn, factors; bool four_axes=false; };

static bool run(ggml_backend_t backend, const config & c, std::vector<float> & out) {
    constexpr int H=16;
    const int B=c.view?2:1;
    const auto ctx=ggml_init({4*1024*1024,nullptr,true});
    auto base=ggml_new_tensor_4d(ctx,c.type,c.view?c.d*2:c.d,c.view?H+2:H,c.view?c.n+1:c.n,B);
    auto x=c.view?ggml_view_4d(ctx,base,c.d,H,c.n,B,base->nb[1],base->nb[2],base->nb[3],c.d*ggml_element_size(base)):base;
    auto pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,c.n*4);
    auto ff=c.factors?ggml_new_tensor_1d(ctx,GGML_TYPE_F32,c.d/2):nullptr;
    const int q=c.four_axes?c.d/8:c.d/4;
    int sections[4]={q,q,q,q};
    auto y=c.inplace?ggml_rope_multi_inplace(ctx,x,pos,ff,c.d/2,sections,GGML_ROPE_TYPE_VISION,32768,10000.0f,c.yarn?0.7f:1.0f,c.yarn?0.5f:0.0f,c.yarn?1.2f:1.0f,32.0f,1.0f)
                    :ggml_rope_multi(ctx,x,pos,ff,c.d/2,sections,GGML_ROPE_TYPE_VISION,32768,10000.0f,c.yarn?0.7f:1.0f,c.yarn?0.5f:0.0f,c.yarn?1.2f:1.0f,32.0f,1.0f);
    ggml_set_input(base);ggml_set_input(pos);if(ff)ggml_set_input(ff);ggml_set_output(y);
    if (!ggml_backend_supports_op(backend,y)) {
        std::fprintf(stderr,"backend=%s does not support VISION RoPE\n",ggml_backend_name(backend));
        ggml_free(ctx);return false;
    }
    const auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    const auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer){ggml_free(ctx);return false;}
    const size_t nel=ggml_nelements(base),esize=ggml_element_size(base);
    std::vector<float> f32(nel);
    std::vector<ggml_fp16_t> f16(nel);
    for(size_t i=0;i<nel;++i) {
        f32[i]=float(int((i*176+31)%2003)-1001)/251.f;
        f16[i]=ggml_fp32_to_fp16(f32[i]);
    }
    ggml_backend_tensor_set(base,c.type==GGML_TYPE_F32?static_cast<const void*>(f32.data()):static_cast<const void*>(f16.data()),0,nel*esize);
    std::vector<int32_t> ids(c.n*4);
    for(int axis=0;axis<4;++axis)for(int i=0;i<c.n;++i)ids[axis*c.n+i]=(i*17+axis*419)%32768;
    ggml_backend_tensor_set(pos,ids.data(),0,ids.size()*sizeof(int32_t));
    if(ff) {
        std::vector<float> factors(c.d/2);
        for(int i=0;i<c.d/2;++i)factors[i]=0.9f+0.01f*(i%21);
        ggml_backend_tensor_set(ff,factors.data(),0,factors.size()*sizeof(float));
    }
    bool ok=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
    std::vector<unsigned char> raw(ggml_nbytes(y));
    if(ok)ggml_backend_tensor_get(y,raw.data(),0,raw.size());
    out.clear();out.reserve(size_t(c.d)*H*c.n*B);
    for(int b=0;b<B;++b)for(int n=0;n<c.n;++n)for(int h=0;h<H;++h)for(int d=0;d<c.d;++d) {
        const auto offset=b*y->nb[3]+n*y->nb[2]+h*y->nb[1]+d*esize;
        float value;
        if(c.type==GGML_TYPE_F32)std::memcpy(&value,raw.data()+offset,4);
        else{ggml_fp16_t v;std::memcpy(&v,raw.data()+offset,2);value=ggml_fp16_to_fp32(v);}
        out.push_back(value);ok=ok&&std::isfinite(value);
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return ok;
}

static bool selector_regression(ggml_backend_t ai) {
    const auto ctx=ggml_init({1024*1024,nullptr,true});
    auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,72,16,17);
    auto pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,68);
    auto ff=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,36);
    int sections[4]={18,18,18,18};
    auto y=ggml_rope_multi(ctx,x,pos,ff,36,sections,GGML_ROPE_TYPE_VISION,32768,10000,1,0,1,32,1);
    bool ok=ggml_backend_supports_op(ai,y);
    int rejected=0;
    const auto reject=[&]() { const bool pass=!ggml_backend_supports_op(ai,y); ok=ok&&pass; rejected+=pass; };
    y->op_params[1]=32;reject();y->op_params[1]=36;
    y->op_params[15]=1;reject();y->op_params[15]=0;
    pos->ne[0]=67;reject();pos->ne[0]=68;
    pos->nb[0]=8;reject();pos->nb[0]=4;
    ff->ne[0]=35;reject();ff->ne[0]=36;
    ff->nb[0]=8;reject();ff->nb[0]=4;
    y->op_params[11]=-1;reject();y->op_params[11]=18;
    y->op_params[14]=19;reject();y->op_params[14]=18;
    ok=ok&&ggml_backend_supports_op(ai,y);
    std::printf("VISION selector rejected=%d/8 %s\n",rejected,ok?"PASS":"FAIL");
    ggml_free(ctx);return ok;
}

int main() {
#ifdef _WIN32
    _putenv_s("GGML_SPACEMIT_VISION_ROPE","1");
#else
    setenv("GGML_SPACEMIT_VISION_ROPE","1",1);
#endif
    ggml_backend_load_all();
    const auto ai_dev=ggml_backend_dev_by_name("SPACEMIT0");
    const auto cpu_dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if(!ai_dev||!cpu_dev||ai_dev==cpu_dev||std::strcmp(ggml_backend_dev_name(cpu_dev),"CPU"))return 2;
    const auto ai=ggml_backend_dev_init(ai_dev,nullptr),cpu=ggml_backend_dev_init(cpu_dev,nullptr);
    if(!ai||!cpu)return 3;
    std::printf("Reference backend=%s; tested backend=%s\n",ggml_backend_name(cpu),ggml_backend_name(ai));
    std::vector<config> cases;
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16}) {
        for(bool view:{false,true})for(bool inplace:{false,true})
            cases.push_back({type,72,17,view,inplace,false,false});
        cases.push_back({type,72,848,false,false,false,false});
        cases.push_back({type,72,1,true,true,true,true});
        cases.push_back({type,80,17,false,false,true,true});
        cases.push_back({type,128,17,true,false,false,true});
        cases.push_back({type,72,17,true,true,false,true,true});
    }
    bool ok=selector_regression(ai);
    for(const auto & c:cases) {
        std::vector<float> ref,out;
        bool pass=run(cpu,c,ref)&&run(ai,c,out)&&ref.size()==out.size();
        double sqerr=0,sqref=0,maxabs=0;size_t identical=0;
        if(pass)for(size_t i=0;i<out.size();++i) {
            const double err=double(out[i])-ref[i];sqerr+=err*err;sqref+=double(ref[i])*ref[i];maxabs=std::max(maxabs,std::abs(err));
            identical+=!std::memcmp(&out[i],&ref[i],sizeof(float));
        }
        const double relative=std::sqrt(sqerr/std::max(sqref,1e-30));
        pass=pass&&relative<(c.type==GGML_TYPE_F32?2e-6:8e-4)&&maxabs<(c.type==GGML_TYPE_F32?2e-5:0.008);
        ok=ok&&pass;
        std::printf("type=%s D=%d N=%d view=%d inplace=%d yarn=%d ff=%d four_axes=%d relative_l2=%.9g maxabs=%.9g identical=%zu/%zu %s\n",ggml_type_name(c.type),c.d,c.n,c.view,c.inplace,c.yarn,c.factors,c.four_axes,relative,maxabs,identical,out.size(),pass?"PASS":"FAIL");
    }
    ggml_backend_free(ai);ggml_backend_free(cpu);return ok?0:1;
}
