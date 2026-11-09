#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-spacemit.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static unsigned mix(unsigned x) { x^=x>>16; x*=0x7feb352d; x^=x>>15; return x*0x846ca68b; }
template<class T> static bool save(const std::string &name,const std::vector<T>&v) {
    std::ofstream f(name,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));return bool(f);
}
int main(int argc,char **argv) {
    if(argc!=3&&argc!=4)return 1;
    const std::string path=argv[1],mode=argc==4?argv[3]:"normal";
    const int n=std::atoi(argv[2]),d=72,h=16;
    if(n<64||n>2048)return 2;
    auto backend=ggml_backend_spacemit_init();if(!backend)return 3;
    ggml_init_params p{};p.no_alloc=true;p.mem_size=2*1024*1024;auto ctx=ggml_init(p);
    auto q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,n,h);
    auto ki=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,h,n);
    auto vi=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,h,n);
    for(auto *t:{q,ki,vi}){ggml_set_input(t);ggml_set_output(t);}
    auto k=ggml_cast(ctx,ggml_permute(ctx,ki,0,2,1,3),GGML_TYPE_F16);
    auto v=ggml_cast(ctx,ggml_permute(ctx,vi,0,2,1,3),GGML_TYPE_F16);
    ggml_set_name(k,"fixture_k_cast");ggml_set_name(v,"fixture_v_cast");
    ggml_set_output(k);ggml_set_output(v);
    auto y=ggml_flash_attn_ext(ctx,q,k,v,nullptr,1.f/std::sqrt(float(d)),0.f,0.f);
    ggml_set_output(y);if(!ggml_prec_set_acc(y,GGML_PREC_F32))return 4;
    auto g=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(g,y);
    for(int i=0;i<ggml_graph_n_nodes(g);++i){
        auto *t=ggml_graph_node(g,i);if(!ggml_backend_supports_op(backend,t))return 5;
    }
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if(!ggml_gallocr_alloc_graph(alloc,g))return 6;
    std::vector<float> qv(d*n*h),kv(qv.size()),vv(qv.size()),out(ggml_nelements(y)),first;
    for(size_t i=0;i<qv.size();++i){
        qv[i]=(int(mix(i+31)%65536)-32768)*(4.f/32768.f);
        kv[i]=(int(mix(i+71)%65536)-32768)*(2.f/32768.f);
        vv[i]=(int(mix(i+113)%65536)-32768)*(1.f/32768.f);
        if(i%541==0)vv[i]=std::ldexp(1.f,-24);
        if(i%127==0)kv[i]=1.f+std::ldexp(1.f,-11);
        if(i%251==0)vv[i]=std::copysign(0.f,i%2?-1.f:1.f);
    }
    if(mode=="kbound")kv[d*h*70+5*d+17]=64.f;
    else if(mode=="vbound")vv[d*h*70+5*d+17]=64.f;
    else if(mode!="normal")return 7;
    ggml_backend_tensor_set(q,qv.data(),0,qv.size()*4);
    std::vector<ggml_fp16_t> kh(kv.size()),vh(vv.size()),checkh(kv.size());
    std::vector<float> check(qv.size());
    for(int rep=-1;rep<5;++rep){
        auto kb=kv,vb=vv;
        if(rep==3)for(size_t i=0;i<kb.size();++i){kb[i]=kb[i]*.5f+.125f;vb[i]=vb[i]*.5f-.25f;}
        ggml_backend_tensor_set(ki,kb.data(),0,kb.size()*4);
        ggml_backend_tensor_set(vi,vb.data(),0,vb.size()*4);
        const auto start=std::chrono::steady_clock::now();
        if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 8;
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
        for(float f:out)if(!std::isfinite(f))return 9;
        if(rep==-1)first=out;
        if((rep==3)==(std::memcmp(first.data(),out.data(),out.size()*4)==0))return 10;
        for(int head=0;head<h;++head)for(int row=0;row<n;++row)for(int c=0;c<d;++c){
            const size_t src=(size_t(row)*h+head)*d+c,dst=(size_t(head)*n+row)*d+c;
            kh[dst]=ggml_fp32_to_fp16(kb[src]);vh[dst]=ggml_fp32_to_fp16(vb[src]);
        }
        ggml_backend_tensor_get(k,checkh.data(),0,checkh.size()*2);
        if(std::memcmp(checkh.data(),kh.data(),kh.size()*2))return 11;
        ggml_backend_tensor_get(v,checkh.data(),0,checkh.size()*2);
        if(std::memcmp(checkh.data(),vh.data(),vh.size()*2))return 12;
        for(const auto &t:std::vector<std::pair<ggml_tensor*,const std::vector<float>*>>{{q,&qv},{ki,&kb},{vi,&vb}}){
            ggml_backend_tensor_get(t.first,check.data(),0,check.size()*4);
            if(std::memcmp(check.data(),t.second->data(),check.size()*4))return 13;
        }
        if(!save(path+"-y.f32.r"+std::to_string(rep),out))return 14;
        std::printf("{\"N\":%d,\"mode\":\"%s\",\"repeat\":%d,\"seconds\":%.9f}\n",n,mode.c_str(),rep,seconds);
        std::fflush(stdout);
    }
    ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_free(backend);
    std::puts("PASS casts=exact replay=A_B_A inputs=preserved finite=true");
}
