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

static unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; return x * 0x846ca68b;
}
template<class T> static bool save(const std::string & path, const std::vector<T> & v) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(v.data()), v.size()*sizeof(T));
    return bool(out);
}
int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return 1;
    const std::string path=argv[1];
    const int n=std::atoi(argv[2]),d=72,h=16;
    if(n<64||n>2048)return 2;
    const float scale=1.f/std::sqrt(float(d));
    auto backend=ggml_backend_spacemit_init();
    if(!backend)return 3;
    ggml_init_params params{};params.mem_size=2*1024*1024;params.no_alloc=true;
    auto ctx=ggml_init(params);
    auto q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,d,n,h);
    auto k=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,d,n,h);
    auto v=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,d,n,h);
    for(auto t:{q,k,v}){ggml_set_input(t);ggml_set_output(t);}
    ggml_set_name(q,"fixture_q");ggml_set_name(k,"fixture_k");ggml_set_name(v,"fixture_v");
    auto y=ggml_flash_attn_ext(ctx,q,k,v,nullptr,scale,0.f,0.f);
    ggml_set_name(y,"fixture_attention");ggml_set_output(y);
    if(!ggml_prec_set_acc(y,GGML_PREC_F32))return 19;
    auto graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(graph,y);
    if(!ggml_backend_supports_op(backend,y))return 4;
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if(!ggml_gallocr_alloc_graph(alloc,graph))return 5;
    std::vector<float> qv(d*n*h),out(ggml_nelements(y)),first;
    std::vector<ggml_fp16_t> kv(qv.size()),vv(qv.size());
    for(size_t i=0;i<qv.size();++i) {
        qv[i]=(int(mix(i+31)%65536)-32768)*(4.f/32768.f);
        kv[i]=ggml_fp32_to_fp16((int(mix(i+71)%65536)-32768)*(2.f/32768.f));
        vv[i]=ggml_fp32_to_fp16((int(mix(i+113)%65536)-32768)*(1.f/32768.f));
        if(i%541==0)vv[i]=ggml_fp32_to_fp16(std::ldexp(1.f,-24));
    }
    // Exercise full-KV rejection after partial progress with finite inputs.
    // Lean preparation must reconstruct legacy K/V locally in these cases.
    if(argc==4){
        const std::string mode=argv[3];
        if(mode=="qbound")qv[d*n*5+17]=1024.f;
        else if(mode=="kbound")kv[d*n*5+17]=ggml_fp32_to_fp16(64.f);
        else if(mode=="vbound")vv[d*n*5+17]=ggml_fp32_to_fp16(64.f);
        else return 20;
    }
    ggml_backend_tensor_set(q,qv.data(),0,qv.size()*4);
    ggml_backend_tensor_set(k,kv.data(),0,kv.size()*2);
    ggml_backend_tensor_set(v,vv.data(),0,vv.size()*2);
    if(!save(path+"-q.f32",qv)||!save(path+"-k.f16",kv)||!save(path+"-v.f16",vv))return 6;
    for(int repeat=-1;repeat<3;++repeat) {
        const auto begin=std::chrono::steady_clock::now();
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 7;
        ggml_backend_synchronize(backend);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
        for(float x:out)if(!std::isfinite(x))return 8;
        if(repeat==-1)first=out;
        if(std::memcmp(first.data(),out.data(),out.size()*4))return 9;
        if(!save(path+"-y.f32.r"+std::to_string(repeat),out))return 10;
        std::printf("{\"N\":%d,\"D\":%d,\"H\":%d,\"scale\":%.9g,\"repeat\":%d,\"seconds\":%.9f}\n",n,d,h,scale,repeat,seconds);
        std::fflush(stdout);
    }
    auto changed=qv;
    for(size_t i=0;i<changed.size();++i)changed[i]+=(int(mix(i+191)%1023)-511)*.001f;
    ggml_backend_tensor_set(q,changed.data(),0,changed.size()*4);
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 11;
    ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
    if(std::memcmp(first.data(),out.data(),out.size()*4)==0)return 12;
    if(!save(path+"-changed-y.f32",out))return 13;
    ggml_backend_tensor_set(q,qv.data(),0,qv.size()*4);
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 14;
    ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
    if(std::memcmp(first.data(),out.data(),out.size()*4))return 15;
    std::vector<float> qcheck(qv.size());std::vector<ggml_fp16_t> check(kv.size());
    ggml_backend_tensor_get(q,qcheck.data(),0,qcheck.size()*4);
    if(std::memcmp(qcheck.data(),qv.data(),qv.size()*4))return 16;
    ggml_backend_tensor_get(k,check.data(),0,check.size()*2);
    if(std::memcmp(check.data(),kv.data(),kv.size()*2))return 17;
    ggml_backend_tensor_get(v,check.data(),0,check.size()*2);
    if(std::memcmp(check.data(),vv.data(),vv.size()*2))return 18;
    ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_free(backend);
    std::puts("PASS replay=A_B_A inputs=preserved finite=true");
}
