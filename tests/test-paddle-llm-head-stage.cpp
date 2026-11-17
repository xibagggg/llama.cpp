#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {
constexpr int K=1024,N=103424,SEEDS=8;
constexpr uint64_t MAGIC=UINT64_C(0x51464b5354414745);
// Fixed-width fields avoid padding bytes in this same-device evidence format.
struct header { uint64_t magic,version,k,n,seeds,bytes,weight_hash; };
static_assert(sizeof(header)==7*sizeof(uint64_t),"fixture header padding");
uint32_t next(uint32_t &v) { v^=v<<13;v^=v>>17;v^=v<<5;return v; }
uint64_t hash_bytes(const void *ptr,size_t bytes) {
    uint64_t h=UINT64_C(14695981039346656037);const auto*p=static_cast<const uint8_t*>(ptr);
    for(size_t i=0;i<bytes;++i)h=(h^p[i])*UINT64_C(1099511628211);
    return h;
}
bool io(FILE*f,bool write,const void*p,size_t bytes) {
    return write?std::fwrite(p,1,bytes,f)==bytes:std::fread(const_cast<void*>(p),1,bytes,f)==bytes;
}
bool compare_or_write(FILE*f,bool write,uint32_t seed,uint64_t input_hash,const std::vector<float>&out) {
    struct record {uint64_t magic,seed,input_hash,bytes,output_hash;};
    static_assert(sizeof(record)==5*sizeof(uint64_t),"fixture record padding");
    record rec{MAGIC,seed,input_hash,out.size()*sizeof(float),hash_bytes(out.data(),out.size()*sizeof(float))};
    if(write)return io(f,true,&rec,sizeof(rec))&&io(f,true,out.data(),rec.bytes);
    record expected{};if(!io(f,false,&expected,sizeof(expected))||std::memcmp(&expected,&rec,sizeof(rec)))return false;
    std::vector<float> ref(out.size());
    return io(f,false,ref.data(),rec.bytes)&&!std::memcmp(ref.data(),out.data(),rec.bytes);
}
}
int main(int argc,char**argv) {
    if(argc!=3||(std::strcmp(argv[1],"--write-reference")&&std::strcmp(argv[1],"--compare-reference")))return 2;
    const bool write=!std::strcmp(argv[1],"--write-reference");
    std::unique_ptr<FILE,decltype(&std::fclose)>file(std::fopen(argv[2],write?"wb":"rb"),&std::fclose);
    if(!file)return 2;
    const char *stage=std::getenv("GGML_SPACEMIT_PADDLE_LLM_HEAD_STAGE_N");
    if(write ? (!stage||std::strcmp(stage,"0")) :
               (!stage||(std::strcmp(stage,"64")&&std::strcmp(stage,"128")))) return 2;
    ggml_backend_load_all();
    auto dev=ggml_backend_dev_by_name("SPACEMIT0");if(!dev)return 3;
    auto backend=ggml_backend_dev_init(dev,nullptr);if(!backend)return 4;
    ggml_init_params params{64*1024*1024,nullptr,true};auto ctx=ggml_init(params);if(!ctx)return 5;
    auto*w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q6_K,K,N);
    auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,1);
    auto*y=ggml_mul_mat(ctx,w,x);
    ggml_set_name(w,"output.weight");ggml_set_name(x,"head_input");ggml_set_name(y,"head_output");ggml_set_input(x);ggml_set_output(y);
    if(!ggml_backend_supports_op(backend,y))return 6;
    auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,y);auto*buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)return 7;
    std::vector<float> weights(size_t(K)*N),features(K),out(N),before(K);
    std::vector<uint8_t> quant(ggml_nbytes(w));uint32_t state=0x51a6e7u;
    for(auto&v:weights)v=float(int(next(state)%65537)-32768)*0x1p-12f;
    if(ggml_quantize_chunk(GGML_TYPE_Q6_K,weights.data(),quant.data(),0,N,K,nullptr)!=quant.size())return 8;
    const uint64_t weight_hash=hash_bytes(quant.data(),quant.size());
    ggml_backend_tensor_set(w,quant.data(),0,quant.size());
    std::vector<float>().swap(weights);std::vector<uint8_t>().swap(quant);
    const std::array<const void*,3> pointers={x->data,y->data,w->data};
    const header expected_header{MAGIC,2,K,N,SEEDS,uint64_t(N)*sizeof(float),weight_hash};
    if(write) {if(!io(file.get(),true,&expected_header,sizeof(expected_header)))return 9;}
    else {header got{};if(!io(file.get(),false,&got,sizeof(got))||std::memcmp(&got,&expected_header,sizeof(got)))return 9;}
    bool ok=true;
    for(uint32_t seed=0;seed<SEEDS;++seed) {
        uint32_t r=0x1a2b3c4du^seed*0x9e3779b9u;
        for(size_t i=0;i<features.size();++i)features[i]=float(int(next(r)%131071)-65535)*0x1p-15f;
        if(seed==0)std::fill(features.begin(),features.end(),0.f);
        if(seed==1)for(size_t i=0;i<features.size();i+=31)features[i]=std::signbit(features[i])?0.f:-0.f;
        before=features;ggml_backend_tensor_set(x,features.data(),0,features.size()*sizeof(float));
        const uint64_t input_hash=hash_bytes(features.data(),features.size()*sizeof(float));
        std::fill(out.begin(),out.end(),std::nanf("stage"));
        ggml_backend_tensor_set(y,out.data(),0,out.size()*sizeof(float));
        if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 10;
        ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float));
        std::vector<float> after(K);ggml_backend_tensor_get(x,after.data(),0,after.size()*sizeof(float));
        const bool finite=std::all_of(out.begin(),out.end(),[](float v){return std::isfinite(v);});
        const bool unchanged=!std::memcmp(before.data(),after.data(),before.size()*sizeof(float));
        const std::array<const void*,3> now={x->data,y->data,w->data};
        const bool pass=finite&&unchanged&&now==pointers&&compare_or_write(file.get(),write,seed,input_hash,out);
        std::printf("stage=%s seed=%u weight_hash=%016llx input_hash=%016llx finite=%d unchanged=%d pointers=%d %s\n",stage,seed,
                    static_cast<unsigned long long>(weight_hash),static_cast<unsigned long long>(input_hash),
                    int(finite),int(unchanged),int(now==pointers),pass?"PASS":"FAIL");
        ok=ok&&pass;
    }
    uint64_t footer[2]={MAGIC,SEEDS};
    if(write)ok=ok&&io(file.get(),true,footer,sizeof(footer))&&!std::fflush(file.get());
    else {uint64_t got[2]{};ok=ok&&io(file.get(),false,got,sizeof(got))&&!std::memcmp(got,footer,sizeof(got))&&std::fgetc(file.get())==EOF;}
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return ok?0:1;
}
