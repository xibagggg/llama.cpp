// Full real-Q6 graph integration. Separate processes force immutable dispatch
// flags; reference=0 and candidate=1 compare all 103424 logits, never argmax only.
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
constexpr int K=1024,N=103424,CASES=16;
constexpr uint64_t MAGIC=UINT64_C(0x5136484541443634);
struct header {uint64_t magic,version,k,n,cases,bytes,weight_hash;};
struct record {uint64_t magic,ordinal,variant,input_hash,bytes,output_hash;};
static_assert(sizeof(header)==7*sizeof(uint64_t)&&sizeof(record)==6*sizeof(uint64_t),"padding");
uint32_t next(uint32_t&v){v^=v<<13;v^=v>>17;v^=v<<5;return v;}
uint64_t hash(const void*p,size_t bytes) {
    const auto*q=static_cast<const uint8_t*>(p);uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<bytes;++i)h=(h^q[i])*UINT64_C(1099511628211);return h;
}
bool io(FILE*f,bool write,const void*p,size_t bytes) {
    return write?std::fwrite(p,1,bytes,f)==bytes:std::fread(const_cast<void*>(p),1,bytes,f)==bytes;
}
bool match(FILE*f,bool write,uint32_t ordinal,uint32_t variant,const std::vector<float>&input,const std::vector<float>&out) {
    record actual{MAGIC,ordinal,variant,hash(input.data(),input.size()*4),out.size()*4,hash(out.data(),out.size()*4)};
    if(write)return io(f,true,&actual,sizeof(actual))&&io(f,true,out.data(),actual.bytes);
    record got{};if(!io(f,false,&got,sizeof(got))||std::memcmp(&got,&actual,sizeof(got)))return false;
    std::vector<float>reference(out.size());return io(f,false,reference.data(),actual.bytes)&&!std::memcmp(reference.data(),out.data(),actual.bytes);
}
}
int main(int argc,char**argv) {
    if(argc!=3||(std::strcmp(argv[1],"--write-reference")&&std::strcmp(argv[1],"--compare-reference")))return 2;
    bool write=!std::strcmp(argv[1],"--write-reference");
    const char*option=std::getenv("GGML_SPACEMIT_PADDLE_Q6_HEAD_N64");
    const char*workers=std::getenv("GGML_SPACEMIT_WORKERS");
    const char*trace=std::getenv("GGML_SPACEMIT_PADDLE_Q6_HEAD_N64_TRACE");
    const char*stage=std::getenv("GGML_SPACEMIT_PADDLE_LLM_HEAD_STAGE_N");
    if(!option||std::strcmp(option,write?"0":"1")||!workers||std::strcmp(workers,"6")||
       !trace||!*trace||!stage||std::strcmp(stage,"0"))return 2;
    std::unique_ptr<FILE,decltype(&std::fclose)>file(std::fopen(argv[2],write?"wb":"rb"),&std::fclose);if(!file)return 2;
    ggml_backend_load_all();auto dev=ggml_backend_dev_by_name("SPACEMIT0");if(!dev)return 3;
    auto backend=ggml_backend_dev_init(dev,nullptr);if(!backend)return 4;
    ggml_init_params params{64*1024*1024,nullptr,true};auto ctx=ggml_init(params);if(!ctx)return 5;
    auto*w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q6_K,K,N);
    auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,1);
    auto*y=ggml_mul_mat(ctx,w,x);
    ggml_set_name(w,"output.weight");ggml_set_name(x,"q6_head_input");ggml_set_name(y,"q6_head_output");ggml_set_input(x);ggml_set_output(y);
    if(!ggml_backend_supports_op(backend,y))return 6;
    auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,y);auto*buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)return 7;
    std::vector<float>weights(size_t(K)*N),features(K),out(N);
    std::vector<uint8_t>quant(ggml_nbytes(w));uint32_t random=0x51a6e7u;
    for(size_t i=0;i<weights.size();++i) {
        weights[i]=float(int(next(random)%65537)-32768)*0x1p-12f;
        if(i/K%257==0)weights[i]=0.f;
        if(i/K%263==0)weights[i]=(i&1)?7.875f:-8.f;
    }
    if(ggml_quantize_chunk(GGML_TYPE_Q6_K,weights.data(),quant.data(),0,N,K,nullptr)!=quant.size())return 8;
    uint64_t weight_hash=hash(quant.data(),quant.size());ggml_backend_tensor_set(w,quant.data(),0,quant.size());
    std::vector<float>().swap(weights);std::vector<uint8_t>().swap(quant);
    std::array<const void*,3>pointers{x->data,y->data,w->data};
    header expected{MAGIC,1,K,N,CASES,uint64_t(N)*4,weight_hash};
    if(write){if(!io(file.get(),true,&expected,sizeof(expected)))return 9;}
    else{header got{};if(!io(file.get(),false,&got,sizeof(got))||std::memcmp(&got,&expected,sizeof(got)))return 9;}
    // Repeat cases after intervening inputs to detect accidental stale activation reuse.
    const uint32_t order[CASES]={0,1,2,3,4,5,6,7,8,9,10,11,4,0,7,11};
    bool ok=true;
    for(uint32_t ordinal=0;ordinal<CASES;++ordinal) {
        uint32_t variant=order[ordinal],r=0x1286173u^variant*0x9e3779b9u;
        for(size_t i=0;i<features.size();++i) {
            float v=float(int(next(r)%131071)-65535)*0x1p-15f;
            switch(variant) {
                case 0:v=0.f;break;
                case 1:v=(i&1)?0.f:-0.f;break;
                case 2:v=std::copysign(0x1p-149f,v);break;
                case 3:v*=0x1p-100f;break;
                case 4:v*=0x1p60f;break;
                case 5:v=(i&1)?127.f:-128.f;break;
                case 6:v=(i%31==0)?v*0x1p12f:0.f;break;
                case 7:v=(i%32==0)?-8.f:(i%32==1)?8.f:v*0x1p-10f;break;
                case 8:v=(i%32==31)?0x1p30f:v*0x1p-60f;break;
                case 9:v=std::nextafter((i&1)?1.f:-1.f,0.f);break;
                default:break;
            }
            features[i]=v;
        }
        ggml_backend_tensor_set(x,features.data(),0,features.size()*4);
        std::fill(out.begin(),out.end(),std::nanf("q6n64"));ggml_backend_tensor_set(y,out.data(),0,out.size()*4);
        if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 10;
        ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
        std::vector<float>after(K);ggml_backend_tensor_get(x,after.data(),0,after.size()*4);
        bool finite=std::all_of(out.begin(),out.end(),[](float v){return std::isfinite(v);});
        std::array<const void*,3>now{x->data,y->data,w->data};
        bool pass=finite&&now==pointers&&!std::memcmp(features.data(),after.data(),after.size()*4)&&match(file.get(),write,ordinal,variant,features,out);
        std::printf("Q6_N64_GRAPH option=%s ordinal=%u variant=%u logits=%d weight_hash=%016llx input_hash=%016llx output_hash=%016llx finite=%d pointers=%d %s\n",option,ordinal,variant,N,
            (unsigned long long)weight_hash,(unsigned long long)hash(features.data(),features.size()*4),
            (unsigned long long)hash(out.data(),out.size()*4),int(finite),int(now==pointers),pass?"PASS":"FAIL");
        ok=ok&&pass;
    }
    uint64_t footer[2]={MAGIC,CASES};
    if(write)ok=ok&&io(file.get(),true,footer,sizeof(footer))&&!std::fflush(file.get());
    else{uint64_t got[2]{};ok=ok&&io(file.get(),false,got,sizeof(got))&&!std::memcmp(got,footer,sizeof(got))&&std::fgetc(file.get())==EOF;}
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);return ok?0:1;
}
