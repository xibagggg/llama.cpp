// Full graph byte comparison in distinct processes. Output/view/use-count
// rejection and both earlier GELU-consuming fusions are exercised explicitly.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

enum class kind:uint32_t {
    direct,down_bias,up_bias,add_gelu,gelu_output,extra_consumer,external_view,
    source_view,wrong_name,ordinary_f16,intervening_view,destination_alias,
    bias_view,computed_bias,matmul_output,
};
static const char*name(kind k) {
    static const char*names[]={"direct","down-bias","up-bias-gelu","add-gelu","gelu-output","extra-consumer",
        "external-view","source-view","wrong-name","ordinary-f16","intervening-view","destination-alias",
        "bias-view","computed-bias","matmul-output"};return names[unsigned(k)];
}
static uint32_t next(uint32_t&x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
struct trace_state {
    std::atomic<int>expected{0},lines{0};std::atomic<bool>mismatch{false};
    ggml_log_callback previous=nullptr;void*previous_data=nullptr;
};
static void capture(enum ggml_log_level level,const char*text,void*opaque) {
    auto&s=*static_cast<trace_state*>(opaque);int n=-1;
    auto*p=std::strstr(text,"PADDLE_F16 GELU+PACK fused=");
    if(p&&std::sscanf(p,"PADDLE_F16 GELU+PACK fused=%d",&n)==1) {
        s.lines.fetch_add(1);if(n!=s.expected.load())s.mismatch.store(true);
    }
    if(s.previous)s.previous(level,text,s.previous_data);
}
static bool record(FILE*f,bool write,kind k,int m,int pass,int slot,const std::vector<float>&values) {
    uint64_t header[]={UINT64_C(0x47454c555041434b),1,uint64_t(k),uint64_t(m),uint64_t(pass),uint64_t(slot),values.size()};
    if(write)return std::fwrite(header,sizeof(header),1,f)==1&&
        std::fwrite(values.data(),4,values.size(),f)==values.size();
    uint64_t actual[7];if(std::fread(actual,sizeof(actual),1,f)!=1||std::memcmp(header,actual,sizeof(header)))return false;
    std::vector<float>reference(values.size());
    if(std::fread(reference.data(),4,reference.size(),f)!=reference.size())return false;
    if(!std::memcmp(reference.data(),values.data(),values.size()*4))return true;
    size_t first=0;while(first<values.size()&&!std::memcmp(&reference[first],&values[first],4))++first;
    uint32_t a=0,b=0;std::memcpy(&a,&reference[first],4);std::memcpy(&b,&values[first],4);
    std::fprintf(stderr,"byte mismatch kind=%s M=%d pass=%d slot=%d float=%zu 0x%08x != 0x%08x\n",name(k),m,pass,slot,first,a,b);
    return false;
}
static bool eligible(kind k) {
    return k!=kind::gelu_output&&k!=kind::extra_consumer&&k!=kind::external_view&&
        k!=kind::source_view&&k!=kind::wrong_name&&k!=kind::ordinary_f16&&k!=kind::intervening_view;
}
static bool run(ggml_backend_t backend,trace_state&trace,bool gate,kind k,int m,FILE*f,bool write) {
    constexpr int K=4304,N=1152;const bool up=k==kind::up_bias;
    const int input_k=up?1152:K;
    ggml_init_params params{32*1024*1024,nullptr,true};auto*ctx=ggml_init(params);if(!ctx)return false;
    auto*w=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,K,N);ggml_set_param(w);
    ggml_set_name(w,k==kind::ordinary_f16?"ordinary.weight":k==kind::wrong_name?"v.blk.0.attn_q.weight":"v.blk.0.ffn_down.weight");
    ggml_tensor*uw=nullptr;
    if(up){uw=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,1152,K);ggml_set_param(uw);ggml_set_name(uw,"v.blk.0.ffn_up.weight");}
    auto*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,input_k,m);ggml_set_input(x);
    ggml_tensor*source=x;
    auto*prebias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,K);
    if(up)source=ggml_add(ctx,ggml_mul_mat(ctx,uw,x),prebias);
    if(k==kind::add_gelu)source=ggml_add(ctx,x,prebias);
    if(k==kind::source_view)source=ggml_view_2d(ctx,x,K,m,x->nb[1],0);
    auto*g=ggml_gelu(ctx,source);
    auto*activation=k==kind::intervening_view?ggml_view_2d(ctx,g,K,m,g->nb[1],0):g;
    auto*matrix=ggml_mul_mat(ctx,w,activation);
    auto*rawbias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,N);ggml_tensor*bias=rawbias;
    if(k==kind::bias_view)bias=ggml_view_1d(ctx,rawbias,N,0);
    if(k==kind::computed_bias)bias=ggml_scale(ctx,rawbias,0.5f);
    const bool has_bias=k!=kind::direct;
    auto*y=has_bias?ggml_add(ctx,matrix,bias):matrix;ggml_set_output(y);
    ggml_tensor*probe=nullptr;
    if(k==kind::gelu_output){ggml_set_output(g);probe=g;}
    if(k==kind::extra_consumer){probe=ggml_scale(ctx,g,0.5f);ggml_set_output(probe);}
    if(k==kind::external_view){probe=ggml_view_2d(ctx,g,K,m,g->nb[1],0);ggml_set_output(probe);}
    if(k==kind::matmul_output){ggml_set_output(matrix);probe=matrix;}
    auto*graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);if(probe)ggml_build_forward_expand(graph,probe);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer){ggml_free(ctx);return false;}
    if(k!=kind::ordinary_f16&&!w->extra){std::fprintf(stderr,"Paddle F16 weight inactive\n");return false;}
    std::vector<ggml_fp16_t>weights(size_t(K)*N),upweights(up?size_t(K)*1152:0);
    std::vector<float>values(size_t(input_k)*m),pvalues(K),bvalues(N);
    uint32_t rng=0x17609873u;
    for(auto&a:weights)a=ggml_fp32_to_fp16(float(int(next(rng)%65537)-32768)*0x1p-20f);
    for(auto&a:upweights)a=ggml_fp32_to_fp16(float(int(next(rng)%65537)-32768)*0x1p-20f);
    for(size_t i=0;i<values.size();++i)values[i]=float(int(next(rng)%65537)-32768)/8192.f;
    if(m==8)std::fill(values.begin(),values.end(),0.f);
    if(m==17)for(auto&a:values)a*=0x1p-100f;
    for(int i=0;i<K;++i)pvalues[i]=float(i%31-15)*0x1p-10f;
    for(int i=0;i<N;++i)bvalues[i]=float(i%43-21)*0x1p-9f;
    ggml_backend_tensor_set(w,weights.data(),0,weights.size()*2);
    if(up)ggml_backend_tensor_set(uw,upweights.data(),0,upweights.size()*2);
    ggml_backend_tensor_set(prebias,pvalues.data(),0,pvalues.size()*4);
    ggml_backend_tensor_set(rawbias,bvalues.data(),0,bvalues.size()*4);
    const bool alias=k==kind::destination_alias;
    void*saved_destination=y->data;if(alias)y->data=x->data;
    bool ok=true;
    for(int pass=0;pass<3;++pass) {
        if(pass==1)for(size_t i=0;i<values.size();++i)values[i]=-0.5f*values[i]+float(int(i%257)-128)*0x1p-13f;
        if(pass==2)for(int i=0;i<N;++i)bvalues[i]=-bvalues[i]+float(i%7)*0x1p-12f;
        ggml_backend_tensor_set(x,values.data(),0,values.size()*4);
        ggml_backend_tensor_set(rawbias,bvalues.data(),0,bvalues.size()*4);
        trace.expected.store(gate&&eligible(k)?1:0);trace.lines.store(0);trace.mismatch.store(false);
        bool status=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
        bool hits=!trace.mismatch.load()&&(pass!=0||trace.lines.load()>0);
        std::vector<float>out(size_t(N)*m);ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
        bool finite=std::all_of(out.begin(),out.end(),[](float a){return std::isfinite(a);});
        bool same=record(f,write,k,m,pass,0,out);
        if(probe){std::vector<float>v(size_t(ggml_nelements(probe)));ggml_backend_tensor_get(probe,v.data(),0,v.size()*4);same=record(f,write,k,m,pass,1,v)&&same;}
        bool inputs=true;
        if(!alias){std::vector<float>v(values.size());ggml_backend_tensor_get(x,v.data(),0,v.size()*4);inputs=!std::memcmp(v.data(),values.data(),v.size()*4);}
        bool valid=status&&hits&&finite&&same&&inputs;ok=ok&&valid;
        std::printf("F16_GELU_PACK_GRAPH kind=%s M=%d pass=%d expected_hit=%d trace=%d matched=%d bytes=%s finite=%d source_unchanged=%d alias=%d %s\n",
            name(k),m,pass,trace.expected.load(),trace.lines.load(),int(hits),write?"write":same?"equal":"different",int(finite),int(inputs),int(alias),valid?"PASS":"FAIL");
    }
    if(alias)y->data=saved_destination;
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return ok;
}
int main(int argc,char**argv) {
    if(argc!=3||(std::strcmp(argv[1],"--write-reference")&&std::strcmp(argv[1],"--compare-reference")))return 2;
    const bool write=!std::strcmp(argv[1],"--write-reference");
    const char*flag=std::getenv("GGML_SPACEMIT_PADDLE_F16_GELU_PACK");
    if(!flag||std::strcmp(flag,write?"0":"1"))return 2;
    const char*workers=std::getenv("GGML_SPACEMIT_WORKERS");if(!workers||std::strcmp(workers,"6"))return 2;
    setenv("GGML_SPACEMIT_PADDLE_F16_IME2","1",1);setenv("GGML_SPACEMIT_PADDLE_F16_PACK_WIDTH","64",1);
    setenv("GGML_SPACEMIT_PADDLE_F16_BIAS_FUSE","1",1);setenv("GGML_SPACEMIT_PADDLE_F16_GELU_FUSE","1",1);
    setenv("GGML_SPACEMIT_ADD_GELU","1",1);setenv("GGML_SPACEMIT_PADDLE_F16_GELU_PACK_TRACE","1",1);
    std::unique_ptr<FILE,decltype(&std::fclose)>file(std::fopen(argv[2],write?"wb":"rb"),&std::fclose);if(!file)return 2;
    trace_state trace;ggml_log_get(&trace.previous,&trace.previous_data);ggml_log_set(capture,&trace);
    ggml_backend_load_all();auto device=ggml_backend_dev_by_name("SPACEMIT0");if(!device)return 3;
    auto backend=ggml_backend_dev_init(device,nullptr);if(!backend)return 4;
    bool ok=true;size_t cases=0;
    for(int m:{1,7,8,9,17,33,848})for(kind k:{kind::direct,kind::down_bias}){ok=run(backend,trace,!write,k,m,file.get(),write)&&ok;++cases;}
    for(kind k:{kind::up_bias,kind::add_gelu,kind::gelu_output,kind::extra_consumer,kind::external_view,kind::source_view,
        kind::wrong_name,kind::ordinary_f16,kind::intervening_view,kind::destination_alias,kind::bias_view,kind::computed_bias,kind::matmul_output})
        {ok=run(backend,trace,!write,k,9,file.get(),write)&&ok;++cases;}
    if(write?std::fflush(file.get())!=0:std::fgetc(file.get())!=EOF||std::ferror(file.get()))ok=false;
    ggml_backend_free(backend);ggml_log_set(trace.previous,trace.previous_data);
    std::printf("F16_GELU_PACK_GRAPH %s cases=%zu passes=%zu all_F32_bytes_and_rejections\n",ok?"PASS":"FAIL",cases,cases*3);
    return ok?0:1;
}
