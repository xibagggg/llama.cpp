// Same-binary off/on native graph regression. This is not a benchmark.
// Physical PC8 weights, the complete production chain, actual graph splits,
// observed fusion counts, A->B->A mutation and byte-exact outputs are required.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml-impl.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

// ggml_graph_view is private/hidden in the current shared ggml-base.
// Link its exact frozen definition into this fixture only; production
// libraries, graph use guards, source and driver remain unchanged. The
// C99 designator comments are replaced by C++17 aggregate field comments.
struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {
    struct ggml_cgraph cgraph = {
        /* size */ 0,
        /* n_nodes */ i1 - i0,
        /* n_leafs */ 0,
        /* nodes */ cgraph0->nodes + i0,
        /* grads */ NULL, // gradients would need visited_hash_set
        /* grad_accs */ NULL,
        /* leafs */ NULL,
        /* use_counts */ cgraph0->use_counts,
        /* visited_hash_set */ cgraph0->visited_hash_set,
        /* order */ cgraph0->order,
        /* uid */ 0
    };

    return cgraph;
}

namespace {
constexpr int K=4304,PK=4352,N=1152,STEPS=3;
constexpr uint64_t MAGIC=UINT64_C(0x50433847454c5531),VERSION=2,FNV=UINT64_C(14695981039346656037);
enum class kind:uint32_t {
    direct,bias,residual_left,residual_right,input_alias,residual_alias,gallocr,
    none_gap,split_before_gelu,split_after_gelu,split_after_pad,
    gelu_output,pad_output,gelu_consumer,pad_consumer,gelu_view,
    source_view,intervening_view,intervening_reshape,intervening_compute,
    wrong_layer,wrong_up_name,wrong_down_name,up_bias_view,up_bias_computed,
    unbiased_gelu,up_bias_left,down_bias_view,down_bias_computed,
    matrix_output,bias_output,preactivation_output,
    view_before_gelu,view_after_gelu,view_after_pad,COUNT
};
const char*name(kind k){static const char*names[]={
    "direct","bias","residual-left","residual-right","input-alias","residual-alias","gallocr",
    "none-gap","split-before-gelu","split-after-gelu","split-after-pad",
    "gelu-output","pad-output","gelu-consumer","pad-consumer","gelu-view",
    "source-view","intervening-view","intervening-reshape","intervening-compute",
    "wrong-layer","wrong-up-name","wrong-down-name","up-bias-view","up-bias-computed",
    "unbiased-gelu","up-bias-left","down-bias-view","down-bias-computed",
    "matrix-output","bias-output","preactivation-output",
    "graph-view-before-gelu","graph-view-after-gelu","graph-view-after-pad"};
    return names[unsigned(k)];}
bool eligible(kind k){switch(k){
    // Public add_node slices lack use metadata and must reject safely.
    // Separate true graph_view cases below preserve native metadata.
    case kind::split_before_gelu:case kind::split_after_gelu:case kind::split_after_pad:
    case kind::view_after_gelu:case kind::view_after_pad:
    case kind::gelu_output:case kind::pad_output:case kind::gelu_consumer:case kind::pad_consumer:
    case kind::gelu_view:case kind::source_view:case kind::intervening_view:case kind::intervening_reshape:
    case kind::intervening_compute:case kind::wrong_layer:case kind::wrong_up_name:case kind::wrong_down_name:
    case kind::up_bias_view:case kind::up_bias_computed:case kind::unbiased_gelu:case kind::up_bias_left:return false;
    default:return true;}}
uint64_t hash(uint64_t h,const void*p,size_t n){const auto*b=static_cast<const uint8_t*>(p);
    for(size_t i=0;i<n;++i)h=(h^b[i])*UINT64_C(1099511628211);return h;}
uint32_t rnd(uint32_t&s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
void set(const char*a,const char*b){if(setenv(a,b,1))std::abort();}
bool io(FILE*f,bool write,void*p,size_t n){return write?std::fwrite(p,1,n,f)==n:std::fread(p,1,n,f)==n;}
struct trace_state{std::atomic<int>lines{0},sum{0};std::atomic<bool>invalid{false};
    ggml_log_callback previous=nullptr;void*previous_data=nullptr;};
void capture(enum ggml_log_level level,const char*text,void*opaque){auto&s=*static_cast<trace_state*>(opaque);
    if(const auto*p=std::strstr(text,"PADDLE_PC8 GELU+PACK fused=")){int n=-1;
        if(std::sscanf(p,"PADDLE_PC8 GELU+PACK fused=%d",&n)!=1||n<0||n>1)s.invalid.store(true);
        else{s.lines.fetch_add(1);s.sum.fetch_add(n);}}
    if(s.previous)s.previous(level,text,s.previous_data);}
struct weights{
    std::array<std::vector<uint8_t>,2>up,down;
    weights(){for(int variant=0;variant<2;++variant){
        std::vector<float>u(size_t(N)*K),d(size_t(PK)*N);uint32_t seed=0x176a321u+variant*7919;
        for(auto&v:u)v=float(int(rnd(seed)%65537)-32768)*0x1p-20f;
        for(int n=0;n<N;++n)for(int k=0;k<PK;++k)
            d[size_t(n)*PK+k]=k<K?float(int(rnd(seed)%65537)-32768)*0x1p-20f:0.f;
        up[variant].resize(ggml_row_size(GGML_TYPE_Q4_0,N)*K);
        down[variant].resize(ggml_row_size(GGML_TYPE_Q4_0,PK)*N);
        if(ggml_quantize_chunk(GGML_TYPE_Q4_0,u.data(),up[variant].data(),0,K,N,nullptr)!=up[variant].size()||
           ggml_quantize_chunk(GGML_TYPE_Q4_0,d.data(),down[variant].data(),0,N,PK,nullptr)!=down[variant].size())std::abort();
    }}
};
struct snapshot{ggml_tensor*t;const void*address;std::vector<uint8_t>bytes;
    explicit snapshot(ggml_tensor*source,size_t size=0):t(source),address(source->data),
        bytes(size?size:ggml_nbytes(source)){
        // SPACEMIT buffers are accessible shared host/AI storage. For weights
        // snapshot the complete physical allocation, not original GGUF nbytes.
        std::memcpy(bytes.data(),source->data,bytes.size());}
    bool unchanged()const{return t->data==address&&!std::memcmp(bytes.data(),t->data,bytes.size());}
};
struct records{FILE*f;bool write;uint64_t count=0,rolling=FNV;
    bool save(kind k,int m,int step,int slot,uint64_t input_hash,const std::vector<float>&out){
        std::array<uint64_t,8>meta={MAGIC,uint64_t(k),uint64_t(m),uint64_t(step),uint64_t(slot),input_hash,out.size(),VERSION},want=meta;
        bool equal=io(f,write,meta.data(),sizeof(meta))&&meta==want;
        if(write)equal=equal&&std::fwrite(out.data(),4,out.size(),f)==out.size();
        else{std::vector<float>ref(out.size());equal=equal&&io(f,false,ref.data(),ref.size()*4);
            if(equal&&std::memcmp(ref.data(),out.data(),out.size()*4)){
                size_t i=0;while(i<out.size()&&!std::memcmp(&ref[i],&out[i],4))++i;
                uint32_t a,b;std::memcpy(&a,&ref[i],4);std::memcpy(&b,&out[i],4);
                std::fprintf(stderr,"PC8 graph mismatch kind=%s M=%d step=%d slot=%d element=%zu %08x != %08x\n",name(k),m,step,slot,i,a,b);equal=false;}}
        rolling=hash(hash(rolling,meta.data(),sizeof(meta)),out.data(),out.size()*4);++count;return equal;
    }
};
bool run(ggml_backend_t backend,trace_state&trace,bool gate,kind k,int m,const weights&w,records&record){
    auto*leaf=ggml_init({16*1024*1024,nullptr,true});auto*ctx=ggml_init({32*1024*1024,nullptr,true});
    if(!leaf||!ctx)return false;
    auto*uw=ggml_new_tensor_2d(leaf,GGML_TYPE_Q4_0,N,K);auto*dw=ggml_new_tensor_2d(leaf,GGML_TYPE_Q4_0,PK,N);
    ggml_set_param(uw);ggml_set_param(dw);
    ggml_set_name(uw,k==kind::wrong_up_name?"v.blk.0.attn_q.weight":"v.blk.0.ffn_up.weight");
    ggml_set_name(dw,k==kind::wrong_down_name?"v.blk.0.attn_o.weight":k==kind::wrong_layer?"v.blk.1.ffn_down.weight":"v.blk.0.ffn_down.weight");
    auto*x=ggml_new_tensor_2d(leaf,GGML_TYPE_F32,N,m);ggml_set_input(x);
    auto*ub=ggml_new_tensor_1d(leaf,GGML_TYPE_F32,K);auto*db=ggml_new_tensor_1d(leaf,GGML_TYPE_F32,N);
    auto*skip=ggml_new_tensor_2d(leaf,GGML_TYPE_F32,N,m);ggml_set_input(skip);
    ggml_tensor*upbias=ub,*downbias=db;
    if(k==kind::up_bias_view)upbias=ggml_view_1d(ctx,ub,K,0);
    if(k==kind::up_bias_computed)upbias=ggml_scale(ctx,ub,0.5f);
    if(k==kind::down_bias_view)downbias=ggml_view_1d(ctx,db,N,0);
    if(k==kind::down_bias_computed)downbias=ggml_scale(ctx,db,0.5f);
    auto*up=ggml_mul_mat(ctx,uw,x);
    auto*added=k==kind::unbiased_gelu?up:k==kind::up_bias_left?
        ggml_add(ctx,ggml_repeat(ctx,upbias,up),up):ggml_add(ctx,up,upbias);
    auto*source=k==kind::source_view?ggml_view_2d(ctx,added,K,m,added->nb[1],0):added;
    auto*gelu=ggml_gelu(ctx,source);ggml_tensor*activation=gelu;
    if(k==kind::intervening_view)activation=ggml_view_2d(ctx,gelu,K,m,gelu->nb[1],0);
    if(k==kind::intervening_reshape)activation=ggml_reshape_2d(ctx,gelu,K,m);
    auto*pad=ggml_pad(ctx,activation,48,0,0,0);auto*matrix=ggml_mul_mat(ctx,dw,pad);
    auto*biased=k==kind::direct?matrix:ggml_add(ctx,matrix,downbias);ggml_tensor*y=biased;
    const bool has_skip=k==kind::residual_left||k==kind::residual_right||k==kind::residual_alias;
    if(has_skip)y=k==kind::residual_left?ggml_add(ctx,skip,biased):ggml_add(ctx,biased,skip);
    ggml_set_output(y);ggml_set_name(y,name(k));
    std::vector<ggml_tensor*>probes;
    if(k==kind::gelu_output){ggml_set_output(gelu);probes.push_back(gelu);}
    if(k==kind::pad_output){ggml_set_output(pad);probes.push_back(pad);}
    if(k==kind::gelu_consumer)probes.push_back(ggml_scale(ctx,gelu,0.5f));
    if(k==kind::pad_consumer)probes.push_back(ggml_scale(ctx,pad,0.5f));
    if(k==kind::gelu_view)probes.push_back(ggml_view_2d(ctx,gelu,K,m,gelu->nb[1],0));
    if(k==kind::matrix_output){ggml_set_output(matrix);probes.push_back(matrix);}
    if(k==kind::bias_output){ggml_set_output(biased);probes.push_back(biased);}
    if(k==kind::preactivation_output){ggml_set_output(added);probes.push_back(added);}
    auto*graph=ggml_new_graph(ctx);
    if(k==kind::none_gap||k==kind::intervening_compute){ggml_build_forward_expand(graph,gelu);
        if(k==kind::none_gap)ggml_graph_add_node(graph,ggml_new_tensor_1d(leaf,GGML_TYPE_F32,1));
        else{auto*other=ggml_scale(ctx,skip,0.5f);ggml_set_output(other);ggml_build_forward_expand(graph,other);probes.push_back(other);}}
    ggml_build_forward_expand(graph,y);
    for(auto*p:probes){ggml_set_output(p);ggml_build_forward_expand(graph,p);}
    const auto leaf_buffer=ggml_backend_alloc_ctx_tensors(leaf,backend);if(!leaf_buffer)return false;
    ggml_backend_buffer_t graph_buffer=nullptr;ggml_gallocr_t allocator=nullptr;
    if(k==kind::gallocr){allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if(!allocator||!ggml_gallocr_alloc_graph(allocator,graph))return false;}
    else{graph_buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!graph_buffer)return false;}
    if(!uw->extra||uw->extra!=dw->extra){std::fprintf(stderr,"physical PC8 trait unavailable\n");return false;}
    const size_t up_bytes=ggml_backend_buffer_get_alloc_size(leaf_buffer,uw),down_bytes=ggml_backend_buffer_get_alloc_size(leaf_buffer,dw);
    if(up_bytes!=128+size_t(GGML_PAD(K,32))*(4+N)||down_bytes!=128+size_t(N)*(4+PK))return false;
    void*saved=y->data;
    if(k==kind::input_alias)y->data=added->data;
    if(k==kind::residual_alias)y->data=skip->data;
    const bool forced_alias=k==kind::input_alias||k==kind::residual_alias;
    if(forced_alias&&y->data!=(k==kind::input_alias?added->data:skip->data))return false;
    std::array<const void*,10>addresses={x->data,uw->data,dw->data,ub->data,db->data,skip->data,added->data,gelu->data,pad->data,y->data};
    std::vector<ggml_cgraph*>parts={graph};std::array<ggml_cgraph,2>views{};
    const bool manual_split=k==kind::split_before_gelu||k==kind::split_after_gelu||k==kind::split_after_pad;
    const bool native_split=k==kind::view_before_gelu||k==kind::view_after_gelu||k==kind::view_after_pad;
    if(manual_split||native_split){
        const auto*boundary=(k==kind::split_before_gelu||k==kind::view_before_gelu)?added:
            (k==kind::split_after_gelu||k==kind::view_after_gelu)?gelu:pad;
        int split=-1;for(int i=0;i<ggml_graph_n_nodes(graph);++i)if(ggml_graph_node(graph,i)==boundary)split=i+1;
        if(split<=0||split>=ggml_graph_n_nodes(graph))return false;
        if(native_split){
            views[0]=ggml_graph_view(graph,0,split);views[1]=ggml_graph_view(graph,split,ggml_graph_n_nodes(graph));
            for(int j=0;j<2;++j){const auto&v=views[j];
                if(v.size!=0||v.n_leafs!=0||v.grads||v.grad_accs||v.leafs||v.uid!=0||
                   v.use_counts!=graph->use_counts||v.visited_hash_set.keys!=graph->visited_hash_set.keys||
                   v.visited_hash_set.used!=graph->visited_hash_set.used||
                   v.visited_hash_set.size!=graph->visited_hash_set.size)return false;
                const int begin=j?split:0;
                for(int i=0;i<v.n_nodes;++i)if(ggml_node_get_use_count(&v,i)!=ggml_node_get_use_count(graph,begin+i))return false;
            }
            parts={&views[0],&views[1]};
        }else{
            auto*a=ggml_new_graph(ctx);auto*b=ggml_new_graph(ctx);
            for(int i=0;i<ggml_graph_n_nodes(graph);++i)ggml_graph_add_node(i<split?a:b,ggml_graph_node(graph,i));parts={a,b};
        }
        std::printf("PC8_GELU_PACK_SPLIT kind=%s native_graph_view=%d full_use_metadata=%d first_nodes=%d second_nodes=%d boundary=%s\n",
            name(k),int(native_split),int(native_split),parts[0]->n_nodes,parts[1]->n_nodes,
            boundary==added?"before-gelu":boundary==gelu?"after-gelu":"after-pad");
    }
    bool ok=true;std::vector<float>first,previous;uint64_t first_input=0,previous_input=0;
    for(int step=0;step<STEPS;++step){const int variant=step==1;uint32_t seed=0x1769763u+variant*277;
        std::vector<float>xs(size_t(N)*m),ubs(K),dbs(N),skips(size_t(N)*m);
        for(auto&v:xs)v=float(int(rnd(seed)%65537)-32768)*0x1p-13f;
        for(auto&v:ubs)v=float(int(rnd(seed)%8193)-4096)*0x1p-13f;
        for(auto&v:dbs)v=float(int(rnd(seed)%8193)-4096)*0x1p-13f;
        for(auto&v:skips)v=float(int(rnd(seed)%65537)-32768)*0x1p-15f;
        ggml_backend_tensor_set(uw,w.up[variant].data(),0,w.up[variant].size());
        ggml_backend_tensor_set(dw,w.down[variant].data(),0,w.down[variant].size());
        ggml_backend_tensor_set(x,xs.data(),0,xs.size()*4);ggml_backend_tensor_set(ub,ubs.data(),0,ubs.size()*4);
        ggml_backend_tensor_set(db,dbs.data(),0,dbs.size()*4);ggml_backend_tensor_set(skip,skips.data(),0,skips.size()*4);
        std::vector<snapshot>unchanged={snapshot(x),snapshot(uw,up_bytes),snapshot(dw,down_bytes),snapshot(ub),snapshot(db)};
        if(k!=kind::residual_alias)unchanged.emplace_back(skip);
        uint64_t ih=FNV;for(const auto&s:unchanged)ih=hash(ih,s.bytes.data(),s.bytes.size());
        if(k==kind::residual_alias)ih=hash(ih,skips.data(),skips.size()*4);
        trace.lines.store(0);trace.sum.store(0);trace.invalid.store(false);bool computed=true;
        for(auto*p:parts)computed=(ggml_backend_graph_compute(backend,p)==GGML_STATUS_SUCCESS)&&computed;
        const int expected=gate&&eligible(k)?1:0;
        const bool hits=!trace.invalid.load()&&(step!=0||trace.lines.load()>0)&&
            (trace.lines.load()==0||trace.sum.load()==expected);
        std::vector<float>out(size_t(N)*m);ggml_backend_tensor_get(y,out.data(),0,out.size()*4);
        bool finite=std::all_of(out.begin(),out.end(),[](float v){return std::isfinite(v);});
        bool immutable=true;for(const auto&s:unchanged)immutable=s.unchanged()&&immutable;
        const std::array<const void*,10>now={x->data,uw->data,dw->data,ub->data,db->data,skip->data,added->data,gelu->data,pad->data,y->data};
        const bool mutation=step==0||(step==1?(ih!=previous_input&&std::memcmp(out.data(),previous.data(),out.size()*4)):
            (ih==first_input&&!std::memcmp(out.data(),first.data(),out.size()*4)));
        bool exact=record.save(k,m,step,0,ih,out);int slot=1;
        for(auto*p:probes){std::vector<float>v(ggml_nelements(p));ggml_backend_tensor_get(p,v.data(),0,v.size()*4);
            finite=finite&&std::all_of(v.begin(),v.end(),[](float z){return std::isfinite(z);});exact=record.save(k,m,step,slot++,ih,v)&&exact;}
        const bool pass=computed&&hits&&finite&&immutable&&addresses==now&&mutation&&exact;ok=ok&&pass;
        std::printf("PC8_GELU_PACK_GRAPH kind=%s M=%d step=%d parts=%zu expected_hit=%d trace_lines=%d trace_sum=%d trace_ok=%d exact=%d finite=%d immutable=%d same_address=%d mutation_ABA=%d forced_alias=%d native_input_alias=%d %s\n",
            name(k),m,step,parts.size(),expected,trace.lines.load(),trace.sum.load(),int(hits),int(exact),int(finite),int(immutable),int(addresses==now),int(mutation),int(forced_alias),int(y->data==added->data),pass?"PASS":"FAIL");
        std::fflush(stdout);if(step==0){first=out;first_input=ih;}previous=out;previous_input=ih;
    }
    y->data=saved;if(allocator)ggml_gallocr_free(allocator);if(graph_buffer)ggml_backend_buffer_free(graph_buffer);
    ggml_backend_buffer_free(leaf_buffer);ggml_free(ctx);ggml_free(leaf);return ok;
}
} // namespace
int main(int argc,char**argv){
    if(argc!=3||(std::strcmp(argv[1],"--write-reference")&&std::strcmp(argv[1],"--compare-reference")))return 2;
    const bool write=!std::strcmp(argv[1],"--write-reference");
    const char*flag=std::getenv("GGML_SPACEMIT_PADDLE_PC8_GELU_PACK");if(!flag||std::strcmp(flag,write?"0":"1"))return 2;
    const std::pair<const char*,const char*>flags[]={
        {"GGML_SPACEMIT_BACKEND","1"},{"GGML_SPACEMIT_WORKERS","6"},{"GGML_SPACEMIT_ALLOW_EIGHT_WORKERS","0"},
        {"SPACEMIT_PERFER_CORE_ID","8,9,10,11,12,13"},{"GGML_SPACEMIT_SHARED_MODE","PINNED_TCM"},
        {"GGML_SPACEMIT_STRICT_AI_CORES","1"},{"SPERT_SHARED_MODE","TCM"},{"SPERT_BACKEND","spacemit-k3"},{"SPERT_RT_PRIO","0"},
        {"GGML_SPACEMIT_OCR_Q4_PC8","2"},{"GGML_SPACEMIT_OCR_Q4_PC8_M8","1"},{"GGML_SPACEMIT_OCR_Q4_PC8_DOWN_M8","1"},
        {"GGML_SPACEMIT_OCR_Q4_PC8_MULTI_N","1"},{"GGML_SPACEMIT_OCR_PC8_PREPACK","1"},
        {"GGML_SPACEMIT_OCR_PC8_REUSE_INPUT","1"},{"GGML_SPACEMIT_OCR_PC8_LOAD_THREADS","6"},
        {"GGML_SPACEMIT_OCR_PC8_BIAS_FUSE","1"},{"GGML_SPACEMIT_OCR_PC8_GELU_FUSE","1"},
        {"GGML_SPACEMIT_OCR_PC8_PAD_FUSE","1"},{"GGML_SPACEMIT_OCR_PC8_DOWN_EPILOGUE","1"},
        {"GGML_SPACEMIT_ADD_GELU","1"},{"GGML_SPACEMIT_PADDLE_PC8_GELU_PACK_TRACE","1"}};
    for(const auto&f:flags)set(f.first,f.second);
    const int fd=write?open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0644):open(argv[2],O_RDONLY);if(fd<0){std::perror(argv[2]);return 2;}
    std::unique_ptr<FILE,decltype(&std::fclose)>file(fdopen(fd,write?"wb":"rb"),&std::fclose);if(!file){close(fd);return 2;}
    std::array<uint64_t,9>header={MAGIC,VERSION,0x01020304,sizeof(float),GGML_TYPE_Q4_0,K,PK,N,STEPS},want=header;
    if(!io(file.get(),write,header.data(),sizeof(header))||header!=want)return 2;
    trace_state trace;ggml_log_get(&trace.previous,&trace.previous_data);ggml_log_set(capture,&trace);
    ggml_backend_load_all();const auto dev=ggml_backend_dev_by_name("SPACEMIT0");if(!dev)return 3;
    const auto backend=ggml_backend_dev_init(dev,nullptr);if(!backend)return 3;
    const weights w;records record{file.get(),write};bool ok=true;size_t cases=0;
    for(int m:{1,3,4,5,7,8,9,13,848,1500})for(kind k:{kind::direct,kind::residual_right}){
        ok=run(backend,trace,!write,k,m,w,record)&&ok;++cases;}
    for(unsigned i=0;i<unsigned(kind::COUNT);++i){const auto k=kind(i);if(k==kind::direct||k==kind::residual_right)continue;
        ok=run(backend,trace,!write,k,9,w,record)&&ok;++cases;}
    ggml_backend_free(backend);ggml_log_set(trace.previous,trace.previous_data);if(!ok)return 1;
    std::array<uint64_t,5>footer={MAGIC^UINT64_MAX,VERSION,cases,record.count,record.rolling},expected=footer;
    if(!io(file.get(),write,footer.data(),sizeof(footer))||footer!=expected)return 1;
    if(!write&&std::fgetc(file.get())!=EOF)return 1;
    if(write&&(std::fflush(file.get())||fsync(fileno(file.get()))))return 1;
    std::printf("PC8_GELU_PACK_GRAPH cases=%zu passes=%zu records=%llu six_workers=1 default_worker_frm_only=1 bitwise_complete=1 PASS\n",cases,cases*STEPS,(unsigned long long)record.count);
    return 0;
}
