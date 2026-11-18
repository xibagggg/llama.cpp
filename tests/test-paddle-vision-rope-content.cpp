#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-spacemit/paddle-vision-rope-content-cache.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

struct trace_state {
    std::atomic<int> lines{0},hits{0},misses{0},writes{0};
    std::atomic<bool> guard_bad{false};
    ggml_log_callback previous=nullptr;void *opaque=nullptr;
    void reset(){lines=hits=misses=writes=0;guard_bad=false;}
};
static void capture(enum ggml_log_level level,const char * text,void * opaque) {
    auto & s=*static_cast<trace_state*>(opaque);
    const auto * p=std::strstr(text,"paddle-vision-rope-content: launch=");
    if(p) {
        unsigned long long launch;size_t hits,misses,inserts,evictions,writes,entries,bytes;int guards;
        if(std::sscanf(p,"paddle-vision-rope-content: launch=%llu hits=%zu misses=%zu inserts=%zu evictions=%zu rejected_writes=%zu entries=%zu bytes=%zu guards=%d",
            &launch,&hits,&misses,&inserts,&evictions,&writes,&entries,&bytes,&guards)==9) {
            ++s.lines;s.hits+=int(hits);s.misses+=int(misses);s.writes+=int(writes);
            if(!guards||bytes>paddle_vision_rope::content_cache::budget||entries>16)s.guard_bad=true;
        } else s.guard_bad=true;
    }
    if(s.previous)s.previous(level,text,s.opaque);
}
static bool fail(const char * why,int n,int kind,int pass) {
    std::fprintf(stderr,"FAIL %s N=%d kind=%d pass=%d\n",why,n,kind,pass);return false;
}
static bool bytes(FILE * f,bool write,uint32_t type,uint32_t n,uint32_t kind,uint32_t pass,uint32_t slot,const std::vector<unsigned char> & v) {
    const uint64_t header[]={0x524f5045434f4e54ULL,type,n,kind,pass,slot,v.size()};
    if(write)return std::fwrite(header,sizeof(header),1,f)==1&&std::fwrite(v.data(),1,v.size(),f)==v.size();
    uint64_t actual[7];if(std::fread(actual,sizeof(actual),1,f)!=1||std::memcmp(header,actual,sizeof(header)))return false;
    std::vector<unsigned char> ref(v.size());
    if(std::fread(ref.data(),1,ref.size(),f)!=ref.size())return false;
    if(!std::memcmp(ref.data(),v.data(),v.size()))return true;
    for(size_t i=0;i<v.size();++i)if(ref[i]!=v[i]){std::fprintf(stderr,"first output byte %zu %02x != %02x\n",i,v[i],ref[i]);break;}
    return false;
}
static bool structural_keys() {
    using namespace paddle_vision_rope;
    const auto ctx=ggml_init({1024*1024,nullptr,true});
    auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,72,16,7);
    auto pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,28);
    int sections[4]={18,18,18,18};
    auto op=ggml_rope_multi(ctx,x,pos,nullptr,36,sections,GGML_ROPE_TYPE_VISION,32768,10000,1,0,1,32,1);
    std::vector<int32_t> ids(28);for(int i=0;i<28;++i)ids[i]=i*7-41;
    pos->data=ids.data();
    std::vector<float> values(7*72);for(size_t i=0;i<values.size();++i)values[i]=float(i)*0.03125f;
    content_cache cache;cache.begin_launch();
    bool ok=!cache.find(op,false);cache.store(op,false,values);
    const float * saved=cache.find(op,false);ok=ok&&saved&&!std::memcmp(saved,values.data(),values.size()*4);
    // Different data address, same complete bytes: genuine content reuse.
    auto elsewhere=ids;pos->data=elsewhere.data();ok=ok&&cache.find(op,false);
    elsewhere.back()^=1;ok=ok&&!cache.find(op,false);elsewhere.back()^=1;
    op->op_params[4]^=1;ok=ok&&!cache.find(op,false);op->op_params[4]^=1;
    ok=ok&&!cache.find(op,true);
    op->ne[0]=80;ok=ok&&!cache.find(op,false);op->ne[0]=72;
    op->ne[2]=8;ok=ok&&!cache.find(op,false);op->ne[2]=7;
    pos->nb[0]=8;ok=ok&&!cache.find(op,false);pos->nb[0]=4;
    op->type=GGML_TYPE_F16;ok=ok&&!cache.find(op,false);op->type=GGML_TYPE_F32;
    // Geometry excludes head count, but includes table width/tokens.
    op->ne[1]=2;ok=ok&&cache.find(op,false);op->ne[1]=16;
    auto & first=cache.entries.front();uint32_t guard;
    std::memcpy(&guard,&first.storage[0],4);first.storage[0]=0;
    ok=ok&&!cache.guards_ok()&&!cache.find(op,false);std::memcpy(&first.storage[0],&guard,4);
    ok=ok&&cache.guards_ok();
    // All entries pinned by this launch must survive attempts to insert a
    // seventeenth key. A new launch may evict one older unpinned entry.
    for(int i=1;i<16;++i){elsewhere[0]=i+100;cache.store(op,false,values);ok=ok&&cache.find(op,false);}
    elsewhere[0]=999;const size_t count=cache.entries.size();cache.store(op,false,values);
    ok=ok&&count==16&&cache.entries.size()==16&&cache.evictions==0;
    cache.begin_launch();cache.store(op,false,values);
    ok=ok&&cache.find(op,false)&&cache.entries.size()==16&&cache.evictions==1&&cache.guards_ok()&&cache.bytes<=content_cache::budget;
    ggml_free(ctx);
    std::printf("content key/full-byte/layout/type/shape/guard/pinning/LRU %s\n",ok?"PASS":"FAIL");return ok;
}
// kind0 ordinary; kind1 padded/inplace; kind2 CPY writes to aliased leaf
// positions before ROPE; kind3 non-leaf VIEW positions must fall back.
static bool run(ggml_backend_t backend,trace_state & trace,FILE * f,bool write,ggml_type type,int n,int kind) {
    const int h=kind==1?2:16,d=72;
    const size_t es=type==GGML_TYPE_F32?4:2;
    auto ctx=ggml_init({8*1024*1024,nullptr,true});
    auto base=ggml_new_tensor_3d(ctx,type,kind==1?d+16:d,h,n);
    auto x=kind==1?ggml_view_3d(ctx,base,d,h,n,base->nb[1],base->nb[2],8*es):base;
    auto pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,4*n);
    // The backend supports F32 CPY. Alias its destination bytes over the
    // I32 leaf positions after allocation: no I32 arithmetic/copy expansion.
    auto replacement=kind==2?ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4*n):nullptr;
    auto copy_target=kind==2?ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4*n):nullptr;
    auto copy=replacement?ggml_cpy(ctx,replacement,copy_target):nullptr;
    auto positions=kind==3?ggml_view_1d(ctx,pos,4*n,0):pos;
    int sections[4]={18,18,18,18};
    auto first=kind==1?ggml_rope_multi_inplace(ctx,x,positions,nullptr,36,sections,GGML_ROPE_TYPE_VISION,32768,10000,1,0,1,32,1)
                      :ggml_rope_multi(ctx,x,positions,nullptr,36,sections,GGML_ROPE_TYPE_VISION,32768,10000,1,0,1,32,1);
    auto second=ggml_rope_multi(ctx,first,positions,nullptr,36,sections,GGML_ROPE_TYPE_VISION,32768,10000,1,0,1,32,1);
    ggml_set_input(base);ggml_set_input(pos);if(replacement)ggml_set_input(replacement);
    ggml_set_output(first);ggml_set_output(second);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer){ggml_free(ctx);return fail("allocation",n,kind,0);}
    if(copy){copy_target->data=pos->data;copy->data=pos->data;}
    const size_t count=ggml_nelements(base);
    std::vector<float> original(count);std::vector<ggml_fp16_t> half(count);
    for(size_t i=0;i<count;++i){original[i]=float(int((i*97+41)%2039)-1019)/317.f;half[i]=ggml_fp32_to_fp16(original[i]);}
    const void * source=type==GGML_TYPE_F32?static_cast<const void*>(original.data()):static_cast<const void*>(half.data());
    std::vector<unsigned char> source_bytes(count*es);std::memcpy(source_bytes.data(),source,source_bytes.size());
    std::vector<int32_t> ids(4*n),changed(4*n);
    bool ok=true;const void * position_address=pos->data;
    const bool enabled=paddle_vision_rope::content_enabled();
    for(int pass=0;pass<5;++pass) {
        for(int axis=0;axis<4;++axis)for(int i=0;i<n;++i) {
            // Negative and large I32 positions exercise exact float conversion.
            const int32_t p=(i%3==0?-1048576:1048576)+i*19+axis*271+kind*65537;
            ids[axis*n+i]=p+(pass>=2?7919:0)+(pass==4&&axis==3?1:0);
            changed[axis*n+i]=-p-911;
        }
        ggml_backend_tensor_set(base,source,0,count*es);
        ggml_backend_tensor_set(pos,ids.data(),0,ids.size()*4);
        if(replacement)ggml_backend_tensor_set(replacement,changed.data(),0,changed.size()*4);
        // A different graph UID every time; position allocation is unchanged.
        auto graph=ggml_new_graph(ctx);
        if(copy)ggml_build_forward_expand(graph,copy);
        ggml_build_forward_expand(graph,first);ggml_build_forward_expand(graph,second);
        trace.reset();
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS){ok=fail("compute",n,kind,pass);break;}
        if(pos->data!=position_address){ok=fail("position allocation moved",n,kind,pass);break;}
        if(enabled) {
            if(trace.lines.load()!=1||trace.guard_bad.load())ok=fail("content trace/guards",n,kind,pass)&&ok;
            if(kind==0||kind==1) {
                if((pass==1||pass==3)&&trace.hits.load()<1)ok=fail("cross-graph same-content hit not proven",n,kind,pass)&&ok;
                if((pass==2||pass==4)&&trace.misses.load()<1)ok=fail("same-address changed bytes not invalidated",n,kind,pass)&&ok;
            }
            if(kind==2&&trace.writes.load()<2)ok=fail("aliased CPY write not rejected",n,kind,pass)&&ok;
            if(kind>=2&&trace.hits.load()!=0)ok=fail("written/non-leaf positions cached",n,kind,pass)&&ok;
        }
        for(uint32_t slot=0;slot<2;++slot) {
            auto y=slot?second:first;std::vector<unsigned char> out(ggml_nbytes(y));
            ggml_backend_tensor_get(y,out.data(),0,out.size());
            if(!bytes(f,write,uint32_t(type),n,kind,pass,slot,out))ok=fail("complete output bytes",n,kind,pass)&&ok;
        }
        std::vector<unsigned char> after(source_bytes.size());ggml_backend_tensor_get(base,after.data(),0,after.size());
        if(kind!=1) {
            if(after!=source_bytes)ok=fail("read-only input modified",n,kind,pass)&&ok;
        } else {
            const size_t rowbytes=size_t(d+16)*es;
            for(int row=0;row<h*n;++row)for(size_t i=0;i<rowbytes;++i)
                if((i<8*es||i>=size_t(d+8)*es)&&after[size_t(row)*rowbytes+i]!=source_bytes[size_t(row)*rowbytes+i])
                    ok=fail("padded input guard",n,kind,pass)&&ok;
        }
        std::vector<int32_t> actual(ids.size());ggml_backend_tensor_get(pos,actual.data(),0,actual.size()*4);
        if(actual!=(kind==2?changed:ids))ok=fail("position bytes changed unexpectedly",n,kind,pass)&&ok;
        std::printf("type=%s N=%d kind=%d pass=%d content_hits=%d misses=%d rejected_writes=%d all_bytes=%s %s\n",
            ggml_type_name(type),n,kind,pass,trace.hits.load(),trace.misses.load(),trace.writes.load(),write?"write":"compare",ok?"PASS":"FAIL");
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return ok;
}
int main(int argc,char ** argv) {
    if(argc!=3||(std::strcmp(argv[1],"--write-reference")&&std::strcmp(argv[1],"--compare-reference")))return 2;
    const bool write=!std::strcmp(argv[1],"--write-reference");
    std::unique_ptr<FILE,decltype(&std::fclose)> f(std::fopen(argv[2],write?"wb":"rb"),&std::fclose);
    if(!f){std::perror(argv[2]);return 2;}
    bool ok=structural_keys();
    ggml_backend_load_all();const auto dev=ggml_backend_dev_by_name("SPACEMIT0");
    if(!dev)return 3;auto backend=ggml_backend_dev_init(dev,nullptr);if(!backend)return 3;
    trace_state trace;ggml_log_get(&trace.previous,&trace.opaque);ggml_log_set(capture,&trace);
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16}) {
        for(int n:{1,7,17,65,212,848})ok=run(backend,trace,f.get(),write,type,n,0)&&ok;
        for(int kind:{1,2,3})ok=run(backend,trace,f.get(),write,type,17,kind)&&ok;
    }
    if(!write&&std::fgetc(f.get())!=EOF)ok=false;
    ggml_backend_free(backend);ggml_log_set(trace.previous,trace.opaque);return ok?0:1;
}
