#pragma once
// Opt-in diagnostic. No additional grid barrier, file I/O or allocation in
// the measured loops. PROFILE=false instantiations compile out every clock.
#include "ggml.h"
#include "spacemit-context.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace ocr_fa_phases_v50 {
inline const char* directory(){static const char *p=std::getenv("GGML_SPACEMIT_OCR_FA_PHASES");return p&&*p?p:nullptr;}
inline bool enabled(){return directory()!=nullptr;}
inline uint64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct item{char name[64]{},stage[12]{};uint64_t start=0,end=0,ns[11]{};int64_t n=0;unsigned worker=0;};
struct storage{
    static constexpr unsigned capacity=1024;
    unsigned used[8]{};item data[8][capacity]{};
    storage(){std::atexit(dump);}
    static storage& get(){static storage *p=new storage;return *p;}
    static void dump(){
        auto &s=get();char path[4096];std::snprintf(path,sizeof(path),"%s/attention-phases.jsonl",directory());
        FILE*f=std::fopen(path,"w");if(!f)return;
        const char* names[]={"shared_pack","prepare_barrier_wait","query_init_pack","mask_bookkeeping","K_copy","QK_compute","softmax_rescale","V_copy","PV_compute","output_store","final_barrier_wait"};
        for(unsigned w=0;w<8;++w){
            std::fprintf(f,"{\"kind\":\"lane\",\"worker\":%u,\"used\":%u,\"dropped\":%u}\n",w,s.used[w],s.used[w]>capacity?s.used[w]-capacity:0);
            for(unsigned i=0;i<s.used[w]&&i<capacity;++i){const auto&r=s.data[w][i];
                std::fprintf(f,"{\"kind\":\"phase\",\"name\":\"%s\",\"stage\":\"%s\",\"N\":%lld,\"worker\":%u,\"start_ns\":%llu,\"end_ns\":%llu",r.name,r.stage,(long long)r.n,w,(unsigned long long)r.start,(unsigned long long)r.end);
                for(int j=0;j<11;++j)std::fprintf(f,",\"%s_ns\":%llu",names[j],(unsigned long long)r.ns[j]);
                std::fprintf(f,"}\n");
            }
        }
        std::fclose(f);
    }
};
struct record{
    item r{};uint64_t previous=0;const ggml::spacemit::context&ctx;const ggml_tensor*op;const char*stage;
    record(const ggml::spacemit::context&c,const ggml_tensor*t,const char*s):ctx(c),op(t),stage(s){}
    void begin(){storage::get();r.worker=ctx.ith;r.n=op->src[0]->ne[1];
        std::snprintf(r.name,sizeof(r.name),"%s",op->name);std::snprintf(r.stage,sizeof(r.stage),"%s",stage);previous=r.start=now();}
    void tick(int i){auto t=now();r.ns[i]+=t-previous;previous=t;}
    void finish(){r.end=now();auto&s=storage::get();GGML_ASSERT(r.worker<8);auto i=s.used[r.worker]++;if(i<storage::capacity)s.data[r.worker][i]=r;}
};
}
