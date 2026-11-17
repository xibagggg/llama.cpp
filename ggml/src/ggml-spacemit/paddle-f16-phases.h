#pragma once
// Bounded opt-in worker timing. Host initializes all storage before launch.
// The unprofiled template instantiation contains no clocks or phase writes.
#include "ggml.h"
#include "spacemit-context.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace paddle_f16_phases {
inline const char *directory(){static const char*p=std::getenv("GGML_SPACEMIT_PADDLE_F16_PHASES");return p&&*p?p:nullptr;}
inline bool enabled(){return directory()!=nullptr;}
inline uint64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct item {char name[64]{};uint64_t start=0,end=0,ns[10]{};int K=0,M=0,N=0,kblock=0,panels=0;unsigned worker=0;};
struct storage {
    static constexpr unsigned capacity=2048;
    unsigned used[8]{};item data[8][capacity]{};
    storage(){std::atexit(dump);}
    static storage& get(){static storage*p=new storage;return *p;}
    static void dump(){
        auto&s=get();char path[4096];std::snprintf(path,sizeof(path),"%s/paddle-f16-phases.jsonl",directory());
        FILE*f=std::fopen(path,"w");if(!f)return;
        const char*names[]={"activation_pack","pack_barrier","weight_copy","activation_copy","partial_reload","IME","partial_save","merge_scale","output_store","setup"};
        for(unsigned w=0;w<8;++w){
            std::fprintf(f,"{\"kind\":\"lane\",\"worker\":%u,\"used\":%u,\"dropped\":%u}\n",w,s.used[w],s.used[w]>capacity?s.used[w]-capacity:0);
            for(unsigned i=0;i<s.used[w]&&i<capacity;++i){const auto&r=s.data[w][i];
                std::fprintf(f,"{\"kind\":\"gemm_phase\",\"name\":\"%s\",\"worker\":%u,\"K\":%d,\"M\":%d,\"N\":%d,\"kblock\":%d,\"panels\":%d,\"start_ns\":%llu,\"end_ns\":%llu",r.name,w,r.K,r.M,r.N,r.kblock,r.panels,(unsigned long long)r.start,(unsigned long long)r.end);
                for(int j=0;j<10;++j)std::fprintf(f,",\"%s_ns\":%llu",names[j],(unsigned long long)r.ns[j]);
                std::fprintf(f,"}\n");
            }
        }
        std::fclose(f);
    }
};
struct record {
    item r{};uint64_t previous=0;
    void begin(const ggml::spacemit::context&c,const ggml_tensor*op){
        r.worker=c.ith;r.K=int(op->src[0]->ne[0]);r.M=int(op->ne[1]);r.N=int(op->ne[0]);
        std::snprintf(r.name,sizeof(r.name),"%s",op->name);
        for(char*p=r.name;*p;++p)if(*p=='"'||*p=='\\'||static_cast<unsigned char>(*p)<32)*p='_';
        previous=r.start=now();
    }
    void tick(int phase){const auto t=now();r.ns[phase]+=t-previous;previous=t;}
    void finish(){tick(9);r.end=previous;auto&s=storage::get();GGML_ASSERT(r.worker<8);const auto i=s.used[r.worker]++;if(i<storage::capacity)s.data[r.worker][i]=r;}
};
}
