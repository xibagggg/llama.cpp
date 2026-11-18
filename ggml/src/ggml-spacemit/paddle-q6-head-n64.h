#pragma once
// Admission and optional evidence only. Reuses the existing activation quantizer,
// existing Q6->Q8 physical weights, and paired staging. No prepared-A cache.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace paddle_q6_head_n64 {
inline constexpr size_t required_tcm_bytes=64*1088+1216;
inline bool requested() {
    static const bool value=[] {
        const char *p=std::getenv("GGML_SPACEMIT_PADDLE_Q6_HEAD_N64");
        return p&&!std::strcmp(p,"1");
    }();
    return value;
}
inline bool eligible(int64_t workers,size_t vlenb,int64_t m,int64_t k,int64_t n,
                     const char*name,size_t row_a,size_t row_b,size_t common_tcm) {
    return requested()&&workers==6&&vlenb==128&&m==1&&k==1024&&n==103424&&
        name&&(!std::strcmp(name,"output.weight")||!std::strcmp(name,"lm_head.weight"))&&
        row_a==1216&&row_b==1088&&common_tcm>=required_tcm_bytes;
}
struct evidence {
    const char *path=nullptr;
    std::atomic<uint64_t> tiles[6]{},columns[6]{},fallbacks[6]{};
    evidence() {
        path=std::getenv("GGML_SPACEMIT_PADDLE_Q6_HEAD_N64_TRACE");
        if(path&&*path)std::atexit(&dump);
    }
    static evidence&instance() {static evidence*p=new evidence;return *p;}
    static void dump() {
        auto&v=instance();if(!v.path||!*v.path)return;
        FILE*f=std::fopen(v.path,"w");if(!f)return;
        std::fprintf(f,"{\"schema\":1,\"requested\":%s,\"workers\":6,\"k\":1024,\"n\":103424,\"lanes\":[",requested()?"true":"false");
        for(int i=0;i<6;++i)std::fprintf(f,"%s{\"worker\":%d,\"tiles\":%llu,\"columns\":%llu,\"fallbacks\":%llu}",
            i?",":"",i,(unsigned long long)v.tiles[i].load(),(unsigned long long)v.columns[i].load(),
            (unsigned long long)v.fallbacks[i].load());
        std::fprintf(f,"]}\n");std::fclose(f);
    }
};
inline evidence*trace() {
    auto&v=evidence::instance();return v.path&&*v.path?&v:nullptr;
}
inline void record_tile(evidence*p,size_t worker,size_t cols,bool selected) {
    if(!p||worker>=6)return;
    if(selected){p->tiles[worker].fetch_add(1,std::memory_order_relaxed);p->columns[worker].fetch_add(cols,std::memory_order_relaxed);}
    else p->fallbacks[worker].fetch_add(1,std::memory_order_relaxed);
}
} // namespace paddle_q6_head_n64
