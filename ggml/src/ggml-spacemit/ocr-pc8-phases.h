#pragma once
// Diagnostic only. Separate from generic Q4 phases: LLM decode must not exhaust
// the bounded Vision PC8 lanes. No math, output reads or extra barriers.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ocr_pc8_phases {
inline const char*path(){
    static const char*p=std::getenv("GGML_SPACEMIT_OCR_PC8_PHASE_PATH");
    return p&&*p?p:nullptr;
}
inline uint64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();}
struct record{
    uint64_t start,packed,synced,end;
    int64_t k,n,m;
    int producer,reused,m8;
    char weight[96];
};
struct storage{
    static constexpr unsigned workers=6,capacity=2048;
    struct alignas(128)lane{unsigned used=0;record records[capacity]{};};
    lane lanes[workers];
    storage(){std::atexit(dump);}
    static storage&get(){static auto*p=new storage;return *p;}
    static void dump(){
        FILE*f=std::fopen(path(),"w");if(!f)return;
        const auto&s=get();
        for(unsigned w=0;w<workers;++w){const auto&l=s.lanes[w];
            std::fprintf(f,"{\"kind\":\"lane\",\"worker\":%u,\"used\":%u,\"dropped\":%u}\n",
                w,l.used,l.used>capacity?l.used-capacity:0);
            for(unsigned i=0;i<l.used&&i<capacity;++i){const auto&r=l.records[i];
                // Weight names admitted by this path are canonical ASCII GGUF
                // v.blk.* names, containing no quotes or escape characters.
                std::fprintf(f,"{\"kind\":\"pc8_linear\",\"worker\":%u,\"ordinal\":%u,"
                    "\"K\":%lld,\"N\":%lld,\"M\":%lld,\"producer\":%d,\"reused\":%d,\"m8\":%d,"
                    "\"weight\":\"%s\",\"start_ns\":%llu,\"packed_ns\":%llu,\"synced_ns\":%llu,\"end_ns\":%llu}\n",
                    w,i,(long long)r.k,(long long)r.n,(long long)r.m,r.producer,r.reused,r.m8,r.weight,
                    (unsigned long long)r.start,(unsigned long long)r.packed,
                    (unsigned long long)r.synced,(unsigned long long)r.end);
            }
        }
        std::fclose(f);
    }
};
inline void prepare_on_host(){if(path())(void)storage::get();}
struct scope{
    bool enabled;unsigned worker;record r;
    scope(unsigned ith,int64_t k,int64_t n,int64_t m,bool producer,bool reused,const char*weight):
        enabled(path()!=nullptr&&ith<storage::workers),worker(ith){
        if(enabled){r.k=k;r.n=n;r.m=m;r.producer=producer;r.reused=reused;r.m8=0;
            std::snprintf(r.weight,sizeof(r.weight),"%s",weight?weight:"");r.start=now();}
    }
    void packed(){if(enabled)r.packed=now();}
    void synced(bool m8){if(enabled){r.synced=now();r.m8=m8;}}
    ~scope(){if(!enabled)return;r.end=now();auto&l=storage::get().lanes[worker];
        const auto i=l.used++;if(i<storage::capacity)l.records[i]=r;}
};
} // namespace ocr_pc8_phases
