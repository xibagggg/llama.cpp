#pragma once
// Diagnostic only: three timestamps around the existing packing and grid
// barrier, with no output reads, callbacks or additional synchronization.
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace ocr_quant_phases {
inline const char * path() {
    static const char * p=std::getenv("GGML_SPACEMIT_OCR_Q4_PHASE_PATH");
    return p&&*p?p:nullptr;
}
inline uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct record {
    uint64_t start,packed,synced,end;
    int64_t k,n,m;
    int type,block,m8;
};
struct storage {
    static constexpr unsigned capacity=2048;
    struct alignas(128) lane {unsigned used=0;record records[capacity]{};};
    lane lanes[8];
    storage(){std::atexit(dump);}
    static storage & get(){static auto * p=new storage;return *p;}
    static void dump(){
        FILE * f=std::fopen(path(),"w");if(!f)return;
        auto & s=get();
        for(unsigned w=0;w<8;++w) {
            const auto & l=s.lanes[w];
            std::fprintf(f,"{\"kind\":\"lane\",\"worker\":%u,\"used\":%u,\"dropped\":%u}\n",
                         w,l.used,l.used>capacity?l.used-capacity:0);
            for(unsigned i=0;i<l.used&&i<capacity;++i) {
                const auto & r=l.records[i];
                std::fprintf(f,"{\"kind\":\"linear\",\"worker\":%u,\"ordinal\":%u,"
                    "\"K\":%lld,\"N\":%lld,\"M\":%lld,\"type\":%d,\"block\":%d,\"m8\":%d,"
                    "\"start_ns\":%llu,\"packed_ns\":%llu,\"synced_ns\":%llu,\"end_ns\":%llu}\n",
                    w,i,(long long)r.k,(long long)r.n,(long long)r.m,r.type,r.block,r.m8,
                    (unsigned long long)r.start,(unsigned long long)r.packed,
                    (unsigned long long)r.synced,(unsigned long long)r.end);
            }
        }
        std::fclose(f);
    }
};
// The first allocation, page writes and atexit registration must precede
// graph dispatch. Workers only append to their already initialized lanes.
inline void prepare_on_host() {
    if(path())(void)storage::get();
}
struct scope {
    bool enabled=false;int worker=0;record r{};
    scope(int ith,int64_t k,int64_t n,int64_t m,int type,int block):
        enabled(path()!=nullptr&&ith>=0&&ith<8),worker(ith) {
        if(enabled){storage::get();r.k=k;r.n=n;r.m=m;r.type=type;r.block=block;r.start=now();}
    }
    void packed(){if(enabled)r.packed=now();}
    void synced(bool m8){if(enabled){r.synced=now();r.m8=m8;}}
    ~scope(){
        if(!enabled)return;
        r.end=now();
        auto & l=storage::get().lanes[worker];const auto i=l.used++;
        if(i<storage::capacity)l.records[i]=r;
    }
};
}
