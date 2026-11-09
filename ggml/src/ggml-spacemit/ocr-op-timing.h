#pragma once
// Optional bounded timing, without graph callbacks, output reads or new
// barriers. Allocate on the host before launch; each worker owns one lane.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ocr_op_timing {
inline const char * path(){static const char*p=std::getenv("GGML_SPACEMIT_OP_TIMING_PATH");return p&&*p?p:nullptr;}
inline uint64_t now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct item {
    uint64_t sequence,start,compute_end,barrier_end;
    uint32_t index;
    int op,opcode,type0,type1;
    int64_t shape0[4],shape1[4];
    char name[64];
};
struct storage {
    static constexpr unsigned capacity=8192;
    struct alignas(128) lane {unsigned used=0;item data[capacity];};
    lane lanes[8];std::atomic<uint64_t> next{0};
    storage(){
        // This constructor runs on the host before launch. Touch every lane's
        // backing page here rather than first faulting it on an AI worker.
        for(auto & l:lanes)std::memset(l.data,0,sizeof(l.data));
        std::atexit(dump);
    }
    static storage& get(){static storage*p=new storage;return *p;}
    static void dump(){
        auto&s=get();FILE*f=std::fopen(path(),"w");if(!f)return;
        for(unsigned w=0;w<8;++w){
            const auto&lane=s.lanes[w];
            std::fprintf(f,"{\"kind\":\"lane\",\"worker\":%u,\"used\":%u,\"dropped\":%u}\n",w,lane.used,lane.used>capacity?lane.used-capacity:0);
            for(unsigned i=0;i<lane.used&&i<capacity;++i){
                const auto&r=lane.data[i];
                std::fprintf(f,"{\"kind\":\"op\",\"worker\":%u,\"sequence\":%llu,\"index\":%u,\"name\":\"%s\",\"op\":\"%s\",\"opcode\":%d,\"type0\":\"%s\",\"type1\":\"%s\",\"shape0\":[%lld,%lld,%lld,%lld],\"shape1\":[%lld,%lld,%lld,%lld],\"start_ns\":%llu,\"compute_end_ns\":%llu,\"barrier_end_ns\":%llu}\n",
                    w,(unsigned long long)r.sequence,r.index,r.name,ggml_op_name(static_cast<ggml_op>(r.op)),r.opcode,
                    r.type0<0?"none":ggml_type_name(static_cast<ggml_type>(r.type0)),r.type1<0?"none":ggml_type_name(static_cast<ggml_type>(r.type1)),
                    (long long)r.shape0[0],(long long)r.shape0[1],(long long)r.shape0[2],(long long)r.shape0[3],
                    (long long)r.shape1[0],(long long)r.shape1[1],(long long)r.shape1[2],(long long)r.shape1[3],
                    (unsigned long long)r.start,(unsigned long long)r.compute_end,(unsigned long long)r.barrier_end);
            }
        }
        std::fclose(f);
    }
};
inline void record(unsigned worker,uint64_t sequence,unsigned index,const ggml_tensor*op,
                   int opcode,uint64_t start,uint64_t compute_end,uint64_t barrier_end){
    if(worker>=8)return;
    auto&lane=storage::get().lanes[worker];const auto i=lane.used++;
    if(i>=storage::capacity)return;
    auto&r=lane.data[i];r.sequence=sequence;r.index=index;r.op=int(op->op);r.opcode=opcode;
    r.start=start;r.compute_end=compute_end;r.barrier_end=barrier_end;
    r.type0=op->src[0]?int(op->src[0]->type):-1;r.type1=op->src[1]?int(op->src[1]->type):-1;
    for(int j=0;j<4;++j){r.shape0[j]=op->src[0]?op->src[0]->ne[j]:0;r.shape1[j]=op->src[1]?op->src[1]->ne[j]:0;}
    std::snprintf(r.name,sizeof(r.name),"%s",op->name);
    for(char*p=r.name;*p;++p)if(*p=='"'||*p=='\\'||static_cast<unsigned char>(*p)<32)*p='_';
}
}
