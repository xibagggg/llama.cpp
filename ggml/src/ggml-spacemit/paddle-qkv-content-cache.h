#pragma once
// Exact-content cache for Paddle F16 Q/K/V activation packs. One owner per
// backend session; host preparation and all launches are externally serialized.
// No input-address/graph-UID key, worker allocation, hashing or TCM ownership.
#include <algorithm>
#include <cassert>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace paddle_qkv_content_cache {
constexpr size_t limit_bytes=64u*1024u*1024u;
constexpr size_t guard_bytes=64;
constexpr uint8_t guard_value=0xa5;
inline bool enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_PADDLE_F16_QKV_CONTENT_CACHE");return p&&!std::strcmp(p,"1");}();
    return on;
}
inline bool trace_enabled() {
    static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_PADDLE_F16_QKV_CACHE_TRACE");return p&&!std::strcmp(p,"1");}();
    return on;
}
inline bool weight_name(const char *name) {
    if(!name||std::strncmp(name,"v.blk.",6))return false;
    const char*p=name+6;
    if(*p<'0'||*p>'9')return false;
    do{++p;}while(*p>='0'&&*p<='9');
    if(*p++!='.')return false;
    return !std::strcmp(p,"attn_q.weight")||!std::strcmp(p,"attn_k.weight")||!std::strcmp(p,"attn_v.weight");
}
inline uint32_t rounding_mode() {
#if defined(__riscv)
    uint32_t value;
    asm volatile("frrm %0":"=r"(value));
    return value;
#else
    return uint32_t(std::fegetround());
#endif
}
inline size_t pad64(size_t value){return (value+63)&~size_t(63);}
struct alignas(64) lane_stats {uint64_t hit=0,miss=0;};

// Every group is written only by g%nth; all consumers wait at the unchanged
// activation-pack barrier. Do not retain this view beyond its owner/launch.
struct view {
    void *packed=nullptr;
    uint8_t *snapshot=nullptr,*valid=nullptr;
    uint32_t *rounding=nullptr;
    lane_stats *stats=nullptr;
    int m=0,k=0,nth=0,groups=0;
    size_t pack_bytes=0;
    bool matches(int M,int K,int workers)const {
        return packed&&snapshot&&valid&&rounding&&m==M&&k==K&&nth==workers;
    }
    size_t source_group_bytes(int group)const {
        assert(group>=0&&group<groups);
        return size_t(std::min(8,m-group*8))*size_t(k)*sizeof(float);
    }
    size_t source_group_offset(int group)const{return size_t(group)*8*size_t(k)*sizeof(float);}
    bool hit(int group,const void *source,uint32_t frm,unsigned worker)const {
        assert(source&&group>=0&&group<groups&&worker<unsigned(nth)&&group%nth==int(worker));
        const size_t offset=source_group_offset(group),bytes=source_group_bytes(group);
        const bool same=valid[group]&&rounding[group]==frm&&
            !std::memcmp(snapshot+offset,static_cast<const uint8_t*>(source)+offset,bytes);
        if(stats){if(same)++stats[worker].hit;else ++stats[worker].miss;}
        // A failed comparison must never leave a partly refreshed pack valid.
        if(!same)valid[group]=0;
        return same;
    }
    void publish(int group,const void *source,uint32_t frm,unsigned worker)const {
        assert(source&&group>=0&&group<groups&&worker<unsigned(nth)&&group%nth==int(worker));
        const size_t offset=source_group_offset(group);
        std::memcpy(snapshot+offset,static_cast<const uint8_t*>(source)+offset,source_group_bytes(group));
        rounding[group]=frm;
        valid[group]=1; // Pack and scale stores precede publication and barrier.
    }
};

class owner {
    void *allocation_=nullptr;
    uint8_t *base_=nullptr;
    size_t bytes_=0,pack_bytes_=0,snapshot_span_=0;
    int m_=0,k_=0,nth_=0;
    std::vector<uint8_t> valid_;
    std::vector<uint32_t> rounding_;
    std::vector<lane_stats> stats_;
    view view_{};
public:
    owner()=default;
    owner(const owner&)=delete;
    owner&operator=(const owner&)=delete;
    ~owner(){std::free(allocation_);}
    void invalidate()noexcept{std::fill(valid_.begin(),valid_.end(),uint8_t(0));}
    bool prepare(int M,int K,int workers,bool trace) {
        // K is fixed before multiplying. The source/pack cap also bounds M and
        // all following integer/size arithmetic, including 32-bit host builds.
        if(M<1||K!=1152||workers<1||workers>8||size_t(M)>limit_bytes/(size_t(K)*8)) {
            invalidate();return false;
        }
        const size_t groups=(size_t(M)+7)/8;
        const size_t pack=pad64(groups*8*size_t(K)*4)+pad64(groups*4);
        const size_t snapshot=pad64(size_t(M)*size_t(K)*sizeof(float));
        const size_t total=3*guard_bytes+pack+snapshot;
        if(total>limit_bytes){invalidate();return false;}
        const bool changed=M!=m_||K!=k_||workers!=nth_||!base_;
        try {
            if(changed) {
                // Allocate metadata before replacing live storage; any failure
                // leaves existing storage invalid and falls back to ordinary pack.
                std::vector<uint8_t> valid(groups,0);
                std::vector<uint32_t> rounding(groups,0);
                std::vector<lane_stats> stats(static_cast<size_t>(workers));
                void *fresh=std::malloc(total+63);
                if(!fresh){invalidate();return false;}
                auto *aligned=reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(fresh)+63)&~uintptr_t(63));
                // Host prefault/initialization, before AI workers can run.
                std::memset(aligned,0,total);
                std::memset(aligned,guard_value,guard_bytes);
                std::memset(aligned+guard_bytes+pack,guard_value,guard_bytes);
                std::memset(aligned+2*guard_bytes+pack+snapshot,guard_value,guard_bytes);
                std::free(allocation_);allocation_=fresh;base_=aligned;
                bytes_=total;pack_bytes_=pack;snapshot_span_=snapshot;
                m_=M;k_=K;nth_=workers;
                valid_.swap(valid);rounding_.swap(rounding);stats_.swap(stats);
            }
            for(auto&s:stats_)s={};
            view_={base_+guard_bytes,base_+2*guard_bytes+pack_bytes_,valid_.data(),rounding_.data(),
                   trace?stats_.data():nullptr,m_,k_,nth_,int(groups),pack_bytes_};
            return true;
        }catch(...){invalidate();return false;}
    }
    view *get(){return base_?&view_:nullptr;}
    const view *get()const{return base_?&view_:nullptr;}
    bool guards_ok()const {
        if(!base_)return true;
        for(size_t i=0;i<guard_bytes;++i)
            if(base_[i]!=guard_value||base_[guard_bytes+pack_bytes_+i]!=guard_value||
               base_[2*guard_bytes+pack_bytes_+snapshot_span_+i]!=guard_value)return false;
        return true;
    }
    size_t storage_bytes()const{return bytes_;}
    uint64_t hits()const{uint64_t n=0;for(const auto&s:stats_)n+=s.hit;return n;}
    uint64_t misses()const{uint64_t n=0;for(const auto&s:stats_)n+=s.miss;return n;}
    void trace()const {
        std::fprintf(stderr,"qkv-content-cache: M=%d K=%d workers=%d groups_hit=%llu groups_miss=%llu storage_bytes=%zu guards=%d\n",
            m_,k_,nth_,static_cast<unsigned long long>(hits()),static_cast<unsigned long long>(misses()),bytes_,int(guards_ok()));
    }
};

// Invalidate on every early return, runtime failure or C++ exception. A success
// preserves content across graph splits; every subsequent use rechecks bytes.
class completion {
    owner &owner_;
    bool committed_=false;
public:
    explicit completion(owner&o):owner_(o){}
    completion(const completion&)=delete;
    completion&operator=(const completion&)=delete;
    ~completion(){if(!committed_)owner_.invalidate();}
    void commit()noexcept{committed_=true;}
};
} // namespace paddle_qkv_content_cache
