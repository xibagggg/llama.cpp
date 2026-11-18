#pragma once
// Session-owned angles, never keyed by graph UID or positions address alone.
// Access is serialized by spacemit_session::stream_mutex. Entries selected in
// a launch are pinned until the next launch, so LRU eviction cannot dangle a
// worker's immutable table pointer. No worker allocates, frees or mutates it.
#include "ggml.h"
#include "ggml-impl.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <new>
#include <vector>
#include <utility>

namespace paddle_vision_rope {
inline bool content_enabled() {
    const char * p=std::getenv("GGML_SPACEMIT_VISION_ROPE_CONTENT_CACHE");
    return p&&!std::strcmp(p,"1");
}
inline bool content_trace_enabled() {
    const char * p=std::getenv("GGML_SPACEMIT_VISION_ROPE_CONTENT_TRACE");
    return p&&!std::strcmp(p,"1");
}
struct content_entry {
    std::array<unsigned char,GGML_MAX_OP_PARAMS> params{};
    std::array<int64_t,GGML_MAX_DIMS> positions_ne{};
    std::array<size_t,GGML_MAX_DIMS> positions_nb{};
    std::vector<unsigned char> positions;
    // Sixteen F32 guard words before and after the actual immutable table.
    std::vector<float> storage;
    int64_t width=0,tokens=0;
    ggml_type type=GGML_TYPE_F32;
    bool planar=false;
    size_t charge=0;
    uint64_t pinned=0,used=0;
    static constexpr size_t guard=16;
    static constexpr uint32_t guard_word=0x4b7ac931;
    const float * values() const {return storage.data()+guard;}
    bool guards_ok() const {
        if(storage.size()<2*guard)return false;
        for(size_t i=0;i<guard;++i) {
            uint32_t a,b;std::memcpy(&a,&storage[i],4);
            std::memcpy(&b,&storage[storage.size()-guard+i],4);
            if(a!=guard_word||b!=guard_word)return false;
        }
        return true;
    }
    bool geometry(const ggml_tensor * op,bool layout) const {
        const auto * pos=op->src[1];
        // Head count does not enter angle geometry; type/layout, D and token
        // count do. Retain every position-axis byte, including unused axes.
        return planar==layout&&width==op->ne[0]&&tokens==op->ne[2]&&type==op->type&&
            !std::memcmp(params.data(),op->op_params,sizeof(op->op_params))&&
            !std::memcmp(positions_ne.data(),pos->ne,sizeof(pos->ne))&&
            !std::memcmp(positions_nb.data(),pos->nb,sizeof(pos->nb))&&
            positions.size()==ggml_nbytes(pos);
    }
};
struct content_cache {
    static constexpr size_t budget=16*1024*1024;
    static constexpr size_t max_entries=16;
    std::list<content_entry> entries;
    size_t bytes=0,hits=0,misses=0,inserts=0,evictions=0,rejected_writes=0;
    uint64_t launch=0;
    void begin_launch() {
        if(launch==UINT64_MAX){entries.clear();bytes=0;launch=0;}
        ++launch;hits=misses=inserts=evictions=rejected_writes=0;
    }
    // The caller must first apply the fixed VISION policy and reject every
    // overlapping write in the raw graph. This method never replaces that
    // proof with a hash or with remembered graph/tensor/data addresses.
    const float * find(const ggml_tensor * op,bool planar) {
        const auto * pos=op->src[1];
        for(auto & e:entries)if(e.geometry(op,planar)&&e.guards_ok()&&
            !std::memcmp(e.positions.data(),pos->data,e.positions.size())) {
            e.pinned=launch;e.used=launch;++hits;return e.values();
        }
        ++misses;return nullptr;
    }
    void store(const ggml_tensor * op,bool planar,const std::vector<float> & values) {
        const auto * pos=op->src[1];
        const size_t pbytes=ggml_nbytes(pos);
        if(values.size()>SIZE_MAX/sizeof(float)-2*content_entry::guard)return;
        const size_t vbytes=(values.size()+2*content_entry::guard)*sizeof(float);
        if(pbytes>budget||vbytes>budget-pbytes||sizeof(content_entry)>budget-pbytes-vbytes)return;
        const size_t charge=pbytes+vbytes+sizeof(content_entry);
        while(bytes>budget-charge||entries.size()>=max_entries) {
            auto victim=entries.end();
            for(auto it=entries.begin();it!=entries.end();++it)
                if(it->pinned!=launch&&(victim==entries.end()||it->used<victim->used))victim=it;
            if(victim==entries.end())return; // Every remaining entry is in use.
            bytes-=victim->charge;entries.erase(victim);++evictions;
        }
        try {
            content_entry e;
            e.width=op->ne[0];e.tokens=op->ne[2];e.type=op->type;e.planar=planar;
            e.charge=charge;e.pinned=launch;e.used=launch;
            std::memcpy(e.params.data(),op->op_params,sizeof(op->op_params));
            std::memcpy(e.positions_ne.data(),pos->ne,sizeof(pos->ne));
            std::memcpy(e.positions_nb.data(),pos->nb,sizeof(pos->nb));
            e.positions.resize(pbytes);std::memcpy(e.positions.data(),pos->data,pbytes);
            e.storage.resize(values.size()+2*content_entry::guard);
            for(size_t i=0;i<content_entry::guard;++i) {
                std::memcpy(&e.storage[i],&content_entry::guard_word,4);
                std::memcpy(&e.storage[e.storage.size()-content_entry::guard+i],&content_entry::guard_word,4);
            }
            std::memcpy(e.storage.data()+content_entry::guard,values.data(),values.size()*sizeof(float));
            entries.push_back(std::move(e));bytes+=charge;++inserts;
        } catch(const std::bad_alloc &) {
            // The launch still owns its newly generated uncached table.
        }
    }
    bool guards_ok() const {for(const auto & e:entries)if(!e.guards_ok())return false;return true;}
    void trace() const {
        if(content_trace_enabled())GGML_LOG_INFO(
            "paddle-vision-rope-content: launch=%llu hits=%zu misses=%zu inserts=%zu evictions=%zu rejected_writes=%zu entries=%zu bytes=%zu guards=%d\n",
            (unsigned long long)launch,hits,misses,inserts,evictions,rejected_writes,entries.size(),bytes,int(guards_ok()));
    }
};
} // namespace paddle_vision_rope
