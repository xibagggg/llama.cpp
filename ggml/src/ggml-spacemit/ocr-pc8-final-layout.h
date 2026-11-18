#pragma once
// Host-only byte permutation. The native VLEN128 prepack oracle must pass
// before this default-off physical layout is used for inference.
#include <algorithm>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ocr_pc8_final_layout {
constexpr size_t header_bytes=128;
constexpr size_t block_bytes=1024;
constexpr unsigned max_load_threads=6;

inline bool eligible(bool enabled,int k,int n,int rounding,int load_threads) {
    return enabled && rounding==FE_TONEAREST && load_threads>=1 &&
        load_threads<=int(max_load_threads) &&
        ((k==1152&&(n==1152||n==4304))||(k==4352&&n==1152));
}

// Each N8/K32 group is split into two consecutive N8/K16 groups.
// This is a byte operation: no conversion, FP environment change or RVV.
inline void permute_panel(int8_t *panel,int k,uint8_t *scratch) {
    for(int kb=0;kb<k/32;++kb) {
        auto *block=reinterpret_cast<uint8_t*>(panel)+size_t(kb)*block_bytes;
        std::memcpy(scratch,block,block_bytes);
        for(int group=0;group<4;++group)for(int half=0;half<2;++half)
            for(int row=0;row<8;++row)
                std::memcpy(block+group*256+half*128+row*16,
                    scratch+group*256+row*32+half*16,16);
    }
}

inline void put_u32(uint8_t *p,uint32_t x) {
    for(unsigned i=0;i<4;++i)p[i]=uint8_t(x>>(8*i));
}
inline void make_header(uint8_t *p,int k,int n) {
    std::memset(p,0,header_bytes);
    constexpr uint8_t signature[16]={'P','C','8','_','V','U','P','1','_','N','3','2','_','V','1',0};
    std::memcpy(p,signature,sizeof(signature));
    put_u32(p+16,uint32_t(k));put_u32(p+20,uint32_t(n));
    put_u32(p+24,uint32_t((n+31)/32*32));
    put_u32(p+28,128); // Required native vector-register bytes.
}
inline void write_header(void *p,int k,int n) {
    make_header(static_cast<uint8_t*>(p),k,n);
}
inline bool valid_header(const void *p,int k,int n,bool final_layout) {
    if(!p)return false;
    uint8_t expected[header_bytes]{};
    if(final_layout)make_header(expected,k,n);
    return !std::memcmp(p,expected,header_bytes);
}
} // namespace ocr_pc8_final_layout
