// Host-only load arithmetic and physical-identity admission test. Native byte
// mapping and kernels require the separate six-worker raw executable.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfenv>
#include <array>
#include <vector>
#include <cstring>
#include <cstdint>
#define GGML_ASSERT(x) do{if(!(x)){std::fprintf(stderr,"assert %s:%d %s\n",__FILE__,__LINE__,#x);std::abort();}}while(0)
struct block_q4_0 {uint16_t d;uint8_t qs[16];};
#include "ocr-pc8-load.h"
#include "test-pc8-final-layout-load-reference.h"
#include "ocr-pc8-final-layout.h"
static uint32_t random_word(uint32_t&x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
static float decode(uint16_t h) {
    uint32_t sign=uint32_t(h&0x8000)<<16,exp=(h>>10)&31,mant=h&1023,bits;
    if(!exp) {
        if(!mant)bits=sign;
        else {
            unsigned shift=0;while(!(mant&1024)){mant<<=1;++shift;}
            bits=sign|((113-shift)<<23)|((mant&1023)<<13);
        }
    }else if(exp==31)bits=sign|0x7f800000|(mant<<13);
    else bits=sign|((exp+112)<<23)|(mant<<13);
    float f;std::memcpy(&f,&bits,4);return f;
}
static void undo_panel(int8_t *panel,int k,uint8_t*scratch) {
    for(int kb=0;kb<k/32;++kb) {
        auto*p=reinterpret_cast<uint8_t*>(panel)+size_t(kb)*1024;
        std::memcpy(scratch,p,1024);
        for(int row=0;row<32;++row)for(int col=0;col<32;++col) {
            const int offset=(row/8)*256+(col/16)*128+(row%8)*16+(col%16);
            p[row*32+col]=scratch[offset];
        }
    }
}
int main() {
    size_t policy=0;
    for(bool on:{false,true})for(int k:{0,64,1151,1152,1153,4304,4352})
    for(int n:{0,1,1152,4304,4352})for(int frm:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO})
    for(int threads:{0,1,2,3,4,5,6,7,8,9}) {
        const bool expected=on&&frm==FE_TONEAREST&&threads>=1&&threads<=6&&
            ((k==1152&&(n==1152||n==4304))||(k==4352&&n==1152));
        GGML_ASSERT(ocr_pc8_final_layout::eligible(on,k,n,frm,threads)==expected);++policy;
    }
    std::array<uint8_t,128> header{};
    GGML_ASSERT(ocr_pc8_final_layout::valid_header(header.data(),1152,4304,false));
    GGML_ASSERT(!ocr_pc8_final_layout::valid_header(header.data(),1152,4304,true));
    ocr_pc8_final_layout::write_header(header.data(),1152,4304);
    GGML_ASSERT(ocr_pc8_final_layout::valid_header(header.data(),1152,4304,true));
    GGML_ASSERT(!ocr_pc8_final_layout::valid_header(header.data(),4352,1152,true));
    GGML_ASSERT(!ocr_pc8_final_layout::valid_header(header.data(),1152,4304,false));
    for(size_t i=0;i<header.size();++i) {
        header[i]^=1;GGML_ASSERT(!ocr_pc8_final_layout::valid_header(header.data(),1152,4304,true));header[i]^=1;
    }
    const int old=std::fegetround();size_t cases=0;
    struct shape {int k,n;};
    for(shape s: {shape{64,1},shape{64,31},shape{64,33},shape{1152,1152},shape{1152,4304},shape{4352,1152}})
    for(unsigned pattern=0;pattern<2;++pattern) {
        const int NP=(s.n+31)/32*32;
        std::vector<block_q4_0> source(size_t(s.k/32)*s.n);uint32_t seed=0x176ade3+pattern;
        for(size_t i=0;i<source.size();++i) {
            auto&v=source[i];
            v.d=uint16_t(random_word(seed));
            if((v.d&0x7c00)==0x7c00)v.d^=0x0400;
            if(pattern==0) {
                constexpr uint16_t values[]={0,0x8000,0x0001,0x03ff,0x0400,0x3c00,0xbc00,0x7bff};
                v.d=values[i%8];
            }
            for(unsigned j=0;j<16;++j)v.qs[j]=pattern?uint8_t(random_word(seed)):uint8_t(i+j*17);
        }
        const auto immutable=source;
        const size_t bytes=128+size_t(NP)*(4+s.k);
        std::vector<uint64_t> reference((bytes+256+7)/8),candidate((bytes+256+7)/8);
        auto *ref=reinterpret_cast<uint8_t*>(reference.data())+128;
        auto *got=reinterpret_cast<uint8_t*>(candidate.data())+128;
        for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
            GGML_ASSERT(!std::fesetround(rounding));
            reference_ocr_pc8_load::repack(source.data(),ref,s.k,s.n,1,decode);
            for(int threads=1;threads<=6;++threads) {
                std::memset(candidate.data(),0xa5,candidate.size()*8);
                std::vector<std::array<uint8_t,1024>> scratch(static_cast<size_t>(threads));
                ocr_pc8_load::repack(source.data(),got,s.k,s.n,threads,decode,
                    [&](int8_t*p,int k,int worker){ocr_pc8_final_layout::permute_panel(p,k,scratch[size_t(worker)].data());});
                auto *weights=reinterpret_cast<int8_t*>(got+128+size_t(NP)*4);
                std::array<uint8_t,1024> inverse_scratch{};
                for(int group=0;group<NP/32;++group)undo_panel(weights+size_t(group)*32*s.k,s.k,inverse_scratch.data());
                GGML_ASSERT(!std::memcmp(ref,got,bytes));
                GGML_ASSERT(std::fegetround()==rounding);
                GGML_ASSERT(!std::memcmp(immutable.data(),source.data(),source.size()*sizeof(block_q4_0)));
                auto *guard=reinterpret_cast<const uint8_t*>(candidate.data());
                for(size_t i=0;i<128;++i)GGML_ASSERT(guard[i]==0xa5&&got[bytes+i]==0xa5);
                ++cases;
            }
        }
    }
    GGML_ASSERT(!std::fesetround(old));
    std::printf("PC8_FINAL_HOST policy=%zu load_cases=%zu four_host_roundings=1 threads_1_to_6=1 exact_legacy_arithmetic=1 physical_header=1 native_mapping_validated=0 PASS\n",policy,cases);
}
