// Runs only on exactly six registered VLEN128 AI workers. --probe recovers
// the complete native byte permutation using two independent index bytes.
#include "test-pc8-final-layout-exact.h"
#include "ocr-pc8-final-layout.h"
#include <spert.hpp>
#include <riscv_vector.h>
#include <sched.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

struct guarded {
    std::vector<uint64_t> storage;size_t bytes;
    explicit guarded(size_t n):storage((n+256+7)/8,UINT64_C(0xa5a5a5a5a5a5a5a5)),bytes(n){}
    void *data(){return reinterpret_cast<uint8_t*>(storage.data())+128;}
    const void *data()const{return reinterpret_cast<const uint8_t*>(storage.data())+128;}
    bool intact()const {
        const auto *p=reinterpret_cast<const uint8_t*>(storage.data());
        for(size_t i=0;i<128;++i)if(p[i]!=0xa5||p[128+bytes+i]!=0xa5)return false;
        return true;
    }
};
static uint32_t random_word(uint32_t&x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
static uint64_t hash_bytes(const void*p,size_t n){
    uint64_t value=UINT64_C(14695981039346656037);const auto*b=static_cast<const uint8_t*>(p);
    for(size_t i=0;i<n;++i)value=(value^b[i])*UINT64_C(1099511628211);
    return value;
}
int main(int argc,char **argv) {
    const bool probe=argc==2&&!std::strcmp(argv[1],"--probe");
    if(argc!=1&&!probe)return 2;
    const char*backend=std::getenv("SPERT_BACKEND");
    if(!backend||std::strcmp(backend,"spacemit-k3"))return 2;
    setenv("GGML_SPACEMIT_WORKERS","6",1);
    setenv("GGML_SPACEMIT_ALLOW_EIGHT_WORKERS","0",1);
    setenv("SPACEMIT_PERFER_CORE_ID","8,9,10,11,12,13",1);
    spert::StreamConfig config;config.n_cores=6;config.core_ids={8,9,10,11,12,13};
    spert::Stream stream(config);if(!stream.valid()||stream.core_count()!=6)return 3;
    if(probe) {
        std::array<std::array<uint8_t,1024>,2> native{};
        auto f=stream.launch(spert::Grid{6},[&](spert::Context *runtime) {
            const unsigned lane=runtime->program_id(0);
            if(lane>=6||sched_getcpu()!=8+int(lane)||__riscv_vlenb()!=128)
                throw std::runtime_error("wrong AI worker");
            if(lane)return;
            auto tcm=runtime->shared_buffer();if(!tcm||tcm.size<1280)throw std::runtime_error("missing TCM");
            auto *p=static_cast<uint8_t*>(tcm.data)+128;
            for(unsigned high=0;high<2;++high) {
                for(unsigned i=0;i<1024;++i)p[i]=uint8_t(i>>(8*high));
                pc8_native_reference::prepack_panel(reinterpret_cast<int8_t*>(p),32);
                std::memcpy(native[high].data(),p,1024);
            }
        });
        if(f.sync()!=spert::Status::Ok)return 4;
        std::array<bool,1024> seen{};
        std::printf("PC8_FINAL_MAPPING [");
        for(unsigned i=0;i<1024;++i) {
            const unsigned source=unsigned(native[0][i])|(unsigned(native[1][i])<<8);
            if(source>=1024||seen[source])return 1;
            seen[source]=true;std::printf("%s%u",i?",":"",source);
        }
        std::printf("]\nPC8_FINAL_MAPPING bytes=1024 bijection=1 native_vlen128=1 PASS\n");
        return 0;
    }
    size_t cases=0;
    for(int K:{64,1152,4352})for(unsigned pattern=0;pattern<6;++pattern)
    for(unsigned rounding=0;rounding<5;++rounding) {
        const size_t panel_bytes=size_t(32)*K;
        guarded source(panel_bytes*6),portable(panel_bytes*6),activation(size_t(8)*K*6),
            ordinary(1024*6),native_result(1024*6),portable_result(1024*6);
        auto *b=static_cast<uint8_t*>(source.data());uint32_t seed=0x176f1a2u+pattern;
        for(size_t i=0;i<panel_bytes*6;++i) {
            uint8_t value=uint8_t(random_word(seed));
            if(pattern==0)value=0;
            if(pattern==1)value=255;
            if(pattern==2)value=uint8_t(i);
            if(pattern==3)value=uint8_t(i>>8);
            if(pattern==4)value=uint8_t((i/32+i%32)%2?0x80:0x7f);
            b[i]=value;
        }
        auto*a=static_cast<uint8_t*>(activation.data());
        for(size_t i=0;i<size_t(8)*K*6;++i)a[i]=uint8_t(random_word(seed));
        const uint64_t before=hash_bytes(source.data(),panel_bytes*6);
        const uint64_t a_before=hash_bytes(activation.data(),size_t(8)*K*6);
        std::memcpy(portable.data(),source.data(),panel_bytes*6);
        std::array<uint8_t,1024> scratch{};
        for(unsigned lane=0;lane<6;++lane)
            ocr_pc8_final_layout::permute_panel(static_cast<int8_t*>(portable.data())+lane*panel_bytes,K,scratch.data());
        std::array<bool,6>entered{},bits{},dots4{},dots8{},frm{},canaries{};
        std::array<void*,6>shared{};std::array<size_t,6>shared_bytes{},vlens{};std::array<int,6>cpus{};
        auto f=stream.launch(spert::Grid{6},[&](spert::Context *runtime) {
            const unsigned lane=runtime->program_id(0);
            if(lane>=6)throw std::runtime_error("bad lane");
            cpus[lane]=sched_getcpu();vlens[lane]=__riscv_vlenb();
            if(cpus[lane]!=8+int(lane)||vlens[lane]!=128)
                throw std::runtime_error("wrong AI worker");
            auto tcm=runtime->shared_buffer();
            if(!tcm||tcm.size<panel_bytes+256)throw std::runtime_error("missing TCM");
            shared[lane]=tcm.data;shared_bytes[lane]=tcm.size;
            auto*guard=static_cast<uint8_t*>(tcm.data);auto*panel=reinterpret_cast<int8_t*>(guard+128);
            std::memset(guard,0xa5,128);std::memset(guard+128+panel_bytes,0xa5,128);
            auto *original=static_cast<const int8_t*>(source.data())+lane*panel_bytes;
            auto *expected=static_cast<const int8_t*>(portable.data())+lane*panel_bytes;
            auto *input=static_cast<const int8_t*>(activation.data())+size_t(lane)*8*K;
            auto *o=reinterpret_cast<int32_t*>(static_cast<uint8_t*>(ordinary.data())+lane*1024);
            auto *n=reinterpret_cast<int32_t*>(static_cast<uint8_t*>(native_result.data())+lane*1024);
            auto *p=reinterpret_cast<int32_t*>(static_cast<uint8_t*>(portable_result.data())+lane*1024);
            unsigned previous;asm volatile("frrm %0\n\tfsrm %1":"=&r"(previous):"r"(rounding):"memory");
            std::memcpy(panel,original,panel_bytes);
            pc8_native_reference::prepack_panel(panel,K);
            bits[lane]=!std::memcmp(panel,expected,panel_bytes);
            pc8_native_reference::kernel(input,original,o,K);
            pc8_native_reference::kernel_prepacked(input,panel,n,K);
            std::memcpy(panel,expected,panel_bytes);
            pc8_native_reference::kernel_prepacked(input,panel,p,K);
            dots4[lane]=!std::memcmp(o,n,512)&&!std::memcmp(o,p,512);
            pc8_native_reference::kernel8(input,input+4*K,original,o,K);
            pc8_native_reference::kernel8_prepacked(input,input+4*K,panel,p,K);
            std::memcpy(panel,original,panel_bytes);
            pc8_native_reference::prepack_panel(panel,K);
            pc8_native_reference::kernel8_prepacked(input,input+4*K,panel,n,K);
            dots8[lane]=!std::memcmp(o,n,1024)&&!std::memcmp(o,p,1024);
            unsigned after;asm volatile("frrm %0":"=r"(after));frm[lane]=after==rounding;
            asm volatile("fsrm %0"::"r"(previous):"memory");
            canaries[lane]=true;
            for(size_t i=0;i<128;++i)canaries[lane]=canaries[lane]&&guard[i]==0xa5&&guard[128+panel_bytes+i]==0xa5;
            entered[lane]=true;
        });
        const auto status=f.sync();bool lanes=true;
        for(unsigned lane=0;lane<6;++lane)
            lanes=lanes&&entered[lane]&&bits[lane]&&dots4[lane]&&dots8[lane]&&frm[lane]&&canaries[lane];
        const bool guards=source.intact()&&portable.intact()&&activation.intact()&&ordinary.intact()&&native_result.intact()&&portable_result.intact();
        const bool immutable=before==hash_bytes(source.data(),panel_bytes*6)&&a_before==hash_bytes(activation.data(),size_t(8)*K*6);
        const bool pass=status==spert::Status::Ok&&lanes&&guards&&immutable;
        if(K==1152&&pattern==0&&rounding==0)for(unsigned lane=0;lane<6;++lane)
            std::printf("PC8_FINAL_TCM lane=%u cpu=%d vlenb=%zu shared=%p shared_bytes=%zu panel=%p guarded_bytes=%zu canaries=%d\n",
                lane,cpus[lane],vlens[lane],shared[lane],shared_bytes[lane],static_cast<uint8_t*>(shared[lane])+128,panel_bytes+256,int(canaries[lane]));
        std::printf("PC8_FINAL_RAW K=%d pattern=%u frm=%u bytes=%d dot4=%d dot8=%d lanes6=%d guards=%d immutable=%d %s\n",
            K,pattern,rounding,int(lanes),int(lanes),int(lanes),int(lanes),int(guards),int(immutable),pass?"PASS":"FAIL");
        std::fflush(stdout);if(!pass)return 1;++cases;
    }
    std::printf("PC8_FINAL_RAW cases=%zu workers=6 ai_mask=0x3f00 five_frm=1 native_byte_oracle=1 PASS\n",cases);
    return 0;
}
