// K3-only, six AI workers. Frozen original instructions versus new N64 helper.
// This exercises physical Q8 repacks directly; full graph fixture separately
// validates real Q6 model-load repacking and the production head admission.
#include <riscv_vector.h>
#include <spert.hpp>
#include <sched.h>
#include "paddle-q6-head-n64-kernel.h"
#include "test-paddle-q6-head-n32-reference.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t next(uint32_t&x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
struct lane {
    std::vector<uint8_t>a,b;
    std::array<float,96>reference{},candidate{};
    int cpu=-1;size_t vlenb=0;bool entered=false;
};
int main() {
    const char*backend=std::getenv("SPERT_BACKEND");
    if(!backend||std::strcmp(backend,"spacemit-k3"))return 2;
    (void)sched_getcpu();
    spert::StreamConfig config;config.n_cores=6;config.core_ids={8,9,10,11,12,13};
    spert::Stream stream(config);
    if(!stream.valid()||stream.core_count()!=6)return 3;
    std::array<lane,6>lanes;
    size_t cases=0;
    const float a_scales[]={0.f,-0.f,0x1p-120f,0x1p-20f,1.f,0x1p60f};
    const uint16_t b_scales[]={0x0000,0x8000,0x0001,0x03ff,0x0400,0x3c00,0xbc00,0x7bff};
    for(size_t kb:{size_t(1),size_t(2),size_t(31),size_t(32),size_t(33)})
    for(size_t n:{size_t(33),size_t(34),size_t(47),size_t(63),size_t(64)})
    for(unsigned variant=0;variant<12;++variant) {
        std::array<uint64_t,6>before{};
        auto hash=[](const std::vector<uint8_t>&a,const std::vector<uint8_t>&b) {
            uint64_t h=UINT64_C(14695981039346656037);
            for(auto&v:a)h=(h^v)*UINT64_C(1099511628211);
            for(auto&v:b)h=(h^v)*UINT64_C(1099511628211);return h;
        };
        for(size_t worker=0;worker<6;++worker) {
            auto&v=lanes[worker];v.a.resize(kb*38);v.b.resize(2*kb*1088);
            uint32_t random=0x7aabc19u^variant*1723u^unsigned(worker)*9281u;
            for(auto&x:v.a)x=uint8_t(next(random));
            for(auto&x:v.b)x=uint8_t(next(random));
            for(size_t k=0;k<kb;++k) {
                float scale=a_scales[(variant+k+worker)%6];
                std::memcpy(v.a.data()+k*38,&scale,4);
                for(size_t i=0;i<32;++i) {
                    if(variant==0)v.a[k*38+6+i]=0;
                    if(variant==1)v.a[k*38+6+i]=uint8_t(i&1?127:128);
                }
                for(size_t panel=0;panel<2;++panel)for(size_t i=0;i<32;++i) {
                    uint16_t scale16=b_scales[(variant+k+i+panel+worker)%8];
                    std::memcpy(v.b.data()+panel*kb*1088+k*1088+i*2,&scale16,2);
                }
            }
            before[worker]=hash(v.a,v.b);
            std::memset(v.reference.data(),0xa5,sizeof(v.reference));v.candidate=v.reference;
            v.cpu=-1;v.vlenb=0;v.entered=false;
        }
        auto future=stream.launch(spert::Grid{6},[&](spert::Context*runtime) {
            const size_t worker=runtime->program_id(0);auto&v=lanes[worker];
            v.cpu=sched_getcpu();v.vlenb=__riscv_vlenb();
            if(v.cpu!=8+int(worker)||v.vlenb!=128)return;
            v.entered=true;
            q6_head_n32_reference(32,v.a.data(),v.b.data(),nullptr,v.reference.data()+16,1,n,kb,n);
            paddle_q6_head_n64_kernel(v.a.data(),v.b.data(),v.candidate.data()+16,n,kb);
        });
        if(future.sync()!=spert::Status::Ok)return 4;
        for(size_t worker=0;worker<6;++worker) {
            auto&v=lanes[worker];
            bool finite=std::all_of(v.candidate.begin()+16,v.candidate.begin()+16+n,[](float x){return std::isfinite(x);});
            if(!v.entered||v.cpu!=8+int(worker)||v.vlenb!=128||!finite||
               std::memcmp(v.reference.data(),v.candidate.data(),sizeof(v.reference))||
               before[worker]!=hash(v.a,v.b)) {
                size_t first=0;while(first<96&&!std::memcmp(&v.reference[first],&v.candidate[first],4))++first;
                std::fprintf(stderr,"Q6_N64_RAW FAIL worker=%zu cpu=%d vlenb=%zu entered=%d kb=%zu n=%zu variant=%u first_float=%zu finite=%d\n",worker,v.cpu,v.vlenb,int(v.entered),kb,n,variant,first,int(finite));
                return 1;
            }
            ++cases;
        }
    }
    std::printf("Q6_N64_RAW PASS six AI workers CPU8-13 VLENB128 cases=%zu tail_columns=33,34,47,63,64 kblocks=1,2,31,32,33 all_bytes_and_canaries_equal inputs_unchanged\n",cases);
}
