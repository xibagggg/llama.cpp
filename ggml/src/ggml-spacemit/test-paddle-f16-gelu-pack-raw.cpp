// Generated exact arithmetic excerpts are fingerprinted beside this fixture.
// They avoid linking hidden backend symbols; the separate graph fixture tests
// real production dispatch and results. No approximate std::tanh reference.
#include "test-paddle-f16-gelu-pack-exact.h"
#include "paddle-f16-gelu-pack-policy.h"
#include <spert.hpp>
#include <sched.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>

struct guarded {
    std::vector<uint64_t> storage;size_t bytes;
    explicit guarded(size_t n):storage((n+256+7)/8,UINT64_C(0xa5a5a5a5a5a5a5a5)),bytes(n){}
    void *data(){return reinterpret_cast<unsigned char*>(storage.data())+128;}
    bool intact()const {
        auto*p=reinterpret_cast<const unsigned char*>(storage.data());
        for(int i=0;i<128;++i)if(p[i]!=0xa5||p[128+bytes+i]!=0xa5)return false;
        return true;
    }
};
static uint32_t next(uint32_t&x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
static uint64_t hash(const void*p,size_t n) {
    auto*q=static_cast<const unsigned char*>(p);uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<n;++i)h=(h^q[i])*UINT64_C(1099511628211);return h;
}
int main(int argc,char**argv) {
    const char*backend=std::getenv("SPERT_BACKEND");
    if(!backend||std::strcmp(backend,"spacemit-k3"))return 2;
    spert::StreamConfig config;config.n_cores=6;config.core_ids={8,9,10,11,12,13};
    spert::Stream stream(config);
    if(!stream.valid()||stream.core_count()!=6)return 3;
    constexpr int K=4304;size_t cases=0;
    const bool death_reference=argc==2&&!std::strcmp(argv[1],"--invalid-reference");
    const bool death_candidate=argc==2&&!std::strcmp(argv[1],"--invalid-candidate");
    if(death_reference||death_candidate) {
        std::vector<float>x(K,std::numeric_limits<float>::infinity()),y(K);
        std::vector<_Float16>high(8*K),low(8*K);float scale=0;
        auto f=stream.launch(spert::Grid{6},[&](spert::Context*runtime) {
            if(runtime->program_id(0))return;
            if(death_reference) {
                reference_gelu_partition(x.data(),y.data(),K,0,1);
                reference_group_pack(y.data(),high.data(),low.data(),&scale,0,1,K);
            } else {
                auto s=runtime->shared_buffer();
                if(!s||s.size<paddle_f16_gelu_pack::scratch_bytes)throw std::runtime_error("TCM missing");
                paddle_f16_ime2::pack_gelu_group(x.data(),reinterpret_cast<float*>(static_cast<char*>(s.data)+128),
                    high.data(),low.data(),&scale,1,K);
            }
        });
        if(f.sync()!=spert::Status::Ok)return 4;
        uint32_t bits=0;std::memcpy(&bits,&scale,4);
        std::printf("F16_GELU_PACK_INVALID accepted scale_bits=0x%08x high_hash=%016llx low_hash=%016llx\n",
            bits,(unsigned long long)hash(high.data(),high.size()*2),(unsigned long long)hash(low.data(),low.size()*2));return 0;
    }
    if(argc!=1)return 2;
    size_t rejects=0;
    for(int M:{1,7,8,9,17,33,848})for(unsigned variant=0;variant<5;++variant)
    for(unsigned rounding=0;rounding<5;++rounding) {
        const int groups=(M+7)/8;const size_t packed=size_t(groups)*8*K*2;
        guarded input(size_t(M)*K*4),gelu(size_t(M)*K*4),rh(packed),rl(packed),ch(packed),cl(packed);
        std::vector<float>rs(groups,-1),cs(groups,-2);
        auto*x=static_cast<float*>(input.data());uint32_t rng=0x8f09ca6bu+variant;
        for(size_t i=0;i<size_t(M)*K;++i) {
            float value=float(int(next(rng)%65537)-32768)/4096.f;
            if(variant==0)value=i%2?-0.f:0.f;
            if(variant==1)value*=0x1p-100f;
            if(variant==2) {
                static const float edges[]={-32.f,-8.f,-1.f,-0x1p-14f,-0x1p-24f,-0x1p-120f,-0.f,
                    0.f,0x1p-120f,0x1p-24f,0x1p-14f,1.f,8.f,32.f};
                value=edges[i%(sizeof(edges)/sizeof(edges[0]))];
            }
            if(variant==4&&i%13<6) {
                static const uint32_t special[]={0x7f800000,0xff800000,0x7fc01234,
                    0xffc05678,0x7f801234,0x00000001};
                std::memcpy(&value,&special[i%13],4);
            }
            x[i]=value;
        }
        const uint64_t before=hash(input.data(),size_t(M)*K*4);
        std::array<int,6>cpus{};std::array<bool,6>entered{},canaries{},intermediate_equal{};
        std::array<size_t,6>lane_rejects{};
        std::array<size_t,6>vlens{};
        auto f=stream.launch(spert::Grid{6},[&](spert::Context*runtime) {
            const unsigned lane=runtime->program_id(0);
            cpus[lane]=sched_getcpu();vlens[lane]=__riscv_vlenb();
            if(cpus[lane]!=8+int(lane)||vlens[lane]!=128)throw std::runtime_error("wrong AI worker");
            auto shared=runtime->shared_buffer();
            if(!shared||shared.size<paddle_f16_gelu_pack::scratch_bytes)
                throw std::runtime_error("real TCM shared view unavailable");
            auto*guard=static_cast<unsigned char*>(shared.data);
            auto*scratch=reinterpret_cast<float*>(guard+128);
            std::memset(guard,0xa5,128);std::memset(scratch+8*K,0xa5,128);
            unsigned old_rounding;
            asm volatile("frrm %0\n\tfsrm %1":"=&r"(old_rounding):"r"(rounding):"memory");
            reference_gelu_partition(x,static_cast<float*>(gelu.data()),size_t(M)*K,lane,6);
            if(runtime->sync()!=spert::Status::Ok)throw std::runtime_error("GELU barrier");
            intermediate_equal[lane]=true;
            for(int g=int(lane);g<groups;g+=6) {
                const int rows=std::min(8,M-g*8);
                if(variant==4) {
                    std::memcpy(scratch,x+size_t(g)*8*K,size_t(rows)*K*4);
                    spacemit_kernels::rvv::forward_gelu_f32_tile(scratch,size_t(rows)*K);
                    intermediate_equal[lane]=intermediate_equal[lane]&&
                        !std::memcmp(scratch,static_cast<float*>(gelu.data())+size_t(g)*8*K,size_t(rows)*K*4);
                    float a=reference_group_maximum(static_cast<float*>(gelu.data()),g,M,K);
                    float b=reference_group_maximum(scratch,0,rows,K);
                    intermediate_equal[lane]=intermediate_equal[lane]&&!std::memcmp(&a,&b,4);
                    if(!std::isfinite(a)) {
                        ++lane_rejects[lane];
                        // Both exact scale_for calls assert on the same input.
                        // Isolated positive-Inf subprocesses test the abort.
                        rs[g]=cs[g]=-3;continue;
                    }
                }
                reference_group_pack(static_cast<const float*>(gelu.data()),static_cast<_Float16*>(rh.data()),
                    static_cast<_Float16*>(rl.data()),rs.data()+g,g,M,K);
                paddle_f16_ime2::pack_gelu_group(x+size_t(g)*8*K,scratch,
                    static_cast<_Float16*>(ch.data())+size_t(g)*8*K,
                    static_cast<_Float16*>(cl.data())+size_t(g)*8*K,cs.data()+g,std::min(8,M-g*8),K);
            }
            asm volatile("fsrm %0"::"r"(old_rounding):"memory");
            canaries[lane]=true;
            for(int b=0;b<128;++b)canaries[lane]=canaries[lane]&&guard[b]==0xa5&&
                reinterpret_cast<unsigned char*>(scratch+8*K)[b]==0xa5;
            entered[lane]=true;
        });
        if(f.sync()!=spert::Status::Ok)return 4;
        bool equal=!std::memcmp(rh.data(),ch.data(),packed)&&!std::memcmp(rl.data(),cl.data(),packed)&&
            !std::memcmp(rs.data(),cs.data(),groups*4);
        bool guards=input.intact()&&gelu.intact()&&rh.intact()&&rl.intact()&&ch.intact()&&cl.intact();
        bool lanes=true;for(int i=0;i<6;++i) {
            lanes=lanes&&entered[i]&&canaries[i]&&intermediate_equal[i]&&cpus[i]==i+8&&vlens[i]==128;
            rejects+=lane_rejects[i];
        }
        bool unchanged=before==hash(input.data(),size_t(M)*K*4);
        if(!equal||!guards||!lanes||!unchanged) {
            std::fprintf(stderr,"F16_GELU_PACK_RAW FAIL M=%d variant=%u rounding=%u equal=%d guards=%d lanes=%d input=%d\n",
                M,variant,rounding,int(equal),int(guards),int(lanes),int(unchanged));return 1;
        }
        ++cases;
    }
    std::printf("F16_GELU_PACK_RAW PASS cases=%zu nonfinite_rejected_groups=%zu six_AI_CPU8_13 VLENB128 exact_full_K_scale_high_residual_bytes TCM_and_DDR_canaries inputs_unchanged rounding=RNE,RTZ,RDN,RUP,RMM\n",cases,rejects);
}
