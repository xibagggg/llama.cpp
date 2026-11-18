// Exact production GELU and frozen PC8 quantizer excerpts are fingerprinted.
// This is a six-AI numerical/TCM test, not a performance benchmark.
#include "test-paddle-pc8-gelu-pack-exact.h"
#include "paddle-pc8-gelu-pack.h"
#include <spert.hpp>
#include <sched.h>
#include <array>
#include <stdexcept>
#include <vector>

struct guarded{
    std::vector<uint64_t> storage;size_t bytes;
    explicit guarded(size_t n):storage((n+256+7)/8,UINT64_C(0xa5a5a5a5a5a5a5a5)),bytes(n){}
    void*data(){return reinterpret_cast<uint8_t*>(storage.data())+128;}
    bool intact()const{const auto*p=reinterpret_cast<const uint8_t*>(storage.data());
        for(size_t i=0;i<128;++i)if(p[i]!=0xa5||p[128+bytes+i]!=0xa5)return false;
        return true;}
};
static uint32_t rnd(uint32_t&x){x^=x<<13;x^=x>>17;x^=x<<5;return x;}
static uint64_t hash(const void*p,size_t n){const auto*b=static_cast<const uint8_t*>(p);uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<n;++i)h=(h^b[i])*UINT64_C(1099511628211);return h;}
int main(int argc,char**argv){
    const char*backend=std::getenv("SPERT_BACKEND");if(!backend||std::strcmp(backend,"spacemit-k3"))return 2;
    setenv("GGML_SPACEMIT_WORKERS","6",1);setenv("GGML_SPACEMIT_ALLOW_EIGHT_WORKERS","0",1);
    setenv("SPACEMIT_PERFER_CORE_ID","8,9,10,11,12,13",1);
    spert::StreamConfig config;config.n_cores=6;config.core_ids={8,9,10,11,12,13};
    spert::Stream stream(config);if(!stream.valid()||stream.core_count()!=6)return 3;
    constexpr int K=4304,PK=4352;
    if(argc==3&&(!std::strcmp(argv[1],"--special-reference")||!std::strcmp(argv[1],"--special-candidate"))){
        const bool candidate=!std::strcmp(argv[1],"--special-candidate");
        const int kind=std::atoi(argv[2]);if(kind<0||kind>3)return 2;
        const uint32_t special[]={0x7f800000u,0xff800000u,0x7fc01234u,0x7f801234u};
        std::vector<float>x(K,0),gelu(K),scale(4);std::vector<int8_t>packed(4*PK);
        std::memcpy(&x[K-1],&special[kind],4);
        auto f=stream.launch(spert::Grid{6},[&](spert::Context*runtime){
            const unsigned lane=runtime->program_id(0);
            if(sched_getcpu()!=8+int(lane)||__riscv_vlenb()!=128)throw std::runtime_error("wrong AI worker");
            if(lane)return;
            if(candidate){auto tcm=runtime->shared_buffer();
                if(!tcm||tcm.size<paddle_pc8_gelu_pack::scratch_bytes)throw std::runtime_error("missing TCM");
                paddle_pc8_gelu_pack::pack_group(x.data(),reinterpret_cast<float*>(static_cast<char*>(tcm.data)+128),packed.data(),scale.data(),1);
            }else{reference_gelu_partition(x.data(),gelu.data(),K,0,1);reference_token_pack(gelu.data(),packed.data(),scale.data(),1,0,1);}
        });
        if(f.sync()!=spert::Status::Ok)return 4;
        std::printf("PC8_GELU_PACK_SPECIAL kind=%d packed_hash=%016llx scale_hash=%016llx\n",kind,
            (unsigned long long)hash(packed.data(),packed.size()),(unsigned long long)hash(scale.data(),16));return 0;
    }
    if(argc!=1)return 2;
    size_t cases=0;
    for(int M:{1,3,4,5,7,8,9,13,17,33,756,828,832,848,940,1500})
    for(unsigned variant=0;variant<5;++variant)for(unsigned rounding=0;rounding<5;++rounding){
        const int groups=(M+3)/4,MP=groups*4;
        guarded input(size_t(M)*K*4),gelu(size_t(M)*K*4),reference(size_t(MP)*PK),candidate(size_t(MP)*PK),
            rs(size_t(MP)*4),cs(size_t(MP)*4);
        auto*x=static_cast<float*>(input.data());uint32_t seed=0x176981au+variant;
        for(size_t i=0;i<size_t(M)*K;++i){
            float v=float(int(rnd(seed)%65537)-32768)*0x1p-12f;
            if(variant==0)v=i%2?-0.f:0.f;
            if(variant==1)v*=0x1p-100f;
            if(variant==2){static const float edges[]={-32.f,-8.f,-1.f,-0x1p-14f,-0x1p-24f,-0x1p-120f,-0.f,0.f,0x1p-120f,0x1p-24f,0x1p-14f,1.f,8.f,32.f};v=edges[i%14];}
            if(variant==4)v=float(int(i%255)-127)*0.00390625f+float(int(i%3)-1)*0x1p-22f;
            x[i]=v;
        }
        const uint64_t before=hash(input.data(),size_t(M)*K*4);
        std::array<int,6>cpus{{-1,-1,-1,-1,-1,-1}},packed_groups{};
        std::array<size_t,6>vlens{},shared_bytes{};std::array<void*,6>shared_ptr{},scratch_ptr{};
        std::array<bool,6>entered{},canaries{},intermediate{},frm{};
        auto f=stream.launch(spert::Grid{6},[&](spert::Context*runtime){
            const unsigned lane=runtime->program_id(0);if(lane>=6)throw std::runtime_error("bad lane");
            cpus[lane]=sched_getcpu();vlens[lane]=__riscv_vlenb();
            if(cpus[lane]!=8+int(lane)||vlens[lane]!=128)throw std::runtime_error("wrong AI worker");
            auto tcm=runtime->shared_buffer();if(!tcm||tcm.size<paddle_pc8_gelu_pack::scratch_bytes)throw std::runtime_error("missing real TCM");
            auto*guard=static_cast<uint8_t*>(tcm.data);auto*scratch=reinterpret_cast<float*>(guard+128);
            shared_ptr[lane]=tcm.data;shared_bytes[lane]=tcm.size;scratch_ptr[lane]=scratch;
            std::memset(guard,0xa5,128);std::memset(guard+128+size_t(4)*K*4,0xa5,128);
            unsigned old_frm;asm volatile("frrm %0\n\tfsrm %1":"=&r"(old_frm):"r"(rounding):"memory");
            reference_gelu_partition(x,static_cast<float*>(gelu.data()),size_t(M)*K,lane,6);
            if(runtime->sync()!=spert::Status::Ok)throw std::runtime_error("reference GELU barrier");
            reference_token_pack(static_cast<const float*>(gelu.data()),static_cast<int8_t*>(reference.data()),static_cast<float*>(rs.data()),M,int(lane),6);
            intermediate[lane]=true;
            for(int g=int(lane);g<groups;g+=6){
                ++packed_groups[lane];
                const int rows=std::min(4,M-g*4);
                paddle_pc8_gelu_pack::pack_group(x+size_t(g)*4*K,scratch,
                    static_cast<int8_t*>(candidate.data())+size_t(g)*4*PK,static_cast<float*>(cs.data())+g*4,rows);
                intermediate[lane]=intermediate[lane]&&!std::memcmp(scratch,static_cast<const float*>(gelu.data())+size_t(g)*4*K,size_t(rows)*K*4);
            }
            unsigned after;asm volatile("frrm %0":"=r"(after));frm[lane]=after==rounding;
            asm volatile("fsrm %0"::"r"(old_frm):"memory");
            canaries[lane]=true;for(int i=0;i<128;++i)canaries[lane]=canaries[lane]&&guard[i]==0xa5&&guard[128+size_t(4)*K*4+i]==0xa5;
            entered[lane]=true;
        });
        const auto status=f.sync();bool lanes=true;
        for(unsigned lane=0;lane<6;++lane)lanes=lanes&&entered[lane]&&cpus[lane]==8+int(lane)&&vlens[lane]==128&&canaries[lane]&&intermediate[lane]&&frm[lane];
        if(M==33&&variant==0&&rounding==0)for(unsigned lane=0;lane<6;++lane)
            std::printf("PC8_GELU_PACK_TCM M=33 lane=%u cpu=%d vlenb=%zu packed_groups=%d shared=%p shared_bytes=%zu scratch=%p guarded_bytes=%zu guarded_end=%p canaries=%d\n",
                lane,cpus[lane],vlens[lane],packed_groups[lane],shared_ptr[lane],shared_bytes[lane],scratch_ptr[lane],
                paddle_pc8_gelu_pack::scratch_bytes,static_cast<unsigned char*>(shared_ptr[lane])+paddle_pc8_gelu_pack::scratch_bytes,int(canaries[lane]));
        const bool bits=!std::memcmp(reference.data(),candidate.data(),size_t(MP)*PK)&&!std::memcmp(rs.data(),cs.data(),size_t(MP)*4);
        const bool guards=input.intact()&&gelu.intact()&&reference.intact()&&candidate.intact()&&rs.intact()&&cs.intact();
        const bool immutable=hash(input.data(),size_t(M)*K*4)==before;
        const bool pass=status==spert::Status::Ok&&lanes&&bits&&guards&&immutable;
        std::printf("PC8_GELU_PACK_RAW M=%d variant=%u frm=%u bits=%d intermediate=%d lanes6=%d guards=%d immutable=%d %s\n",
            M,variant,rounding,int(bits),int(lanes),int(lanes),int(guards),int(immutable),pass?"PASS":"FAIL");
        std::fflush(stdout);if(!pass)return 1;++cases;
    }
    std::printf("PC8_GELU_PACK_RAW cases=%zu workers=6 ai_mask=0x3f00 five_frm=1 exact_token_scale_and_pack=1 PASS\n",cases);
    return 0;
}
