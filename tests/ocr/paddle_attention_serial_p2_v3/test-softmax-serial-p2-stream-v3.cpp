#include <riscv_vector.h>
#include <spert.hpp>
#include <sched.h>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "ocr-fa-fullkv.h"
#include "ocr-fa-serial-p2-v3-plan.h"

// Exact f449 vector exp and reduction definitions, generated from the checked
// base rvv_kernels.cpp. This is an RVV pack/lifetime test, not an IME benchmark.
#include "raw-math-f449.inc"
#include "softmax-v3-only.inc"

namespace {
constexpr size_t GUARD=128;
constexpr uint8_t SENTINEL=0xa5;
struct guarded {
    std::vector<uint64_t> words;
    size_t size;
    explicit guarded(size_t bytes):words((bytes+2*GUARD+7)/8),size(bytes) {
        std::memset(words.data(),SENTINEL,words.size()*8);
    }
    void* data(){return reinterpret_cast<uint8_t*>(words.data())+GUARD;}
    bool intact()const{
        const auto*p=reinterpret_cast<const uint8_t*>(words.data());
        for(size_t i=0;i<GUARD;++i)if(p[i]!=SENTINEL)return false;
        for(size_t i=GUARD+size;i<words.size()*8;++i)if(p[i]!=SENTINEL)return false;
        return true;
    }
};
uint32_t random_u32(uint32_t&s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
bool same(const void*a,const void*b,size_t n){return !std::memcmp(a,b,n);}
}

static int run_cases_on_ai_lane(){
    bool ok=true;size_t count=0;
    for(int n:{1,7,8,9,63,64,65,71,72,73,127,128,129,848,1120,1121,1496,1500,1504,1536,1537}){
        const int padded=(n+7)/8*8;
        for(int rows:{1,7,8,15,16})for(int variant=0;variant<5;++variant){
            const size_t score_bytes=size_t(rows)*n*sizeof(float),pack_bytes=size_t(16)*padded*sizeof(_Float16);
            guarded scores(score_bytes),saved(score_bytes),high(pack_bytes),low(pack_bytes),single(pack_bytes),
                inverse_a(size_t(16)*4),inverse_b(size_t(16)*4);
            auto*s=static_cast<float*>(scores.data());auto*e=static_cast<float*>(saved.data());
            uint32_t rng=0x176a981fu^uint32_t(n*31+rows*13+variant);
            for(size_t i=0;i<size_t(rows)*n;++i){
                s[i]=float(int(random_u32(rng)%65537)-32768)*0x1p-11f;
                if(variant==1)s[i]=float(int(i%3)-1)*1000.f;
                if(variant==2)s[i]=(i%2)?-0.f:0.f;
            }
            if(variant==3)s[size_t(rows)*n-1]=std::numeric_limits<float>::infinity();
            if(variant==4)s[size_t(rows)*n-1]=std::numeric_limits<float>::quiet_NaN();
            std::memcpy(e,s,score_bytes);
            const bool a=ocr_serial_v3_softmax_pack<false>(s,rows,n,padded,16,static_cast<_Float16*>(high.data()),
                static_cast<_Float16*>(low.data()),2,static_cast<float*>(inverse_a.data()));
            const bool b=ocr_serial_v3_softmax_pack<false>(e,rows,n,padded,16,static_cast<_Float16*>(single.data()),
                nullptr,2,static_cast<float*>(inverse_b.data()),e);
            bool pass=a==b;
            if(a&&b){
                pass=pass&&same(high.data(),single.data(),pack_bytes)&&
                    same(inverse_a.data(),inverse_b.data(),size_t(rows)*4);
                pass=pass&&ocr_serial_v3_pack_saved_residual(e,rows,n,padded,16,static_cast<_Float16*>(single.data()));
                pass=pass&&same(low.data(),single.data(),pack_bytes);
            }
            for(const auto*g:{&scores,&saved,&high,&low,&single,&inverse_a,&inverse_b})pass=pass&&g->intact();
            if(n>=64&&n<=1536)pass=pass&&ocr_fa_serial_p2_v3::bytes(n)<=393216;
            std::printf("raw N=%d rows=%d variant=%d legacy=%d serial=%d scratch=%zu %s\n",
                n,rows,variant,int(a),int(b),ocr_fa_serial_p2_v3::bytes(n),pass?"PASS":"FAIL");
            ok=ok&&pass;++count;
        }
    }
    std::printf("raw cases=%zu guards=128B mode1_exp_sum_high_low_bits=%s\n",count,ok?"PASS":"FAIL");
    return ok?0:1;
}


// Host setup contains no RVV instructions. The unchanged 525-case arithmetic
// runs only on AI lane zero; the other five lanes record actual placement.
// Each lane writes a distinct array element, and the host reads after sync.
int main(){
    const char*backend=std::getenv("SPERT_BACKEND");
    if(!backend||std::strcmp(backend,"spacemit-k3"))return 2;
    const std::pair<const char*,const char*> flags[]={
        {"GGML_SPACEMIT_WORKERS","6"},
        {"GGML_SPACEMIT_ALLOW_EIGHT_WORKERS","0"},
        {"SPACEMIT_PERFER_CORE_ID","8,9,10,11,12,13"},
        {"GGML_SPACEMIT_OCR_FA_SOFTMAX_PACK","1"},
        {"GGML_SPACEMIT_OCR_FA_FIXED_INDEX","1"},
        {"GGML_SPACEMIT_OCR_FA_P_STORE64","1"},
        {"GGML_SPACEMIT_OCR_FA_TRUST_BOUNDS","0"},
        {"GGML_SPACEMIT_OCR_FA_BATCH_CHECK","0"},
    };
    for(const auto&f:flags)if(setenv(f.first,f.second,1))return 2;
    // Resolve first-use getenv/static guards on the dispatching host.
    (void)ocr_fa_fullkv::softmax_pack_mode();
    (void)ocr_fa_fullkv::fixed_index_enabled();
    (void)ocr_fa_fullkv::p_store64_enabled();
    (void)ocr_fa_fullkv::trust_bounds_enabled();
    (void)ocr_fa_fullkv::batch_check_enabled();
    spert::StreamConfig config;config.n_cores=6;config.core_ids={8,9,10,11,12,13};
    spert::Stream stream(config);
    if(!stream.valid()||stream.core_count()!=6)return 3;
    std::array<int,6> cpus{{-1,-1,-1,-1,-1,-1}};
    std::array<size_t,6> vlens{};
    std::array<bool,6> entered{};
    int result=4;
    auto future=stream.launch(spert::Grid{6},[&](spert::Context*runtime){
        const unsigned lane=runtime->program_id(0);
        if(lane>=6)throw std::runtime_error("unexpected AI lane");
        cpus[lane]=sched_getcpu();vlens[lane]=__riscv_vlenb();entered[lane]=true;
        if(cpus[lane]!=8+int(lane)||vlens[lane]!=128)
            throw std::runtime_error("wrong AI worker or VLEN");
        if(lane==0)result=run_cases_on_ai_lane();
    });
    const auto status=future.sync();
    bool placement=stream.core_count()==6;
    for(unsigned lane=0;lane<6;++lane){
        const bool pass=entered[lane]&&cpus[lane]==8+int(lane)&&vlens[lane]==128;
        placement=placement&&pass;
        std::printf("raw stream_v3 lane=%u cpu=%d vlenb=%zu %s\n",lane,cpus[lane],vlens[lane],pass?"PASS":"FAIL");
    }
    std::printf("raw stream_v3 workers=6 expected_ai_mask=0x3f00 placement=%s cases_result=%d runtime_ok=%d\n",
        placement?"PASS":"FAIL",result,int(status==spert::Status::Ok));
    return status==spert::Status::Ok&&placement&&result==0?0:4;
}
