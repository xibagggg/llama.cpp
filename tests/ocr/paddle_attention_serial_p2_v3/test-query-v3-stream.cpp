#include <riscv_vector.h>
#include <spert.hpp>
#include <sched.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// Compile the exact candidate helper; no scalar substitute is exercised.
#include "ocr-fa-serial-p2-v3-query.h"

namespace {
constexpr int D=72;
constexpr size_t GUARD=128;
struct guarded {
    std::vector<uint64_t> words;
    size_t size;
    explicit guarded(size_t n):words((n+2*GUARD+7)/8),size(n){
        std::memset(words.data(),0xa5,words.size()*8);
    }
    void *data(){return reinterpret_cast<uint8_t*>(words.data())+GUARD;}
    bool intact()const{
        const auto*p=reinterpret_cast<const uint8_t*>(words.data());
        for(size_t i=0;i<GUARD;++i)if(p[i]!=0xa5)return false;
        for(size_t i=GUARD+size;i<words.size()*8;++i)if(p[i]!=0xa5)return false;
        return true;
    }
};
static float scalar_mul(float a,float b){
    float r;asm volatile("fmul.s %0,%1,%2,dyn":"=f"(r):"f"(a),"f"(b));return r;
}
static float scalar_sub(float a,float b){
    float r;asm volatile("fsub.s %0,%1,%2,dyn":"=f"(r):"f"(a),"f"(b));return r;
}
static _Float16 scalar_half(float a){
    _Float16 h;asm volatile("fcvt.h.s %0,%1,dyn":"=f"(h):"f"(a));return h;
}
static float scalar_float(_Float16 h){
    float r;asm volatile("fcvt.s.h %0,%1":"=f"(r):"f"(h));return r;
}
uint32_t rng_next(uint32_t&s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
float from_bits(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
void set_frm(unsigned value){asm volatile("csrw frm,%0"::"r"(value):"memory");}

int run_cases(){
    const unsigned incoming=ocr_serial_v3_frm();
    set_frm(0);
    bool ok=ocr_serial_v3_frm()==0;size_t cases=0;
    // Scalar reference uses explicit dyn conversions/mul/sub instructions,
    // independent of vector packing/layout and compiler FP-expression folding.
    for(int rows:{1,7,8,9,15,16})for(int padding:{0,5})
    for(float scale:{1.f,0.1178511301977579f,0.25f})for(int variant=0;variant<10;++variant){
        const int stride=D+padding,padded=(rows+7)/8*8;
        const size_t input_bytes=size_t(rows)*stride*4,pack_bytes=size_t(padded)*D*2;
        guarded input(input_bytes),high(pack_bytes),low(pack_bytes);
        auto*x=static_cast<float*>(input.data());
        uint32_t seed=0xb1726f43u^uint32_t(rows*631+padding*313+variant*97);
        for(int r=0;r<rows;++r)for(int d=0;d<stride;++d){
            float value=float(int(rng_next(seed)%32769)-16384)*0x1p-11f;
            if(variant==1)value=(d&1)?-0.f:0.f;
            if(variant==2)value=from_bits((d&1?0x80000000u:0u)|(1u+(rng_next(seed)%0x00ffffffu)));
            if(variant==3)value=(d&1?-1.f:1.f)*(1.f+float(d%3)*0x1p-11f);
            if(variant==4)value=(d&1?-1.f:1.f)*63.f;
            if(variant==5)value=(d&1?-1.f:1.f)*64.f;
            if(variant==6)value=from_bits(0x7fc01234u);
            if(variant==7)value=std::numeric_limits<float>::infinity();
            if(variant==8)value=-std::numeric_limits<float>::infinity();
            if(variant==9)value=(d&1?-1.f:1.f)*0x1p-24f;
            x[size_t(r)*stride+d]=value;
        }
        std::vector<uint8_t> original(input_bytes);
        std::memcpy(original.data(),input.data(),input_bytes);
        std::vector<_Float16> expected_h(size_t(padded)*D,(_Float16)0);
        std::vector<_Float16> expected_l(size_t(padded)*D,(_Float16)0);
        bool valid=true;
        for(int r=0;r<rows;++r)for(int d=0;d<D;++d){
            const float a=scalar_mul(x[size_t(r)*stride+d],scale);
            const auto h=scalar_half(a);const float f=scalar_float(h);
            const float residual=scalar_mul(scalar_sub(a,f),1048576.f);
            valid=valid&&std::isfinite(f)&&std::fabs(f)<=63.f&&
                std::isfinite(residual)&&std::fabs(residual)<=65504.f;
            const size_t id=size_t(r/8)*8*D+size_t(d/8)*64+(r%8)*8+d%8;
            expected_h[id]=scalar_half(scalar_mul(f,1024.f));
            expected_l[id]=scalar_half(residual);
        }
        const bool admission=ocr_serial_v3_query<false,false>(input.data(),stride*4,rows,padded,scale,nullptr);
        const bool actual_h=ocr_serial_v3_query<false,true>(input.data(),stride*4,rows,padded,scale,
            static_cast<_Float16*>(high.data()));
        const bool actual_l=ocr_serial_v3_query<true,true>(input.data(),stride*4,rows,padded,scale,
            static_cast<_Float16*>(low.data()));
        bool pass=admission==valid&&actual_h==valid&&actual_l==valid;
        if(valid)pass=pass&&!std::memcmp(high.data(),expected_h.data(),pack_bytes)&&
            !std::memcmp(low.data(),expected_l.data(),pack_bytes);
        pass=pass&&input.intact()&&high.intact()&&low.intact()&&
            !std::memcmp(input.data(),original.data(),input_bytes);
        if(!pass)std::printf("query FAIL rows=%d stride=%d scale=%.9g variant=%d wanted=%d got=%d/%d/%d\n",
            rows,stride,scale,variant,int(valid),int(admission),int(actual_h),int(actual_l));
        ok=ok&&pass;++cases;
    }
    // Verify the CSR guard sees all supported modes and restore the incoming
    // mode. This is a helper/CSR test, not an engine non-RNE fallback result.
    for(unsigned mode=0;mode<=4;++mode){set_frm(mode);ok=ok&&ocr_serial_v3_frm()==mode;}
    set_frm(incoming);
    std::printf("query_v3 cases=%zu high_low_bits_scalar_dyn_oracle=%s guards=128B input_unchanged=1 frm_restored=%u\n",
        cases,ok?"PASS":"FAIL",ocr_serial_v3_frm());
    return ok?0:1;
}
} // namespace

int main(){
    const char*backend=std::getenv("SPERT_BACKEND");
    if(!backend||std::strcmp(backend,"spacemit-k3"))return 2;
    for(const auto&p:std::array<std::pair<const char*,const char*>,3>{{
        {"GGML_SPACEMIT_WORKERS","6"},{"GGML_SPACEMIT_ALLOW_EIGHT_WORKERS","0"},
        {"SPACEMIT_PERFER_CORE_ID","8,9,10,11,12,13"}}})if(setenv(p.first,p.second,1))return 2;
    spert::StreamConfig cfg;cfg.n_cores=6;cfg.core_ids={8,9,10,11,12,13};
    spert::Stream stream(cfg);if(!stream.valid()||stream.core_count()!=6)return 3;
    std::array<int,6> cpus{{-1,-1,-1,-1,-1,-1}};std::array<size_t,6> vlens{};
    std::array<bool,6> entered{};int result=4;
    auto future=stream.launch(spert::Grid{6},[&](spert::Context*runtime){
        const unsigned lane=runtime->program_id(0);if(lane>=6)throw std::runtime_error("unexpected lane");
        cpus[lane]=sched_getcpu();vlens[lane]=__riscv_vlenb();entered[lane]=true;
        if(cpus[lane]!=8+int(lane)||vlens[lane]!=128)throw std::runtime_error("wrong AI worker/VLEN");
        if(lane==0)result=run_cases();
    });
    const auto status=future.sync();bool placement=true;
    for(unsigned lane=0;lane<6;++lane){
        const bool pass=entered[lane]&&cpus[lane]==8+int(lane)&&vlens[lane]==128;placement=placement&&pass;
        std::printf("query_v3 lane=%u cpu=%d vlenb=%zu %s\n",lane,cpus[lane],vlens[lane],pass?"PASS":"FAIL");
    }
    std::printf("query_v3 workers=6 placement=%s result=%d runtime_ok=%d\n",
        placement?"PASS":"FAIL",result,int(status==spert::Status::Ok));
    return status==spert::Status::Ok&&placement&&result==0?0:4;
}
