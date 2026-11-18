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
#include <stdexcept>
#include <vector>

// Test the actual existing helper and exact candidate wrapper, not a scalar
// algebraic replacement. Original query packing stays unchanged on both sides.
#include "ocr-qk-ime2-v53.h"
#include "ocr-fa-serial-p2-v3-query.h"
#include "ocr-fa-serial-p2-v3-qk-pair.h"

namespace {
constexpr int D=72;
constexpr size_t GUARD=128;
struct region {
    uint8_t *raw;size_t size;
    void*data()const{return raw+GUARD;}
    void reset()const{std::memset(raw,0xa5,size+2*GUARD);}
    bool intact()const{
        for(size_t i=0;i<GUARD;++i)if(raw[i]!=0xa5||raw[GUARD+size+i]!=0xa5)return false;
        return true;
    }
};
struct arena {
    uint8_t*base;size_t capacity,used=0;
    region take(size_t size){
        used=(used+127)/128*128;
        if(size+2*GUARD>capacity-used)throw std::runtime_error("TCM fixture layout exceeds grant");
        region r{base+used,size};used+=size+2*GUARD;r.reset();return r;
    }
};
uint32_t next(uint32_t&s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
_Float16 half_from_bits(uint16_t bits){_Float16 value;std::memcpy(&value,&bits,2);return value;}
static void original_merge(float*high,const float*low,int stride,int rows,int count){
    for(int row=0;row<rows;++row){
        const size_t vl=count;
        const auto l=__riscv_vle32_v_f32m2(low+row*64,vl);
        const auto h=__riscv_vle32_v_f32m2(high+size_t(row)*stride,vl);
        __riscv_vse32_v_f32m2(high+size_t(row)*stride,
            __riscv_vfmacc_vf_f32m2(h,1.f/1024.f,l,vl),vl);
    }
}

int run_cases(void*base,size_t bytes){
    if(ocr_serial_v3_frm()!=0)return 4;
    arena memory{static_cast<uint8_t*>(base),bytes};
    auto input=memory.take(16*(D+5)*sizeof(float));
    auto raw_k=memory.take(64*D*sizeof(_Float16));
    auto packed_k=memory.take(64*D*sizeof(_Float16));
    auto high_q=memory.take(16*D*sizeof(_Float16)),low_q=memory.take(16*D*sizeof(_Float16));
    auto ref_high=memory.take(16*80*sizeof(float)),got_high=memory.take(16*80*sizeof(float));
    auto ref_low=memory.take(16*64*sizeof(float)),got_low=memory.take(16*64*sizeof(float));
    auto ref_scratch=memory.take(64*D*sizeof(_Float16)+512*sizeof(float));
    auto got_scratch=memory.take(64*D*sizeof(_Float16)+512*sizeof(float));
    const std::array<region,11> regions{{input,raw_k,packed_k,high_q,low_q,
        ref_high,got_high,ref_low,got_low,ref_scratch,got_scratch}};
    bool ok=true;size_t cases=0;size_t zero_low_tiles=0;
    for(int rows:{1,7,8,9,15,16})for(int count=1;count<=64;++count)for(int variant=0;variant<4;++variant){
        for(const auto&r:regions)r.reset();
        const int q_stride=D+5,padded=(rows+7)/8*8,score_stride=count+13;
        auto*q=static_cast<float*>(input.data());auto*k=static_cast<_Float16*>(raw_k.data());
        uint32_t seed=0x176a1fabu^uint32_t(rows*173+count*59+variant*397);
        for(int row=0;row<16;++row)for(int d=0;d<q_stride;++d){
            float value=float(int(next(seed)%131073)-65536)*0x1p-15f;
            if(variant==1)value=(d&1)?-0.f:0.f;
            if(variant==2)value=float(half_from_bits(uint16_t((d&1?0x8000u:0u)|uint16_t(1+next(seed)%1023))));
            if(variant==3)value=(d&1?-1.f:1.f)*(1.f+float(d%3)*0x1p-11f);
            q[row*q_stride+d]=value;
        }
        for(int d=0;d<D;++d)for(int token=0;token<64;++token){
            float value=float(int(next(seed)%131073)-65536)*0x1p-15f;
            if(variant==1)value=(d&1)?-0.f:0.f;
            k[d*64+token]=(_Float16)value;
            if(variant==2)k[d*64+token]=half_from_bits(uint16_t((d&1?0x8000u:0u)|uint16_t(1+next(seed)%1023)));
        }
        const float scale=variant==2?1.f:1.f/std::sqrt(float(D));
        bool pass=ocr_qk_ime2_v53::pack_k(k,static_cast<_Float16*>(packed_k.data()),64,count);
        pass=pass&&ocr_serial_v3_query<false,true>(q,q_stride*4,rows,padded,scale,
            static_cast<_Float16*>(high_q.data()));
        pass=pass&&ocr_serial_v3_query<true,true>(q,q_stride*4,rows,padded,scale,
            static_cast<_Float16*>(low_q.data()));
        const auto*low=static_cast<const _Float16*>(low_q.data());bool all_zero=true;
        for(int i=0;i<padded*D;++i)all_zero=all_zero&&float(low[i])==0.f;
        zero_low_tiles+=all_zero;
        std::vector<uint8_t> input_before(input.size),k_before(raw_k.size),pk_before(packed_k.size),
            hq_before(high_q.size),lq_before(low_q.size);
        for(const auto&pair:std::array<std::pair<region,std::vector<uint8_t>*>,5>{{
            {input,&input_before},{raw_k,&k_before},{packed_k,&pk_before},{high_q,&hq_before},{low_q,&lq_before}}})
            std::memcpy(pair.second->data(),pair.first.data(),pair.first.size);
        pass=pass&&ocr_qk_ime2_v53::qk(static_cast<float*>(ref_high.data()),
            static_cast<const _Float16*>(high_q.data()),static_cast<const _Float16*>(packed_k.data()),
            score_stride,rows,count,ref_scratch.data());
        pass=pass&&ocr_qk_ime2_v53::qk(static_cast<float*>(ref_low.data()),
            static_cast<const _Float16*>(low_q.data()),static_cast<const _Float16*>(packed_k.data()),
            64,rows,count,ref_scratch.data());
        original_merge(static_cast<float*>(ref_high.data()),static_cast<const float*>(ref_low.data()),
            score_stride,rows,count);
        pass=pass&&ocr_serial_v3_qk_pair(static_cast<float*>(got_high.data()),static_cast<float*>(got_low.data()),
            static_cast<const _Float16*>(high_q.data()),static_cast<const _Float16*>(low_q.data()),
            static_cast<const _Float16*>(packed_k.data()),score_stride,rows,count,got_scratch.data());
        // Compare the entire poisoned output regions, not just valid cells.
        pass=pass&&!std::memcmp(ref_high.data(),got_high.data(),ref_high.size)&&
            !std::memcmp(ref_low.data(),got_low.data(),ref_low.size);
        for(const auto&r:regions)pass=pass&&r.intact();
        for(const auto&pair:std::array<std::pair<region,const std::vector<uint8_t>*>,5>{{
            {input,&input_before},{raw_k,&k_before},{packed_k,&pk_before},{high_q,&hq_before},{low_q,&lq_before}}})
            pass=pass&&!std::memcmp(pair.first.data(),pair.second->data(),pair.first.size);
        if(!pass)std::printf("qk_pair_a1 rows=%d count=%d variant=%d FAIL\n",rows,count,variant);
        ok=ok&&pass;++cases;
    }
    std::printf("qk_pair_a1 cases=%zu complete_high_merge_low_F32_bits=%s TCM_used=%zu grant=%zu guards=128B inputs_unchanged=1 zero_low_fixture_tiles=%zu\n",
        cases,ok?"PASS":"FAIL",memory.used,bytes,zero_low_tiles);
    return ok?0:1;
}
}

int main(int argc,char**argv){
    if(argc!=3||(std::strcmp(argv[1],"0")&&std::strcmp(argv[1],"1"))||
       (std::strcmp(argv[2],"0")&&std::strcmp(argv[2],"1"))){
        std::fprintf(stderr,"Usage: %s SCATTER_0_OR_1 PIPELINE_0_OR_1\n",argv[0]);return 2;
    }
    const char*backend=std::getenv("SPERT_BACKEND");if(!backend||std::strcmp(backend,"spacemit-k3"))return 2;
    const std::pair<const char*,const char*> options[]={
        {"GGML_SPACEMIT_WORKERS","6"},{"GGML_SPACEMIT_ALLOW_EIGHT_WORKERS","0"},
        {"SPACEMIT_PERFER_CORE_ID","8,9,10,11,12,13"},
        {"GGML_SPACEMIT_OCR_FA_QK_SCATTER",argv[1]},{"GGML_SPACEMIT_OCR_QK_PIPELINE",argv[2]}};
    for(const auto&p:options)if(setenv(p.first,p.second,1))return 2;
    // Warm static environment options before launching AI workers.
    (void)ocr_qk_ime2_v53::scatter_enabled();(void)ocr_qk_pipeline::enabled();
    spert::StreamConfig cfg;cfg.n_cores=6;cfg.core_ids={8,9,10,11,12,13};
    spert::Stream stream(cfg);if(!stream.valid()||stream.core_count()!=6)return 3;
    std::array<int,6> cpus{{-1,-1,-1,-1,-1,-1}};std::array<size_t,6> vlens{},grants{};
    std::array<bool,6> entered{};int result=4;
    auto future=stream.launch(spert::Grid{6},[&](spert::Context*runtime){
        const unsigned lane=runtime->program_id(0);if(lane>=6)throw std::runtime_error("unexpected lane");
        cpus[lane]=sched_getcpu();vlens[lane]=__riscv_vlenb();entered[lane]=true;
        if(cpus[lane]!=8+int(lane)||vlens[lane]!=128)throw std::runtime_error("wrong AI worker/VLEN");
        const auto shared=runtime->shared_buffer();grants[lane]=shared.size;
        if(!shared.data||shared.size<393216)throw std::runtime_error("missing expected shared/TCM grant");
        if(lane==0)result=run_cases(shared.data,shared.size);
    });
    const auto status=future.sync();bool placement=true;
    for(unsigned lane=0;lane<6;++lane){
        const bool pass=entered[lane]&&cpus[lane]==8+int(lane)&&vlens[lane]==128&&grants[lane]>=393216;
        placement=placement&&pass;
        std::printf("qk_pair_a1 lane=%u cpu=%d vlenb=%zu shared=%zu %s\n",lane,cpus[lane],vlens[lane],grants[lane],pass?"PASS":"FAIL");
    }
    std::printf("qk_pair_a1 workers=6 scatter=%s pipeline=%s placement=%s result=%d runtime_ok=%d\n",
        argv[1],argv[2],placement?"PASS":"FAIL",result,int(status==spert::Status::Ok));
    return status==spert::Status::Ok&&placement&&result==0?0:4;
}
