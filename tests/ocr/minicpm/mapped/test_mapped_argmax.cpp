// Independent exactness check for the mapped ARGMAX candidate.
// Build with -fno-fast-math -fno-tree-vectorize -fno-tree-slp-vectorize.
// Host prepares and prefaults static fixtures; only AI lane 0 runs RVV. No device
// allocation, model invocation, or cached inferred result is used by the test.
// The SPERT launch is captureless. Earlier failed harnesses and their exact
// source/ELF/log identities remain archived. The observed scalar-pointer
// clobber is isolated below without changing the production RVV helper.
#include "ocr-mapped-argmax-rvv.h"
#include <spert.hpp>
#include <sched.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#if !defined(__riscv_vector)
#error "Requires the real RISC-V vector implementation, not host fallback"
#endif

namespace {
constexpr uint32_t guard32=UINT32_C(0x5ac31e79);
constexpr int64_t guard64=INT64_C(0x3179ad56ef02468b);
constexpr int32_t sentinel=INT32_C(0x3b5a7d19);
constexpr size_t guards=16;
float from_bits(uint32_t bits) { float v;std::memcpy(&v,&bits,4);return v; }
uint32_t to_bits(float v) { uint32_t bits;std::memcpy(&bits,&v,4);return bits; }
bool nan_bits(float v) { return (to_bits(v)&UINT32_C(0x7fffffff))>UINT32_C(0x7f800000); }
struct fixture {
    int64_t count=0, vocabulary=0;
    unsigned mode=0, mapping=0;
    std::vector<float> storage;
    std::vector<int64_t> id_storage;
    int32_t expected=0;
    bool has_nan=false;
    const float * values() const {return storage.data()+guards;}
    const int64_t * ids() const {return id_storage.data()+guards;}
};
struct failure {int code=0;size_t item=0;int64_t actual=0,expected=0;};

// Independent oracle: physically create the original full target vocabulary,
// scatter in draft order, and scan ascending full target order. No compact
// maximum or candidate algorithm is reused in computing the expected ID.
int32_t full_vocabulary_oracle(const float * values,const int64_t * ids,
                              int64_t count,int64_t vocabulary) {
    std::vector<float> full(size_t(vocabulary),-INFINITY);
    for(int64_t i=0;i<count;++i) full[size_t(ids[i])]=values[i];
    float best=-INFINITY;
    int32_t answer=0;
    for(int32_t target=0;target<int32_t(vocabulary);++target) {
        const float value=full[size_t(target)];
        best=best>value?best:value;
        if(best==value)answer=target;
    }
    return answer;
}

// Frozen original compact callback reference, including its full-target-order
// chunk fallback. Used only after the candidate explicitly reports fallback;
// its result must separately match the materialized full-vocabulary oracle.
// Keep scalar reference code out of the RVV caller. In the frozen failing
// executable the compiler inserted vsetvli t1,zero inside this inlined scalar
// pointer loop, overwriting its live t1 values pointer with VL=64; the next
// load used address0x44. This is a harness code-generation boundary only.
__attribute__((noinline,noclone))
int32_t original_callback(const float * values,const int64_t * ids,
                          int64_t count,int64_t vocabulary) {
    float best=-INFINITY;
    int32_t index=int32_t(vocabulary-1);
    bool has_nan=false;
    for(int64_t i=0;i<count;++i) {
        const float value=values[i];
        if(nan_bits(value)){has_nan=true;break;}
        const int32_t target=int32_t(ids[i]);
        if(value>best||(value==best&&target>index)){best=value;index=target;}
    }
    if(!has_nan)return index;
    float chunk[256];best=-INFINITY;index=0;
    for(int64_t begin=0;begin<vocabulary;begin+=256) {
        const int64_t size=std::min<int64_t>(256,vocabulary-begin);
        for(int64_t i=0;i<size;++i)chunk[i]=-INFINITY;
        for(int64_t i=0;i<count;++i) {
            const int64_t offset=ids[i]-begin;
            if(offset>=0&&offset<size)chunk[offset]=values[i];
        }
        for(int64_t i=0;i<size;++i) {
            const float value=chunk[i];best=best>value?best:value;
            if(best==value)index=int32_t(begin+i);
        }
    }
    return index;
}

std::vector<fixture> prepare() {
    std::mt19937 random(0x51ac790d);
    std::vector<fixture> fixtures;
    const float inf=INFINITY;
    // e32m2 at the device VLEN128B has 64 lanes; include every relevant tail
    // boundary and the real 114688-row shape plus its immediate neighbors.
    for(int64_t count:{1,2,3,31,32,33,63,64,65,127,128,129,255,256,257,
                       511,512,513,114687,114688,114689}) {
        const int64_t vocabulary=count>=114687?248100:count+19;
        for(unsigned mapping=0;mapping<3;++mapping) {
            std::vector<int64_t> pool(static_cast<size_t>(vocabulary));
            std::iota(pool.begin(),pool.end(),int64_t(0));
            if(mapping==0) std::reverse(pool.begin(),pool.end());
            if(mapping==1) std::shuffle(pool.begin(),pool.end(),random);
            // mapping2 deliberately excludes the final target for all -inf.
            for(unsigned mode=0;mode<22;++mode) {
                // Keep expensive original NaN fallback on moderate shapes,
                // plus first/middle/last NaNs for the actual production size.
                if(count>=114687&&mode>=14&&mode<=20&&
                   !(count==114688&&mapping==1&&mode<=16))continue;
                fixture f;f.count=count;f.vocabulary=vocabulary;f.mode=mode;f.mapping=mapping;
                f.storage.assign(size_t(count)+2*guards,from_bits(guard32));
                f.id_storage.assign(size_t(count)+2*guards,guard64);
                float * values=f.storage.data()+guards;
                int64_t * ids=f.id_storage.data()+guards;
                for(int64_t i=0;i<count;++i) {
                    ids[i]=pool[size_t(i)];
                    switch(mode) {
                        case 0:values[i]=-inf;break;
                        case 1:values[i]=inf;break;
                        case 2:values[i]=0.0f;break;
                        case 3:values[i]=(i&1)?-0.0f:0.0f;break;
                        case 4:values[i]=float(i);break;
                        case 5:values[i]=float(count-i);break;
                        case 6:values[i]=float(random()%5);break;
                        case 7:values[i]=(random()%13==0)?inf:float(random()%17);break;
                        case 8:values[i]=(random()%13==0)?-inf:-float(random()%17);break;
                        case 9:values[i]=(i&1)?std::numeric_limits<float>::max():-std::numeric_limits<float>::max();break;
                        case 10:values[i]=from_bits(uint32_t(1+random()%127)|((i&1)?UINT32_C(0x80000000):0));break;
                        case 11:values[i]=from_bits(UINT32_C(0x00800000)+uint32_t(random()%127));break;
                        case 12:values[i]=(i==0||i==count-1)?10.0f:-1.0f;break;
                        case 13:values[i]=(i==std::min<int64_t>(63,count-1)||i==std::min<int64_t>(64,count-1))?11.0f:-2.0f;break;
                        case 14:values[i]=i==0?from_bits(UINT32_C(0x7fc12345)):float(random()%7);break;
                        case 15:values[i]=i==count/2?from_bits(UINT32_C(0xffc54321)):float(random()%7);break;
                        case 16:values[i]=i==count-1?from_bits(UINT32_C(0x7f812345)):float(random()%7);break;
                        case 17:values[i]=(i&1)?from_bits(UINT32_C(0x7fc00001)):inf;break;
                        case 18:values[i]=from_bits(UINT32_C(0xff812345));break;
                        case 19:values[i]=(i==0||i==count-1)?from_bits(UINT32_C(0x7fc01001)):-inf;break;
                        case 20:values[i]=ids[i]==pool[0]?from_bits(UINT32_C(0x7f800001)):-0.0f;break;
                        default:values[i]=float(int(random()%200001)-100000)*0.0009765625f;break;
                    }
                    f.has_nan|=nan_bits(values[i]);
                }
                f.expected=full_vocabulary_oracle(values,ids,count,vocabulary);
                fixtures.emplace_back(std::move(f));
            }
        }
    }
    return fixtures;
}

failure run(const std::vector<fixture> & fixtures,size_t & fast,size_t & fallback) {
    for(size_t label=0;label<fixtures.size();++label) {
        const auto & f=fixtures[label];
        std::array<int32_t,3> output={int32_t(guard32),sentinel,int32_t(guard32)};
        const bool accepted=ocr_mapped_argmax_rvv::compute(f.values(),f.ids(),f.count,f.vocabulary,output[1]);
        if(accepted==f.has_nan)return {1,label,accepted,!f.has_nan};
        if(!accepted) {
            ++fallback;
            if(output[1]!=sentinel)return {2,label,output[1],sentinel};
            output[1]=original_callback(f.values(),f.ids(),f.count,f.vocabulary);
        } else ++fast;
        if(output[1]!=f.expected)return {3,label,output[1],f.expected};
        if(output[0]!=int32_t(guard32)||output[2]!=int32_t(guard32))return {4,label,0,1};
        for(size_t i=0;i<guards;++i)
            if(to_bits(f.storage[i])!=guard32||to_bits(f.storage[size_t(f.count)+guards+i])!=guard32||
               f.id_storage[i]!=guard64||f.id_storage[size_t(f.count)+guards+i]!=guard64)return {5,label,0,1};
    }
    int32_t out=sentinel;
    const std::array<std::array<int64_t,2>,4> invalid={{{0,248100},{-1,248100},{1,0},{1,int64_t(INT32_MAX)+1}}};
    for(const auto & pair:invalid) {
        if(ocr_mapped_argmax_rvv::compute(nullptr,nullptr,pair[0],pair[1],out)||out!=sentinel)
            return {6,fixtures.size(),out,sentinel};
    }
    // Reuse precisely the same live addresses with changed values and IDs.
    // These stack fixtures need no worker allocations and detect stale caching.
    std::array<float,257> values;std::array<int64_t,257> ids;
    for(int i=0;i<257;++i){values[size_t(i)]=-INFINITY;ids[size_t(i)]=i;}
    for(int step=0;step<128;++step) {
        values.fill(-INFINITY);
        const int a=(step*37)%257,b=(step*61+1)%257;
        values[size_t(a)]=3.0f;values[size_t(b)]=3.0f;
        std::rotate(ids.begin(),ids.begin()+1,ids.end());
        const int32_t expected=int32_t(std::max(ids[size_t(a)],ids[size_t(b)]));
        out=sentinel;
        if(!ocr_mapped_argmax_rvv::compute(values.data(),ids.data(),257,276,out)||out!=expected)
            return {7,size_t(step),out,expected};
    }
    return {};
}
}

int main() {
    const char * backend=std::getenv("SPERT_BACKEND");
    if(!backend||std::strcmp(backend,"spacemit-k3")){std::puts("FAIL requires SPERT_BACKEND=spacemit-k3");return 10;}
    if(sched_getcpu()<0){std::puts("FAIL sched_getcpu unavailable");return 11;}
    // Keep every object referenced by the device closure in static storage.
    // A worker must never follow a capture into the launching host's stack.
    static auto fixtures=prepare();
    static std::array<std::atomic<int>,6> cpus;for(auto & c:cpus)c.store(-1,std::memory_order_relaxed);
    static std::array<int,6> entry{},exit{};
    static failure result;static size_t fast=0,fallback=0,vlenb=0;
    spert::StreamConfig config;config.n_cores=6;config.core_ids={8,9,10,11,12,13};
    spert::Stream stream(config);
    if(!stream.valid()||stream.core_count()!=6){std::puts("FAIL AI6 stream unavailable");return 12;}
    auto future=stream.launch(spert::Grid{6},[](spert::Context * runtime){
        const unsigned lane=runtime->program_id(0);
        cpus[lane].store(sched_getcpu(),std::memory_order_release);
        entry[lane]=int(runtime->sync());
        if(lane==0) {
            unsigned mask=0;
            for(const auto & c:cpus){const int id=c.load(std::memory_order_acquire);if(id<8||id>13){result={13,0,id,8};break;}mask|=1U<<unsigned(id-8);}
            if(!result.code&&mask!=63)result={14,0,mask,63};
            if(!result.code){vlenb=__riscv_vlenb();if(vlenb!=128)result={15,0,int64_t(vlenb),128};}
            if(!result.code)result=run(fixtures,fast,fallback);
        }
        exit[lane]=int(runtime->sync());
    });
    const auto status=future.sync();
    std::printf("SPERT status=%d workers=6 cpus=",int(status));for(const auto & c:cpus)std::printf(" %d",c.load());
    std::printf(" vlenb=%zu executor_lane=0 fast=%zu fallback=%zu\n",vlenb,fast,fallback);
    for(size_t i=0;i<6;++i)if(entry[i]!=int(spert::Status::Ok)||exit[i]!=int(spert::Status::Ok)){std::printf("FAIL barrier lane=%zu entry=%d exit=%d\n",i,entry[i],exit[i]);return 16;}
    if(status!=spert::Status::Ok||result.code){std::printf("FAIL code=%d case=%zu actual=%lld expected=%lld\n",result.code,result.item,(long long)result.actual,(long long)result.expected);return result.code?result.code:17;}
    std::printf("PASS %zu independently materialized full-vocab cases; exact mapped target-ID ties, signed zeros, infinities, subnormals, qNaN/sNaN original fallback, VL tails, guards, 128 same-address mutations. No model latency claim.\n",fixtures.size());
    return 0;
}
