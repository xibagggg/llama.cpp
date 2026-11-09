#include <spert.hpp>
#include <riscv_vector.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ocr-fa-fullkv.h"

struct alignas(128) result {
    int passed=0;
    bool inputs=true,guards=true;
    double seconds[2][4]{};
};

int main(int argc,char **argv) {
    const int workers=argc==2?std::atoi(argv[1]):1;
    if(workers!=1&&workers!=7)return 1;
    constexpr int iterations=64;
    std::array<result,7> results{};
    const auto info=spert::backend_info();
    if(info.vlen!=128||info.shared_mem_size<128*1024)return 2;
    spert::StreamConfig cfg;cfg.n_cores=workers;
    cfg.core_ids=workers==1?std::vector<int>{8}:std::vector<int>{8,9,10,11,13,14,15};
    std::puts("HOST PV stream-open");std::fflush(stdout);
    spert::Stream stream(cfg);if(!stream.valid())return 3;
    std::puts("HOST PV launch");std::fflush(stdout);
    const auto future=stream.launch(spert::Grid{uint32_t(workers)},[&](spert::Context *rt) {
        auto &r=results[rt->program_id(0)];auto shared=rt->shared_buffer();
        auto *raw=static_cast<unsigned char*>(shared.data);
        auto hash=[](const void *data,size_t bytes) {
            const auto *p=static_cast<const unsigned char*>(data);uint64_t h=1469598103934665603ull;
            for(size_t i=0;i<bytes;++i)h=(h^p[i])*1099511628211ull;
            return h;
        };
        for(int k:{8,16,24,64,72,992,1008,1120})for(int sample=0;sample<4;++sample) {
            const size_t qb=16*k*2,wb=32*k*2;
            const size_t qo=128,wo=qo+qb+128,ao=wo+wb+128,bo=ao+2048+128,total=bo+2048+128;
            std::memset(raw,0xa5,total);
            auto *q=reinterpret_cast<_Float16*>(raw+qo),*w=reinterpret_cast<_Float16*>(raw+wo);
            auto *a=reinterpret_cast<float*>(raw+ao),*b=reinterpret_cast<float*>(raw+bo);
            for(size_t i=0;i<qb/2;++i)q[i]=sample==0?(_Float16)0:(_Float16)((int(i*71%67)-33)*.125f);
            for(size_t i=0;i<wb/2;++i)w[i]=(_Float16)((int(i*41%83)-41)*.0625f);
            if(sample==2)for(size_t i=0;i<wb/2;i+=31)w[i]=(_Float16)std::ldexp(1.f,-24);
            if(sample==3)for(size_t i=0;i<qb/2;i+=17)q[i]=(_Float16)-0.f;
            const auto qhash=hash(q,qb),whash=hash(w,wb);
            for(int tail=0;tail<2;++tail) {
                std::memset(a,0xa5,2048);std::memset(b,0xa5,2048);
                if(tail){ocr_fa_fullkv::pv16x8_reference(q,w,a,k);ocr_pv_pipeline::core8(q,w,b,k);}
                else {ocr_fa_fullkv::pv16_reference(q,w,a,k);ocr_pv_pipeline::core16(q,w,b,k);}
                bool equal=std::memcmp(a,b,2048)==0;
                for(int offset=0;offset<512;++offset)
                    if(!tail||(offset<64||(offset>=256&&offset<320)))equal=equal&&std::isfinite(a[offset])&&std::isfinite(b[offset]);
                if(equal)++r.passed;
                r.inputs=r.inputs&&hash(q,qb)==qhash&&hash(w,wb)==whash;
                for(size_t start:{size_t(0),qo+qb,wo+wb,ao+2048,bo+2048})
                    for(size_t i=0;i<128;++i)r.guards=r.guards&&raw[start+i]==0xa5;
            }
            if(k==1008&&sample==1)for(int tail=0;tail<2;++tail)for(int arm=0;arm<4;++arm) {
                const auto begin=std::chrono::steady_clock::now();
                for(int i=0;i<iterations;++i) {
                    if(tail){if(arm==1||arm==2)ocr_pv_pipeline::core8(q,w,b,k);else ocr_fa_fullkv::pv16x8_reference(q,w,a,k);}
                    else {if(arm==1||arm==2)ocr_pv_pipeline::core16(q,w,b,k);else ocr_fa_fullkv::pv16_reference(q,w,a,k);}
                }
                r.seconds[tail][arm]=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            }
        }
    });
    if(future.sync()!=spert::Status::Ok)return 4;
    for(int i=0;i<workers;++i){const auto&r=results[i];
        std::printf("{\"worker\":%d,\"passed\":%d,\"guards\":%s,\"inputs\":%s,\"iterations\":%d,\"pv32_abba_s\":[%.9f,%.9f,%.9f,%.9f],\"pv8_abba_s\":[%.9f,%.9f,%.9f,%.9f]}\n",
            i,r.passed,r.guards?"true":"false",r.inputs?"true":"false",iterations,
            r.seconds[0][0],r.seconds[0][1],r.seconds[0][2],r.seconds[0][3],
            r.seconds[1][0],r.seconds[1][1],r.seconds[1][2],r.seconds[1][3]);
        if(r.passed!=64||!r.guards||!r.inputs)return 5;
    }
    std::puts("PASS PV32/PV8 byte_exact finite guards inputs");
}
