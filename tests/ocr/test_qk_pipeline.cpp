#include <riscv_vector.h>
#include <spert.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "ocr-qk-ime2-v53.h"

struct result { int passed=0; bool guards=true,inputs=true; double seconds[4]{}; };

int main() {
    constexpr int workers=7,cases=12,iterations=400;
    std::array<result,workers> results{};
    alignas(64) std::array<_Float16,16*72> queries{};
    alignas(64) std::array<_Float16,32*72> keys{};
    for (size_t i=0;i<queries.size();++i) queries[i]=_Float16((int(i*71%521)-260)*.0625f);
    for (size_t i=0;i<keys.size();++i) keys[i]=_Float16((int(i*41%733)-366)*.03125f);
    std::puts("HOST backend-info-begin");std::fflush(stdout);
    const auto info=spert::backend_info();
    if(info.vlen!=128||info.shared_mem_size<16384)return 2;
    std::puts("HOST backend-info-end stream-open-begin");std::fflush(stdout);
    spert::StreamConfig cfg;cfg.n_cores=workers;cfg.core_ids={8,9,10,11,13,14,15};
    spert::Stream stream(cfg);if(!stream.valid())return 3;
    std::puts("HOST stream-open-end launch-begin");std::fflush(stdout);
    auto future=stream.launch(spert::Grid{workers},[&](spert::Context * rt) {
        auto & r=results[rt->program_id(0)];auto shared=rt->shared_buffer();
        auto * raw=static_cast<unsigned char*>(shared.data);
        // All inputs, outputs and guards live in this worker's own TCM.
        auto * q=reinterpret_cast<_Float16*>(raw+128);
        auto * k=reinterpret_cast<_Float16*>(raw+128+sizeof(queries));
        auto * a=reinterpret_cast<float*>(raw+8192+128);
        auto * b=reinterpret_cast<float*>(raw+12288+128);
        for(int sample=0;sample<cases;++sample) {
            std::memset(raw,0xa5,16384);
            std::memcpy(q,queries.data(),sizeof(queries));std::memcpy(k,keys.data(),sizeof(keys));
            if(sample==1)std::memset(q,0,sizeof(queries));
            if(sample==2)std::memset(k,0,sizeof(keys));
            if(sample>=3) {
                for(size_t i=0;i<queries.size();++i) q[i]=_Float16(float(q[i])*(sample-2)*.125f);
                for(size_t i=0;i<keys.size();i+=31)k[i]=_Float16(std::ldexp(1.f,-24));
            }
            // Fingerprint bytes on the worker before and after both calls.
            auto hash=[](const void * data,size_t size) {
                const auto * p=static_cast<const unsigned char*>(data);uint64_t h=1469598103934665603ull;
                for(size_t i=0;i<size;++i)h=(h^p[i])*1099511628211ull;
                return h;
            };
            const auto input_hash=hash(q,sizeof(queries)+sizeof(keys));
            ocr_qk_ime2_v53::core16(q,k,a);ocr_qk_pipeline::core16(q,k,b);
            r.inputs=r.inputs&&input_hash==hash(q,sizeof(queries)+sizeof(keys));
            bool equal=std::memcmp(a,b,2048)==0;
            for(int i=0;i<512;++i)equal=equal&&std::isfinite(a[i])&&std::isfinite(b[i]);
            if(equal)++r.passed;
            for(int i=0;i<128;++i)r.guards=r.guards&&raw[i]==0xa5&&raw[8192+i]==0xa5&&
                raw[8192+128+2048+i]==0xa5&&raw[12288+i]==0xa5&&raw[12288+128+2048+i]==0xa5;
        }
        for(int arm=0;arm<4;++arm) {
            const auto start=std::chrono::steady_clock::now();
            for(int iteration=0;iteration<iterations;++iteration) {
                if(arm==1||arm==2)ocr_qk_pipeline::core16(q,k,b);
                else ocr_qk_ime2_v53::core16(q,k,a);
            }
            r.seconds[arm]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        }
    });
    if(future.sync()!=spert::Status::Ok)return 4;
    std::puts("HOST sync-end");
    for(int worker=0;worker<workers;++worker) {
        const auto & r=results[worker];
        std::printf("{\"worker\":%d,\"passed\":%d,\"guards\":%s,\"inputs\":%s,\"iterations_per_arm\":%d,\"abba_seconds\":[%.9f,%.9f,%.9f,%.9f]}\n",
            worker,r.passed,r.guards?"true":"false",r.inputs?"true":"false",iterations,
            r.seconds[0],r.seconds[1],r.seconds[2],r.seconds[3]);
        if(r.passed!=cases||!r.guards||!r.inputs)return 5;
    }
    std::puts("PASS QK old/new=byte_exact cases=84 guards=preserved inputs=preserved");
}
