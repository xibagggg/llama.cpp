#pragma once
// Host-only Q4 -> per-channel INT8 conversion. Each row is independent;
// preserve the scalar dequantization, maximum and nearbyint order exactly.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace ocr_pc8_load {
template<class Decode>
void repack(const block_q4_0 * q, void * output, int k_size, int rows,
            int requested_threads, Decode decode) {
    const int padded_rows=(rows+31)/32*32;
    const int threads=std::max(1,std::min({requested_threads,8,(rows+31)/32}));
    std::memset(output,0,128+size_t(padded_rows)*(sizeof(float)+k_size));
    auto * scales=reinterpret_cast<float*>(static_cast<char*>(output)+128);
    auto * packed=reinterpret_cast<int8_t*>(scales+padded_rows);
    // Allocate and touch scratch on the caller; workers do not use runtime,
    // AI registration, TCM, the ggml allocator, or any shared row storage.
    std::vector<float> scratch(size_t(threads)*k_size);
    std::fenv_t caller_environment;
    std::fegetenv(&caller_environment);
    auto work=[&](int worker) {
        std::fesetenv(&caller_environment);
        float * row=scratch.data()+size_t(worker)*k_size;
        const int groups=(rows+31)/32;
        const int begin=(groups*worker/threads)*32;
        const int end=std::min(rows,(groups*(worker+1)/threads)*32);
        for(int r=begin;r<end;++r) {
            float maximum=0;
            for(int k=0;k<k_size;k+=32) {
                const auto & block=q[size_t(r)*(k_size/32)+k/32];
                const float d=decode(block.d);
                for(int j=0;j<32;++j) {
                    const int v=(j<16?block.qs[j]&15:block.qs[j-16]>>4)-8;
                    row[k+j]=d*v;
                    maximum=std::max(maximum,std::fabs(row[k+j]));
                }
            }
            GGML_ASSERT(std::isfinite(maximum));
            const float scale=maximum/127;
            scales[r]=scale;
            const float inv=scale?1/scale:0;
            for(int k=0;k<k_size;k+=32)for(int j=0;j<32;++j)
                packed[size_t(r/32)*32*k_size+(k/32)*1024+(r%32)*32+j]=
                    int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(row[k+j]*inv))));
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(threads-1);
    int next=1;
    try {
        for(;next<threads;++next)workers.emplace_back(work,next);
    } catch (...) {
        // Resource pressure may prevent thread creation. Complete every
        // unassigned partition on the caller and join existing workers.
        for(;next<threads;++next)work(next);
    }
    work(0);
    for(auto & worker:workers)worker.join();
}
} // namespace ocr_pc8_load
