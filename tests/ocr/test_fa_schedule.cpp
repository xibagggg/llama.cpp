#include "../../ggml/src/ggml-spacemit/ocr-fa-schedule.h"
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
    int cases=0;
    for(int64_t rows:{64,252,256,1008,4096})
    for(int64_t heads:{1,7,16})
    for(int64_t chunk:{128,256,512,1024})
    for(int workers:{1,7,8}) {
        const auto total=ocr_fa_schedule::units(rows,heads,chunk);
        std::atomic<int64_t> queue{0};
        std::vector<std::atomic<int>> visits(rows*heads);
        for(auto &n:visits)n.store(0);
        std::vector<std::thread> threads;
        for(int i=0;i<workers;++i)threads.emplace_back([&]{
            for(;;) {
                const auto id=queue.fetch_add(1);
                if(id>=total)break;
                const auto r=ocr_fa_schedule::task(id,rows,chunk);
                if(r.begin<0||r.end>rows*heads||r.end<=r.begin||
                   r.begin/rows!=(r.end-1)/rows)std::abort();
                for(auto n=r.begin;n<r.end;++n)visits[n].fetch_add(1);
            }
        });
        for(auto &t:threads)t.join();
        for(auto &n:visits)if(n.load()!=1)return 1;
        ++cases;
    }
    std::printf("PASS %d dynamic attention range/ownership configurations\n",cases);
}
