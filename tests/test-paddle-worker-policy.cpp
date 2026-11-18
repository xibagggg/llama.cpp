#include "../ggml/src/ggml-spacemit/ocr-worker-policy.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
static void require(bool x){if(!x)std::abort();}
int main() {
    using namespace ocr_worker_policy;
    std::size_t policies=0,grants=0;
    for(int available=0;available<=16;++available)for(bool allow:{false,true})
    for(int requested=-2;requested<=10;++requested) {
        const bool expected=requested>=1&&requested<=(allow?8:6)&&requested<=available;
        require(requested_count(requested,available,allow)==expected);++policies;
    }
    std::array<int,8> preferred{{8,9,10,11,12,13,14,15}},observed=preferred;
    for(std::size_t n:{std::size_t(6),std::size_t(7),std::size_t(8)}) {
        do {
            require(exact_grant(n,n,preferred.data(),n,observed.data(),n));
            require(!exact_grant(n,n-1,preferred.data(),n,observed.data(),n));
            require(!exact_grant(n,n,preferred.data(),n-1,observed.data(),n));
            require(!exact_grant(n,n,preferred.data(),n,observed.data(),n-1));
            ++grants;
        } while(std::next_permutation(observed.begin(),observed.begin()+n));
        observed=preferred;
    }
    for(int slot=0;slot<8;++slot)for(int bad:{-1,0,7,16,255}) {
        auto copy=preferred;copy[slot]=bad;require(!ai_mask(copy.data(),copy.size()));
    }
    for(int a=0;a<8;++a)for(int b=0;b<8;++b)if(a!=b) {
        auto copy=preferred;copy[b]=copy[a];require(!ai_mask(copy.data(),copy.size()));
    }
    require(!ai_mask(preferred.data(),0));require(!ai_mask(preferred.data(),9));
    auto different=preferred;different[6]=15;
    require(!exact_grant(7,7,preferred.data(),7,different.data(),7));
    std::printf("worker policy PASS count_cases=%zu unique_exact_grant_permutations=%zu\n",policies,grants);
}
