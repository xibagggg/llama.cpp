#include "paddle-f16-gelu-pack-policy.h"
#include <array>
#include <cstdio>
#include <limits>
int main() {
    using namespace paddle_f16_gelu_pack;
    size_t cases=0;
    for(int workers=1;workers<=8;++workers)
    for(size_t vlen:{size_t(64),size_t(128),size_t(256)})
    for(int k:{1152,4304,4608})for(int n:{1024,1152,4304})
    for(int m:{0,1,7,8,9,848})
    for(size_t bytes:{size_t(0),scratch_bytes-1,scratch_bytes,scratch_bytes+1})
    for(bool wide:{false,true}) {
        bool wanted=workers==6&&vlen==128&&k==4304&&n==1152&&m>0&&
            bytes>=scratch_bytes&&wide;
        if(shape_eligible(workers,vlen,k,n,m,bytes,wide)!=wanted)return 1;
        if(eligible(workers,vlen,k,n,m,bytes,wide)!=(wanted&&requested()))return 2;
        ++cases;
    }
    std::array<unsigned char,2048> b{};
    auto*p=b.data();
    // Exact adjacency is safe. Input/destination reuse is intentionally safe;
    // every input group is packed before any destination store.
    if(!storage_eligible(p,512,p,512,p+512,512,p+1024,512))return 3;
    if(!storage_eligible(p,512,p+128,384,p+512,512,nullptr,0))return 4;
    if(storage_eligible(p,512,p+128,512,p+512,512,nullptr,0))return 5;
    if(storage_eligible(p,512,p,512,p+1024,512,p+511,512))return 6;
    if(storage_eligible(nullptr,512,p,512,p+512,512,nullptr,0))return 7;
    if(storage_eligible(p,512,nullptr,512,p+512,512,nullptr,0))return 8;
    if(storage_eligible(p,512,p,512,nullptr,512,nullptr,0))return 9;
    if(storage_eligible(p,512,p,0,p+512,512,nullptr,0))return 10;
    if(storage_eligible(p,512,p,512,p+512,0,nullptr,0))return 11;
    // Subtraction-based interval checks do not add addresses and overflow.
    auto high=reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-31);
    if(!nonoverlap(p,512,high,32)||nonoverlap(high,32,high,32))return 12;
    std::printf("F16_GELU_PACK_POLICY PASS cases=%zu requested=%d six_only precise_shape_tcm_wide_pack alias_guards\n",cases,int(requested()));
}
