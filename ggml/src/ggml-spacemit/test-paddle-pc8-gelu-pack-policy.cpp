#include "paddle-pc8-gelu-pack-policy.h"
#include <array>
#include <cstdio>
#include <limits>

int main(){
    using namespace paddle_pc8_gelu_pack;
    size_t cases=0;bool ok=true;
    auto check=[&](bool value){++cases;ok=ok&&value;};
    check(scratch_bytes==69120);
    for(int worker:{0,1,5,6,7,8})for(uint32_t arch:{0u,0xa064u})for(size_t vlen:{16,32,64,128})
    for(int64_t k:{4304,4352})for(int64_t n:{1024,1152})for(int64_t m:std::array<int64_t,8>{0,1,4,1500,2147483644LL,2147483645LL,2147483647LL,2147483648LL})
    for(size_t shared:{0,69119,69120,393216}){
        const bool want=worker==6&&arch==0xa064&&vlen==128&&k==4352&&n==1152&&m>0&&m<=INT32_MAX-3&&shared>=69120;
        check(eligible(worker,arch,vlen,k,n,m,shared)==want);
    }
    for(int layer_index=0;layer_index<27;++layer_index){
        char up[80],down[80];std::snprintf(up,sizeof(up),"v.blk.%d.ffn_up.weight",layer_index);
        std::snprintf(down,sizeof(down),"v.blk.%d.ffn_down.weight",layer_index);check(same_layer(up,down));
    }
    check(!same_layer("v.blk.0.ffn_up.weight","v.blk.1.ffn_down.weight"));
    for(const char*bad:{"v.blk.27.ffn_up.weight","v.blk.-1.ffn_up.weight","v.blk.0.ffn_up.weight.extra","v.blk.999999999999999999999.ffn_up.weight","mm.1.weight", ""})
        check(!same_layer(bad,"v.blk.0.ffn_down.weight"));
    std::array<unsigned char,300000> memory{};
    const void*input=memory.data()+100000;const void*dst=memory.data()+120000;
    const void*weights=memory.data()+160000;const void*bias=memory.data()+180000;const void*residual=memory.data()+200000;
    check(storage_eligible(input,1000,dst,1000,weights,1000,bias,100,residual,1000));
    check(storage_eligible(input,1000,input,1000,weights,1000,bias,100,residual,1000));
    check(storage_eligible(input,1000,dst,1000,weights,1000,bias,100,dst,1000));
    check(!storage_eligible(input,1000,dst,1000,dst,1000,bias,100,residual,1000));
    check(!storage_eligible(input,1000,dst,1000,weights,1000,dst,100,residual,1000));
    check(!storage_eligible(input,1000,dst,1000,weights,1000,bias,100,static_cast<const unsigned char*>(dst)+1,1000));
    check(scratch_eligible(memory.data(),scratch_bytes,input,1000,memory.data()+80000,1000,dst,1000,weights,1000,bias,100,residual,1000));
    check(!scratch_eligible(memory.data(),scratch_bytes-1,input,1000,memory.data()+80000,1000,dst,1000,weights,1000,bias,100,residual,1000));
    for(size_t offset:{0,128,69119})check(!scratch_eligible(memory.data(),scratch_bytes,memory.data()+offset,1000,memory.data()+80000,1000,dst,1000,weights,1000,bias,100,residual,1000));
    const auto hi=reinterpret_cast<const void*>(std::numeric_limits<uintptr_t>::max()-100);
    check(overlap(hi,100,hi,100));check(!overlap(hi,100,reinterpret_cast<const void*>(1),100));
    std::printf("PC8_GELU_PACK_POLICY cases=%zu status=%s\n",cases,ok?"PASS":"FAIL");return ok?0:1;
}
