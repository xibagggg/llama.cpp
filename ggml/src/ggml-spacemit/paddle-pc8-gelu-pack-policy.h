#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>

namespace paddle_pc8_gelu_pack {
inline constexpr int input_k=4304,packed_k=4352,group_rows=4;
inline constexpr size_t scratch_bytes=128+size_t(group_rows)*input_k*sizeof(float)+128;
inline bool overlap(const void*a,size_t an,const void*b,size_t bn){
    if(!a||!b||!an||!bn)return false;
    const auto ap=reinterpret_cast<uintptr_t>(a),bp=reinterpret_cast<uintptr_t>(b);
    return ap<=bp?bp-ap<an:ap-bp<bn;
}
inline int layer(const char*name,const char*suffix){
    if(!name||std::strncmp(name,"v.blk.",6))return -1;
    const char*p=name+6;if(*p<'0'||*p>'9')return -1;
    unsigned value=0;while(*p>='0'&&*p<='9'){
        if(value>26)return -1;
        value=value*10+unsigned(*p++-'0');
    }
    return value<=26&&!std::strcmp(p,suffix)?int(value):-1;
}
inline bool same_layer(const char*up,const char*down){
    const int index=layer(up,".ffn_up.weight");return index>=0&&index==layer(down,".ffn_down.weight");
}
inline bool eligible(int workers,uint32_t arch,size_t vlen,int64_t k,int64_t n,int64_t m,size_t shared){
    return workers==6&&arch==0xa064&&vlen==128&&k==packed_k&&n==1152&&m>0&&
        m<=std::numeric_limits<int32_t>::max()-3&&shared>=scratch_bytes;
}
inline bool storage_eligible(const void*input,size_t in_bytes,const void*dst,size_t dst_bytes,
    const void*weights,size_t weight_bytes,const void*bias,size_t bias_bytes,
    const void*residual,size_t residual_bytes){
    if(!input||!in_bytes||!dst||!dst_bytes||!weights||!weight_bytes||
       (bias_bytes&&!bias)||(residual_bytes&&!residual))return false;
    // Input is consumed in full before the original barrier. Its allocation
    // may be reused by dst. Constant weights/bias may not be overwritten;
    // residual accepts only disjoint or exact in-place elementwise storage.
    return !overlap(dst,dst_bytes,weights,weight_bytes)&&!overlap(dst,dst_bytes,bias,bias_bytes)&&
        (!overlap(dst,dst_bytes,residual,residual_bytes)||
         (dst==residual&&dst_bytes==residual_bytes));
}
inline bool scratch_eligible(const void*shared,size_t shared_bytes,const void*input,size_t input_bytes,
    const void*workspace,size_t workspace_bytes,const void*dst,size_t dst_bytes,
    const void*weights,size_t weight_bytes,const void*bias,size_t bias_bytes,
    const void*residual,size_t residual_bytes){
    if(!shared||shared_bytes<scratch_bytes||!workspace||!workspace_bytes)return false;
    // Validate the complete guarded view, not just real tail rows.
    return !overlap(shared,scratch_bytes,input,input_bytes)&&
        !overlap(shared,scratch_bytes,workspace,workspace_bytes)&&
        !overlap(shared,scratch_bytes,dst,dst_bytes)&&
        !overlap(shared,scratch_bytes,weights,weight_bytes)&&
        !overlap(shared,scratch_bytes,bias,bias_bytes)&&
        !overlap(shared,scratch_bytes,residual,residual_bytes);
}
} // namespace paddle_pc8_gelu_pack
