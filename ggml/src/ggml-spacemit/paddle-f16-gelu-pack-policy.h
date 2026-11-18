#pragma once
// Explicit, narrow, default-off admission. No worker or precision expansion.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
namespace paddle_f16_gelu_pack {
inline constexpr size_t scratch_bytes=128+size_t(8)*4304*sizeof(float)+128;
inline bool requested() {
    static const bool v=[] {const char*p=std::getenv("GGML_SPACEMIT_PADDLE_F16_GELU_PACK");
        return p&&!std::strcmp(p,"1");}();
    return v;
}
inline bool shape_eligible(int64_t workers,size_t vlenb,int64_t k,int64_t n,int64_t m,
                           size_t tcm_bytes,bool wide_pack) {
    return workers==6&&vlenb==128&&k==4304&&n==1152&&m>0&&
        tcm_bytes>=scratch_bytes&&wide_pack;
}
inline bool eligible(int64_t workers,size_t vlenb,int64_t k,int64_t n,int64_t m,
                     size_t tcm_bytes,bool wide_pack) {
    return requested()&&shape_eligible(workers,vlenb,k,n,m,tcm_bytes,wide_pack);
}
inline bool nonoverlap(const void *a,size_t an,const void *b,size_t bn) {
    if(!a||!an||!b||!bn)return false;
    const auto ap=reinterpret_cast<uintptr_t>(a),bp=reinterpret_cast<uintptr_t>(b);
    return ap<=bp?bp-ap>=an:ap-bp>=bn;
}
inline bool storage_eligible(const void *input,size_t input_bytes,
    const void *destination,size_t destination_bytes,const void *weight,size_t weight_bytes,
    const void *bias,size_t bias_bytes) {
    return input&&input_bytes&&nonoverlap(destination,destination_bytes,weight,weight_bytes)&&
        (!bias||nonoverlap(destination,destination_bytes,bias,bias_bytes));
}
} // namespace paddle_f16_gelu_pack
