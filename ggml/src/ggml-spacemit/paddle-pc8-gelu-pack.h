#pragma once
#include "paddle-pc8-gelu-pack-policy.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <riscv_vector.h>

namespace paddle_pc8_gelu_pack {
// Exact current PC8 token quantizer: do not share max/scale between rows,
// change FRM, narrow GELU to F16, or include uninitialised padded elements.
inline void pack_group(const float*input,float*scratch,int8_t*packed,float*scales,int rows){
    GGML_ASSERT(rows>=1&&rows<=group_rows);
    std::memcpy(scratch,input,size_t(rows)*input_k*sizeof(float));
    spacemit_kernels::rvv::forward_gelu_f32_tile(scratch,size_t(rows)*input_k);
    for(int r=0;r<group_rows;++r){
        if(r>=rows){
            scales[r]=0;
            for(int k=0;k<packed_k;k+=32)std::memset(packed+k*4+r*32,0,32);
            continue;
        }
        auto vmax=__riscv_vfmv_v_f_f32m2(0,64);
        for(int k=0;k<input_k;k+=64){
            const size_t vl=std::min(64,input_k-k);
            const auto a=__riscv_vfabs_v_f32m2(__riscv_vle32_v_f32m2(scratch+size_t(r)*input_k+k,vl),vl);
            vmax=__riscv_vfmax_vv_f32m2_tu(vmax,vmax,a,vl);
        }
        const float maximum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(
            vmax,__riscv_vfmv_v_f_f32m1(0,1),64));
        const float scale=maximum/127;GGML_ASSERT(std::isfinite(maximum));
        scales[r]=scale;const float inv=scale?1/scale:0;
        for(int k=0;k<packed_k;k+=32){
            auto*target=packed+k*4+r*32;
            const size_t vl=std::min(32,std::max(0,input_k-k));
            if(vl<32)std::memset(target,0,32);if(!vl)continue;
            const auto a=__riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(scratch+size_t(r)*input_k+k,vl),inv,vl);
            const auto q=__riscv_vncvt_x_x_w_i8mf4(__riscv_vfncvt_x_f_w_i16mf2(a,vl),vl);
            __riscv_vse8_v_i8mf4(target,q,vl);
        }
    }
}
} // namespace paddle_pc8_gelu_pack
