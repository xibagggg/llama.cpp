#pragma once
#include <cstddef>

namespace paddle_f16_ime2 {
// Keep this decision independent of tensor addresses and advertised capacity.
// Only the actual per-worker TCM grant may admit a larger staging layout.
struct kblock_plan { int block_k; int panels; };
constexpr std::size_t kblock_tcm_bytes(int block_k,int panels) {
    return std::size_t(panels)*32*block_k*2 + 2*std::size_t(16)*block_k*2 +
           2*std::size_t(panels)*512*sizeof(float) + 256;
}
static_assert(sizeof(float)==4,"K3 accumulators require FP32");
static_assert(kblock_tcm_bytes(1024,4)==344320,"K1024 staging budget changed");
static_assert(kblock_tcm_bytes(1440,3)==381184,"K1440 staging budget changed");
constexpr kblock_plan make_kblock_plan(int requested,std::size_t available) {
    return requested==1440 && available>=kblock_tcm_bytes(1440,3) ? kblock_plan{1440,3} :
           (requested==1024 || requested==1440) && available>=kblock_tcm_bytes(1024,4) ?
           kblock_plan{1024,4} : kblock_plan{0,0};
}
}
