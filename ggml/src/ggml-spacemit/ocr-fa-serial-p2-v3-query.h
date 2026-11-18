#pragma once
// Included inside the existing RVV namespace. This header has no ggml/context
// dependency so the same real-RVV query arithmetic can be tested in a stream.
static unsigned ocr_serial_v3_frm() {
    unsigned value;
    asm volatile("csrr %0, frm" : "=r"(value));
    return value;
}

// x = F32(q * attention_scale). High preserves the old two FP16 roundings.
// Low captures F32(x-F32(F16(x))) at 2^20, then the qk helper's 2^-20 result
// is merged at 2^-10 because packed K uses 2^10. Under RNE and |F16(x)|<=63,
// |residual|<=1/64 and the low FP16 input is bounded by 16384 (no overflow).
// STORE=false performs the identical high/low admission test without writes.
template<bool LOW,bool STORE>
static bool ocr_serial_v3_query(const void *src,ptrdiff_t stride_bytes,int rows,
                               int padded_rows,float scale,_Float16 *dst) {
    constexpr int D=72;
    if(!src||rows<1||rows>16||rows>padded_rows||padded_rows%8||
       !std::isfinite(scale)||(STORE&&!dst))return false;
    if constexpr(STORE)if(rows!=padded_rows)
        std::memset(dst,0,size_t(padded_rows)*D*sizeof(_Float16));
    constexpr size_t vl=D;
    const auto ch=__riscv_vid_v_u32m4(vl);
    const auto offsets=__riscv_vsll_vx_u32m4(__riscv_vadd_vv_u32m4(
        __riscv_vsll_vx_u32m4(__riscv_vsrl_vx_u32m4(ch,3,vl),6,vl),
        __riscv_vand_vx_u32m4(ch,7,vl),vl),1,vl);
    for(int row=0;row<rows;++row) {
        const auto *input=reinterpret_cast<const float*>(
            static_cast<const char*>(src)+row*stride_bytes);
        const auto x=__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(input,vl),scale,vl);
        const auto h=__riscv_vfncvt_f_f_w_f16m2(x,vl);
        const auto f=__riscv_vfwcvt_f_f_v_f32m4(h,vl);
        const auto low=__riscv_vfmul_vf_f32m4(__riscv_vfsub_vv_f32m4(x,f,vl),1048576.f,vl);
        const auto bad_high=__riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(f,f,vl),
            __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(f,vl),63.f,vl),vl);
        const auto bad_low=__riscv_vmor_mm_b8(__riscv_vmfne_vv_f32m4_b8(low,low,vl),
            __riscv_vmfgt_vf_f32m4_b8(__riscv_vfabs_v_f32m4(low,vl),65504.f,vl),vl);
        if(__riscv_vcpop_m_b8(__riscv_vmor_mm_b8(bad_high,bad_low,vl),vl))return false;
        if constexpr(STORE) {
            const auto value=LOW?__riscv_vfncvt_f_f_w_f16m2(low,vl):
                __riscv_vfncvt_f_f_w_f16m2(__riscv_vfmul_vf_f32m4(f,1024.f,vl),vl);
            auto *out=dst+size_t(row/8)*8*D+(row%8)*8;
            __riscv_vsuxei32_v_f16m2(out,offsets,value,vl);
        }
    }
    return true;
}
