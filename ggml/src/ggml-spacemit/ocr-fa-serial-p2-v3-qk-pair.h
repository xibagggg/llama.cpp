#pragma once
// Included inside the existing RVV namespace after original v53 QK. This
// helper changes only K movement and independent high/low tile scheduling.
static bool ocr_serial_v3_qk_pair(float*high_scores,float*low_scores,
        const _Float16*high_q,const _Float16*low_q,const _Float16*packed_k,
        int high_stride,int rows,int count,void*scratch) {
    constexpr int D=72;
    if(!high_scores||!low_scores||!high_q||!low_q||!packed_k||!scratch||
       rows<1||rows>16||count<1||count>64||high_stride<count)return false;
    auto *local_k=static_cast<_Float16*>(scratch);
    auto *core_tmp=reinterpret_cast<float*>(local_k+64*D);
    // Copy the same complete original padded tile once, then invoke the exact
    // two original kernels with their explicit preloaded-K interface.
    std::memcpy(local_k,packed_k,64*D*sizeof(_Float16));
    if(!ocr_qk_ime2_v53::qk(high_scores,high_q,local_k,
                          high_stride,rows,count,core_tmp,true))return false;
    if(!ocr_qk_ime2_v53::qk(low_scores,low_q,local_k,
                          64,rows,count,core_tmp,true))return false;
    for(int row=0;row<rows;++row) {
        const size_t vl=count;
        const auto low=__riscv_vle32_v_f32m2(low_scores+row*64,vl);
        const auto high=__riscv_vle32_v_f32m2(high_scores+size_t(row)*high_stride,vl);
        __riscv_vse32_v_f32m2(high_scores+size_t(row)*high_stride,
            __riscv_vfmacc_vf_f32m2(high,1.f/1024.f,low,vl),vl);
    }
    return true;
}
