#pragma once
// Included after the existing RVV exp/reduction helpers, within their namespace.
// Kept separate from the original f449 softmax; old small paths are unchanged.
template<bool PROFILE>
static bool ocr_serial_v3_softmax_pack(const float*scores,int rows,int n,int padded,int capacity,
                            _Float16*high,_Float16*low,int passes,float*inverse,float*saved_exp=nullptr){
    const int mode=ocr_fa_fullkv::softmax_pack_mode();
    const bool fixed_index=ocr_fa_fullkv::fixed_index_enabled();
    const bool store64=ocr_fa_fullkv::p_store64_enabled();
    // Full-KV validates every Q/K FP16 value before IME2, so every score is
    // finite and exp(score-max) is in [0,1]. Avoid rechecking that invariant
    // for every probability vector. Keep max and positive-sum guards.
    const bool trusted=ocr_fa_fullkv::trust_bounds_enabled();
    const auto lanes=__riscv_vid_v_u32m2(64);
    const auto lane_offsets=__riscv_vsll_vx_u32m2(__riscv_vadd_vv_u32m2(
        __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(lanes,3,64),6,64),
        __riscv_vand_vx_u32m2(lanes,7,64),64),1,64);
    // Each aligned group of four FP16 values is contiguous in the IME2
    // panel. Store its bits as one 64-bit element, reducing scatter lanes
    // without changing exp, reduction grouping or either FP16 rounding.
    const auto words=__riscv_vid_v_u32mf2(16);
    const auto word_offsets=__riscv_vadd_vv_u32mf2(
        __riscv_vsll_vx_u32mf2(__riscv_vsrl_vx_u32mf2(words,1,16),7,16),
        __riscv_vsll_vx_u32mf2(__riscv_vand_vx_u32mf2(words,1,16),3,16),16);
    if(rows!=capacity||n!=padded){
        std::memset(high,0,size_t(capacity)*padded*sizeof(_Float16));
        if(passes==2&&!saved_exp)std::memset(low,0,size_t(capacity)*padded*sizeof(_Float16));
    }
    for(int row=0;row<rows;++row){
        const float*p=scores+size_t(row)*n;
        auto vmax=__riscv_vfmv_v_f_f32m4(-INFINITY,128);
        for(int k=0;k<n;k+=128){const size_t vl=std::min(128,n-k);
            vmax=__riscv_vfmax_vv_f32m4_tu(vmax,vmax,__riscv_vle32_v_f32m4(p+k,vl),vl);}
        const float maximum=__riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(
            vmax,__riscv_vfmv_v_f_f32m1(-INFINITY,1),128));
        if(!std::isfinite(maximum))return false;
        float sum=0;auto vsum=__riscv_vfmv_v_f_f32m2(0,64);
        const bool batch_check=ocr_fa_fullkv::batch_check_enabled();
        auto bad_any=__riscv_vmclr_m_b16(64);
        const size_t base=size_t(row/8)*8*padded+(row%8)*8;
        // vsetvl may split the final AVL in (VLMAX,2*VLMAX) into two
        // balanced vectors. Mode 1 retains the original sum grouping.
        for(int k=0;k<n;){const size_t vl=mode==1?__riscv_vsetvl_e32m2(n-k):std::min(64,n-k);
            const auto f=rvv_expf_approx_f32m2(__riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(p+k,vl),maximum,vl),vl);
            if(!trusted){
                const auto bad=__riscv_vmor_mm_b16(__riscv_vmfne_vv_f32m2_b16(f,f,vl),
                    __riscv_vmor_mm_b16(__riscv_vmflt_vf_f32m2_b16(f,0.f,vl),__riscv_vmfgt_vf_f32m2_b16(f,1.001f,vl),vl),vl);
                // Only full masks can be combined; tail bits are unspecified.
                if(batch_check&&vl==64)bad_any=__riscv_vmor_mm_b16(bad_any,bad,64);
                else if(__riscv_vcpop_m_b16(bad,vl))return false;
            }
            // All maxima for this row are already consumed. Preserve the exact
            // unnormalised exp bits for the later low-P pass: no second exp,
            // changed reduction grouping, approximation or extra FP16 rounding.
            if(saved_exp)__riscv_vse32_v_f32m2(saved_exp+size_t(row)*n+k,f,vl);
            if(mode==2)vsum=__riscv_vfadd_vv_f32m2_tu(vsum,vsum,f,vl);
            else sum+=reduce_sum_f32m2_vlen1024(f,vl);
            const bool aligned=fixed_index&&!(k&7);
            const auto ids=__riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vl),k,vl);
            const auto offsets=aligned?lane_offsets:__riscv_vsll_vx_u32m2(__riscv_vadd_vv_u32m2(
                __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(ids,3,vl),6,vl),
                __riscv_vand_vx_u32m2(ids,7,vl),vl),1,vl);
            const size_t offset_base=base+(aligned?size_t(k)*8:0);
            const auto h=__riscv_vfncvt_f_f_w_f16m1(__riscv_vfmul_vf_f32m2(f,1024.f,vl),vl);
            const bool wide=store64&&!(k&7)&&!(vl&7);
            if(wide)__riscv_vsuxei32_v_u64m1(reinterpret_cast<uint64_t*>(high+base+size_t(k)*8),
                word_offsets,__riscv_vreinterpret_v_u16m1_u64m1(__riscv_vreinterpret_v_f16m1_u16m1(h)),vl/4);
            else __riscv_vsuxei32_v_f16m1(high+offset_base,offsets,h,vl);
            if(passes==2&&!saved_exp){const auto restored=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(h,vl),1.f/1024.f,vl);
                const auto l=__riscv_vfncvt_f_f_w_f16m1(
                    __riscv_vfmul_vf_f32m2(__riscv_vfsub_vv_f32m2(f,restored,vl),16777216.f,vl),vl);
                if(wide)__riscv_vsuxei32_v_u64m1(reinterpret_cast<uint64_t*>(low+base+size_t(k)*8),
                    word_offsets,__riscv_vreinterpret_v_u16m1_u64m1(__riscv_vreinterpret_v_f16m1_u16m1(l)),vl/4);
                else __riscv_vsuxei32_v_f16m1(low+offset_base,offsets,l,vl);}
            k+=int(vl);
        }
        if(mode==2)sum=reduce_sum_f32m2_vlen1024(vsum,64);
        if(!trusted&&batch_check&&__riscv_vcpop_m_b16(bad_any,64))return false;
        if(!(sum>0.f)||!std::isfinite(sum))return false;
        inverse[row]=1.f/sum;
    }
    return true;
}


// Reuse the dead high-P storage after every high-PV panel is complete. The
// saved exp values are identical F32 bits from the first pass. Reconstruct
// the same high half and the same residual low half; never renormalise here.
static bool ocr_serial_v3_pack_saved_residual(const float*saved_exp,int rows,int n,int padded,
                                    int capacity,_Float16*low){
    if(!saved_exp||!low||rows<1||rows>capacity||capacity!=16||n<1||padded<n||padded%8)return false;
    const bool fixed_index=ocr_fa_fullkv::fixed_index_enabled();
    const bool store64=ocr_fa_fullkv::p_store64_enabled();
    const auto lanes=__riscv_vid_v_u32m2(64);
    const auto lane_offsets=__riscv_vsll_vx_u32m2(__riscv_vadd_vv_u32m2(
        __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(lanes,3,64),6,64),
        __riscv_vand_vx_u32m2(lanes,7,64),64),1,64);
    const auto words=__riscv_vid_v_u32mf2(16);
    const auto word_offsets=__riscv_vadd_vv_u32mf2(
        __riscv_vsll_vx_u32mf2(__riscv_vsrl_vx_u32mf2(words,1,16),7,16),
        __riscv_vsll_vx_u32mf2(__riscv_vand_vx_u32mf2(words,1,16),3,16),16);
    if(rows!=capacity||n!=padded)std::memset(low,0,size_t(capacity)*padded*sizeof(_Float16));
    for(int row=0;row<rows;++row){
        const size_t base=size_t(row/8)*8*padded+(row%8)*8;
        for(int k=0;k<n;){
            // Serial P2 is admitted only with production softmax mode 1.
            const size_t vl=__riscv_vsetvl_e32m2(n-k);
            const auto f=__riscv_vle32_v_f32m2(saved_exp+size_t(row)*n+k,vl);
            const auto h=__riscv_vfncvt_f_f_w_f16m1(__riscv_vfmul_vf_f32m2(f,1024.f,vl),vl);
            const auto restored=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(h,vl),1.f/1024.f,vl);
            const auto l=__riscv_vfncvt_f_f_w_f16m1(
                __riscv_vfmul_vf_f32m2(__riscv_vfsub_vv_f32m2(f,restored,vl),16777216.f,vl),vl);
            const bool aligned=fixed_index&&!(k&7);
            const auto ids=__riscv_vadd_vx_u32m2(__riscv_vid_v_u32m2(vl),k,vl);
            const auto offsets=aligned?lane_offsets:__riscv_vsll_vx_u32m2(__riscv_vadd_vv_u32m2(
                __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(ids,3,vl),6,vl),
                __riscv_vand_vx_u32m2(ids,7,vl),vl),1,vl);
            const bool wide=store64&&!(k&7)&&!(vl&7);
            if(wide)__riscv_vsuxei32_v_u64m1(reinterpret_cast<uint64_t*>(low+base+size_t(k)*8),
                word_offsets,__riscv_vreinterpret_v_u16m1_u64m1(__riscv_vreinterpret_v_f16m1_u16m1(l)),vl/4);
            else __riscv_vsuxei32_v_f16m1(low+base+(aligned?size_t(k)*8:0),offsets,l,vl);
            k+=int(vl);
        }
    }
    return true;
}

// A large shape enters only after the complete operation was validated at
// preparation. Invalid/nonfinite operations return before any serial output
// store and retain exactly the original legacy prepared online fallback.
template<bool PROFILE>
static bool forward_ocr_serial_p2_v3(ggml::spacemit::context &ctx,ggml_tensor *dst,int ir0,int ir1) {
    if(!ocr_fa_pack_v40::serial_v3(dst)||
       ocr_fa_pack_v40::serial_v3_lane(ctx,dst,ctx.ith)[1]!=1)return false;
    constexpr int D=72,Q=16;
    const auto *q=dst->src[0];const int n=int(q->ne[1]);
    const int padded=int(ocr_fa_pack_v40::full_k(dst));
    const size_t qk_bytes=64*D*sizeof(_Float16)+512*sizeof(float);
    const size_t needed=ocr_fa_serial_p2_v3::bytes(n);
    GGML_ASSERT(ctx.shared.data&&needed<=ctx.shared.size&&ir0>=0&&ir1<=n*16);
    float scale;std::memcpy(&scale,dst->op_params,sizeof(scale));
    auto *raw=static_cast<_Float16*>(ctx.shared.data);
    auto *packed_q=raw+Q*D;
    void *qk_scratch=packed_q+Q*D;
    auto *scores=reinterpret_cast<float*>(static_cast<char*>(qk_scratch)+qk_bytes);
    auto *prob=reinterpret_cast<_Float16*>(scores+Q*n);
    auto *weights=prob+Q*padded;
    auto *acc_high=reinterpret_cast<float*>(weights+D*padded);
    auto *acc_low=acc_high+3*512;
    // P does not become live until after both QK passes. One 16x64 F32 tile
    // fits in that dead P region, independent of the raw/packed Q buffers.
    auto *low_scores=reinterpret_cast<float*>(prob);
    static_assert(Q*64*sizeof(float)<=Q*1128*sizeof(_Float16),"low-score alias capacity");
    ocr_fa_phases_v50::record phase(ctx,dst,"fullkv-s3");
    if constexpr(PROFILE)phase.begin();
    int loaded_head=-1;
    for(int ir=ir0;ir<ir1;) {
        const int head=ir/n,iq=ir%n,rows=std::min(Q,std::min(ir1-ir,n-iq));
        if(head!=loaded_head) {
            const auto *shared_v=ocr_fa_pack_v40::full_value_head(ctx,dst,head);
            std::memcpy(weights,shared_v,size_t(D)*padded*sizeof(_Float16));
            loaded_head=head;
        }
        if constexpr(PROFILE)phase.tick(7);
        const auto *src=reinterpret_cast<const uint8_t*>(q->data)+iq*q->nb[1]+head*q->nb[2];
        GGML_ASSERT((ocr_serial_v3_query<false,true>(src,q->nb[1],rows,(rows+7)/8*8,scale,packed_q)));
        if constexpr(PROFILE)phase.tick(2);
        for(int ic=0;ic<n;ic+=64) {
            const auto *key=ocr_fa_pack_v40::ime_key_tile(ctx,dst,
                size_t(head)*ocr_fa_pack_v40::tiles(dst)+ic/64);
            GGML_ASSERT(ocr_qk_ime2_v53::qk(scores+ic,packed_q,key,n,rows,std::min(64,n-ic),qk_scratch));
        }
        // Re-read Q and recreate the same rounded high half. No content cache,
        // address tag, extra Q buffer, or changed high-QK accumulation order.
        GGML_ASSERT((ocr_serial_v3_query<true,true>(src,q->nb[1],rows,(rows+7)/8*8,scale,packed_q)));
        if constexpr(PROFILE)phase.tick(2);
        for(int ic=0;ic<n;ic+=64) {
            const int count=std::min(64,n-ic);
            const auto *key=ocr_fa_pack_v40::ime_key_tile(ctx,dst,
                size_t(head)*ocr_fa_pack_v40::tiles(dst)+ic/64);
            GGML_ASSERT(ocr_qk_ime2_v53::qk(low_scores,packed_q,key,64,rows,count,qk_scratch));
            for(int row=0;row<rows;++row) {
                const size_t vl=count;
                const auto low=__riscv_vle32_v_f32m2(low_scores+row*64,vl);
                const auto high=__riscv_vle32_v_f32m2(scores+size_t(row)*n+ic,vl);
                __riscv_vse32_v_f32m2(scores+size_t(row)*n+ic,
                    __riscv_vfmacc_vf_f32m2(high,1.f/1024.f,low,vl),vl);
            }
        }
        if constexpr(PROFILE)phase.tick(5);
        float inverse[Q];
        GGML_ASSERT(ocr_serial_v3_softmax_pack<PROFILE>(scores,rows,n,padded,Q,prob,nullptr,2,inverse,scores));
        if constexpr(PROFILE)phase.tick(6);
        for(int channel=0;channel<D;channel+=32) {
            const auto multiply=channel==64?ocr_fa_fullkv::pv16x8:ocr_fa_fullkv::pv16;
            multiply(prob,weights+channel*padded,acc_high+(channel/32)*512,padded);
        }
        if constexpr(PROFILE)phase.tick(8);
        GGML_ASSERT(ocr_serial_v3_pack_saved_residual(scores,rows,n,padded,Q,prob));
        if constexpr(PROFILE)phase.tick(6);
        for(int channel=0;channel<D;channel+=32) {
            const auto multiply=channel==64?ocr_fa_fullkv::pv16x8:ocr_fa_fullkv::pv16;
            multiply(prob,weights+channel*padded,acc_low,padded);
            if constexpr(PROFILE)phase.tick(8);
            const int columns=std::min(32,D-channel);
            const auto ids=__riscv_vid_v_u32m1(columns);
            const auto offsets=__riscv_vsll_vx_u32m1(__riscv_vadd_vv_u32m1(
                __riscv_vsll_vx_u32m1(__riscv_vsrl_vx_u32m1(ids,3,columns),6,columns),
                __riscv_vand_vx_u32m1(ids,7,columns),columns),2,columns);
            for(int row=0;row<rows;++row) {
                const int offset=(row/8)*256+(row%8)*8;
                auto value=__riscv_vfmul_vf_f32m1(__riscv_vluxei32_v_f32m1(
                    acc_high+(channel/32)*512+offset,offsets,columns),1.f/1048576.f,columns);
                value=__riscv_vfmacc_vf_f32m1(value,1.f/17179869184.f,
                    __riscv_vluxei32_v_f32m1(acc_low+offset,offsets,columns),columns);
                value=__riscv_vfmul_vf_f32m1(value,inverse[row],columns);
                auto *out=reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data)+
                    (size_t(iq+row)*dst->ne[1]+head)*dst->nb[1])+channel;
                __riscv_vse32_v_f32m1(out,value,columns);
            }
            if constexpr(PROFILE)phase.tick(9);
        }
        ir+=rows;
    }
    if constexpr(PROFILE)phase.finish();
    ocr_worker_events::publish(ocr_worker_events::kind::full_kv,
        {uint64_t(n),uint64_t(Q),uint64_t(2),uint64_t(0),needed,ctx.shared.size});
    return true;
}
