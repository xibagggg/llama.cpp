#pragma once

#include "ocr-im2col-f32.h"
#include "ggml-impl.h"

namespace ocr_im2col_pack {

// Only this exact adjacent chain is elided. The original graph allocator still
// reserves its FP32 patch storage; this optimization removes writes/reads, not
// that allocation. Packing completes before any GEMM output is written.
inline int match(const ggml_cgraph * graph,int index) {
    if(!graph||index<0||index+2>=graph->n_nodes)return -1;
    const auto * patch=graph->nodes[index];
    const auto * flat=graph->nodes[index+1];
    const auto * matrix=graph->nodes[index+2];
    if(!ocr_im2col_f32::supported(patch)||flat->op!=GGML_OP_RESHAPE||
       flat->src[0]!=patch||flat->view_src!=patch||flat->view_offs!=0||
       matrix->op!=GGML_OP_MUL_MAT||matrix->src[1]!=flat||!matrix->src[0]||
       (patch->flags&GGML_TENSOR_FLAG_OUTPUT)||(flat->flags&GGML_TENSOR_FLAG_OUTPUT)||
       ggml_node_get_use_count(graph,index)!=1||ggml_node_get_use_count(graph,index+1)!=1)return -1;
    const auto * weights=matrix->src[0];
    const int64_t k=patch->ne[0],rows=patch->ne[1]*patch->ne[2]*patch->ne[3];
    if((weights->type!=GGML_TYPE_Q4_0&&weights->type!=GGML_TYPE_Q8_0)||
       weights->ne[0]!=k||k%32||weights->ne[2]!=1||weights->ne[3]!=1||
       flat->ne[0]!=k||flat->ne[1]!=rows||flat->ne[2]!=1||flat->ne[3]!=1||
       !ggml_is_contiguous(flat)||matrix->type!=GGML_TYPE_F32||!ggml_is_contiguous(matrix)||
       matrix->ne[0]!=weights->ne[1]||matrix->ne[1]!=rows||matrix->ne[2]!=1||matrix->ne[3]!=1)return -1;
    return index+2;
}

inline size_t scratch_stride(const ggml_tensor * matrix) {
    return GGML_PAD(size_t(matrix->src[1]->ne[0])*4*sizeof(float),64);
}

inline size_t scratch_bytes(const ggml_tensor * matrix,int threads) {
    GGML_ASSERT(threads>0);
    return scratch_stride(matrix)*size_t(threads);
}

// Quantizers are the same production functions as the unfused path. Four-row
// interleaving and single-row tails remain their responsibility.
template<class QuantizeRow,class QuantizeFour>
inline void pack(const ggml_tensor * patch,int64_t ith,int64_t nth,int64_t block,
                 size_t row_stride,uint8_t * packed,float * scratch,
                 QuantizeRow quantize_row,QuantizeFour quantize_four) {
    GGML_ASSERT(ocr_im2col_f32::supported(patch)&&ith>=0&&ith<nth&&nth>0&&block>0);
    const int64_t k=patch->ne[0],rows=patch->ne[1]*patch->ne[2]*patch->ne[3];
    GGML_ASSERT(k%block==0&&scratch&&packed);
    const int64_t tiles=(rows+3)/4,per_worker=(tiles+nth-1)/nth;
    const int64_t begin=std::min(tiles,ith*per_worker),end=std::min(tiles,begin+per_worker);
    for(int64_t tile=begin;tile<end;++tile) {
        const int64_t first=tile*4,count=std::min(int64_t(4),rows-first);
        ocr_im2col_f32::extract_rows(patch,first,count,scratch);
        if(count==4)quantize_four(size_t(block),scratch,size_t(k),packed+first*row_stride);
        else for(int64_t row=0;row<count;++row)
            quantize_row(size_t(block),scratch+row*k,size_t(k),packed+(first+row)*row_stride);
    }
}
} // namespace ocr_im2col_pack
