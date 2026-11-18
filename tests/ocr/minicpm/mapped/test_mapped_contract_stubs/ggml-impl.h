#pragma once
// Metadata-only fixture ABI. This does not represent the model/backend ABI.
#include <cstddef>
#include <cstdint>
constexpr int GGML_MAX_SRC=10;
enum ggml_type {GGML_TYPE_F32,GGML_TYPE_I64,GGML_TYPE_I32,GGML_TYPE_F16};
enum ggml_op {GGML_OP_NONE,GGML_OP_CUSTOM,GGML_OP_ADD,GGML_OP_MAP_CUSTOM1};
struct ggml_tensor {
    char name[64]{};ggml_type type=GGML_TYPE_F32;ggml_op op=GGML_OP_NONE;
    int64_t ne[4]{};size_t nb[4]{};void * data=nullptr;
    ggml_tensor * view_src=nullptr;ggml_tensor * src[GGML_MAX_SRC]{};
    alignas(void *) int32_t op_params[16]{};
};
using ggml_custom_op_t=void(*)(ggml_tensor *,int,int,void *);
struct ggml_custom_op_params {ggml_custom_op_t fun;int n_tasks;void * userdata;};
inline size_t ggml_nbytes(const ggml_tensor * t){
    for(int i=0;i<4;++i)if(t->ne[i]<=0)return 0;
    size_t bytes=t->type==GGML_TYPE_I64?8:(t->type==GGML_TYPE_F16?2:4);
    for(int i=0;i<4;++i)bytes+=size_t(t->ne[i]-1)*t->nb[i];
    return bytes;
}
inline bool ggml_is_contiguous(const ggml_tensor * t){
    size_t bytes=t->type==GGML_TYPE_I64?8:(t->type==GGML_TYPE_F16?2:4);
    // Matches current ggml.c:1485-1510 for these unblocked scalar types:
    // strides in dimensions of extent1 are irrelevant, including ne[0]==1.
    if(t->ne[0]!=1&&t->nb[0]!=bytes)return false;
    bytes*=size_t(t->ne[0]);
    for(int i=1;i<4;++i){if(t->ne[i]!=1&&t->nb[i]!=bytes)return false;bytes*=size_t(t->ne[i]);}
    return true;
}
