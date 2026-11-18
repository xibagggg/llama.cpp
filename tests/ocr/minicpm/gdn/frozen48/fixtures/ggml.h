#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
constexpr int GGML_MAX_SRC=10;
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_I64, GGML_TYPE_F16 };
enum ggml_op { GGML_OP_NONE, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_GATED_DELTA_NET, GGML_OP_CPY, GGML_OP_ADD };
constexpr int GGML_TENSOR_FLAG_OUTPUT=1;
constexpr int GGML_TENSOR_FLAG_COMPUTE=2;
struct ggml_tensor {
    ggml_type type=GGML_TYPE_F32;
    ggml_op op=GGML_OP_NONE;
    int flags=0;
    int64_t ne[4]{1,1,1,1};
    size_t nb[4]{4,4,4,4};
    void * data=nullptr;
    ggml_tensor * src[GGML_MAX_SRC]{};
    ggml_tensor * view_src=nullptr;
    size_t view_offs=0;
    int32_t op_params[16]{};
};
struct ggml_cgraph { int n_nodes=0; ggml_tensor ** nodes=nullptr; };
inline int64_t ggml_nelements(const ggml_tensor * t) { return t->ne[0]*t->ne[1]*t->ne[2]*t->ne[3]; }
inline bool ggml_is_contiguous(const ggml_tensor * t) {
    size_t stride=t->type==GGML_TYPE_F32?4:t->type==GGML_TYPE_I64?8:2;
    for(int d=0;d<4;++d){if(t->ne[d]>1 && t->nb[d]!=stride)return false;stride*=size_t(t->ne[d]);}return true;
}
inline size_t ggml_nbytes(const ggml_tensor * t) {
    size_t n=t->type==GGML_TYPE_F32?4:8;
    for(int d=0;d<4;++d)n+=size_t(t->ne[d]-1)*t->nb[d];
    return n;
}
inline bool ggml_is_contiguous_rows(const ggml_tensor * t) { return t->nb[0]==4; }
inline int32_t ggml_get_op_params_i32(const ggml_tensor * t,size_t index) { return t->op_params[index]; }
#define GGML_TENSOR_LOCALS(type,prefix,tensor,field) \
    const type prefix##0=(tensor)->field[0],prefix##1=(tensor)->field[1],prefix##2=(tensor)->field[2],prefix##3=(tensor)->field[3]
#define GGML_ASSERT(x) assert(x)
