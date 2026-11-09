#pragma once

// Additional OCR operators executed by SPERT workers, never by the host CPU
// backend. Matrix kernels retain IME2 dispatch; irregular/indexing operators
// use the AI cores' scalar/vector instruction sets.
#include "ggml-impl.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cfloat>

namespace ocr_ai_complete {
inline bool floating(const ggml_tensor * t) {
    return t && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
}
inline size_t offset(const ggml_tensor * t, int64_t index) {
    size_t result = 0;
    for (int d = 0; d < 4; ++d) { result += (index % t->ne[d]) * t->nb[d]; index /= t->ne[d]; }
    return result;
}
inline float read(const ggml_tensor * t, size_t pos) {
    const char * p = static_cast<const char *>(t->data) + pos;
    return t->type == GGML_TYPE_F32 ? *reinterpret_cast<const float *>(p)
                                   : ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(p));
}
inline void write(ggml_tensor * t, int64_t index, float value) {
    char * p = static_cast<char *>(t->data) + offset(t, index);
    if (t->type == GGML_TYPE_F32) *reinterpret_cast<float *>(p) = value;
    else *reinterpret_cast<ggml_fp16_t *>(p) = ggml_fp32_to_fp16(value);
}
inline bool supported(const ggml_tensor * t) {
    if (!t) return false;
    const auto * a = t->src[0]; const auto * b = t->src[1];
    switch (t->op) {
        case GGML_OP_DUP:
            return a && a->type == t->type && ggml_blck_size(t->type) == 1 &&
                   ggml_nelements(a) == ggml_nelements(t);
        case GGML_OP_CUSTOM: case GGML_OP_MAP_CUSTOM1: case GGML_OP_MAP_CUSTOM2: case GGML_OP_MAP_CUSTOM3:
            return true;
        case GGML_OP_MUL_MAT:
            return a && b && a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                   a->nb[0] == 4 && b->nb[0] == 4 && t->type == GGML_TYPE_F32;
        case GGML_OP_IM2COL:
            return floating(a) && floating(b) && floating(t) && t->op_params[6] == 1 &&
                   ggml_is_contiguous(t) && b->ne[2] == a->ne[2] &&
                   (t->type == GGML_TYPE_F16 || b->type == GGML_TYPE_F16);
        case GGML_OP_UNARY:
            if (!floating(a) || t->type != a->type || !ggml_are_same_shape(a,t)) return false;
            if (a->type == GGML_TYPE_F32 && ggml_is_contiguous(a) &&
                ggml_get_unary_op(t) != GGML_UNARY_OP_RELU && ggml_get_unary_op(t) != GGML_UNARY_OP_ABS &&
                ggml_get_unary_op(t) != GGML_UNARY_OP_GELU_ERF) return false;
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_RELU: case GGML_UNARY_OP_GELU: case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_SILU: case GGML_UNARY_OP_TANH: case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_EXP: case GGML_UNARY_OP_NEG: case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SOFTPLUS: return true;
                default: return false;
            }
        case GGML_OP_POOL_2D:
            return floating(a) && t->type == GGML_TYPE_F32 &&
                   (t->op_params[0] == GGML_OP_POOL_MAX || t->op_params[0] == GGML_OP_POOL_AVG);
        case GGML_OP_CONV_2D_DW:
            return floating(a) && b && b->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(t);
        case GGML_OP_GET_ROWS:
            return a && b && !a->extra && ggml_is_quantized(a->type) &&
                   ggml_get_type_traits(a->type)->to_float && a->nb[0] == ggml_type_size(a->type) &&
                   b->type == GGML_TYPE_I32 && t->type == GGML_TYPE_F32 && t->ne[0] == a->ne[0] &&
                   a->ne[2] == b->ne[1] && ggml_nrows(t) == ggml_nelements(b);
        default: return false;
    }
}
inline void custom(ggml_tensor * t, int ith, int nth) {
    auto tasks = [nth](int requested) { return requested == GGML_N_TASKS_MAX ? nth : std::min(nth, requested); };
    if (t->op == GGML_OP_CUSTOM) {
        ggml_custom_op_params p; memcpy(&p,t->op_params,sizeof(p));
        const int n = tasks(p.n_tasks); if (ith < n) p.fun(t,ith,n,p.userdata);
    } else if (t->op == GGML_OP_MAP_CUSTOM1) {
        ggml_map_custom1_op_params p; memcpy(&p,t->op_params,sizeof(p));
        const int n = tasks(p.n_tasks); if (ith < n) p.fun(t,t->src[0],ith,n,p.userdata);
    } else if (t->op == GGML_OP_MAP_CUSTOM2) {
        ggml_map_custom2_op_params p; memcpy(&p,t->op_params,sizeof(p));
        const int n = tasks(p.n_tasks); if (ith < n) p.fun(t,t->src[0],t->src[1],ith,n,p.userdata);
    } else {
        ggml_map_custom3_op_params p; memcpy(&p,t->op_params,sizeof(p));
        const int n = tasks(p.n_tasks); if (ith < n) p.fun(t,t->src[0],t->src[1],t->src[2],ith,n,p.userdata);
    }
}
inline bool forward(ggml_tensor * t, int ith, int nth) {
    if (!supported(t)) return false;
    const auto * a = t->src[0]; const auto * b = t->src[1];
    if (t->op == GGML_OP_CUSTOM || t->op == GGML_OP_MAP_CUSTOM1 ||
        t->op == GGML_OP_MAP_CUSTOM2 || t->op == GGML_OP_MAP_CUSTOM3) {
        custom(t,ith,nth); return true;
    }
    if (t->op == GGML_OP_MUL_MAT) return false; // Uses existing stride-aware vector dot implementation.
    if (t->op == GGML_OP_DUP) {
        const int64_t count=ggml_nelements(t),begin=count*ith/nth,end=count*(ith+1)/nth;
        const size_t unit=ggml_type_size(t->type);
        for(int64_t i=begin;i<end;++i)
            memcpy(static_cast<char *>(t->data)+offset(t,i),static_cast<const char *>(a->data)+offset(a,i),unit);
        return true;
    }
    if (t->op == GGML_OP_GET_ROWS) {
        const auto convert = ggml_get_type_traits(a->type)->to_float;
        for (int64_t r=ith;r<ggml_nrows(t);r+=nth) {
            const int64_t i0=r%b->ne[0],i1=(r/b->ne[0])%b->ne[1],i2=r/(b->ne[0]*b->ne[1]);
            int32_t row; memcpy(&row,static_cast<const char *>(b->data)+i0*b->nb[0]+i1*b->nb[1]+i2*b->nb[2],4);
            GGML_ASSERT(row>=0&&row<a->ne[1]);
            const auto * src=static_cast<const char *>(a->data)+row*a->nb[1]+i1*a->nb[2]+i2*a->nb[3];
            auto * dst=reinterpret_cast<float *>(static_cast<char *>(t->data)+offset(t,r*t->ne[0]));
            convert(src,dst,a->ne[0]);
        }
        return true;
    }
    const int64_t count=ggml_nelements(t),begin=count*ith/nth,end=count*(ith+1)/nth;
    if (t->op == GGML_OP_UNARY) {
        for (int64_t i=begin;i<end;++i) {
            const float x=read(a,offset(a,i)); float y;
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_RELU: y=std::max(x,0.f);break;
                case GGML_UNARY_OP_GELU: y=.5f*x*(1.f+tanhf(.7978845608028654f*x*(1.f+.044715f*x*x)));break;
                case GGML_UNARY_OP_GELU_ERF: y=.5f*x*(1.f+erff(x*.7071067811865475f));break;
                case GGML_UNARY_OP_SILU: y=x/(1.f+expf(-x));break;
                case GGML_UNARY_OP_TANH: y=tanhf(x);break;
                case GGML_UNARY_OP_SIGMOID: y=1.f/(1.f+expf(-x));break;
                case GGML_UNARY_OP_EXP: y=expf(x);break;
                case GGML_UNARY_OP_NEG: y=-x;break;
                case GGML_UNARY_OP_ABS: y=fabsf(x);break;
                case GGML_UNARY_OP_SOFTPLUS: y=x>20.f?x:log1pf(expf(x));break;
                default: GGML_ABORT("unexpected OCR unary");
            }
            write(t,i,y);
        }
    } else if (t->op == GGML_OP_IM2COL) {
        const int64_t kw=a->ne[0],kh=a->ne[1],patch=t->ne[0];
        for (int64_t i=begin;i<end;++i) {
            int64_t q=i, kx=q%kw;q/=kw;const int64_t ky=q%kh;q/=kh;
            const int64_t c=q%b->ne[2];q=i/patch;const int64_t ox=q%t->ne[1];q/=t->ne[1];
            const int64_t oy=q%t->ne[2],n=q/t->ne[2];
            const int64_t x=ox*t->op_params[0]-t->op_params[2]+kx*t->op_params[4];
            const int64_t y=oy*t->op_params[1]-t->op_params[3]+ky*t->op_params[5];
            write(t,i,x<0||x>=b->ne[0]||y<0||y>=b->ne[1]?0.f:
                  read(b,x*b->nb[0]+y*b->nb[1]+c*b->nb[2]+n*b->nb[3]));
        }
    } else if (t->op == GGML_OP_POOL_2D) {
        const int kx=t->op_params[1],ky=t->op_params[2],sx=t->op_params[3],sy=t->op_params[4];
        const int px=t->op_params[5],py=t->op_params[6];
        for (int64_t i=begin;i<end;++i) {
            int64_t q=i;const int64_t ox=q%t->ne[0];q/=t->ne[0];const int64_t oy=q%t->ne[1];q/=t->ne[1];
            const int64_t c=q%t->ne[2],n=q/t->ne[2];
            float value=t->op_params[0]==GGML_OP_POOL_MAX?-FLT_MAX:0.f;
            for(int y=0;y<ky;++y)for(int x=0;x<kx;++x) {
                const int64_t ix=ox*sx-px+x,iy=oy*sy-py+y;
                if(ix<0||ix>=a->ne[0]||iy<0||iy>=a->ne[1])continue;
                const float v=read(a,ix*a->nb[0]+iy*a->nb[1]+c*a->nb[2]+n*a->nb[3]);
                value=t->op_params[0]==GGML_OP_POOL_MAX?std::max(v,value):value+v;
            }
            write(t,i,t->op_params[0]==GGML_OP_POOL_AVG?value/(kx*ky):value);
        }
    } else if (t->op == GGML_OP_CONV_2D_DW) {
        const int64_t kw=a->ne[0],kh=a->ne[1];
        for(int64_t i=begin;i<end;++i) {
            int64_t q=i;const int64_t ox=q%t->ne[0];q/=t->ne[0];const int64_t oy=q%t->ne[1];q/=t->ne[1];
            const int64_t c=q%t->ne[2],n=q/t->ne[2];float value=0.f;
            for(int64_t y=0;y<kh;++y)for(int64_t x=0;x<kw;++x) {
                const int64_t ix=ox*t->op_params[0]+x*t->op_params[4]-t->op_params[2];
                const int64_t iy=oy*t->op_params[1]+y*t->op_params[5]-t->op_params[3];
                if(ix<0||ix>=b->ne[0]||iy<0||iy>=b->ne[1])continue;
                value+=read(a,x*a->nb[0]+y*a->nb[1]+c*a->nb[2])*
                       read(b,ix*b->nb[0]+iy*b->nb[1]+c*b->nb[2]+n*b->nb[3]);
            }
            write(t,i,value);
        }
    }
    return true;
}
} // namespace ocr_ai_complete
