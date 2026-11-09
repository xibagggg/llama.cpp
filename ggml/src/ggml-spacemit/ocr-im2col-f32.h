#pragma once

#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

// Float patch extraction in the destination backend's buffer. It reads only
// the kernel shape, never its values. No quantization, TCM claim or scratch.
namespace ocr_im2col_f32 {

inline bool supported(const ggml_tensor * out) {
    if(!out||out->op!=GGML_OP_IM2COL||!out->src[0]||!out->src[1]||
       out->type!=GGML_TYPE_F32||out->src[1]->type!=GGML_TYPE_F32||out->op_params[6]!=1||
       !ggml_is_contiguous(out)||!ggml_is_contiguous(out->src[1]))return false;
    const auto * k=out->src[0];const auto * x=out->src[1];
    for(int d=0;d<4;++d)if(x->ne[d]<=0||out->ne[d]<=0)return false;
    const int64_t kw=k->ne[0],kh=k->ne[1],c=x->ne[2];
    const int64_t sx=out->op_params[0],sy=out->op_params[1],px=out->op_params[2],py=out->op_params[3];
    const int64_t dx=out->op_params[4],dy=out->op_params[5];
    if(kw<=0||kh<=0||k->ne[2]!=c||sx<=0||sy<=0||dx<=0||dy<=0||px<0||py<0)return false;
    const __int128 cols=__int128(kw)*kh*c;
    const __int128 nx=__int128(x->ne[0])+2*px-__int128(dx)*(kw-1)-1;
    const __int128 ny=__int128(x->ne[1])+2*py-__int128(dy)*(kh-1)-1;
    if(cols>std::numeric_limits<int64_t>::max()||nx<0||ny<0||
       nx/sx+1!=out->ne[1]||ny/sy+1!=out->ne[2])return false;
    return out->ne[0]==cols&&out->ne[3]==x->ne[3];
}

// Extract a bounded set of patch rows without materializing the full tensor.
inline void extract_rows(const ggml_tensor * out,int64_t begin,int64_t rows,float * target_rows) {
    GGML_ASSERT(supported(out)&&out->src[1]->data&&target_rows);
    const auto * k=out->src[0];const auto * x=out->src[1];
    const int64_t kw=k->ne[0],kh=k->ne[1],c=x->ne[2],iw=x->ne[0],ih=x->ne[1];
    const int64_t ow=out->ne[1],oh=out->ne[2],patch=out->ne[0];
    const int64_t sx=out->op_params[0],sy=out->op_params[1],px=out->op_params[2],py=out->op_params[3];
    const int64_t dx=out->op_params[4],dy=out->op_params[5];
    const int64_t count=ow*oh*x->ne[3];
    GGML_ASSERT(begin>=0&&rows>=0&&begin<=count&&rows<=count-begin);
    const int64_t end=begin+rows;
    for(int64_t r=begin;r<end;++r) {
        const int64_t n=r/(ow*oh),y=(r/ow)%oh,x0=(r%ow)*sx-px,y0=y*sy-py;
        auto * target=target_rows+(r-begin)*patch;
        for(int64_t channel=0;channel<c;++channel) {
            const auto * image=static_cast<const char *>(x->data)+n*x->nb[3]+channel*x->nb[2];
            for(int64_t ky=0;ky<kh;++ky) {
                auto * row=target+(channel*kh+ky)*kw;const int64_t iy=y0+ky*dy;
                if(iy<0||iy>=ih){std::memset(row,0,size_t(kw)*sizeof(float));continue;}
                const auto * pixels=reinterpret_cast<const float *>(image+iy*x->nb[1]);
                if(dx==1) {
                    const int64_t first=std::min(kw,std::max(int64_t(0),-x0));
                    const int64_t last=std::max(first,std::min(kw,iw-x0));
                    if(first)std::memset(row,0,size_t(first)*sizeof(float));
                    if(last>first)std::memcpy(row+first,pixels+x0+first,size_t(last-first)*sizeof(float));
                    if(last<kw)std::memset(row+last,0,size_t(kw-last)*sizeof(float));
                }else {
                    for(int64_t kx=0;kx<kw;++kx) {
                        const int64_t ix=x0+kx*dx;row[kx]=(ix<0||ix>=iw)?0.f:pixels[ix];
                    }
                }
            }
        }
    }
}

inline void forward(ggml_tensor * out,int64_t ith,int64_t nth) {
    GGML_ASSERT(supported(out)&&nth>0&&ith>=0&&ith<nth&&out->data);
    const int64_t count=out->ne[1]*out->ne[2]*out->ne[3],step=(count+nth-1)/nth;
    const int64_t begin=std::min(count,ith*step),rows=std::min(count-begin,step);
    extract_rows(out,begin,rows,static_cast<float *>(out->data)+begin*out->ne[0]);
}
} // namespace ocr_im2col_f32
