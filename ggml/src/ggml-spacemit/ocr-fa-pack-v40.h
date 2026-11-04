#pragma once
#include "ggml.h"
#include "spacemit-context.h"
#include "../ggml-cpu/common.h"
#include <cstdlib>
#include <cstring>
namespace ocr_fa_pack_v40 {
inline bool ime_qk(const ggml_tensor*op){
 static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_COMPUTE");return p&&(std::strcmp(p,"3")==0||std::strcmp(p,"4")==0);}();
 return on&&op&&op->src[3]==nullptr&&op->src[0]->ne[1]>=988;
}
inline bool pipeline_pv(){
 static const bool on=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_COMPUTE");return p&&std::strcmp(p,"4")==0;}();return on;
}
inline int direct_v44(){
    static const int v=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_DIRECT");
        return p&&p[0]>='1'&&p[0]<='3'&&!p[1]?int(p[0]-'0'):0;}();return v;
}
inline bool lazy_init(){
    static const bool value=[] {const char *p=std::getenv("GGML_SPACEMIT_OCR_FA_LAZY_INIT");return p&&std::strcmp(p,"1")==0;}();return value;
}
inline bool eligible(const ggml_tensor *op){
    static const bool enabled=[] {const char *v=std::getenv("GGML_SPACEMIT_OCR_FA_PACK");return v&&std::strcmp(v,"1")==0;}();
    if(!enabled||!op||op->op!=GGML_OP_FLASH_ATTN_EXT||!op->src[0]||!op->src[1]||!op->src[2])return false;
    const auto *q=op->src[0],*k=op->src[1],*v=op->src[2];
    return q->type==GGML_TYPE_F32&&k->type==GGML_TYPE_F16&&v->type==GGML_TYPE_F16&&
        q->ne[0]==72&&k->ne[0]==72&&v->ne[0]==72&&q->ne[1]>=GGML_FA_TILE_Q&&q->ne[1]<=4096&&
        q->ne[1]==k->ne[1]&&q->ne[1]==v->ne[1]&&q->ne[2]==16&&k->ne[2]==16&&v->ne[2]==16&&
        q->ne[3]==1&&k->ne[3]==1&&v->ne[3]==1&&k->nb[0]==2&&v->nb[0]==2;
}
inline size_t scratch_offset(int threads){
    constexpr size_t d=72,Q=GGML_FA_TILE_Q,KV=GGML_FA_TILE_KV;
    const size_t bytes=sizeof(float)*(Q*d+2*Q*KV+Q*d+KV*d+KV*d+ggml::spacemit::cache_line_size_f32)*threads;
    return (bytes+127)/128*128;
}
inline size_t tiles(const ggml_tensor *op){return (size_t(op->src[1]->ne[1])+GGML_FA_TILE_KV-1)/GGML_FA_TILE_KV;}
inline size_t packed_elements(const ggml_tensor *op){return tiles(op)*16*72*GGML_FA_TILE_KV;}
inline size_t work_bytes(int threads,const ggml_tensor *op){return scratch_offset(threads)+((ime_qk(op)?3:2)*packed_elements(op)+(ime_qk(op)?tiles(op)*16*32:0))*sizeof(_Float16);}
inline _Float16 *keys(ggml::spacemit::context &ctx){return reinterpret_cast<_Float16*>(static_cast<char*>(ctx.workspace)+scratch_offset(ctx.nth));}
inline _Float16 *values(ggml::spacemit::context &ctx,const ggml_tensor *op){return keys(ctx)+packed_elements(op);}
inline _Float16 *ime_key_tile(ggml::spacemit::context &ctx,const ggml_tensor*op,size_t tile){return keys(ctx)+2*packed_elements(op)+tile*(72*GGML_FA_TILE_KV+32);}

}
