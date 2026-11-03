#pragma once
// Explicit opt-in only. Full tensor I/O and barriers perturb execution; never
// use these captures as performance measurements.
#include "ggml.h"
#include "ggml-impl.h"
#include "spacemit-context.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ocr_down_diagnostic_v36 {
inline const char *directory() {
    static const char *value=std::getenv("GGML_SPACEMIT_OCR_DOWN_CAPTURE");
    return value;
}
inline void write_file(const std::string &name,const void *data,size_t size) {
    FILE *f=std::fopen(name.c_str(),"wbx");
    if(!f)GGML_ABORT("OCR diagnostic cannot create %s",name.c_str());
    if(std::fwrite(data,1,size,f)!=size)GGML_ABORT("OCR diagnostic write failed %s",name.c_str());
    if(std::fclose(f)!=0)GGML_ABORT("OCR diagnostic close failed %s",name.c_str());
}
struct scope {
    ggml::spacemit::context &ctx;
    ggml_tensor *op;
    bool active=false;
    int layer=-1;
    unsigned sequence=0;
    std::string prefix;
    scope(ggml::spacemit::context &c,ggml_tensor *t):ctx(c),op(t) {
        const char *dir=directory();
        if(!dir||!*dir||!op||!op->src[0]||!op->src[1])return;
        const auto *w=op->src[0],*x=op->src[1];
        char suffix=0;
        if(std::sscanf(w->name,"v.blk.%d.ffn_down.weight%c",&layer,&suffix)!=1||
           layer<0||layer>=27||w->type!=GGML_TYPE_F16||x->type!=GGML_TYPE_F32||
           op->type!=GGML_TYPE_F32||w->ne[0]!=4304||w->ne[1]!=1152||
           !ggml_is_contiguous(w)||!ggml_is_contiguous(x)||!ggml_is_contiguous(op))return;
        active=true;
        GGML_ASSERT(ctx.sync()==spert::Status::Ok);
        if(ctx.ith==0){
            static unsigned next_sequence=0;
            sequence=next_sequence++;
            if(sequence>=108)GGML_ABORT("OCR diagnostic capture limit reached");
            char stem[96];std::snprintf(stem,sizeof(stem),"/%04u-layer-%02d",sequence,layer);
            prefix=std::string(dir)+stem;
            write_file(prefix+"-x.f32",x->data,ggml_nbytes(x));
            // Save weights once per layer. Later image passes use the same model.
            if(sequence<27)write_file(std::string(dir)+"/weight-"+std::to_string(layer)+".f16",w->data,ggml_nbytes(w));
        }
        GGML_ASSERT(ctx.sync()==spert::Status::Ok);
    }
    ~scope(){
        if(!active)return;
        GGML_ASSERT(ctx.sync()==spert::Status::Ok);
        if(ctx.ith==0){
            write_file(prefix+"-y.f32",op->data,ggml_nbytes(op));
            const std::string manifest=std::string(directory())+"/captures.jsonl";
            FILE *f=std::fopen(manifest.c_str(),"a");
            if(!f)GGML_ABORT("OCR diagnostic cannot open manifest");
            std::fprintf(f,"{\"sequence\":%u,\"layer\":%d,\"K\":4304,\"M\":1152,\"N\":%lld,\"workers\":%d,\"diagnostic_only\":true}\n",sequence,layer,(long long)op->src[1]->ne[1],ctx.nth);
            if(std::fclose(f)!=0)GGML_ABORT("OCR diagnostic manifest close failed");
        }
        GGML_ASSERT(ctx.sync()==spert::Status::Ok);
    }
};
}
