#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

struct config { int w, h, channels; bool view; };
static constexpr int mode = GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS;

static uint32_t bits(float x) { uint32_t value; std::memcpy(&value,&x,4); return value; }

static float input_value(size_t i, int replay) {
    // Exact finite edges plus signed, cancellation-prone values. Avoid extreme
    // finite magnitudes whose mathematically valid weighted sum could overflow.
    const float edges[] = {0.f,-0.f,std::numeric_limits<float>::denorm_min(),
        -std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::min(),
        -std::numeric_limits<float>::min(),0x1.fffffep-1f,1.f,0x1.000002p0f,
        -0x1.fffffep-1f,-1.f,-0x1.000002p0f,0x1p-20f,-0x1p-20f,0x1p20f,-0x1p20f};
    const size_t seed=i+size_t(replay)*773;
    if (seed%3!=2) return edges[(seed+seed/27)% (sizeof(edges)/sizeof(edges[0]))];
    return std::ldexp(float(int((seed*137+17)%2047)-1023)/1024.f,int((seed*17+11)%31)-15);
}

static bool run(ggml_backend_t backend,const config & c,std::vector<float> & out) {
    const auto ctx=ggml_init({4*1024*1024,nullptr,true});
    if(!ctx)return false;
    auto source_base=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,c.channels+(c.view?5:0),
                                      27+(c.view?2:0),27+(c.view?2:0),1);
    auto source=c.view?ggml_view_4d(ctx,source_base,c.channels,27,27,1,
        source_base->nb[1],source_base->nb[2],source_base->nb[3],2*sizeof(float)):source_base;
    source=ggml_permute(ctx,source,2,0,1,3); // [27,27,C], channel stride4.
    const size_t elements=size_t(c.w)*c.h*c.channels;
    auto destination_base=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,elements+32);
    auto y=ggml_view_4d(ctx,destination_base,c.w,c.h,c.channels,1,
        size_t(c.w)*4,size_t(c.w)*c.h*4,elements*4,16*sizeof(float));
    // A guarded destination view retains the normal UPSCALE strides/operator.
    // This test graph uses actual backend kernels, without a scheduler fallback.
    y->op=GGML_OP_UPSCALE;y->src[0]=source;y->op_params[0]=mode;
    ggml_set_input(source_base);ggml_set_output(y);
    if(!ggml_backend_supports_op(backend,y)) {
        std::fprintf(stderr,"backend=%s unsupported position resize\n",ggml_backend_name(backend));
        ggml_free(ctx);return false;
    }
    const auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    const auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer){ggml_free(ctx);return false;}
    const auto * source_address=source->data;
    const auto * output_address=y->data;
    std::vector<float> input(size_t(ggml_nelements(source_base))),after(input.size());
    std::vector<float> guard(elements+32,-12345.25f),result(guard.size());
    bool ok=true;
    out.clear();out.reserve(elements*2);
    for(int replay=0;replay<2&&ok;++replay) {
        for(size_t i=0;i<input.size();++i)input[i]=input_value(i,replay);
        ggml_backend_tensor_set(source_base,input.data(),0,input.size()*4);
        ggml_backend_tensor_set(destination_base,guard.data(),0,guard.size()*4);
        ok=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS &&
           source->data==source_address && y->data==output_address;
        if(!ok)break;
        ggml_backend_tensor_get(source_base,after.data(),0,after.size()*4);
        ggml_backend_tensor_get(destination_base,result.data(),0,result.size()*4);
        ok=!std::memcmp(input.data(),after.data(),input.size()*4);
        for(size_t i=0;i<16;++i)
            ok=ok&&bits(result[i])==bits(guard[i])&&bits(result[16+elements+i])==bits(guard[16+elements+i]);
        for(size_t i=0;i<elements;++i) {
            ok=ok&&std::isfinite(result[16+i]);out.push_back(result[16+i]);
        }
    }
    if(ok)ok=std::memcmp(out.data(),out.data()+elements,elements*4)!=0;
    if(!ok)std::fprintf(stderr,"backend=%s failed compute, finite/source/guard/same-address replay gate\n",ggml_backend_name(backend));
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return ok;
}

static bool selector_regression(ggml_backend_t ai,bool disabled) {
    const auto ctx=ggml_init({1024*1024,nullptr,true});
    auto base=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1152,27,27);
    auto src=ggml_permute(ctx,base,2,0,1,3);
    auto y=ggml_interpolate(ctx,src,106,8,1152,1,mode);
    bool ok=ggml_backend_supports_op(ai,y)==!disabled;
    if(disabled) {
        std::printf("POS_RESIZE disabled selector=%s\n",ok?"PASS":"FAIL");ggml_free(ctx);return ok;
    }
    int rejected=0;
    const auto reject=[&]() {const bool pass=!ggml_backend_supports_op(ai,y);ok=ok&&pass;rejected+=pass;};
    y->op_params[0]=GGML_SCALE_MODE_BILINEAR;reject();y->op_params[0]=mode;
    y->op_params[0]=mode|GGML_SCALE_FLAG_ALIGN_CORNERS;reject();y->op_params[0]=mode;
    y->op_params[0]=GGML_SCALE_MODE_NEAREST;reject();y->op_params[0]=mode;
    src->ne[0]=26;reject();src->ne[0]=27;
    y->ne[2]=1151;reject();y->ne[2]=1152;
    y->ne[3]=2;reject();y->ne[3]=1;
    src->nb[2]=8;reject();src->nb[2]=4;
    const auto old_stride=src->nb[1];src->nb[1]=src->nb[0];reject();src->nb[1]=old_stride;
    y->ne[0]=0;reject();y->ne[0]=106;
    src->type=GGML_TYPE_F16;reject();src->type=GGML_TYPE_F32;
    // supports_op must also reject real/future aliasing once pointers are known.
    src->data=reinterpret_cast<void *>(uintptr_t(0x100000));y->data=src->data;reject();
    src->data=nullptr;y->data=nullptr;
    ok=ok&&ggml_backend_supports_op(ai,y);
    std::printf("POS_RESIZE selector rejected=%d/11 %s\n",rejected,ok?"PASS":"FAIL");
    ggml_free(ctx);return ok;
}

int main(int argc,char ** argv) {
    const bool disabled=argc==2&&!std::strcmp(argv[1],"--disabled");
    if(argc>1&&!disabled)return 2;
#ifdef _WIN32
    _putenv_s("GGML_SPACEMIT_PADDLE_POS_RESIZE",disabled?"0":"1");
#else
    setenv("GGML_SPACEMIT_PADDLE_POS_RESIZE",disabled?"0":"1",1);
#endif
    ggml_backend_load_all();
    const auto ai_dev=ggml_backend_dev_by_name("SPACEMIT0");
    const auto cpu_dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if(!ai_dev||!cpu_dev||ai_dev==cpu_dev||std::strcmp(ggml_backend_dev_name(cpu_dev),"CPU"))return 3;
    const auto ai=ggml_backend_dev_init(ai_dev,nullptr),cpu=ggml_backend_dev_init(cpu_dev,nullptr);
    if(!ai||!cpu)return 4;
    std::printf("Reference backend=%s; tested backend=%s\n",ggml_backend_name(cpu),ggml_backend_name(ai));
    bool ok=selector_regression(ai,disabled);
    std::vector<config> cases={{106,8,1152,false},{106,8,1152,true},{8,106,1152,false}};
    for(const auto & wh:std::vector<std::pair<int,int>>{{1,1},{1,8},{8,1},{27,27},{26,28},{28,26},{54,54},{13,13},{7,7}})
        cases.push_back({wh.first,wh.second,17,true});
    for(int channels:{1,7,8,9,31,32,33,63,64,65,1153})cases.push_back({9,11,channels,true});
    size_t total=0,case_count=0;
    if(!disabled)for(const auto & c:cases) {
        std::vector<float> reference,output;
        bool pass=run(cpu,c,reference)&&run(ai,c,output)&&reference.size()==output.size();
        size_t different=0;double maximum=0;
        if(pass)for(size_t i=0;i<output.size();++i) {
            if(bits(reference[i])!=bits(output[i])) {
                if(different<4)std::fprintf(stderr,"position resize mismatch i=%zu cpu=%a/%08x ai=%a/%08x\n",
                    i,double(reference[i]),bits(reference[i]),double(output[i]),bits(output[i]));
                ++different;
            }
            maximum=std::max(maximum,std::abs(double(reference[i])-double(output[i])));
        }
        pass=pass&&!different;ok=ok&&pass;total+=output.size();++case_count;
        std::printf("W=%d H=%d C=%d view=%d replays=2 elements=%zu different_f32_words=%zu max_abs=%.9g %s\n",
            c.w,c.h,c.channels,int(c.view),output.size(),different,maximum,pass?"PASS":"FAIL");
    }
    std::printf("POS_RESIZE cases=%zu compared_elements=%zu exact_f32_bytes=%s result=%s\n",
        case_count,total,ok?"PASS":"FAIL",ok?"PASS":"FAIL");
    ggml_backend_free(ai);ggml_backend_free(cpu);return ok?0:1;
}
