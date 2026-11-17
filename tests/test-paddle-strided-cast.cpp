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
#include <vector>

// Direct CPU versus AI comparison. No scheduler is involved, and AI support
// is checked before allocation so a CPU fallback cannot hide a missing kernel.
struct config { int d, n, h, b; bool view, permute; };

static float from_bits(uint32_t bits) {
    float value;std::memcpy(&value,&bits,sizeof(value));return value;
}

static std::vector<float> edge_values(bool nonfinite) {
    const float inf=std::numeric_limits<float>::infinity();
    std::vector<float> values={0.0f,-0.0f,
        std::numeric_limits<float>::denorm_min(),-std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(),-std::numeric_limits<float>::min()};
    // Round-to-even ties, normal/subnormal transition, and overflow threshold.
    const float ties[]={0x1p-25f,0x1.8p-24f,0x1.ffcp-15f,
                        1.0f+0x1p-11f,1.0f+3*0x1p-11f,65520.0f};
    for(float tie:ties)for(float value:{std::nextafter(tie,0.0f),tie,std::nextafter(tie,inf)}) {
        values.push_back(value);values.push_back(-value);
    }
    for(float value:{0x1p-24f,0x1.ff8p-15f,0x1p-14f,0.125f,1.0f,
                     17.75f,65504.0f,65519.0f,65536.0f,
                     std::numeric_limits<float>::max()}) {
        values.push_back(value);values.push_back(-value);
    }
    if(nonfinite)for(uint32_t bits:{0x7f800000u,0xff800000u,0x7fc00000u,
                                  0xffc00000u,0x7fc12345u,0xffc12345u,
                                  0x7f800001u,0xff800001u})values.push_back(from_bits(bits));
    return values;
}

static bool run(ggml_backend_t backend,const config &c,bool nonfinite,
                std::vector<ggml_fp16_t> &out,std::vector<float> &logical) {
    const auto ctx=ggml_init({4*1024*1024,nullptr,true});
    if(!ctx)return false;
    // Start from the native [D,H,T,B] layout, then permute to [D,T,H,B]
    // exactly as clip::build_attn does for K/V. Views add real row padding
    // and a nonzero buffer offset instead of only changing metadata labels.
    auto base=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,
        c.view?c.d+5:c.d,c.view?c.h+2:c.h,c.view?c.n+1:c.n,c.b);
    auto x=c.view?ggml_view_4d(ctx,base,c.d,c.h,c.n,c.b,
        base->nb[1],base->nb[2],base->nb[3],2*sizeof(float)):base;
    if(c.permute)x=ggml_permute(ctx,x,0,2,1,3);
    auto y=ggml_cast(ctx,x,GGML_TYPE_F16);
    ggml_set_input(base);ggml_set_output(y);
    if(!ggml_backend_supports_op(backend,y)) {
        std::fprintf(stderr,"backend=%s does not support strided F32->F16\n",ggml_backend_name(backend));
        ggml_free(ctx);return false;
    }
    const auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    const auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!buffer){ggml_free(ctx);return false;}
    const auto edges=edge_values(nonfinite);
    std::vector<float> input(size_t(ggml_nelements(base)));
    for(size_t i=0;i<input.size();++i) {
        const int exponent=int((i*17+11)%47)-24;
        input[i]=std::ldexp(float(int((i*137+17)%2047)-1023)/1024.0f,exponent);
    }
    logical.clear();logical.reserve(size_t(ggml_nelements(x)));
    size_t serial=0;
    const size_t view_offset=c.view?2*sizeof(float):0;
    for(int64_t i3=0;i3<x->ne[3];++i3)for(int64_t i2=0;i2<x->ne[2];++i2)
        for(int64_t i1=0;i1<x->ne[1];++i1)for(int64_t i0=0;i0<x->ne[0];++i0) {
            const size_t at=(view_offset+i0*x->nb[0]+i1*x->nb[1]+i2*x->nb[2]+i3*x->nb[3])/sizeof(float);
            // Spread edge cases through both vector body/tail and every row.
            if(serial%3!=2)input[at]=edges[(serial+serial/size_t(c.d))%edges.size()];
            logical.push_back(input[at]);++serial;
        }
    ggml_backend_tensor_set(base,input.data(),0,input.size()*sizeof(float));
    const bool computed=ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
    bool ok=computed;
    out.resize(size_t(ggml_nelements(y)));
    if(ok)ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(ggml_fp16_t));
    std::vector<float> after(input.size());
    if(ok)ggml_backend_tensor_get(base,after.data(),0,after.size()*sizeof(float));
    ok=ok&&!std::memcmp(after.data(),input.data(),input.size()*sizeof(float));
    if(!ok)std::fprintf(stderr,"backend=%s compute failed or source/guard bytes changed\n",ggml_backend_name(backend));
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return ok;
}

int main() {
#ifdef _WIN32
    _putenv_s("GGML_SPACEMIT_OCR_STRIDED_CAST","1");
#else
    setenv("GGML_SPACEMIT_OCR_STRIDED_CAST","1",1);
#endif
    ggml_backend_load_all();
    const auto ai_dev=ggml_backend_dev_by_name("SPACEMIT0");
    const auto cpu_dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if(!ai_dev||!cpu_dev){std::fprintf(stderr,"CPU or SPACEMIT0 unavailable\n");return 2;}
    const auto ai=ggml_backend_dev_init(ai_dev,nullptr),cpu=ggml_backend_dev_init(cpu_dev,nullptr);
    if(!ai||!cpu)return 3;
    std::vector<config> cases={{72,1,16,1,false,true},{72,17,16,1,false,true},
        {72,848,16,1,false,true},{72,848,16,2,true,true},{72,17,16,2,true,false}};
    for(int d:{1,7,8,9,63,64,65,80})cases.push_back({d,17,3,2,true,true});
    bool ok=true;size_t total=0;
    for(const auto &c:cases) {
        std::vector<ggml_fp16_t> ref,out;std::vector<float> ref_input,ai_input;
        bool pass=run(cpu,c,false,ref,ref_input)&&run(ai,c,false,out,ai_input)&&
                  ref.size()==out.size()&&ref_input.size()==ai_input.size()&&
                  !std::memcmp(ref_input.data(),ai_input.data(),ref_input.size()*sizeof(float));
        size_t different=0,overflows=0;
        if(pass)for(size_t i=0;i<out.size();++i) {
            if(ref[i]!=out[i]) {
                if(different<4)std::fprintf(stderr,"finite mismatch i=%zu input=%a cpu=%04x ai=%04x\n",i,
                    double(ref_input[i]),unsigned(ref[i]),unsigned(out[i]));
                ++different;
            }
            overflows+=(ref[i]&0x7fffu)==0x7c00u;
            // Finite inputs may overflow to +/-inf but must never become NaN.
            if((out[i]&0x7c00u)==0x7c00u&&(out[i]&0x03ffu))pass=false;
        }
        pass=pass&&!different;ok=ok&&pass;total+=out.size();
        std::printf("D=%d T=%d H=%d B=%d view=%d permute=%d finite=%zu different=%zu expected_overflow=%zu %s\n",
            c.d,c.n,c.h,c.b,c.view,c.permute,out.size(),different,overflows,pass?"PASS":"FAIL");
    }
    // Nonfinite semantics are diagnostic, separated from finite byte gates.
    // Infinities and NaN classifications must agree; NaN sign/payload may differ
    // because RISC-V canonicalNaN and ggml's software helper have distinct rules.
    {
        const config c={72,17,16,1,true,true};
        std::vector<ggml_fp16_t> ref,out;std::vector<float> ref_input,ai_input;
        bool pass=run(cpu,c,true,ref,ref_input)&&run(ai,c,true,out,ai_input)&&ref.size()==out.size();
        size_t nan_payload_differences=0,other_differences=0;
        if(pass)for(size_t i=0;i<out.size();++i)if(ref[i]!=out[i]) {
            const bool both_nan=(ref[i]&0x7c00u)==0x7c00u&&(ref[i]&0x03ffu)&&
                                (out[i]&0x7c00u)==0x7c00u&&(out[i]&0x03ffu);
            if(both_nan)++nan_payload_differences;else ++other_differences;
        }
        pass=pass&&!other_differences;ok=ok&&pass;
        std::printf("nonfinite diagnostic nan_sign_payload_differences=%zu other_differences=%zu %s\n",
            nan_payload_differences,other_differences,pass?"PASS":"FAIL");
    }
    std::printf("finite_cases=%zu finite_elements=%zu result=%s\n",cases.size(),total,ok?"PASS":"FAIL");
    ggml_backend_free(ai);ggml_backend_free(cpu);return ok?0:1;
}
