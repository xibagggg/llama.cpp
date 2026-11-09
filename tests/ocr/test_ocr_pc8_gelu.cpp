#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-spacemit.h"
#include "ggml-quants.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Same original synthetic weights in every arm. q4asq8 is an EXACT storage
// expansion of Q4 integers/scales, not a new quantization of the float weights.
// Timings exclude allocation, repacking and output readback.
static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b;
    return x ^ (x >> 16);
}
static void fill(std::vector<float> & v, uint32_t seed, float scale) {
    for (size_t i=0; i<v.size(); ++i)
        v[i]=(int32_t(mix(uint32_t(i)+seed)%20001)-10000)*scale;
}
static bool write(const std::string & p, const void * data, size_t bytes) {
    std::ofstream f(p,std::ios::binary);
    f.write(static_cast<const char *>(data),bytes);
    return bool(f);
}
int main(int argc,char ** argv) {
    if(argc!=6) return 1;
    const int bias_mode=std::atoi(argv[5]);if(bias_mode<0||bias_mode>7)return 1;
    const std::string format=argv[1], prefix=argv[4];
    const int ci=std::atoi(argv[2]), repeats=std::atoi(argv[3]);
    struct Shape {int64_t k,n,m;};
    const Shape shapes[]={{1152,1152,1008},{1152,4304,1008},
                          {4352,1152,1008},{1152,1152,7},{1152,4304,5},
                          {1152,4304,13},{1152,1152,9},
                          {1280,1152,1008},{1280,4320,1008},
                          {4352,1152,7},{4352,1152,9},{4352,1152,13},
                          {4352,1152,1008},{4352,1152,7},{4352,1152,9},{4352,1152,13},
                          {4352,1152,9},{4352,1152,9},{4352,1152,9}};
    if(ci<0||ci>=19||repeats<1||repeats>3) return 1;
    const Shape s=shapes[ci];
    const bool padded=ci==7||ci==8;
    const bool down_pad=ci>=12;
    const int source_case=padded?ci-7:ci;
    const int64_t input_k=down_pad?4304:padded?1152:s.k;
    const int64_t output_n=ci==8?4304:s.n;
    ggml_type type;
    if(format=="q4") type=GGML_TYPE_Q4_0;
    else if(format=="q8"||format=="q4asq8") type=GGML_TYPE_Q8_0;
    else if(format=="f16"||format=="q4asf16") type=GGML_TYPE_F16;
    else return 1;
    auto backend=ggml_backend_spacemit_init();
    if(!backend) return 2;
    ggml_init_params ip{}; ip.mem_size=4*1024*1024;ip.no_alloc=true;
    auto ctx=ggml_init(ip);if(!ctx)return 3;
    auto w=ggml_new_tensor_2d(ctx,type,s.k,s.n);
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,input_k,s.m);
    ggml_set_name(w,"v.blk.0.attn_q.weight");
    ggml_set_input(w);ggml_set_output(w);
    ggml_set_input(x);ggml_set_output(x);
    auto a=(padded||down_pad)?ggml_pad(ctx,x,s.k-input_k,0,0,0):x;
    if(ci==16)ggml_set_output(a); // observable PAD must remain materialized
    auto matmul=ggml_mul_mat(ctx,w,a);
    auto bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,s.n);ggml_set_input(bias);ggml_set_output(bias);
    if(bias_mode==2)ggml_set_output(matmul);
    auto added=ggml_add(ctx,matmul,bias);
    if(bias_mode==5)ggml_set_output(added);
    auto full=bias_mode?ggml_gelu(ctx,added):matmul;
    auto y=output_n==s.n?full:ggml_cont(ctx,ggml_view_2d(ctx,full,output_n,s.m,s.n*sizeof(float),0));
    ggml_set_name(y,"quant-format-result");
    ggml_set_output(y);
    auto g=ggml_new_graph_custom(ctx,16,false);ggml_build_forward_expand(g,y);
    ggml_tensor * extra=ci==16?a:ci==17?ggml_scale(ctx,a,.5f):ci==18?
        ggml_cont(ctx,ggml_view_2d(ctx,a,100,s.m,s.k*sizeof(float),0)):nullptr;
    if(bias_mode==2)extra=matmul;
    if(bias_mode==3)extra=ggml_scale(ctx,matmul,.5f);
    if(bias_mode==4)extra=ggml_cont(ctx,ggml_view_2d(ctx,matmul,100,s.m,s.n*sizeof(float),0));
    if(bias_mode==5)extra=added;
    if(bias_mode==6)extra=ggml_scale(ctx,added,.5f);
    if(bias_mode==7)extra=ggml_cont(ctx,ggml_view_2d(ctx,added,100,s.m,s.n*sizeof(float),0));
    if(extra){ggml_set_output(extra);ggml_build_forward_expand(g,extra);}
    if(!ggml_backend_supports_op(backend,y)) return 4;
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if(!ggml_gallocr_alloc_graph(alloc,g)) return 5;
    std::vector<float> wf(size_t(s.k*s.n)),xf(size_t(input_k*s.m));
    fill(wf,91u+source_case,0.00003f);fill(xf,177u+source_case,0.001f);
    if(padded) {
        std::vector<float> original(size_t(input_k*output_n));
        fill(original,91u+source_case,0.00003f);
        std::fill(wf.begin(),wf.end(),0);
        for(int64_t r=0;r<output_n;++r)
            std::copy(original.begin()+r*input_k,original.begin()+(r+1)*input_k,wf.begin()+r*s.k);
    }
    if(s.k==4352) {
        for(int64_t r=0;r<s.n;++r)std::fill(wf.begin()+r*s.k+4304,wf.begin()+(r+1)*s.k,0);
        if(!down_pad)for(int64_t r=0;r<s.m;++r)std::fill(xf.begin()+r*s.k+4304,xf.begin()+(r+1)*s.k,0);
    }
    std::vector<uint8_t> packed(ggml_nbytes(w));
    std::vector<block_q4_0> q4(wf.size()/32);
    if(ggml_quantize_chunk(GGML_TYPE_Q4_0,wf.data(),q4.data(),0,s.n,s.k,nullptr)!=q4.size()*sizeof(block_q4_0))
        return 6;
    if(std::getenv("GGML_OCR_TEST_REFERENCE")) {
        if(!write(prefix+"-weights.q4",q4.data(),q4.size()*sizeof(block_q4_0))||
           !write(prefix+"-input.f32",xf.data(),xf.size()*4))return 15;
    }
    if(format=="q4asq8") {
        auto * q8=reinterpret_cast<block_q8_0 *>(packed.data());
        for(size_t b=0;b<q4.size();++b) {
            q8[b].d=q4[b].d;
            for(int j=0;j<16;++j) {
                q8[b].qs[j]=int(q4[b].qs[j]&15)-8;
                q8[b].qs[j+16]=int(q4[b].qs[j]>>4)-8;
            }
        }
        // Assert exact reconstructed real weights before any backend repack.
        std::vector<float> a(wf.size()),b(wf.size());
        dequantize_row_q4_0(q4.data(),a.data(),wf.size());
        dequantize_row_q8_0(q8,b.data(),wf.size());
        if(std::memcmp(a.data(),b.data(),a.size()*4)) return 7;
    } else if(format=="q4") std::memcpy(packed.data(),q4.data(),packed.size());
    else {
        if(format=="q4asf16")dequantize_row_q4_0(q4.data(),wf.data(),wf.size());
        if(ggml_quantize_chunk(type,wf.data(),packed.data(),0,s.n,s.k,nullptr)!=packed.size()) return 8;
    }
    ggml_backend_tensor_set(w,packed.data(),0,packed.size());
    std::vector<float> bf(size_t(s.n));fill(bf,971u+source_case,.00013f);
    ggml_backend_tensor_set(bias,bf.data(),0,bf.size()*4);
    ggml_backend_tensor_set(x,xf.data(),0,xf.size()*4);
    const size_t main_elements=size_t(output_n*s.m);
    std::vector<float> out(main_elements+(extra?ggml_nelements(extra):0)),reference,changed=xf;
    auto read_output=[&]{
        ggml_backend_tensor_get(y,out.data(),0,main_elements*4);
        if(extra)ggml_backend_tensor_get(extra,out.data()+main_elements,0,ggml_nbytes(extra));
    };
    bool stable=true;
    double total=0;
    for(int r=-1;r<repeats;++r) {
        const auto begin=std::chrono::steady_clock::now();
        if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 9;
        ggml_backend_synchronize(backend);
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        read_output();
        if(!std::all_of(out.begin(),out.end(),[](float f){return std::isfinite(f);}))return 10;
        if(r==-1)reference=out;
        else {total+=ms;stable&=std::memcmp(reference.data(),out.data(),out.size()*4)==0;}
        std::printf("{\"format\":\"%s\",\"case\":%d,\"run\":%d,\"ms\":%.6f}\n",format.c_str(),ci,r,ms);
    }
    // A -> B -> A prevents an address-only output/activation cache from passing.
    for(auto & f:changed)f=f*0.75f+0.125f;
    ggml_backend_tensor_set(x,changed.data(),0,changed.size()*4);
    if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 11;
    read_output();
    const bool responsive=std::memcmp(reference.data(),out.data(),out.size()*4)!=0;
    ggml_backend_tensor_set(x,xf.data(),0,xf.size()*4);
    if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 12;
    read_output();
    const bool restored=std::memcmp(reference.data(),out.data(),out.size()*4)==0;
    ggml_backend_tensor_get(x,changed.data(),0,changed.size()*4);
    bool immutable=std::memcmp(xf.data(),changed.data(),xf.size()*4)==0;
    std::vector<float> bc(bf.size());ggml_backend_tensor_get(bias,bc.data(),0,bc.size()*4);
    immutable&=!std::memcmp(bf.data(),bc.data(),bf.size()*4);
    if(!write(prefix+".f32",reference.data(),reference.size()*4))return 13;
    std::printf("{\"kind\":\"summary\",\"format\":\"%s\",\"case\":%d,\"K\":%lld,\"N\":%lld,\"M\":%lld,"
                "\"mean_ms\":%.6f,\"stable\":%s,\"responsive\":%s,\"restored\":%s,\"input_immutable\":%s}\n",
                format.c_str(),ci,(long long)s.k,(long long)s.n,(long long)s.m,total/repeats,
                stable?"true":"false",responsive?"true":"false",restored?"true":"false",immutable?"true":"false");
    ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_free(backend);
    return stable&&responsive&&restored&&immutable?0:14;
}
