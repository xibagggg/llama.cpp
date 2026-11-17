#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// A separate process is used per fusion combination. Compare every F32 byte,
// including a second execution with changed values at identical input addresses.
static bool fixture(FILE * f, bool write, int width, int rows, int kind, int pass,
                    int output, const std::vector<float> & values) {
    const uint32_t header[] = {0x50454655,1,uint32_t(width),uint32_t(rows),uint32_t(kind),
                               uint32_t(pass),uint32_t(output),uint32_t(values.size())};
    if (write) return std::fwrite(header,sizeof(header),1,f)==1 &&
        std::fwrite(values.data(),sizeof(float),values.size(),f)==values.size();
    uint32_t actual[8];
    std::vector<float> expected(values.size());
    if (std::fread(actual,sizeof(actual),1,f)!=1 || std::memcmp(actual,header,sizeof(header)) ||
        std::fread(expected.data(),sizeof(float),expected.size(),f)!=expected.size()) return false;
    if (std::memcmp(expected.data(),values.data(),values.size()*sizeof(float))) {
        for (size_t i=0;i<values.size();++i) if (std::memcmp(&expected[i],&values[i],sizeof(float))) {
            std::fprintf(stderr,"DIFF width=%d rows=%d kind=%d pass=%d output=%d index=%zu: %.9g vs %.9g\n",
                         width,rows,kind,pass,output,i,values[i],expected[i]); break;
        }
        return false;
    }
    return true;
}

int main(int argc,char **argv) {
    if (argc!=3 || (std::strcmp(argv[1],"--write-reference") && std::strcmp(argv[1],"--compare-reference"))) return 2;
    const bool write=!std::strcmp(argv[1],"--write-reference");
    FILE * reference=std::fopen(argv[2],write?"wb":"rb");
    if (!reference) return 2;
    ggml_backend_load_all();
    auto dev=ggml_backend_dev_by_name("SPACEMIT0");
    if (!dev) return 3;
    auto backend=ggml_backend_dev_init(dev,nullptr);
    if (!backend) return 3;
    std::mt19937 rng(17619);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    int outputs_checked=0,cases=0;
    bool ok=true;
    struct shape {int w,m;};
    for (auto shape : {shape{1152,1},shape{1152,7},shape{1152,8},shape{1152,9},
                       shape{1152,17},shape{1152,33},shape{1152,216},shape{1152,848},
                       shape{1024,17},shape{4304,9},shape{4304,33},shape{4608,17}}) {
        // plain affine, residual affine with live residual output, exposed norm
        // intermediate (must reject fusion), row-bias GELU, scalar-bias GELU,
        // and exposed ADD intermediate (must reject ADD+GELU fusion).
        for (int kind=0;kind<6;++kind) {
            const int w=shape.w,m=shape.m;
            auto ctx=ggml_init({4*1024*1024,nullptr,true});
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,w,m);
            auto skip=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,w,m);
            auto scale=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,w);
            auto bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,kind==4?1:w);
            ggml_set_input(x);ggml_set_input(skip);ggml_set_input(scale);ggml_set_input(bias);
            auto graph=ggml_new_graph(ctx);
            std::vector<ggml_tensor*> outputs;
            ggml_tensor * y;
            if (kind<3) {
                auto input=kind==1?ggml_add(ctx,x,skip):x;
                auto norm=ggml_norm(ctx,input,1e-5f);
                auto scaled=ggml_mul(ctx,norm,scale);
                y=ggml_add(ctx,scaled,bias);
                if (kind==1) {ggml_set_output(input);outputs.push_back(input);}
                if (kind==2) {ggml_set_output(norm);outputs.push_back(norm);}
            } else {
                auto sum=ggml_add(ctx,x,bias);
                y=ggml_gelu(ctx,sum);
                if (kind==5) {ggml_set_output(sum);outputs.push_back(sum);}
            }
            ggml_set_output(y);outputs.push_back(y);
            ggml_build_forward_expand(graph,y);
            auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
            if (!buffer) return 4;
            std::vector<float> xv(size_t(w)*m),sv(xv.size()),wv(w),bv(kind==4?1:w);
            for (auto &v:xv) v=dist(rng)*8.f;
            for (auto &v:sv) v=dist(rng)*2.f;
            for (auto &v:wv) v=dist(rng);
            for (auto &v:bv) v=dist(rng)*0.2f;
            const void * address=x->data;
            for (int pass=0;pass<2;++pass) {
                if (pass) {
                    for (size_t i=0;i<xv.size();++i) xv[i]=i%13?xv[i]*-0.25f:0.f;
                    for (auto &v:sv) v=-v;
                    for (auto &v:wv) v+=0.125f;
                    for (auto &v:bv) v=-v+0.03125f;
                }
                ggml_backend_tensor_set(x,xv.data(),0,xv.size()*4);
                ggml_backend_tensor_set(skip,sv.data(),0,sv.size()*4);
                ggml_backend_tensor_set(scale,wv.data(),0,wv.size()*4);
                ggml_backend_tensor_set(bias,bv.data(),0,bv.size()*4);
                if (x->data!=address || ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 5;
                for (size_t j=0;j<outputs.size();++j) {
                    std::vector<float> result(size_t(ggml_nelements(outputs[j])));
                    ggml_backend_tensor_get(outputs[j],result.data(),0,result.size()*4);
                    for (float v:result) if (!std::isfinite(v)) ok=false;
                    if (!fixture(reference,write,w,m,kind,pass,int(j),result)) ok=false;
                    ++outputs_checked;
                }
            }
            ++cases;
            ggml_backend_buffer_free(buffer);ggml_free(ctx);
        }
    }
    if (!write && std::fgetc(reference)!=EOF) ok=false;
    std::fclose(reference);ggml_backend_free(backend);
    std::printf("Elementwise fusion cases=%d passes=%d full_outputs=%d reference=%s result=%s\n",
                cases,2*cases,outputs_checked,write?"written":"byte-identical",ok?"PASS":"FAIL");
    return ok?0:1;
}
