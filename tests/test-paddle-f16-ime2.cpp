#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

// Separate processes are required: the kernel reads its panel count once.
// The CTest fixture writes every output element with PANEL_TILES=1, then the
// other panel counts compare their full outputs byte for byte against it.
static bool reference_output(FILE * file, bool write, int k, int n, int m,
                             const std::vector<float> & output) {
    const uint32_t header[] = {0x50494d45, 1, uint32_t(k), uint32_t(n), uint32_t(m), uint32_t(output.size())};
    if (write) {
        return std::fwrite(header, sizeof(header), 1, file) == 1 &&
               std::fwrite(output.data(), sizeof(float), output.size(), file) == output.size();
    }
    uint32_t actual_header[6];
    if (std::fread(actual_header, sizeof(actual_header), 1, file) != 1 ||
            std::memcmp(header, actual_header, sizeof(header))) {
        std::fprintf(stderr, "Reference shape/header mismatch at K=%d N=%d M=%d\n", k, n, m);
        return false;
    }
    std::vector<float> expected(output.size());
    if (std::fread(expected.data(), sizeof(float), expected.size(), file) != expected.size()) {
        std::fprintf(stderr, "Truncated reference at K=%d N=%d M=%d\n", k, n, m);
        return false;
    }
    if (std::memcmp(expected.data(), output.data(), output.size() * sizeof(float))) {
        for (size_t i = 0; i < output.size(); ++i) {
            uint32_t want, have;
            std::memcpy(&want, &expected[i], sizeof(want));
            std::memcpy(&have, &output[i], sizeof(have));
            if (want != have) {
                std::fprintf(stderr, "Panel output differs at K=%d N=%d M=%d row=%zu col=%zu: 0x%08x != 0x%08x\n",
                             k, n, m, i / n, i % n, unsigned(have), unsigned(want));
                break;
            }
        }
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 1 && (argc != 3 || (std::strcmp(argv[1], "--write-reference") &&
                                      std::strcmp(argv[1], "--compare-reference")))) {
        std::fprintf(stderr, "Usage: %s [--write-reference FILE | --compare-reference FILE]\n", argv[0]);
        return 2;
    }
    const bool write_reference = argc == 3 && !std::strcmp(argv[1], "--write-reference");
    std::unique_ptr<FILE, decltype(&std::fclose)> reference(nullptr, &std::fclose);
    if (argc == 3) {
        reference.reset(std::fopen(argv[2], write_reference ? "wb" : "rb"));
        if (!reference) {
            std::perror(argv[2]);
            return 2;
        }
    }
    setenv("GGML_SPACEMIT_PADDLE_F16_IME2", "1", 1);
    setenv("GGML_SPACEMIT_PADDLE_F16_PROJECTOR", "1", 1);
    ggml_backend_load_all();
    auto dev=ggml_backend_dev_by_name("SPACEMIT0");
    if (!dev) return 2;
    auto backend=ggml_backend_dev_init(dev,nullptr);
    if (!backend) return 3;
    std::mt19937 rng(176);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    bool ok=true;
    for (auto shape : {std::pair<int,int>{1152,1152},{1152,4304},{4304,1152},{4608,4608},{4608,1024}}) {
        for (int m : {1,7,8,9,17,33,216}) {
            const int k=shape.first,n=shape.second;
            ggml_init_params params{32*1024*1024,nullptr,true};
            auto ctx=ggml_init(params);
            auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,k,n);
            ggml_set_name(w,k==4608?(n==4608?"mm.1.weight":"mm.2.weight"):"v.blk.0.attn_q.weight");
            ggml_set_param(w);
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,m);
            ggml_set_input(x);
            auto y=ggml_mul_mat(ctx,w,x);
            ggml_set_output(y);
            auto graph=ggml_new_graph(ctx);
            ggml_build_forward_expand(graph,y);
            auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
            if (!buffer) return 4;
            if (!w->extra) { std::fprintf(stderr,"Paddle F16 IME2 trait is inactive\n"); return 6; }
            std::vector<ggml_fp16_t> weights(size_t(k)*n);
            std::vector<float> values(size_t(k)*m),output(size_t(n)*m);
            for (auto &a:weights) a=ggml_fp32_to_fp16(dist(rng)*0.1f);
            // Include subnormals, signs, zeros and tiny activations.
            for (size_t i=0;i<weights.size();i+=31) weights[i]=ggml_fp32_to_fp16(i%62 ? -0x1p-23f : 0x1p-23f);
            for (size_t i=0;i<values.size();++i) values[i]=i%11 ? dist(rng)*2.f : dist(rng)*1e-5f;
            if (m==17) for (auto &a:values) a*=1e-10f;
            if (m==8) std::fill(values.begin(),values.end(),0.f);
            if (m==9) std::fill_n(values.begin(),k,0.f);
            ggml_backend_tensor_set(w,weights.data(),0,weights.size()*2);
            ggml_backend_tensor_set(x,values.data(),0,values.size()*4);
            const auto status=ggml_backend_graph_compute(backend,graph);
            if (status!=GGML_STATUS_SUCCESS) return 5;
            ggml_backend_tensor_get(y,output.data(),0,output.size()*4);
            double sqerr=0,sqref=0,maxabs=0;
            bool finite=true,zero=true;
            for (float a:output) finite=finite&&std::isfinite(a);
            const size_t zero_count=m==8?output.size():(m==9?size_t(n):0);
            for (size_t i=0;i<zero_count;++i) zero=zero&&output[i]==0.f;
            std::vector<int> columns;
            for(int j=0;j<64;++j) columns.push_back((j*71)%n);
            // Covers every valid lane in the final N32 panel, including all
            // 16 valid channels when N=4304 and the padded lanes are omitted.
            for(int col=std::max(0,n-32);col<n;++col) columns.push_back(col);
            std::sort(columns.begin(),columns.end());
            columns.erase(std::unique(columns.begin(),columns.end()),columns.end());
            for (int row=0;row<m;++row) for(int col:columns) {
                double ref=0;
                for(int z=0;z<k;++z) ref+=double(ggml_fp16_to_fp32(weights[size_t(col)*k+z]))*values[size_t(row)*k+z];
                const double err=output[size_t(row)*n+col]-ref;
                sqerr+=err*err;sqref+=ref*ref;maxabs=std::max(maxabs,std::abs(err));
            }
            const double relative=std::sqrt(sqerr/std::max(sqref,1e-30));
            const bool same=!reference||reference_output(reference.get(),write_reference,k,n,m,output);
            const bool pass=finite&&zero&&same&&relative<2e-5&&maxabs<0.002;
            ok=ok&&pass;
            std::printf("K=%d N=%d M=%d relative_l2=%.9g max_abs=%.9g finite=%d zero=%d reference=%s %s\n",
                        k,n,m,relative,maxabs,int(finite),int(zero),
                        !reference?"none":(write_reference?"write":(same?"identical":"different")),pass?"PASS":"FAIL");
            ggml_backend_buffer_free(buffer);ggml_free(ctx);
        }
    }
    if (reference && (write_reference ? std::fflush(reference.get()) != 0 :
                     (std::fgetc(reference.get()) != EOF || std::ferror(reference.get())))) {
        std::fprintf(stderr,"Reference I/O error or trailing data\n");
        ok=false;
    }
    ggml_backend_free(backend);
    return ok?0:1;
}
