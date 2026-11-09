#include "../../ggml/src/ggml-spacemit/ocr-im2col-pack.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef OCR_PACK_RVV_TEST
#include <riscv_vector.h>
#include "rvv-quant-under-test.h"
#endif

static void check(bool value,const char * message){if(!value)throw std::runtime_error(message);}
using quantizer=std::function<void(size_t,const float *,size_t,uint8_t *)>;

static int run(int w,int h,int c,int batch,int kw,int kh,int stride,int pad,int dilation) {
    auto * ctx=ggml_init({64*1024*1024,nullptr,false});check(ctx,"context failed");
    auto * image=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,w,h,c,batch);
    auto * geometry=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,kw,kh,c,1);
    auto * patch=ggml_im2col(ctx,geometry,image,stride,stride,pad,pad,dilation,dilation,true,GGML_TYPE_F32);
    const int64_t k=patch->ne[0],rows=patch->ne[1]*patch->ne[2]*patch->ne[3];
    auto * weights=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,k,25);
    auto * flat=ggml_reshape_2d(ctx,patch,k,rows);auto * matrix=ggml_mul_mat(ctx,weights,flat);
    auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,matrix);
    int index=-1;for(int i=0;i<graph->n_nodes;++i)if(graph->nodes[i]==patch)index=i;
    check(ocr_im2col_pack::match(graph,index)==index+2,"valid fusion rejected");
    ggml_set_output(patch);check(ocr_im2col_pack::match(graph,index)<0,"observable patch fused");
    patch->flags&=~GGML_TENSOR_FLAG_OUTPUT;
    ggml_set_output(flat);check(ocr_im2col_pack::match(graph,index)<0,"observable reshape fused");
    flat->flags&=~GGML_TENSOR_FLAG_OUTPUT;
    auto * consumer=ggml_scale(ctx,flat,.5f);ggml_build_forward_expand(graph,consumer);
    check(ocr_im2col_pack::match(graph,index)<0,"shared patch view fused");
    auto * pixels=static_cast<float *>(image->data);
    for(int64_t i=0;i<ggml_nelements(image);++i)pixels[i]=(float(i%127)-63.f)*.125f;
    pixels[0]=-0.f;
    std::vector<uint8_t> before(static_cast<uint8_t *>(image->data),static_cast<uint8_t *>(image->data)+ggml_nbytes(image));
    std::memset(patch->data,0x5b,ggml_nbytes(patch));
    std::vector<float> reference(size_t(k*rows));
    for(int n=0;n<batch;++n)for(int y=0;y<patch->ne[2];++y)for(int x=0;x<patch->ne[1];++x)
        for(int channel=0;channel<c;++channel)for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx) {
            const int iy=y*stride+ky*dilation-pad,ix=x*stride+kx*dilation-pad;
            const float expected=(iy<0||iy>=h||ix<0||ix>=w)?0.f:pixels[((n*c+channel)*h+iy)*w+ix];
            const size_t offset=(((n*size_t(patch->ne[2])+y)*patch->ne[1]+x)*c+channel)*kh*kw+ky*kw+kx;
            reference[offset]=expected;
        }
    int checks=0;
    for(int block:{32,256}) {
        if(k%block)continue;
        quantizer one,four;size_t row_stride;
#ifdef OCR_PACK_RVV_TEST
        if(block==32) {
            one=spacemit_kernels::rvv::quantize_a_row_i8;four=spacemit_kernels::rvv::quantize_a_4row_i8;
            row_stride=size_t(k/block)*spacemit_kernels::q8_blk_size(block,true);
        }else {
            one=spacemit_kernels::rvv::quantize_a_row_i8_hp;four=spacemit_kernels::rvv::quantize_a_4row_i8_hp;
            row_stride=size_t(k/block)*spacemit_kernels::q8_hp_blk_size(block,true,true);
        }
#else
        // Portable coverage of row extraction and callback boundaries. This is
        // standard GGML Q8_0, not the SpacemiT physical packed representation.
        if(block!=32)continue;
        row_stride=ggml_row_size(GGML_TYPE_Q8_0,k);
        one=[](size_t,const float * src,size_t count,uint8_t * dst){ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(src,dst,int64_t(count));};
        four=[&](size_t b,const float * src,size_t count,uint8_t * dst){for(int r=0;r<4;++r)one(b,src+r*count,count,dst+r*row_stride);};
#endif
        // HP four-row blocks reserve some scale slots that their shared-scale
        // format does not write. Compare like-initialized buffers, including
        // those slots, instead of assuming that the quantizer zero-fills them.
        std::vector<uint8_t> expected(size_t(rows)*row_stride,0x6a);
        for(int64_t r=0;r<rows;) {
            if(rows-r>=4){four(block,reference.data()+r*k,size_t(k),expected.data()+r*row_stride);r+=4;}
            else{one(block,reference.data()+r*k,size_t(k),expected.data()+r*row_stride);++r;}
        }
        for(int nth:{1,4,8}) {
            std::vector<uint8_t> actual(expected.size()+128,0x6a);
            std::vector<std::vector<float>> scratch(size_t(nth),std::vector<float>(size_t(4*k+32),12345.f));
            std::vector<std::thread> workers;
            for(int ith=0;ith<nth;++ith)workers.emplace_back([&,ith]{
                ocr_im2col_pack::pack(patch,ith,nth,block,row_stride,actual.data()+64,scratch[ith].data()+16,one,four);
            });
            for(auto & worker:workers)worker.join();
            if(std::memcmp(actual.data()+64,expected.data(),expected.size())) {
                size_t offset=0;
                while(offset<expected.size()&&actual[offset+64]==expected[offset])++offset;
                std::fprintf(stderr,"packed mismatch K=%lld rows=%lld block=%d threads=%d byte=%zu expected=%u actual=%u\n",
                    (long long)k,(long long)rows,block,nth,offset,unsigned(expected[offset]),unsigned(actual[offset+64]));
                throw std::runtime_error("packed bytes differ");
            }
            for(int i=0;i<64;++i)check(actual[i]==0x6a&&actual[actual.size()-1-i]==0x6a,"packed buffer guard changed");
            for(const auto & data:scratch)for(int i=0;i<16;++i)
                check(data[i]==12345.f&&data[data.size()-1-i]==12345.f,"scratch guard changed");
            for(size_t i=0;i<ggml_nbytes(patch);++i)check(static_cast<uint8_t *>(patch->data)[i]==0x5b,"full FP32 patch was written");
            check(!std::memcmp(image->data,before.data(),before.size()),"image changed");++checks;
        }
    }
    ggml_free(ctx);return checks;
}

int main() {
    try {
        int checks=0;
        for(const auto & v:std::vector<std::array<int,9>>{{1,1,256,1,1,1,1,0,1},{3,1,256,1,1,1,1,0,1},
            {5,1,256,1,1,1,1,0,1},{7,5,32,1,3,3,2,1,1},{5,7,32,2,2,4,1,3,2},
            {3,2,256,1,3,3,2,4,1}})
            checks+=run(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]);
        std::printf("{\"passed\":true,\"geometries\":6,\"checks\":%d,\"packed_bytes_exact\":true,\"float_patch_untouched\":true,\"input_scratch_guards\":true,\"observable_or_shared_patch_rejected\":true,\"board_executed\":false}\n",checks);
        return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
