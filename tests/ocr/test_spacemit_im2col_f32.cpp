#include "../../tools/ocr-ggml/ocr-float-graph.h"
#include "../../ggml/src/ggml-spacemit/ocr-im2col-f32.h"
#include <thread>

static void check(int w,int h,int c,int batch,int kw,int kh,int sx,int sy,int px,int py,int dx,int dy) {
    auto * ctx=ggml_init({16*1024*1024,nullptr,false});require(ctx,"context unavailable");
    auto * image=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,w,h,c,batch);
    auto * geometry=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,kw,kh,c,1);
    auto * reference=ggml_im2col(ctx,geometry,image,sx,sy,px,py,dx,dy,true,GGML_TYPE_F32);
    require(ocr_im2col_f32::supported(reference),"valid geometry rejected");
    auto * pixels=static_cast<float *>(image->data);
    for(int64_t i=0;i<ggml_nelements(image);++i)pixels[i]=(float(i%127)-63.f)*.125f;
    pixels[0]=-0.f;
    std::vector<char> before(static_cast<const char *>(image->data),static_cast<const char *>(image->data)+ggml_nbytes(image));
    std::memset(geometry->data,0x55,ggml_nbytes(geometry));
    auto * cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(cpu,4);
    auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,reference);
    require(ggml_backend_graph_compute(cpu,graph)==GGML_STATUS_SUCCESS,"CPU reference failed");
    for(int nth:{1,4,8}) {
        std::vector<uint32_t> output(size_t(ggml_nelements(reference))+32,0x5b5b5b5b);
        auto candidate=*reference;candidate.data=output.data()+16;
        std::vector<std::thread> workers;
        for(int ith=0;ith<nth;++ith)workers.emplace_back([&,ith]{ocr_im2col_f32::forward(&candidate,ith,nth);});
        for(auto & worker:workers)worker.join();
        require(std::memcmp(candidate.data,reference->data,ggml_nbytes(reference))==0,"IM2COL differs from GGML CPU");
        const auto * actual=static_cast<const float *>(candidate.data);
        for(int n=0;n<batch;++n)for(int y=0;y<reference->ne[2];++y)for(int x=0;x<reference->ne[1];++x)
            for(int channel=0;channel<c;++channel)for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx) {
                const int iy=y*sy+ky*dy-py,ix=x*sx+kx*dx-px;
                const float expected=(iy<0||iy>=h||ix<0||ix>=w)?0.f:pixels[((n*c+channel)*h+iy)*w+ix];
                const size_t index=(((n*size_t(reference->ne[2])+y)*reference->ne[1]+x)*c+channel)*kh*kw+ky*kw+kx;
                require(std::memcmp(actual+index,&expected,sizeof(float))==0,"IM2COL differs from independent coordinates");
            }
        for(size_t i=0;i<16;++i)require(output[i]==0x5b5b5b5b&&output[output.size()-1-i]==0x5b5b5b5b,"output guard overwritten");
        require(std::memcmp(image->data,before.data(),before.size())==0,"image modified");
    }
    auto bad=*reference;bad.op_params[0]=0;require(!ocr_im2col_f32::supported(&bad),"zero stride accepted");
    bad=*reference;bad.op_params[6]=0;require(!ocr_im2col_f32::supported(&bad),"1D operation accepted");
    bad=*reference;bad.op_params[2]=-1;require(!ocr_im2col_f32::supported(&bad),"negative padding accepted");
    bad=*reference;bad.type=GGML_TYPE_F16;require(!ocr_im2col_f32::supported(&bad),"FP16 output accepted");
    bad=*reference;bad.ne[0]++;require(!ocr_im2col_f32::supported(&bad),"wrong output shape accepted");
    auto bad_image=*image;bad_image.nb[1]+=4;bad=*reference;bad.src[1]=&bad_image;
    require(!ocr_im2col_f32::supported(&bad),"noncontiguous image accepted");
    ggml_backend_free(cpu);ggml_free(ctx);
}

int main() {
    try {
        check(7,9,3,1,3,3,1,1,1,1,1,1);
        check(11,7,5,2,3,2,2,1,1,0,1,1);
        check(9,13,4,2,2,3,1,2,2,1,2,2);
        check(4,5,2,1,1,1,1,1,0,0,1,1);
        check(3,2,1,1,3,3,2,2,4,3,1,1);
        check(17,11,32,1,3,3,2,2,1,1,1,1);
        check(5,4,2,2,1,3,1,1,0,1,1,1);
        check(5,4,2,1,3,1,1,1,1,0,1,1);
        std::puts("{\"passed\":true,\"geometries\":8,\"thread_counts\":[1,4,8],\"cpu_reference_byte_exact\":true,\"independent_coordinates_byte_exact\":true,\"guards\":true,\"input_preserved\":true,\"board_executed\":false}");return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
