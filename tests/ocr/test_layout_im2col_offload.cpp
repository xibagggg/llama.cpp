#include "../../tools/ocr-ggml/ocr-float-graph.h"
#include "../../tools/ocr-ggml/doclayout-backend.h"
#include "../../ggml/src/ggml-spacemit/ocr-im2col-f32.h"
#include "layout-test-nonhost.h"
#include <thread>

static int im2col_calls=0;
static bool special_compute(ggml_tensor * out) {
    if(out->op!=GGML_OP_IM2COL)return false;
    require(ocr_im2col_f32::supported(out),"scheduled unsupported IM2COL");
    ++im2col_calls;std::vector<std::thread> workers;
    for(int ith=0;ith<4;++ith)workers.emplace_back([=]{ocr_im2col_f32::forward(out,ith,4);});
    for(auto & worker:workers)worker.join();return true;
}

static json run(bool offload,int batch,std::vector<float> & final) {
    const int w=8,h=9,c=4,kw=3,kh=3,m=32,k=c*kw*kh;
    setenv("OCR_LAYOUT_OFFLOAD_IM2COL",offload?"1":"0",1);
    auto * cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(cpu,4);
    layout_test_nonhost other;other.special_compute=special_compute;
    auto * ctx=ggml_init({8*1024*1024,nullptr,true});require(ctx,"metadata unavailable");
    auto * weight_ctx=ggml_init({1024*1024,nullptr,true});require(weight_ctx,"weight metadata unavailable");
    auto * weights=ggml_new_tensor_2d(weight_ctx,GGML_TYPE_F32,k,m);
    auto * weight_buffer=ggml_backend_alloc_ctx_tensors(weight_ctx,&other.frontend);require(weight_buffer,"test weight allocation failed");
    ggml_backend_buffer_set_usage(weight_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> weight_data(size_t(k*m));
    for(int j=0;j<m;++j)for(int i=0;i<k;++i)weight_data[size_t(j*k+i)]=float((i+j)%3-1);
    ggml_backend_tensor_set(weights,weight_data.data(),0,weight_data.size()*sizeof(float));
    auto * image=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,w,h,c,batch);ggml_set_input(image);ggml_set_output(image);
    auto * geometry=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,kw,kh,c,1);
    std::vector<float> geometry_values(size_t(k),0.f);geometry->data=geometry_values.data();
    auto * patches=ggml_im2col(ctx,geometry,image,1,1,1,1,1,1,true,GGML_TYPE_F32);
    auto * flattened=ggml_reshape_2d(ctx,patches,k,w*h*batch);
    auto * matrix=ggml_mul_mat(ctx,weights,flattened);
    auto * output=ggml_scale(ctx,matrix,.5f);ggml_set_output(output);
    auto * graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(graph,output);
    json info;
    const int calls_before=im2col_calls;
    {
        layout_backend execution("cpu-sched",cpu);
        // Test injection only. No SpacemiT API or physical repacking is invoked.
        execution.accelerator=&other.frontend;execution.accelerated_weights.insert(weights);
        execution.allocate(graph,image);
        require(execution.accelerated_matrices==1&&execution.accelerated_im2cols==int(offload),"layout assignment ignored the offload option");
        std::vector<float> input(size_t(w*h*c*batch)),preserved(input.size());final.resize(size_t(w*h*batch*m));
        const auto uploads=other.uploaded_bytes,downloads=other.downloaded_bytes;
        for(int repeat=0;repeat<2;++repeat) {
            for(size_t i=0;i<input.size();++i)input[i]=(float(i%31)-15.f)*.125f+float(repeat)*.25f;
            ggml_backend_tensor_set(image,input.data(),0,input.size()*sizeof(float));
            require(execution.compute(graph)==GGML_STATUS_SUCCESS,"scheduled convolution failed");
            ggml_backend_tensor_get(output,final.data(),0,final.size()*sizeof(float));
            ggml_backend_tensor_get(image,preserved.data(),0,preserved.size()*sizeof(float));require(preserved==input,"input changed");
            for(int n=0;n<batch;++n)for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int row=0;row<m;++row) {
                float expected=0;
                for(int channel=0;channel<c;++channel)for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx) {
                    const int iy=y+ky-1,ix=x+kx-1;
                    if(iy>=0&&iy<h&&ix>=0&&ix<w)expected+=input[size_t(((n*c+channel)*h+iy)*w+ix)]*weight_data[size_t(row*k+(channel*kh+ky)*kw+kx)];
                }
                require(final[size_t(((n*h+y)*w+x)*m+row)]==expected*.5f,"offloaded convolution differs from independent formula");
            }
        }
        require(im2col_calls-calls_before==(offload?2:0),"new primitive did not execute as assigned");
        info={{"offload",offload},{"batch",batch},{"repeats",2},
            {"uploaded_bytes_compute_only",other.uploaded_bytes-uploads},
            {"downloaded_bytes_compute_only",other.downloaded_bytes-downloads},
            {"new_im2col_calls",im2col_calls-calls_before},{"backend_info",execution.describe()},{"passed",true}};
    }
    ggml_backend_buffer_free(weight_buffer);require(other.live_buffers==0,"non-host buffers leaked");
    ggml_free(weight_ctx);ggml_free(ctx);ggml_backend_free(cpu);return info;
}

int main() {
    try {
        json results=json::array();
        for(int batch:{1,2}) {
            std::vector<float> before,after;
            const auto a=run(false,batch,before),b=run(true,batch,after);
            require(before==after,"switch changed convolution output");
            require(b.at("uploaded_bytes_compute_only").get<size_t>()<a.at("uploaded_bytes_compute_only").get<size_t>(),"moving IM2COL did not reduce actual scheduler uploads");
            require(b.at("downloaded_bytes_compute_only")==a.at("downloaded_bytes_compute_only"),"unexpected output-copy traffic change");
            results.push_back(a);results.push_back(b);
        }
        std::printf("%s\n",json{{"passed",true},{"states",results.size()},{"real_ggml_scheduler_copies",true},
            {"new_im2col_kernel_executed",true},{"switch_outputs_exact",true},{"compute_engine","x86 CPU; distinct non-host test buffers"},
            {"spacemit_executed",false},{"board_executed",false},{"results",results}}.dump().c_str());return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
