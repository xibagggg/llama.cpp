#include "../../tools/ocr-ggml/ocr-float-graph.h"
#include "../../tools/ocr-ggml/doclayout-backend.h"
#include "../../ggml/src/ggml-spacemit/ocr-im2col-pack.h"
#include "layout-test-nonhost.h"
#include <thread>

// The test backend uses real scheduler copies but CPU math and standard Q8_0.
// Observe the same matcher used by production before computing the split.
static int matches=0,rebound_inputs=0,fused_calls=0;
static bool execute_fusion=false;
static ggml_tensor * original_input=nullptr;
static std::vector<float> current_pixels;

static ggml_status inspect(layout_test_nonhost * device,ggml_cgraph * graph) {
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
        const int end=ocr_im2col_pack::match(graph,i);
        if(end<0) {
            if(execute_fusion) {
                auto single=*graph;single.size=0;single.n_nodes=1;single.n_leafs=0;
                single.nodes=graph->nodes+i;single.leafs=nullptr;single.grads=nullptr;single.grad_accs=nullptr;single.uid=0;
                const auto status=ggml_backend_graph_compute(device->executor,&single);
                if(status!=GGML_STATUS_SUCCESS)return status;
            }
            continue;
        }
        ++matches;auto * patch=ggml_graph_node(graph,i);
        auto * matrix=ggml_graph_node(graph,end);
        require(patch->src[1]!=original_input,"scheduler did not copy image input");
        require(patch->src[1]->data!=original_input->data,"scheduler reused foreign image buffer");
        ++rebound_inputs;
        require(!std::memcmp(patch->src[1]->data,current_pixels.data(),current_pixels.size()*sizeof(float)),"copied image input is stale");
        const int64_t k=patch->ne[0],rows=ggml_nelements(patch)/k;
        const size_t row_stride=ggml_row_size(GGML_TYPE_Q8_0,k);
        const std::vector<uint8_t> patch_before(static_cast<uint8_t *>(patch->data),
            static_cast<uint8_t *>(patch->data)+ggml_nbytes(patch));
        std::vector<uint8_t> packed(size_t(rows)*row_stride);
        std::vector<std::vector<float>> scratch(4,std::vector<float>(size_t(4*k)));
        auto one=[](size_t,const float * src,size_t count,uint8_t * dst) {
            ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(src,dst,int64_t(count));
        };
        auto four=[&](size_t block,const float * src,size_t count,uint8_t * dst) {
            for(int r=0;r<4;++r)one(block,src+r*count,count,dst+r*row_stride);
        };
        std::vector<std::thread> workers;
        for(int ith=0;ith<4;++ith)workers.emplace_back([&,ith] {
            ocr_im2col_pack::pack(patch,ith,4,32,row_stride,packed.data(),scratch[ith].data(),one,four);
        });
        for(auto & worker:workers)worker.join();
        // Cross-check extraction/packing against coordinates on the original
        // CPU input, independently of the scheduler's replacement tensors.
        const auto * geometry=patch->src[0];
        const int64_t kw=geometry->ne[0],kh=geometry->ne[1],c=original_input->ne[2];
        std::vector<float> reference(size_t(k*rows));
        const int64_t iw=original_input->ne[0],ih=original_input->ne[1];
        const int64_t ow=patch->ne[1],oh=patch->ne[2];
        for(int64_t r=0;r<rows;++r)for(int64_t channel=0;channel<c;++channel)
            for(int64_t ky=0;ky<kh;++ky)for(int64_t kx=0;kx<kw;++kx) {
                const int64_t ix=(r%ow)*patch->op_params[0]+kx*patch->op_params[4]-patch->op_params[2];
                const int64_t iy=((r/ow)%oh)*patch->op_params[1]+ky*patch->op_params[5]-patch->op_params[3];
                const int64_t batch=r/(ow*oh);
                reference[size_t(r*k+(channel*kh+ky)*kw+kx)]=(ix<0||ix>=iw||iy<0||iy>=ih)?0.f:
                    current_pixels[size_t(((batch*c+channel)*ih+iy)*iw+ix)];
            }
        std::vector<uint8_t> expected(packed.size());
        for(int64_t r=0;r<rows;++r)one(32,reference.data()+r*k,size_t(k),expected.data()+r*row_stride);
        require(packed==expected,"scheduled fused packing differs from independent coordinates");
        require(!std::memcmp(patch->data,patch_before.data(),patch_before.size()),"packing wrote the full patch");
        require(matrix->src[1]->view_src==patch,"scheduled matrix view lost the patch");
        if(execute_fusion) {
            const auto * weights=matrix->src[0];
            const auto * traits=ggml_get_type_traits_cpu(weights->type);
            require(traits->vec_dot&&traits->vec_dot_type==GGML_TYPE_Q8_0,"test dot requires Q8_0 activation packing");
            for(int64_t row=0;row<rows;++row)for(int64_t col=0;col<matrix->ne[0];++col)
                traits->vec_dot(int(k),static_cast<float *>(matrix->data)+row*matrix->ne[0]+col,0,
                    static_cast<const char *>(weights->data)+col*weights->nb[1],0,packed.data()+row*row_stride,0,1);
            ++fused_calls;i=end;
        }
    }
    return execute_fusion?GGML_STATUS_SUCCESS:ggml_backend_graph_compute(device->executor,graph);
}

static json run(bool offload,int batch,int visibility,bool fused) {
    const int w=7,h=5,c=32,kw=3,kh=3,n=32,k=c*kw*kh;
    setenv("OCR_LAYOUT_OFFLOAD_IM2COL",offload?"1":"0",1);
    execute_fusion=fused;
    auto * cpu=ggml_backend_cpu_init();require(cpu,"CPU allocation failed");ggml_backend_cpu_set_n_threads(cpu,4);
    layout_test_nonhost other;other.inspect_graph=inspect;
    auto * ctx=ggml_init({8*1024*1024,nullptr,true});require(ctx,"metadata failed");
    auto * weights_context=ggml_init({1024*1024,nullptr,true});require(weights_context,"weight metadata failed");
    auto * weights=ggml_new_tensor_2d(weights_context,GGML_TYPE_Q8_0,k,n);
    auto * weight_buffer=ggml_backend_alloc_ctx_tensors(weights_context,&other.frontend);require(weight_buffer,"weight buffer failed");
    ggml_backend_buffer_set_usage(weight_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> raw(size_t(k*n));
    for(size_t i=0;i<raw.size();++i)raw[i]=i%3==0?127.f*.03125f:0.f;
    std::vector<uint8_t> quantized(ggml_nbytes(weights));
    require(ggml_quantize_chunk(GGML_TYPE_Q8_0,raw.data(),quantized.data(),0,n,k,nullptr)==quantized.size(),"weight quantization failed");
    ggml_backend_tensor_set(weights,quantized.data(),0,quantized.size());
    auto * image=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,w,h,c,batch);ggml_set_input(image);ggml_set_output(image);
    original_input=image;
    auto * geometry=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,kw,kh,c,1);
    std::vector<float> geometry_data(size_t(k),0.f);geometry->data=geometry_data.data();
    auto * patch=ggml_im2col(ctx,geometry,image,1,1,1,1,1,1,true,GGML_TYPE_F32);
    auto * flat=ggml_reshape_2d(ctx,patch,k,w*h*batch);
    auto * matrix=ggml_mul_mat(ctx,weights,flat);
    auto * output=ggml_scale(ctx,matrix,.5f);ggml_set_output(output);
    if(visibility==1)ggml_set_output(patch);
    if(visibility==2)ggml_set_output(flat);
    auto * graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(graph,output);
    ggml_tensor * extra=nullptr;
    if(visibility==3) {
        extra=ggml_scale(ctx,flat,.25f);ggml_set_output(extra);ggml_build_forward_expand(graph,extra);
    }
    const int before_matches=matches,before_rebound=rebound_inputs,before_fused=fused_calls;
    json result;
    {
        layout_backend execution("cpu-sched",cpu);
        execution.accelerator=&other.frontend;execution.accelerated_weights.insert(weights);
        execution.allocate(graph,image);
        const auto uploads=other.uploaded_bytes;
        for(int repeat=0;repeat<3;++repeat) {
            current_pixels.resize(size_t(w*h*c*batch));
            for(size_t i=0;i<current_pixels.size();++i)current_pixels[i]=(int(i)+repeat)%5==0?127.f:0.f;
            ggml_backend_tensor_set(image,current_pixels.data(),0,current_pixels.size()*sizeof(float));
            require(execution.compute(graph)==GGML_STATUS_SUCCESS,"scheduled graph failed");
            std::vector<float> out(size_t(w*h*batch*n));ggml_backend_tensor_get(output,out.data(),0,out.size()*sizeof(float));
            for(int b=0;b<batch;++b)for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int row=0;row<n;++row) {
                float expected=0;
                for(int channel=0;channel<c;++channel)for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx) {
                    const int iy=y+ky-1,ix=x+kx-1;
                    if(iy>=0&&iy<h&&ix>=0&&ix<w)
                        expected+=current_pixels[size_t(((b*c+channel)*h+iy)*w+ix)]*raw[size_t(row*k+(channel*kh+ky)*kw+kx)];
                }
                require(out[size_t(((b*h+y)*w+x)*n+row)]==expected*.5f,"matrix output differs from independent convolution");
            }
            std::vector<float> preserved(current_pixels.size());ggml_backend_tensor_get(image,preserved.data(),0,preserved.size()*sizeof(float));
            require(preserved==current_pixels,"input changed");
            if(visibility>0) {
                auto * retained=visibility==1?patch:visibility==2?flat:extra;
                std::vector<float> retained_values(size_t(ggml_nelements(retained)));
                ggml_backend_tensor_get(retained,retained_values.data(),0,retained_values.size()*sizeof(float));
                for(int b=0;b<batch;++b)for(int y=0;y<h;++y)for(int x=0;x<w;++x)
                    for(int channel=0;channel<c;++channel)for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx) {
                        const int iy=y+ky-1,ix=x+kx-1;
                        const float expected=iy<0||iy>=h||ix<0||ix>=w?0.f:
                            current_pixels[size_t(((b*c+channel)*h+iy)*w+ix)]*(visibility==3?.25f:1.f);
                        require(retained_values[size_t(((b*h+y)*w+x)*k+(channel*kh+ky)*kw+kx)]==expected,"retained output or shared branch changed");
                    }
            }
        }
        const int expected_matches=offload&&visibility==0?3:0;
        require(matches-before_matches==expected_matches,"runtime graph eligibility differs from expected");
        require(rebound_inputs-before_rebound==expected_matches,"replacement input was not checked");
        require(fused_calls-before_fused==(fused?expected_matches:0),"unexpected fused execution count");
        result={{"offload",offload},{"batch",batch},{"visibility",visibility},{"repeats",3},{"fused",fused},
            {"matched_scheduled_chains",matches-before_matches},{"rebound_inputs",rebound_inputs-before_rebound},
            {"fused_calls",fused_calls-before_fused},
            {"uploaded_bytes_compute_only",other.uploaded_bytes-uploads},{"backend_info",execution.describe()},{"passed",true}};
    }
    ggml_backend_buffer_free(weight_buffer);require(other.live_buffers==0,"device buffer leaked");
    ggml_free(weights_context);ggml_free(ctx);ggml_backend_free(cpu);return result;
}

int main() {
    try {
        json results=json::array();
        for(bool fused:{false,true})for(int batch:{1,2})for(bool offload:{false,true})for(int visibility=0;visibility<4;++visibility)
            results.push_back(run(offload,batch,visibility,fused));
        std::printf("%s\n",json{{"passed",true},{"configurations",results.size()},{"inferences",results.size()*3},
            {"real_scheduler_copies",true},{"results",results},{"board_executed",false},
            {"spacemit_executed",false},{"physical_spacemit_packing_executed",false}}.dump().c_str());return 0;
    }catch(const std::exception & error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
