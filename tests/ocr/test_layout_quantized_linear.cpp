#define OCR_LAYOUT_LIBRARY 1
#include "../../tools/ocr-ggml/doclayout-ggml.cpp"
#include <array>

static void run(enum ggml_type type,int k,int m,int rows) {
    doclayout_graph model;
    model.context=ggml_init({16*1024*1024,nullptr,false});require(model.context,"metadata failed");
    auto * x=ggml_new_tensor_2d(model.context,GGML_TYPE_F32,k,rows);
    std::vector<float> input(size_t(k*rows)),weights(size_t(k*m));
    for(size_t i=0;i<input.size();++i)input[i]=i%3==0?127.f:0.f;
    for(size_t i=0;i<weights.size();++i)
        weights[i]=type==GGML_TYPE_Q4_0?float(int(i%16)-8)*.03125f:(i%3==0?127.f*.03125f:0.f);
    std::memcpy(x->data,input.data(),input.size()*sizeof(float));
    model.values["x"]=packed(x,{rows,k});
    auto * w=ggml_new_tensor_2d(model.context,type,k,m);
    require(ggml_quantize_chunk(type,weights.data(),w->data,0,m,k,nullptr)==ggml_nbytes(w),"bad quantization size");
    const std::vector<char> weight_bytes(static_cast<char *>(w->data),static_cast<char *>(w->data)+ggml_nbytes(w));
    model.quantized_weights["w"]=w;
    auto * b=ggml_new_tensor_1d(model.context,GGML_TYPE_F32,m);
    for(int i=0;i<m;++i)static_cast<float *>(b->data)[i]=float(i)*.5f;
    model.values["b"]=packed(b,{m});
    auto node=json{{"op","Gemm"},{"name","quantized-linear-test"},{"inputs",{"x","w","b"}},
        {"attrs",{{"transA",0},{"transB",1},{"alpha",1.f},{"beta",1.f},{"ggml_linear_weight_shape",{m,k}}}},
        {"outputs",json::array({{{"id","y"},{"shape",{rows,m}},{"dtype","f32"}}})}};
    for(int invalid=0;invalid<3;++invalid) {
        auto bad=node;
        if(invalid==0)bad["attrs"]["transB"]=0;
        if(invalid==1)bad["attrs"].erase("ggml_linear_weight_shape");
        if(invalid==2)bad["attrs"]["ggml_linear_weight_shape"]={m+1,k};
        bool rejected=false;try{model.append_layout(bad);}catch(const std::exception &){rejected=true;}
        require(rejected,"malformed quantized Gemm accepted");
    }
    require(model.quantized_linear_count==0,"rejected nodes changed counters");
    model.append_layout(node);
    require(model.quantized_linear_count==1 && model.q4_linear_count==int(type==GGML_TYPE_Q4_0)
        && model.q8_linear_count==int(type==GGML_TYPE_Q8_0),"linear counts incorrect");
    auto * y=model.get("y");auto * graph=ggml_new_graph(model.context);ggml_build_forward_expand(graph,y);
    for(int repeat=0;repeat<2;++repeat) {
        require(ggml_graph_compute_with_ctx(model.context,graph,4)==GGML_STATUS_SUCCESS,"compute failed");
        for(int r=0;r<rows;++r)for(int j=0;j<m;++j) {
            float expected=float(j)*.5f;
            for(int i=0;i<k;++i)expected+=input[size_t(r*k+i)]*weights[size_t(j*k+i)];
            require(static_cast<float *>(y->data)[r*m+j]==expected,"quantized linear differs from independent exact-grid dot");
        }
        require(!std::memcmp(x->data,input.data(),input.size()*sizeof(float)),"input changed");
        require(!std::memcmp(w->data,weight_bytes.data(),weight_bytes.size()),"weight changed");
    }
}

int main(int argc,char ** argv) {
    try {
        if(argc==2) {
            doclayout_graph model;model.load_layout(argv[1],4,{});
            int matrices=0,quantized=0;double total_ops=0,quantized_ops=0;
            for(int i=0;i<ggml_graph_n_nodes(model.graph);++i) {
                auto * node=ggml_graph_node(model.graph,i);if(node->op!=GGML_OP_MUL_MAT)continue;
                ++matrices;const double ops=2.*node->src[0]->ne[0]*ggml_nelements(node);total_ops+=ops;
                if(ggml_is_quantized(node->src[0]->type)){++quantized;quantized_ops+=ops;}
            }
            std::printf("%s\n",json{{"matrix_nodes",matrices},{"quantized_matrix_nodes",quantized},
                {"matrix_dense_ops",total_ops},{"quantized_matrix_dense_ops",quantized_ops},
                {"quantized_dense_ops_fraction",quantized_ops/total_ops},{"backend_info",model.backend_info()},
                {"compute_called",false},{"hardware_coverage_verified",false}}.dump().c_str());return 0;
        }
        int cases=0;
        for(auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q8_0})
            for(const auto & v:std::vector<std::array<int,3>>{{32,4,1},{32,25,3},{256,32,17},{256,96,3},{512,256,1}}) {
                run(type,v[0],v[1],v[2]);++cases;
            }
        std::printf("%s\n",json{{"passed",true},{"cases",cases},{"repeats",2},
            {"invalid_signatures_rejected",30},{"independent_dot_exact",true},{"input_weights_preserved",true},
            {"board_executed",false}}.dump().c_str());return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
