#define OCR_LAYOUT_LIBRARY 1
#include "../../tools/ocr-ggml/doclayout-ggml.cpp"

static std::vector<float> run(int mode,int n,int c,int h,int width,int m,int kh,int kw,int stride,int pad,int dilation,bool bias) {
    doclayout_graph model;
    model.context=ggml_init({64*1024*1024,nullptr,false});
    require(model.context,"test allocation failed");
    auto * x=ggml_new_tensor_4d(model.context,GGML_TYPE_F32,width,h,c,n);
    for(int64_t i=0;i<ggml_nelements(x);++i)static_cast<float *>(x->data)[i]=i%3==0?127.f:0.f;
    const std::vector<float> input(static_cast<float *>(x->data),static_cast<float *>(x->data)+ggml_nelements(x));
    const int64_t k=c*kh*kw;
    const bool matrix=mode>0&&mode<4;
    std::vector<float> raw(size_t(m*k));
    for(size_t i=0;i<raw.size();++i)raw[i]=(mode==3||mode==4)?(i%3==0?127.f*.03125f:0.f):float(int(i%16)-8)*.03125f;
    model.values["x"]=packed(x,{n,c,h,width});
    if(mode==2||mode==3) {
        auto type=mode==2?GGML_TYPE_Q4_0:GGML_TYPE_Q8_0;
        auto * w=ggml_new_tensor_2d(model.context,type,k,m);
        require(ggml_quantize_chunk(type,raw.data(),w->data,0,m,k,nullptr)==ggml_nbytes(w),"quantized size mismatch");
        model.quantized_weights["w"]=w;
    } else {
        auto * w=matrix?ggml_new_tensor_2d(model.context,GGML_TYPE_F32,k,m):ggml_new_tensor_4d(model.context,GGML_TYPE_F32,kw,kh,c,m);
        std::memcpy(w->data,raw.data(),raw.size()*sizeof(float));
        model.values["w"]=packed(w,matrix?shape({m,k}):shape({m,c,kh,kw}));
    }
    auto * b=ggml_new_tensor_1d(model.context,GGML_TYPE_F32,m);
    for(int i=0;i<m;++i)static_cast<float *>(b->data)[i]=float(i)*.5f;
    model.values["b"]=packed(b,{m});
    const int oh=(h+2*pad-dilation*(kh-1)-1)/stride+1;
    const int ow=(width+2*pad-dilation*(kw-1)-1)/stride+1;
    json attrs={{"group",1},{"strides",{stride,stride}},{"pads",{pad,pad,pad,pad}},{"dilations",{dilation,dilation}},{"kernel_shape",{kh,kw}}};
    if(matrix)attrs["ggml_conv_weight_shape"]={m,c,kh,kw};
    std::vector<std::string> ins={"x","w"};if(bias)ins.push_back("b");
    model.append_layout({{"op","Conv"},{"name","test"},{"inputs",ins},{"attrs",attrs},
        {"outputs",json::array({{{"id","y"},{"shape",{n,m,oh,ow}},{"dtype","f32"}}})}});
    require(model.matrix_conv_count==int(matrix)&&model.q4_matrix_conv_count==int(mode==2)
            &&model.q8_matrix_conv_count==int(mode==3),"wrong path counters");
    auto * y=model.get("y");auto * graph=ggml_new_graph(model.context);ggml_build_forward_expand(graph,y);
    require(ggml_graph_compute_with_ctx(model.context,graph,4)==GGML_STATUS_SUCCESS,"test compute failed");
    require(std::memcmp(x->data,input.data(),input.size()*sizeof(float))==0,"input changed");
    std::vector<float> out(size_t(ggml_nelements(y)));std::memcpy(out.data(),y->data,out.size()*sizeof(float));
    for(float v:out)require(std::isfinite(v),"nonfinite output");
    return out;
}

int main() {
    try {
        const int cases[][10]={{1,32,5,7,3,1,1,1,0,1},{2,8,5,7,5,2,2,1,0,1},
            {1,32,9,11,7,3,3,2,1,1},{2,32,9,11,4,3,3,1,2,2},
            {1,32,2,3,1,1,1,1,0,1},{2,4,7,9,5,2,4,1,1,1}};
        for(const auto & v:cases) {
            auto reference=run(0,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],true);
            for(int mode:{1,2,3}) {
                auto got=run(mode,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],true);
                auto expected=mode==3?run(4,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],true):reference;
                require(got.size()==expected.size()&&!std::memcmp(got.data(),expected.data(),got.size()*sizeof(float)),"matrix Conv layout differs");
            }
        }
        std::puts("{\"passed\":true,\"cases\":6,\"modes\":4,\"batch2\":true,\"non_square_kernel\":true,\"q4_exact_grid\":true,\"q8_exact_grid\":true}");
        return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
