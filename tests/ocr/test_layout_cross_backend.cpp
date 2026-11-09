#define OCR_LAYOUT_LIBRARY 1
#include "../../tools/ocr-ggml/doclayout-ggml.cpp"
#include "layout-test-nonhost.h"

struct observed_operation {
    layout_operation operation;
    int replaced_calls=0;
};

static void observed_compute(ggml_tensor * out,int ith,int nth,void * raw) {
    auto & observed=*static_cast<observed_operation *>(raw);
    if(ith==0) {
        bool replaced=false;
        for(size_t i=0;i<observed.operation.inputs.size();++i)
            replaced|=out->src[i]!=observed.operation.inputs[i].storage;
        require(replaced,"scheduler did not substitute the cross-backend input");
        ++observed.replaced_calls;
    }
    layout_compute(out,ith,nth,&observed.operation);
}

static json check(layout_code code,bool packed_path) {
    auto * cpu=ggml_backend_cpu_init();require(cpu,"CPU unavailable");ggml_backend_cpu_set_n_threads(cpu,4);
    layout_test_nonhost other;
    require(!ggml_backend_supports_buft(cpu,&other.buffer_type),"CPU accepted test non-host buffer");
    require(!ggml_backend_supports_buft(&other.frontend,ggml_backend_get_default_buffer_type(cpu)),"test backend accepted CPU buffer");
    auto * context=ggml_init({8*1024*1024,nullptr,true});require(context,"metadata unavailable");
    auto * x=ggml_new_tensor_1d(context,GGML_TYPE_F32,102);ggml_set_input(x);ggml_set_output(x);ggml_set_name(x,"input");
    auto * bias=ggml_new_tensor_1d(context,GGML_TYPE_F32,1);ggml_set_input(bias);ggml_set_output(bias);
    auto * indices=ggml_new_tensor_1d(context,GGML_TYPE_I64,3);ggml_set_input(indices);ggml_set_output(indices);
    auto * seed=ggml_scale(context,x,2.f);ggml_set_name(seed,"nonhost.seed");
    observed_operation first{};auto & op=first.operation;
    op.code=code;op.dims={2,3,17};op.axis=1;op.inputs={packed(seed,{2,3,17})};
    op.packed_concat=packed_path&&code==layout_code::concat;
    if(code==layout_code::add)op.inputs.push_back(packed(bias,{}));
    if(code==layout_code::gather)op.inputs.push_back(packed(indices,{3}));
    if(code==layout_code::concat){op.inputs.push_back(packed(seed,{2,3,17}));op.dims={2,6,17};}
    if(code==layout_code::reduce_sum){op.axes={1};op.keepdims=true;op.dims={2,1,17};}
    std::vector<ggml_tensor *> args;for(const auto & v:op.inputs)args.push_back(v.storage);
    auto * custom=ggml_custom_4d(context,GGML_TYPE_F32,elements(op.dims),1,1,1,args.data(),int(args.size()),observed_compute,4,&first);
    ggml_set_name(custom,"cpu.custom");
    auto * scaled=ggml_scale(context,custom,.5f);ggml_set_name(scaled,"nonhost.scale");
    observed_operation second{{layout_code::copy,{packed(scaled,op.dims)},op.dims}};
    ggml_tensor * argument[]={scaled};
    auto * output=ggml_custom_4d(context,GGML_TYPE_F32,elements(op.dims),1,1,1,argument,1,observed_compute,4,&second);
    ggml_set_name(output,"cpu.output");ggml_set_output(output);
    auto * graph=ggml_new_graph_custom(context,128,false);ggml_build_forward_expand(graph,output);
    ggml_backend_t backends[]={&other.frontend,cpu};
    auto * scheduler=ggml_backend_sched_new(backends,nullptr,2,128,false,false);require(scheduler,"scheduler unavailable");
    for(auto * t:{x,bias,indices,custom,output})ggml_backend_sched_set_tensor_backend(scheduler,t,cpu);
    for(auto * t:{seed,scaled})ggml_backend_sched_set_tensor_backend(scheduler,t,&other.frontend);
    require(ggml_backend_sched_alloc_graph(scheduler,graph),"scheduler allocation failed");
    const auto first_storage=op.inputs[0].storage;
    const auto second_storage=second.operation.inputs[0].storage;
    require(ggml_backend_sched_get_n_splits(scheduler)==4,"expected four alternating backend splits");
    std::vector<float> input(102),actual(size_t(elements(op.dims))),preserved(102);
    const int64_t indexes[3]={-1,0,1};const float scalar=5.f;
    if(bias->data)ggml_backend_tensor_set(bias,&scalar,0,sizeof(scalar));
    if(indices->data)ggml_backend_tensor_set(indices,indexes,0,sizeof(indexes));
    for(int repeat=0;repeat<3;++repeat) {
        for(size_t i=0;i<input.size();++i)input[i]=(float(i)-50.f)*.25f+float(repeat)*2.f;
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        const auto uploads=other.upload_calls,downloads=other.download_calls;
        require(ggml_backend_sched_graph_compute(scheduler,graph)==GGML_STATUS_SUCCESS,"cross-backend compute failed");
        require(other.upload_calls>uploads&&other.download_calls>downloads,"scheduler failed to copy in both directions");
        ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
        ggml_backend_tensor_get(x,preserved.data(),0,input.size()*sizeof(float));require(input==preserved,"input modified");
        require(op.inputs[0].storage==first_storage&&second.operation.inputs[0].storage==second_storage,"shared operation plan changed");
        for(size_t i=0;i<actual.size();++i) {
            float expected=0;
            if(code==layout_code::add)expected=input[i]+2.5f;
            if(code==layout_code::copy)expected=input[i];
            if(code==layout_code::concat)expected=input[(i/102)*51+(i/17%3)*17+i%17];
            if(code==layout_code::gather){int64_t j=indexes[(i/17)%3];if(j<0)j+=3;expected=input[(i/51)*51+size_t(j)*17+i%17];}
            if(code==layout_code::reduce_sum)for(int j=0;j<3;++j)expected+=input[(i/17)*51+size_t(j)*17+i%17];
            require(actual[i]==expected,"cross-backend output differs from independent formula");
        }
    }
    require(first.replaced_calls==3&&second.replaced_calls==3,"custom callback did not observe each copied input");
    json result={{"operation",int(code)},{"packed_path",packed_path},{"repeats",3},{"splits",4},
        {"actual_src_substitutions",first.replaced_calls+second.replaced_calls},
        {"upload_calls",other.upload_calls},{"download_calls",other.download_calls},
        {"uploaded_bytes",other.uploaded_bytes},{"downloaded_bytes",other.downloaded_bytes},
        {"nonhost_compute_calls",other.compute_calls},{"passed",true}};
    ggml_backend_sched_free(scheduler);require(other.live_buffers==0,"scheduler leaked non-host buffers");
    ggml_free(context);ggml_backend_free(cpu);return result;
}

int main() {
    try {
        json results=json::array();
        for(bool packed:{false,true})for(auto code:{layout_code::add,layout_code::copy,layout_code::concat,layout_code::gather,layout_code::reduce_sum})
            results.push_back(check(code,packed));
        std::printf("%s\n",json{{"passed",true},{"cases",results.size()},{"threads",4},{"repeats_per_case",3},
            {"real_ggml_scheduler_copies",true},{"compute_engine","x86 CPU"},{"spacemit_executed",false},
            {"board_executed",false},{"results",results}}.dump().c_str());return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
