#define OCR_LAYOUT_LIBRARY 1
#include "../../tools/ocr-ggml/doclayout-ggml.cpp"
#include <thread>

static void check(layout_code code,bool packed_path) {
    auto * ctx=ggml_init({8*1024*1024,nullptr,false});require(ctx,"context unavailable");
    layout_operation op{};op.code=code;op.dims={2,3,17};op.axis=1;
    op.packed_masks=packed_path && (code==layout_code::where||code==layout_code::cast);
    op.packed_concat=packed_path && code==layout_code::concat;
    const int64_t count=elements(op.dims);
    std::vector<ggml_tensor *> actuals;
    auto add=[&](const shape & dims,enum ggml_type type,int source) {
        auto * old=ggml_new_tensor_1d(ctx,type,elements(dims)+32);
        auto * actual=ggml_dup_tensor(ctx,old);
        const size_t unit=ggml_type_size(type);
        std::memset(old->data,0,ggml_nbytes(old));std::memset(actual->data,0x4b,ggml_nbytes(actual));
        for(int64_t i=0;i<elements(dims);++i) {
            auto * ptr=static_cast<char *>(actual->data)+(16+size_t(i))*unit;
            if(type==GGML_TYPE_F32) {
                float v=source==0?float(i)*.25f:source==1?5.f:-7.f;
                std::memcpy(ptr,&v,sizeof(v));
            } else if(type==GGML_TYPE_I64) {
                int64_t v=(i%3)-1;std::memcpy(ptr,&v,sizeof(v));
            } else *ptr=char(i%3==0);
        }
        value v{old,dims,std::vector<size_t>(dims.size()),16*unit};
        size_t stride=unit;
        for(int d=int(dims.size())-1;d>=0;--d){v.strides[d]=stride;stride*=size_t(dims[d]);}
        op.inputs.push_back(v);actuals.push_back(actual);
    };
    if(code==layout_code::add) {add(op.dims,GGML_TYPE_F32,0);add({},GGML_TYPE_F32,1);}
    else if(code==layout_code::where) {add(op.dims,GGML_TYPE_I8,0);add({},GGML_TYPE_F32,1);add(op.dims,GGML_TYPE_F32,2);}
    else if(code==layout_code::cast) add(op.dims,GGML_TYPE_I8,0);
    else if(code==layout_code::concat) {add({2,1,17},GGML_TYPE_F32,1);add({2,2,17},GGML_TYPE_F32,2);}
    else if(code==layout_code::gather) {
        op.dims={2,3,17};op.axis=1;add({2,3,17},GGML_TYPE_F32,0);add({3},GGML_TYPE_I64,0);
    }
    auto * out=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,count+32);
    auto * allocation=static_cast<char *>(out->data);std::memset(allocation,0x6b,ggml_nbytes(out));
    out->data=allocation+16*sizeof(float);
    for(size_t p=0;p<actuals.size();++p)out->src[p]=actuals[p];
    std::vector<std::vector<char>> before;
    for(auto * input:actuals)before.emplace_back(static_cast<const char *>(input->data),static_cast<const char *>(input->data)+ggml_nbytes(input));
    std::vector<std::thread> workers;
    for(int t=0;t<4;++t)workers.emplace_back([&,t]{layout_compute(out,t,4,&op);});
    for(auto & t:workers)t.join();
    for(int64_t i=0;i<count;++i) {
        float expected=0;
        if(code==layout_code::add)expected=float(i)*.25f+5.f;
        else if(code==layout_code::where)expected=i%3==0?5.f:-7.f;
        else if(code==layout_code::cast)expected=float(i%3==0);
        else if(code==layout_code::concat)expected=(i/17)%3==0?5.f:-7.f;
        else if(code==layout_code::gather) {
            int64_t query=(i/17)%3-1;if(query<0)query+=3;
            expected=float((i/51)*51+query*17+i%17)*.25f;
        }
        require(static_cast<float *>(out->data)[i]==expected,"custom callback read stale storage");
    }
    for(size_t p=0;p<actuals.size();++p) {
        require(op.inputs[p].storage!=actuals[p],"shared plan mutated");
        require(std::memcmp(actuals[p]->data,before[p].data(),before[p].size())==0,"actual input modified");
        const auto * bytes=static_cast<const unsigned char *>(op.inputs[p].storage->data);
        for(size_t i=0;i<ggml_nbytes(op.inputs[p].storage);++i)require(bytes[i]==0,"original input modified");
    }
    for(size_t i=0;i<16*sizeof(float);++i)
        require(allocation[i]==char(0x6b)&&allocation[16*sizeof(float)+size_t(count)*sizeof(float)+i]==char(0x6b),"output guard modified");
    auto * saved=out->src[0];out->src[0]=nullptr;
    bool rejected=false;try{layout_compute(out,0,1,&op);}catch(const std::runtime_error &){rejected=true;}
    require(rejected,"missing runtime input accepted");out->src[0]=saved;
    auto bad=*saved;bad.nb[1]+=sizeof(float);out->src[0]=&bad;
    rejected=false;try{layout_compute(out,0,1,&op);}catch(const std::runtime_error &){rejected=true;}
    require(rejected,"changed copy layout accepted");out->src[0]=saved;
    out->data=allocation;ggml_free(ctx);
}

int main() {
    try {
        for(bool packed:{false,true})for(auto code:{layout_code::add,layout_code::where,layout_code::cast,layout_code::concat,layout_code::gather})check(code,packed);
        std::puts("{\"passed\":true,\"cases\":10,\"threads\":4,\"replaced_sources\":true,\"offsets\":true,\"shared_plan_immutable\":true,\"guards\":true}");return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
