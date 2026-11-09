#define OCR_LAYOUT_LIBRARY 1
#include "../../tools/ocr-ggml/doclayout-ggml.cpp"
#include <thread>

static int cases = 0;
static void check(layout_code code, const shape & dims, enum ggml_type type,
                  const std::vector<shape> & input_dims, const std::vector<enum ggml_type> & types) {
    auto * ctx = ggml_init({32 * 1024 * 1024, nullptr, false});
    require(ctx != nullptr, "test allocation failed");
    layout_operation op{}; op.code=code; op.dims=dims;
    std::vector<std::vector<char>> before;
    for (size_t p=0; p<input_dims.size(); ++p) {
        auto * tensor=ggml_new_tensor_1d(ctx,types[p],elements(input_dims[p])+32);
        const size_t unit=ggml_type_size(types[p]);
        std::memset(tensor->data,0x5a,ggml_nbytes(tensor));
        const size_t offset=16*unit;
        for (int64_t i=0; i<elements(input_dims[p]); ++i) {
            auto * ptr=static_cast<char *>(tensor->data)+offset+size_t(i)*unit;
            if (types[p]==GGML_TYPE_I8) *ptr=char(int(i%5)-2);
            else {
                // Include exact comparison boundaries, signed zero, infinities and quiet NaN.
                const float values[]={-1.f,0.f,-0.f,.5f,1.f,std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()};
                const float x=values[(size_t(i)+p*3)%8];
                std::memcpy(ptr,&x,sizeof(x));
            }
        }
        auto v=packed(ggml_view_1d(ctx,tensor,elements(input_dims[p]),offset),input_dims[p]);
        // Keep backing storage and explicit nonzero byte offset, as the graph does.
        v.storage=tensor; v.offset=offset;
        op.inputs.push_back(v);
        before.emplace_back(static_cast<const char *>(tensor->data),static_cast<const char *>(tensor->data)+ggml_nbytes(tensor));
    }
    require(can_linear_masks(op,type),"valid linear masks rejected");
    auto * out=ggml_new_tensor_1d(ctx,type,elements(dims)+32);
    auto * allocation=static_cast<char *>(out->data);
    const size_t unit=ggml_type_size(type),guard=16*unit,bytes=size_t(elements(dims))*unit;
    std::memset(allocation,0x6b,ggml_nbytes(out)); out->data=allocation+guard;
    // Reference ONNX broadcasting coordinates; copy selected F32 bits without arithmetic.
    std::vector<char> expected(bytes);
    shape coord(dims.size());
    auto pointer=[&](const value & v) {
        size_t offset=v.offset;
        const size_t shift=dims.size()-v.dims.size();
        for(size_t d=0; d<v.dims.size(); ++d)offset+=size_t(v.dims[d]==1?0:coord[d+shift])*v.strides[d];
        return static_cast<const char *>(v.storage->data)+offset;
    };
    for(int64_t i=0;i<elements(dims);++i) {
        coordinates(i,dims,coord);
        auto * dst=expected.data()+size_t(i)*unit;
        if(code==layout_code::where) {
            const bool selected=*reinterpret_cast<const int8_t *>(pointer(op.inputs[0]))!=0;
            std::memcpy(dst,pointer(op.inputs[selected?1:2]),unit);
        } else if(code==layout_code::greater) {
            *dst=char(*reinterpret_cast<const float *>(pointer(op.inputs[0])) > *reinterpret_cast<const float *>(pointer(op.inputs[1])));
        } else if(type==GGML_TYPE_F32) {
            const float x=float(*reinterpret_cast<const int8_t *>(pointer(op.inputs[0]))); std::memcpy(dst,&x,unit);
        } else {
            const int64_t x=int64_t(*reinterpret_cast<const int8_t *>(pointer(op.inputs[0]))); std::memcpy(dst,&x,unit);
        }
    }
    op.packed_masks=true;
    for(size_t i=0;i<op.inputs.size();++i)out->src[i]=op.inputs[i].storage;
    std::vector<std::thread> workers;
    for(int t=0;t<4;++t)workers.emplace_back([&,t]{layout_compute(out,t,4,&op);});
    for(auto & worker:workers)worker.join();
    require(std::memcmp(out->data,expected.data(),bytes)==0,"fast masks differ from reference");
    // The existing fallback must preserve these quiet NaNs and signed zeros too.
    const std::vector<char> fast(static_cast<const char *>(out->data),static_cast<const char *>(out->data)+bytes);
    op.packed_masks=false;
    for(int t=0;t<4;++t)layout_compute(out,t,4,&op);
    require(std::memcmp(out->data,fast.data(),bytes)==0,"fallback changed mask values");
    for(size_t i=0;i<guard;++i) {
        require(allocation[i]==char(0x6b),"prefix guard changed");
        require(allocation[guard+bytes+i]==char(0x6b),"suffix guard changed");
    }
    for(size_t i=0;i<before.size();++i)
        require(std::memcmp(op.inputs[i].storage->data,before[i].data(),before[i].size())==0,"input changed");
    auto strided=op;
    if(!strided.inputs[0].dims.empty()&&strided.inputs[0].dims.back()>1) {
        strided.inputs[0].strides.back()*=2;
        require(!can_linear_masks(strided,type),"strided input accepted");
    }
    auto bad=op; bad.inputs[0].offset=ggml_nbytes(bad.inputs[0].storage);
    require(!can_linear_masks(bad,type),"out-of-range input accepted");
    out->data=allocation; ggml_free(ctx); ++cases;
}

int main() {
    try {
        check(layout_code::greater,{2,3,17},GGML_TYPE_I8,{{2,3,17},{}},{GGML_TYPE_F32,GGML_TYPE_F32});
        check(layout_code::greater,{2,3,17},GGML_TYPE_I8,{{1,1},{2,3,17}},{GGML_TYPE_F32,GGML_TYPE_F32});
        check(layout_code::greater,{2,3,17},GGML_TYPE_I8,{{2,3,17},{2,3,17}},{GGML_TYPE_F32,GGML_TYPE_F32});
        check(layout_code::cast,{2,3,17},GGML_TYPE_F32,{{2,3,17}},{GGML_TYPE_I8});
        check(layout_code::cast,{2,3,17},GGML_TYPE_I64,{{2,3,17}},{GGML_TYPE_I8});
        check(layout_code::cast,{1,1,1},GGML_TYPE_F32,{{1,1,1}},{GGML_TYPE_I8});
        check(layout_code::where,{2,3,17},GGML_TYPE_F32,{{2,3,17},{2,3,17},{}},{GGML_TYPE_I8,GGML_TYPE_F32,GGML_TYPE_F32});
        check(layout_code::where,{2,3,17},GGML_TYPE_F32,{{2,3,17},{1},{2,3,17}},{GGML_TYPE_I8,GGML_TYPE_F32,GGML_TYPE_F32});
        check(layout_code::where,{2,3,17},GGML_TYPE_F32,{{},{2,3,17},{2,3,17}},{GGML_TYPE_I8,GGML_TYPE_F32,GGML_TYPE_F32});
        check(layout_code::where,{2,3,17},GGML_TYPE_F32,{{2,3,17},{2,3,17},{2,3,17}},{GGML_TYPE_I8,GGML_TYPE_F32,GGML_TYPE_F32});
        check(layout_code::where,{2,1,3,1,17},GGML_TYPE_F32,{{2,1,3,1,17},{2,1,3,1,17},{}},{GGML_TYPE_I8,GGML_TYPE_F32,GGML_TYPE_F32});
        std::printf("{\"passed\":true,\"cases\":%d,\"threads\":4,\"guards\":true,\"offsets\":true,\"rank5\":true,\"special_values\":true}\n",cases);
        return 0;
    }catch(const std::exception & error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
