// Host-only metadata/refusal test. Compile with:
// c++ -std=c++17 -O2 -Itest_mapped_contract_stubs -Imapped-argmax-next
//     test_mapped_contract.cpp -o test_mapped_contract
// The RVV implementation is deliberately disabled in this metadata-only TU;
// real numerical RVV is separately checked by test_mapped_argmax on AI6.
#ifdef __riscv_vector
#undef __riscv_vector
#endif
#include "ocr-dspark-custom-contract.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <vector>

namespace {
int callback_calls=0,other_calls=0;
void trusted(ggml_tensor *,int,int,void *){++callback_calls;}
void impostor(ggml_tensor *,int,int,void *){++other_calls;}
void option(const char * value){
#ifdef _WIN32
    _putenv_s("GGML_SPACEMIT_DSPARK_MAPPED_ARGMAX_RVV",value);
#else
    setenv("GGML_SPACEMIT_DSPARK_MAPPED_ARGMAX_RVV",value,1);
#endif
}
ggml_tensor tensor(void * data,ggml_type type,std::array<int64_t,4> dimensions){
    ggml_tensor t{};t.data=data;t.type=type;
    size_t stride=type==GGML_TYPE_I64?8:4;
    for(int i=0;i<4;++i){t.ne[i]=dimensions[size_t(i)];t.nb[i]=stride;stride*=size_t(t.ne[i]);}
    return t;
}
void params(ggml_tensor & t,ggml_custom_op_t fun=trusted,int tasks=1,uintptr_t vocabulary=248100){
    ggml_custom_op_params p{fun,tasks,reinterpret_cast<void *>(vocabulary)};
    std::memcpy(t.op_params,&p,sizeof(p));
}
}
int main(){
    using namespace ocr_dspark_custom_contract;
    size_t cases=0;
    auto check=[&](bool value){if(!value){std::printf("FAIL metadata case=%zu\n",cases);std::abort();}++cases;};
    check(registered_callback.load()==nullptr);check(!vector_enabled.load());
    check(!register_callback(nullptr));check(registered_callback.load()==nullptr);
    option("0");check(register_callback(trusted));check(registered_callback.load()==trusted);check(!vector_enabled.load());
    check(register_callback(trusted));check(!register_callback(impostor));check(!register_callback(nullptr));
    check(registered_callback.load()==trusted);check(!vector_enabled.load());
    option("1");check(register_callback(trusted));check(vector_enabled.load());
    option("0");check(!register_callback(impostor));check(vector_enabled.load());
    check(register_callback(trusted));check(!vector_enabled.load());
    for(const char * v:{"","true","2","01"," 1","1 "}){option(v);check(register_callback(trusted));check(!vector_enabled.load());}
    option("1");check(register_callback(trusted));check(vector_enabled.load());
    std::vector<float> values(114688);
    std::vector<int64_t> ids(114688);
    std::array<int32_t,3> output={0x13579bdf,0x2468ace,0x13579bdf};
    auto a=tensor(values.data(),GGML_TYPE_F32,{114688,1,1,1});
    auto b=tensor(ids.data(),GGML_TYPE_I64,{114688,1,1,1});
    auto op=tensor(&output[1],GGML_TYPE_I32,{1,1,1,1});op.op=GGML_OP_CUSTOM;op.src[0]=&a;op.src[1]=&b;params(op);
    auto valid=[&]{return writes_only_output(&op,&b);};
    auto refuse=[&]{check(!valid());};
    check(valid());check(!writes_only_output(nullptr,&b));check(!writes_only_output(&op,nullptr));
    std::strcpy(op.name,"dspark_compact_mapped_argmax");params(op,impostor);refuse();
    params(op,nullptr);refuse();params(op);
    for(int tasks:{0,2,6,-1}){params(op,trusted,tasks);refuse();}params(op);
    for(uintptr_t vocab:{uintptr_t(0),uintptr_t(1),uintptr_t(248099),uintptr_t(248101),UINTPTR_MAX}){params(op,trusted,1,vocab);refuse();}params(op);
    op.op=GGML_OP_ADD;refuse();op.op=GGML_OP_MAP_CUSTOM1;refuse();op.op=GGML_OP_CUSTOM;
    const auto original_op=op,original_a=a,original_b=b;
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_I64,GGML_TYPE_F16}){op.type=type;refuse();}op=original_op;
    for(auto type:{GGML_TYPE_I32,GGML_TYPE_I64,GGML_TYPE_F16}){a.type=type;refuse();}a=original_a;
    for(auto type:{GGML_TYPE_F32,GGML_TYPE_I32,GGML_TYPE_F16}){b.type=type;refuse();}b=original_b;
    for(int src:{0,1}){op.src[src]=nullptr;refuse();op=original_op;}
    for(int src=2;src<GGML_MAX_SRC;++src){op.src[src]=&a;refuse();op=original_op;}
    op.data=nullptr;refuse();op=original_op;a.data=nullptr;refuse();a=original_a;b.data=nullptr;refuse();b=original_b;
    for(int d=0;d<4;++d){op.ne[d]+=1;refuse();op=original_op;a.ne[d]+=1;refuse();a=original_a;b.ne[d]+=1;refuse();b=original_b;}
    // Unit-extent strides do not change storage and remain admitted, exactly
    // as ggml.c's real contiguous predicate. Only active row stride rejects.
    for(int d=0;d<4;++d){
        op.nb[d]+=4;check(valid());op=original_op;
        a.nb[d]+=4;check(valid()==(d!=0));a=original_a;
        b.nb[d]+=8;check(valid()==(d!=0));b=original_b;
    }
    op.view_src=&a;refuse();op=original_op;
    op.data=reinterpret_cast<char *>(&output[1])+1;refuse();op=original_op;
    a.data=reinterpret_cast<char *>(values.data())+1;refuse();a=original_a;
    b.data=reinterpret_cast<char *>(ids.data())+1;refuse();b=original_b;
    auto root=b;root.data=ids.data()+1;check(!writes_only_output(&op,&root));
    root=b;root.ne[0]+=1;root.nb[1]=root.nb[2]=root.nb[3]=size_t(root.ne[0])*8;check(!writes_only_output(&op,&root));
    for(void * alias:{static_cast<void *>(values.data()),static_cast<void *>(values.data()+114687),static_cast<void *>(ids.data()),static_cast<void *>(ids.data()+114687)}){op.data=alias;refuse();op=original_op;}
    op.data=reinterpret_cast<void *>(UINTPTR_MAX&~uintptr_t(3));refuse();op=original_op;
    a.data=reinterpret_cast<void *>(UINTPTR_MAX&~uintptr_t(3));refuse();a=original_a;
    b.data=reinterpret_cast<void *>(UINTPTR_MAX&~uintptr_t(7));refuse();b=original_b;
    check(!disjoint(reinterpret_cast<void *>(UINTPTR_MAX-3),8,values.data(),4));
    check(!disjoint(values.data(),4,reinterpret_cast<void *>(UINTPTR_MAX-3),8));
    check(disjoint(values.data(),0,values.data(),4));
    check(valid());
    // Metadata-only scalar TU always returns false from helper without writing.
    // Cover disabled, enabled-valid, and enabled-refused integration entry paths.
    const auto saved=output;
    option("0");check(register_callback(trusted));check(!vector_forward(&op));check(output==saved);
    option("1");check(register_callback(trusted));check(!vector_forward(&op));check(output==saved);
    params(op,impostor);check(!vector_forward(&op));check(output==saved);params(op);
    op.src[1]=nullptr;check(!vector_forward(&op));op=original_op;
    check(!vector_forward(nullptr));check(output==saved);
    // No metadata admission or registration may run either supplied callback.
    check(callback_calls==0);check(other_calls==0);
    std::printf("PASS mapped callback metadata/refusal cases=%zu; trusted identity, n_tasks/userdata, nulls, dtype/shape/stride, aliases, alignment, extra sources, overflow, host option parsing; no callback or RVV executed.\n",cases);
}
