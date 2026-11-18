#include "ggml.h"
#include "../ocr-gdn-snapshot-store.h"
#include <array>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace spec=ocr_gdn_snapshot_store;
static void tensor(ggml_tensor & t,std::array<int64_t,4> ne,void * data) {
    t={};size_t stride=4;
    for(int d=0;d<4;++d){t.ne[d]=ne[d];t.nb[d]=stride;stride*=size_t(ne[d]);}
    t.data=data;
}
struct fixture {
    std::vector<float> inputs=std::vector<float>(2*128*16*16+128*16*16+2*16*16+128*128*16);
    std::vector<float> output=std::vector<float>((spec::attn_bytes+16*spec::state_bytes)/4);
    std::vector<float> cache=std::vector<float>(3*16*spec::state_elems);
    ggml_tensor q,k,v,gate,beta,state,g,from,cache_root,dest,store,attn,consumer,other;
    std::vector<ggml_tensor *> nodes;
    fixture(){reset();}
    void reset() {
        float * p=inputs.data();
        tensor(q,{128,16,16,1},p);p+=128*16*16;
        tensor(k,{128,16,16,1},p);p+=128*16*16;
        tensor(v,{128,16,16,1},p);p+=128*16*16;
        tensor(gate,{1,16,16,1},p);p+=16*16;
        tensor(beta,{1,16,16,1},p);p+=16*16;
        tensor(state,{128,128,16,1},p);
        tensor(g,{int64_t(output.size()),1,1,1},output.data());
        g.op=GGML_OP_GATED_DELTA_NET;g.flags=GGML_TENSOR_FLAG_COMPUTE;g.op_params[0]=16;
        ggml_tensor * source[]={&q,&k,&v,&gate,&beta,&state};
        for(int i=0;i<6;++i)g.src[i]=source[i];
        tensor(from,{spec::state_elems,1,16,1},output.data()+spec::attn_bytes/4);
        from.op=GGML_OP_VIEW;from.view_src=&g;from.view_offs=spec::attn_bytes;
        tensor(cache_root,{spec::state_elems,48,1,1},cache.data());
        tensor(dest,{spec::state_elems,1,16,1},cache.data()+spec::state_elems);
        dest.nb[2]=3*spec::state_bytes;dest.nb[3]=16*dest.nb[2];
        dest.view_src=&cache_root;dest.view_offs=spec::state_bytes;
        store=dest;store.op=GGML_OP_CPY;store.flags=GGML_TENSOR_FLAG_COMPUTE;
        store.src[0]=&from;store.src[1]=&dest;
        tensor(attn,{128,16,16,1},output.data());attn.op=GGML_OP_VIEW;attn.view_src=&g;
        tensor(consumer,{1,1,1,1},inputs.data());consumer.op=GGML_OP_ADD;
        consumer.flags=GGML_TENSOR_FLAG_COMPUTE;consumer.src[0]=&attn;
        tensor(other,{1,1,1,1},inputs.data());other.op=GGML_OP_ADD;other.flags=GGML_TENSOR_FLAG_COMPUTE;
        nodes={&g,&from,&attn,&store,&consumer};
    }
    bool match() {
        ggml_cgraph graph{int(nodes.size()),nodes.data()};
        return spec::match_store(&graph,0,[](ggml_tensor * t){return t->flags&GGML_TENSOR_FLAG_COMPUTE;})==&store;
    }
};

int main(){
    fixture f;unsigned tests=0,failures=0;
    auto check=[&](const char * name,bool expected,const std::function<void()> & change){
        f.reset();change();bool actual=f.match();++tests;
        if(actual!=expected){++failures;std::fprintf(stderr,"FAIL %s actual=%d expected=%d\n",name,int(actual),int(expected));}
    };
    check("normal K16T16",true,[]{});
    check("groupedQK4",true,[&]{f.q.ne[1]=4;f.k.ne[1]=4;});
    check("K1",false,[&]{f.g.op_params[0]=1;});
    check("K15",false,[&]{f.g.op_params[0]=15;});
    check("T15",false,[&]{f.v.ne[2]=15;});
    check("T17",false,[&]{f.v.ne[2]=17;});
    check("two sequences",false,[&]{f.v.ne[3]=2;});
    check("KDA",false,[&]{f.gate.ne[0]=128;});
    check("invalid Q heads",false,[&]{f.q.ne[1]=3;});
    check("missing input",false,[&]{f.g.src[4]=nullptr;});
    check("state padded",false,[&]{f.state.nb[1]+=16;});
    check("non F32",false,[&]{f.beta.type=GGML_TYPE_F16;});
    check("zero dimension",false,[&]{f.beta.ne[1]=0;});
    check("output as view",false,[&]{f.g.view_src=&f.other;});
    check("whole output exposed",false,[&]{f.g.flags|=GGML_TENSOR_FLAG_OUTPUT;});
    check("state output exposed",false,[&]{f.from.flags|=GGML_TENSOR_FLAG_OUTPUT;});
    check("attention output exposed",true,[&]{f.attn.flags|=GGML_TENSOR_FLAG_OUTPUT;});
    check("whole output reader",false,[&]{f.consumer.src[0]=&f.g;});
    check("extra state reader",false,[&]{f.consumer.src[1]=&f.from;});
    check("extra attention reader",true,[&]{f.consumer.src[1]=&f.attn;});
    check("duplicate state CPY",false,[&]{f.nodes.push_back(&f.store);});
    check("intervening computation",false,[&]{f.nodes.insert(f.nodes.begin()+1,&f.other);});
    check("source offset",false,[&]{f.from.view_offs+=4;});
    check("source pointer",false,[&]{f.from.data=static_cast<char *>(f.from.data)+4;});
    check("source wrong root",false,[&]{f.from.view_src=&f.other;});
    check("source wrong stride",false,[&]{f.from.nb[2]+=4;});
    check("source wrong shape",false,[&]{f.from.ne[0]-=1;});
    check("CPY shape mismatch",false,[&]{f.store.ne[2]-=1;});
    check("destination shape mismatch",false,[&]{f.dest.ne[2]-=1;});
    check("destination pointer mismatch",false,[&]{f.dest.data=static_cast<char *>(f.dest.data)+4;});
    check("destination stride mismatch",false,[&]{f.dest.nb[2]+=4;});
    check("destination overlapping slots",false,[&]{f.dest.nb[2]=f.store.nb[2]=spec::state_bytes-4;});
    check("destination unaligned",false,[&]{f.dest.data=f.store.data=static_cast<char *>(f.dest.data)+1;});
    check("alias state input",false,[&]{f.state.data=f.dest.data;});
    check("alias other input",false,[&]{f.q.data=f.dest.data;});
    check("alias output",false,[&]{f.dest.data=f.store.data=f.g.data;});
    check("alias output tail",false,[&]{f.dest.data=f.store.data=static_cast<char *>(f.g.data)+spec::attn_bytes;});
    check("alias source and G output",false,[&]{f.state.data=f.g.data;});
    check("missing cache view",false,[&]{f.dest.view_src=nullptr;});
    check("cache root too short",false,[&]{f.cache_root.ne[1]=2;});
    check("cache offset wrong",false,[&]{f.dest.view_offs+=4;});
    check("overflow stride",false,[&]{f.dest.nb[2]=f.store.nb[2]=size_t(-4);});
    check("nested snapshot output",false,[&]{f.other=f.from;f.other.view_src=&f.from;f.other.flags=GGML_TENSOR_FLAG_OUTPUT;f.nodes.push_back(&f.other);});
    check("nested attention output",true,[&]{f.other=f.attn;f.other.view_src=&f.attn;f.other.flags=GGML_TENSOR_FLAG_OUTPUT;f.nodes.push_back(&f.other);});
    // Exercise the exact production destination helper. Each head writes all
    // 16 reverse slots. Compare complete cache byte-for-byte, including the
    // initial state row, unused sequence rows and all padding.
    unsigned copy_cases=0;
    uint32_t random=0x736e6170;
    for(size_t mem_size:{size_t(1),size_t(2),size_t(3),size_t(5)}) {
        for(size_t row:{size_t(0),mem_size-1}) {
            std::vector<uint32_t> actual(16*mem_size*spec::state_elems+64,0xcafe5921u),expected=actual;
            std::vector<uint32_t> temp(16*spec::state_elems),head(128*128);
            for(int64_t h=0;h<16;++h)for(int64_t token=0;token<16;++token) {
                for(auto & x:head){random=random*1664525u+1013904223u;x=random;}
                const int64_t slot=15-token;
                std::memcpy(temp.data()+slot*spec::state_elems+h*128*128,head.data(),head.size()*4);
                std::memcpy(spec::snapshot_pointer(actual.data()+32+row*spec::state_elems,
                    mem_size*spec::state_bytes,slot,h),head.data(),head.size()*4);
            }
            for(size_t slot=0;slot<16;++slot)std::memcpy(expected.data()+32+(slot*mem_size+row)*spec::state_elems,
                temp.data()+slot*spec::state_elems,spec::state_bytes);
            ++copy_cases;
            if(actual!=expected){++failures;std::fprintf(stderr,"FAIL snapshot copy mem=%zu row=%zu\n",mem_size,row);}
        }
    }
    std::printf("{\"status\":\"%s\",\"admission_cases\":%u,\"copy_cases\":%u,\"failures\":%u}\n",
        failures?"FAIL":"PASS",tests,copy_cases,failures);
    return failures?1:0;
}
