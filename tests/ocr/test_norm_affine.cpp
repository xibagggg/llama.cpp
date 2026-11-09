#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-spacemit.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// Compare OFF/ON output files, including a downstream residual consumer,
// exposed intermediates, a view consumer, in-place output and matrix scales.
struct fixture { int width, rows, guard; bool residual; };
static unsigned mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; return x * 0x846ca68b;
}
int main(int argc, char ** argv) {
    if (argc != 2) return 1;
    const fixture cases[] = {
        {1,1,0,false}, {31,7,0,false}, {129,7,0,true},
        {1152,252,0,true}, {1152,1008,0,true}, {4304,7,0,false},
        {4352,3,0,true}, {129,7,1,true}, {129,7,2,true},
        {129,7,3,true}, {129,7,4,true}, {129,7,5,true},
    };
    std::ofstream file(argv[1], std::ios::binary);
    auto backend = ggml_backend_spacemit_init();
    if (!file || !backend) return 2;
    for (size_t ci=0;ci<sizeof(cases)/sizeof(cases[0]);++ci) {
        const auto c=cases[ci];
        ggml_init_params params{};params.mem_size=2*1024*1024;params.no_alloc=true;
        auto ctx=ggml_init(params);
        auto xb=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,c.width,c.rows);
        auto x=xb;
        auto r=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,c.width,c.rows);
        auto w=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,c.width,c.guard==4?c.rows:1);
        auto b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,c.width);
        for(auto t:{xb,r,w,b}) {ggml_set_input(t);ggml_set_output(t);}
        auto residual=c.residual?ggml_add(ctx,x,r):x;
        // This fixture reads the residual after the entire graph, including
        // the extra view branch. Keep it live past its ordinary last consumer.
        if(c.residual)ggml_set_output(residual);
        auto norm=ggml_norm(ctx,residual,1.e-5f);
        auto scale=ggml_mul(ctx,norm,w);
        auto y=c.guard==5?ggml_add_inplace(ctx,scale,b):ggml_add(ctx,scale,b);
        if(c.guard==1)ggml_set_output(norm);
        if(c.guard==2)ggml_set_output(scale);
        auto result=c.residual?ggml_add(ctx,y,residual):y;
        ggml_tensor * extra=nullptr;
        if(c.guard==3) {
            auto view=ggml_view_2d(ctx,norm,c.width,c.rows,norm->nb[1],0);
            extra=ggml_scale(ctx,view,0.25f);
            ggml_set_output(extra);
        }
        ggml_set_output(result);
        auto graph=ggml_new_graph_custom(ctx,128,false);
        ggml_build_forward_expand(graph,result);
        if(extra)ggml_build_forward_expand(graph,extra);
        if(!ggml_backend_supports_op(backend,norm))return 3;
        auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if(!ggml_gallocr_alloc_graph(alloc,graph))return 4;
        std::vector<float> xv(ggml_nelements(xb)),rv(ggml_nelements(r)),wv(ggml_nelements(w)),bv(c.width);
        for(size_t i=0;i<xv.size();++i)xv[i]=(int(mix(i+31)%20001)-10000)*.0001f;
        for(size_t i=0;i<rv.size();++i)rv[i]=(int(mix(i+71)%2001)-1000)*.00003f;
        for(size_t i=0;i<wv.size();++i)wv[i]=.75f+(mix(i+3)%1000)*.0005f;
        for(int i=0;i<c.width;++i)bv[i]=(int(mix(i+7)%1000)-500)*.0001f;
        ggml_backend_tensor_set(xb,xv.data(),0,xv.size()*4);
        if(c.residual)ggml_backend_tensor_set(r,rv.data(),0,rv.size()*4);
        ggml_backend_tensor_set(w,wv.data(),0,wv.size()*4);
        ggml_backend_tensor_set(b,bv.data(),0,bv.size()*4);
        std::vector<float> values(ggml_nelements(result)),first;
        double elapsed=0.;
        for(int repeat=0;repeat<4;++repeat) {
            const auto begin=std::chrono::steady_clock::now();
            if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 5;
            ggml_backend_synchronize(backend);
            const auto end=std::chrono::steady_clock::now();
            ggml_backend_tensor_get(result,values.data(),0,values.size()*4);
            for(float value:values)if(!std::isfinite(value))return 6;
            if(!repeat)first=values;
            else {
                if(std::memcmp(first.data(),values.data(),values.size()*4))return 7;
                elapsed+=std::chrono::duration<double,std::milli>(end-begin).count();
            }
        }
        // A new input must invalidate all graph-derived data, including a
        // cached fusion plan. Restore A afterward and demand an exact replay.
        auto changed=xv;
        for(size_t i=0;i<changed.size();++i)
            changed[i]+=(int(mix(i+113)%1023)-511)*.0001f;
        ggml_backend_tensor_set(xb,changed.data(),0,changed.size()*4);
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 8;
        ggml_backend_tensor_get(result,values.data(),0,values.size()*4);
        for(float value:values)if(!std::isfinite(value))return 9;
        file.write(reinterpret_cast<const char*>(values.data()),values.size()*4);
        ggml_backend_tensor_set(xb,xv.data(),0,xv.size()*4);
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 10;
        ggml_backend_tensor_get(result,values.data(),0,values.size()*4);
        if(std::memcmp(first.data(),values.data(),values.size()*4))return 11;
        file.write(reinterpret_cast<const char*>(values.data()),values.size()*4);
        if(c.residual) {
            std::vector<float> saved(rv.size());
            ggml_backend_tensor_get(residual,saved.data(),0,saved.size()*4);
            for(size_t i=0;i<saved.size();++i)
                if(saved[i]!=xv[i]+rv[i])return 12;
            file.write(reinterpret_cast<const char*>(saved.data()),saved.size()*4);
        }
        for(auto t:{xb,r,w,b}) {
            if(t==r&&!c.residual)continue;
            const auto &expected=t==xb?xv:t==r?rv:t==w?wv:bv;
            std::vector<float> check(expected.size());
            ggml_backend_tensor_get(t,check.data(),0,check.size()*4);
            if(std::memcmp(check.data(),expected.data(),check.size()*4))return 13;
        }
        for(auto t:{c.guard==1?norm:nullptr,c.guard==2?scale:nullptr,extra})if(t) {
            std::vector<float> check(ggml_nelements(t));
            ggml_backend_tensor_get(t,check.data(),0,check.size()*4);
            file.write(reinterpret_cast<const char*>(check.data()),check.size()*4);
        }
        std::printf("{\"case\":%zu,\"width\":%d,\"rows\":%d,\"guard\":%d,\"residual\":%s,\"mean_ms\":%.6f,\"replay_exact\":true}\n",
            ci,c.width,c.rows,c.guard,c.residual?"true":"false",elapsed/3);
        std::fflush(stdout);
        ggml_gallocr_free(alloc);ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return file?0:14;
}
