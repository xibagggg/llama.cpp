#pragma once
// Graph-launch-owned tables for Paddle's fixed VISION RoPE policy. The table
// is rebuilt from current input values every call and never aliases op scratch.
#include "ggml.h"
#include "spacemit-opnode.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace paddle_vision_rope {
inline bool table_enabled() {
    const char * p=std::getenv("GGML_SPACEMIT_VISION_ROPE");
    const char * c=std::getenv("GGML_SPACEMIT_VISION_ROPE_CACHE");
    return p&&!std::strcmp(p,"1")&&(!c||std::strcmp(c,"0"));
}
inline bool policy(const ggml_tensor * op) {
    if (!op||op->op!=GGML_OP_ROPE||op->ne[0]!=72||op->op_params[1]!=36||
        op->op_params[2]!=GGML_ROPE_TYPE_VISION||op->op_params[15]!=0||op->src[2]||
        (op->type!=GGML_TYPE_F32&&op->type!=GGML_TYPE_F16)) return false;
    const float expected[4]={10000,1,0,1};
    if(std::memcmp(op->op_params+5,expected,sizeof(expected)))return false;
    for(int i=11;i<15;++i)if(op->op_params[i]!=18)return false;
    const auto * p=op->src[1];
    // The scheduler's cross-backend input clone may omit the INPUT flag.
    // Require an external leaf and reject every graph write to its range.
    return p&&p->op==GGML_OP_NONE&&p->data&&
        p->type==GGML_TYPE_I32&&ggml_is_contiguous(p)&&p->ne[0]==4*op->ne[2]&&
        p->ne[1]==1&&p->ne[2]==1&&p->ne[3]==1;
}
inline bool overlaps(const ggml_tensor * a,const ggml_tensor * b) {
    if(!a||!b||!a->data||!b->data)return false;
    const auto x=reinterpret_cast<uintptr_t>(a->data),y=reinterpret_cast<uintptr_t>(b->data);
    return x<=y ? y-x<ggml_nbytes(a) : x-y<ggml_nbytes(b);
}
struct entry {
    const ggml_tensor * positions;
    std::vector<float> values;
};
struct launch_tables {
    std::vector<entry> entries;
    std::vector<const float *> nodes;
    launch_tables(const ggml_cgraph * graph,const std::vector<spacemit_opnode> & ops) {
        const char * trace=std::getenv("GGML_SPACEMIT_VISION_ROPE_TRACE");
        if(trace&&!std::strcmp(trace,"1"))std::fprintf(stderr,"paddle-vision-rope: host enabled=%d nodes=%zu\n",int(table_enabled()),ops.size());
        if(!table_enabled())return;
        nodes.resize(ops.size(),nullptr);
        for(size_t i=0;i<ops.size();++i) {
            const auto * op=ops[i].node;
            if(trace&&!std::strcmp(trace,"1")&&op->op==GGML_OP_ROPE&&op->op_params[2]==GGML_ROPE_TYPE_VISION) {
                const auto * p=op->src[1];
                std::fprintf(stderr,"paddle-vision-rope: policy=%d D=%lld n_dims=%d pos_op=%d pos_flags=%d pos_ne=%lld,%lld,%lld,%lld\n",
                    int(policy(op)),(long long)op->ne[0],op->op_params[1],p?int(p->op):-1,p?p->flags:0,
                    p?(long long)p->ne[0]:0,p?(long long)p->ne[1]:0,p?(long long)p->ne[2]:0,p?(long long)p->ne[3]:0);
            }
            if(!ops[i].fused.empty()||!policy(op))continue;
            const auto * pos=op->src[1];
            bool immutable=true;
            // Reject graph writes, including aliased views/CPY targets. Raw
            // graph nodes include intermediate destinations hidden by fusion.
            for(int j=0;j<graph->n_nodes;++j) {
                const auto * dst=graph->nodes[j];
                if(!ggml_op_is_empty(dst->op)&&dst->op!=GGML_OP_NONE&&overlaps(pos,dst)) {
                    immutable=false;break;
                }
            }
            if(!immutable)continue;
            for(const auto & e:entries)if(e.positions==pos)nodes[i]=e.values.data();
            if(nodes[i])continue;
            const int64_t n=op->ne[2];
            if(n<=0||uint64_t(n)>SIZE_MAX/(72*sizeof(float)))continue;
            entries.push_back({pos,std::vector<float>(size_t(n)*72)});
            auto & table=entries.back().values;
            const auto * ids=static_cast<const int32_t*>(pos->data);
            const float theta_scale=powf(10000.0f,-2.0f/36);
            for(int64_t row=0;row<n;++row) {
                float theta_t=float(ids[row]),theta_h=float(ids[n+row]);
                for(int pair=0;pair<36;++pair) {
                    // Paddle uses two independent 18-pair sections. Match
                    // the original per-pair FP32 theta multiplication order.
                    if(pair==18)theta_h=float(ids[n+row]);
                    const float theta=pair<18?theta_t:theta_h;
                    table[size_t(row)*72+2*pair]=cosf(theta);
                    table[size_t(row)*72+2*pair+1]=sinf(theta);
                    theta_t*=theta_scale;theta_h*=theta_scale;
                }
            }
            nodes[i]=table.data();
        }
        if(trace&&!std::strcmp(trace,"1")&&!entries.empty()) {
            size_t hits=0;for(const auto * p:nodes)hits+=p!=nullptr;
            std::fprintf(stderr,"paddle-vision-rope: tables=%zu consumers=%zu (fresh launch)\n",entries.size(),hits);
        }
    }
    const float * get(size_t i) const {return nodes.empty()?nullptr:nodes[i];}
};
} // namespace paddle_vision_rope
