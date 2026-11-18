// Standalone, frozen RVV arithmetic. Parent owns every device runtime slot.
#include <riscv_vector.h>
#include <spert.hpp>
#include "ggml.h"
#include "ocr-gdn-tcm.h"
#include "ocr-gdn-rollback-tcm.h"
#include "ocr-gdn-snapshot-store.h"
#include "real-tcm-standalone.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <thread>
#include <unistd.h>
#include <vector>
#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif
namespace ggml::spacemit {struct context {unsigned ith,nth;spert::SharedBufferView shared;bool check_tcm_view;};}
#include "gdn-reference.h"
#include "gdn-candidate.h"
#include "gdn-copy-reference.h"

namespace {
constexpr size_t guard=128;
constexpr unsigned char canary=0x6d;
namespace spec=ocr_gdn_snapshot_store;
struct storage {
    ggml_tensor tensor{};
    std::vector<unsigned char> bytes;
    storage(std::array<int64_t,4> ne,bool padded=false){
        size_t stride=4;
        for(unsigned d=0;d<4;++d){tensor.ne[d]=ne[d];tensor.nb[d]=stride;stride*=size_t(ne[d]);
            if(padded&&d==0)stride+=16;if(padded&&d==1)stride+=64;}
        tensor.type=GGML_TYPE_F32;
        bytes.assign(2*guard+ggml_nbytes(&tensor),canary);tensor.data=bytes.data()+guard;
    }
    void fill(uint32_t & seed,float scale,float offset=0.f){
        for(int64_t s=0;s<tensor.ne[3];++s)for(int64_t t=0;t<tensor.ne[2];++t)
            for(int64_t h=0;h<tensor.ne[1];++h)for(int64_t d=0;d<tensor.ne[0];++d){
                seed=seed*1664525u+1013904223u;
                const float value=float(int32_t(seed>>8)-0x7fffff)/float(0x800000)*scale+offset;
                std::memcpy(static_cast<char *>(tensor.data)+s*tensor.nb[3]+t*tensor.nb[2]+h*tensor.nb[1]+d*4,&value,4);
            }
    }
    size_t guards()const {size_t bad=0;for(size_t i=0;i<guard;++i)bad+=bytes[i]!=canary||bytes[bytes.size()-1-i]!=canary;return bad;}
};
void install(ggml_tensor & g,storage & q,storage & k,storage & v,storage & gate,storage & beta,storage & state){
    g.op=GGML_OP_GATED_DELTA_NET;g.flags=GGML_TENSOR_FLAG_COMPUTE;
    int32_t slots=16;std::memcpy(g.op_params,&slots,4);
    ggml_tensor * inputs[]={&q.tensor,&k.tensor,&v.tensor,&gate.tensor,&beta.tensor,&state.tensor};
    for(unsigned i=0;i<6;++i)g.src[i]=inputs[i];
}
struct binding {
    ggml_tensor source{},destination{},copy{};
    binding(ggml_tensor & g,ggml_tensor & cache,size_t memory,size_t row){
        source.type=destination.type=GGML_TYPE_F32;
        const int64_t ne[4]={spec::state_elems,1,16,1};
        size_t stride=4;
        for(unsigned d=0;d<4;++d){source.ne[d]=destination.ne[d]=ne[d];source.nb[d]=destination.nb[d]=stride;stride*=size_t(ne[d]);}
        source.op=destination.op=GGML_OP_VIEW;
        source.view_src=&g;source.view_offs=spec::attn_bytes;
        source.data=static_cast<char *>(g.data)+spec::attn_bytes;
        destination.view_src=&cache;destination.view_offs=row*spec::state_bytes;
        destination.data=static_cast<char *>(cache.data)+destination.view_offs;
        destination.nb[2]=memory*spec::state_bytes;destination.nb[3]=16*destination.nb[2];
        copy=destination;copy.op=GGML_OP_CPY;copy.flags=GGML_TENSOR_FLAG_COMPUTE;
        copy.src[0]=&source;copy.src[1]=&destination;
    }
};
}

int main(int argc,char ** argv){
    const int reps=argc>1?std::max(1,std::atoi(argv[1])):3;
    const int hold=argc>2?std::clamp(std::atoi(argv[2]),0,3000):1500;
    // No caller option can turn this into AI8. The actual worker CPU set is
    // additionally validated after every launch.
    spert::StreamConfig config;config.n_cores=6;config.core_ids={8,9,10,11,12,13};
    spert::Stream stream(config);if(!stream.valid()||stream.core_count()!=6)return 2;
    standalone_real_tcm::host platform;ocr_gdn_tcm::prepare();
    ocr_gdn_rollback_tcm::prepare_on_host(6,platform.bytes,true);
    unsigned cases=0,failures=0;
    for(int qheads:{4,16})for(bool padded:{false,true})for(size_t memory:{size_t(1),size_t(3)})for(uint32_t initial:{1u,0x9811410u}){
        uint32_t seed=initial;
        storage q({128,qheads,16,1},padded),k({128,qheads,16,1},padded),v({128,16,16,1},padded);
        storage gate({1,16,16,1},padded),beta({1,16,16,1},padded),state({128,128,16,1});
        storage reference({128*16,16+16*128,1,1}),candidate({128*16,16+16*128,1,1});
        storage cache_ref({spec::state_elems,int64_t(16*memory),1,1}),cache_cand({spec::state_elems,int64_t(16*memory),1,1});
        q.fill(seed,.05f);k.fill(seed,.05f);v.fill(seed,.1f);gate.fill(seed,.02f,-.05f);beta.fill(seed,.02f,.2f);state.fill(seed,.01f);
        install(reference.tensor,q,k,v,gate,beta,state);install(candidate.tensor,q,k,v,gate,beta,state);
        binding copy_ref(reference.tensor,cache_ref.tensor,memory,memory-1),copy_cand(candidate.tensor,cache_cand.tensor,memory,memory-1);
        if(!spec::storage_eligible(&candidate.tensor,&copy_cand.copy))return 4;
        // Reuse exactly the same pointers for three requests with rewritten
        // inputs; no preparation cache or pointer-identity assumption allowed.
        for(unsigned request=0;request<3;++request){
            if(request){q.fill(seed,.05f);k.fill(seed,.05f);v.fill(seed,.1f);state.fill(seed,.01f);}
            std::fill(reference.bytes.begin(),reference.bytes.end(),canary);
            std::fill(candidate.bytes.begin(),candidate.bytes.end(),canary);
            std::fill(cache_ref.bytes.begin(),cache_ref.bytes.end(),canary);
            std::fill(cache_cand.bytes.begin(),cache_cand.bytes.end(),canary);
            const auto qb=q.bytes,kb=k.bytes,vb=v.bytes,gb=gate.bytes,bb=beta.bytes,sb=state.bytes;
            std::array<double,4> timings{};std::array<int,6> cpus{};
            standalone_real_tcm::policy::launch_state ownership;
            auto launch=stream.launch(spert::Grid{6},[&](spert::Context * runtime){
                const unsigned lane=runtime->program_id(0);
                standalone_real_tcm::worker owned(*runtime,platform,ownership,lane);
                if(!owned.ready())return;const auto full=owned.full();cpus[lane]=sched_getcpu();
                ggml::spacemit::context ctx{lane,6,full,true};
                if(cases==0&&lane==0)std::this_thread::sleep_for(std::chrono::milliseconds(hold));
                if(runtime->sync()!=spert::Status::Ok)return;
                gdn_reference::forward_gated_delta_net(ctx,&reference.tensor);
                if(runtime->sync()!=spert::Status::Ok)return;
                gdn_copy::forward_cpy_f32_row_strided(ctx,&copy_ref.copy);
                if(runtime->sync()!=spert::Status::Ok)return;
                gdn_candidate::forward_gated_delta_net_snapshot_store(ctx,&candidate.tensor,&copy_cand.copy);
                if(runtime->sync()!=spert::Status::Ok)return;
                // Keep the graph's skipped-CPY boundary in the candidate path.
                if(runtime->sync()!=spert::Status::Ok)return;
                if(request==0&&initial==1){
                    for(unsigned sample=0;sample<4;++sample){
                        const bool direct=sample==1||sample==2;
                        const auto began=std::chrono::steady_clock::now();
                        for(int rep=0;rep<reps;++rep){
                            if(direct)gdn_candidate::forward_gated_delta_net_snapshot_store(ctx,&candidate.tensor,&copy_cand.copy);
                            else gdn_reference::forward_gated_delta_net(ctx,&reference.tensor);
                            if(runtime->sync()!=spert::Status::Ok)return;
                            if(!direct)gdn_copy::forward_cpy_f32_row_strided(ctx,&copy_ref.copy);
                            if(runtime->sync()!=spert::Status::Ok)return;
                        }
                        if(lane==0)timings[sample]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count()/reps;
                        if(runtime->sync()!=spert::Status::Ok)return;
                    }
                }
                owned.release();
            });
            if(!launch.valid()||launch.sync()!=spert::Status::Ok)return 3;
            size_t attention_diffs=0,cache_diffs=0,snapshot_writes=0,nonfinite=0,core_errors=0;
            const auto * rp=static_cast<const unsigned char *>(reference.tensor.data);
            const auto * cp=static_cast<const unsigned char *>(candidate.tensor.data);
            for(size_t i=0;i<spec::attn_bytes;++i)attention_diffs+=rp[i]!=cp[i];
            for(size_t i=0;i<cache_ref.bytes.size();++i)cache_diffs+=cache_ref.bytes[i]!=cache_cand.bytes[i];
            for(size_t i=spec::attn_bytes;i<ggml_nbytes(&candidate.tensor);++i)snapshot_writes+=cp[i]!=canary;
            for(size_t i=0;i<spec::attn_bytes/4;++i)nonfinite+=!std::isfinite(static_cast<const float *>(candidate.tensor.data)[i]);
            for(size_t slot=0;slot<16;++slot){
                const auto * snapshot=spec::snapshot_pointer(copy_cand.copy.data,copy_cand.copy.nb[2],int64_t(slot),0);
                for(size_t i=0;i<size_t(spec::state_elems);++i)nonfinite+=!std::isfinite(snapshot[i]);
            }
            for(int cpu:cpus)core_errors+=cpu<8||cpu>13;
            const size_t guard_errors=reference.guards()+candidate.guards()+cache_ref.guards()+cache_cand.guards();
            const unsigned owner_errors=standalone_real_tcm::errors(ownership,6);
            const bool inputs_changed=qb!=q.bytes||kb!=k.bytes||vb!=v.bytes||gb!=gate.bytes||bb!=beta.bytes||sb!=state.bytes;
            const bool bad=attention_diffs||cache_diffs||snapshot_writes||nonfinite||core_errors||guard_errors||owner_errors||inputs_changed;
            ++cases;failures+=bad;
            std::printf("{\"q_heads\":%d,\"padded\":%s,\"mem_size\":%zu,\"row\":%zu,\"seed\":%u,\"request\":%u,\"attention_byte_differences\":%zu,\"cache_byte_differences\":%zu,\"temporary_snapshot_byte_changes\":%zu,\"nonfinite\":%zu,\"guard_errors\":%zu,\"core_errors\":%zu,\"input_changed\":%s,\"timing_ms\":[%.6f,%.6f,%.6f,%.6f],\"worker_cpus\":[",
                qheads,padded?"true":"false",memory,memory-1,initial,request,attention_diffs,cache_diffs,snapshot_writes,nonfinite,guard_errors,core_errors,inputs_changed?"true":"false",timings[0],timings[1],timings[2],timings[3]);
            for(unsigned lane=0;lane<6;++lane)std::printf("%s%d",lane?",":"",cpus[lane]);
            std::printf("],\"worker_tids\":[");for(unsigned lane=0;lane<6;++lane)std::printf("%s%d",lane?",":"",ownership.lanes[lane].tid);
            std::printf("],\"tcm_block_ids\":[");for(unsigned lane=0;lane<6;++lane)std::printf("%s%d",lane?",":"",ownership.lanes[lane].id);
            std::printf("],\"tcm_ownership_errors\":%u,\"process_pid\":%d,\"status\":\"%s\"}\n",owner_errors,int(getpid()),bad?"FAIL":"PASS");
            std::fflush(stdout);
        }
    }
    std::fprintf(stderr,"GDN direct snapshot cases=%u failures=%u\n",cases,failures);
    return failures?1:0;
}
