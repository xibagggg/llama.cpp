#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// Independent negative-admission regression, not a benchmark. One gate per
// fresh process: all static backend options are fixed before backend creation.
namespace {
constexpr int D=72,H=16,STEPS=2;
constexpr uint64_t MAGIC=UINT64_C(0x5350334558434c33),VERSION=3;
constexpr uint64_t FNV=UINT64_C(14695981039346656037);
const char*const gates[]={"mask","softcap","sinks","compact0","direct0",
    "softmax0","softmax2","fullkv1","k_inf","v_inf","q_nan"};
uint64_t hash(uint64_t h,const void*p,size_t n) {
    const auto*b=static_cast<const uint8_t*>(p);
    for(size_t i=0;i<n;++i)h=(h^b[i])*UINT64_C(1099511628211);
    return h;
}
uint32_t rnd(uint32_t&s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
bool same(const char*a,const char*b){return !std::strcmp(a,b);}
bool io(FILE*f,bool write,void*p,size_t n){return write?std::fwrite(p,1,n,f)==n:std::fread(p,1,n,f)==n;}
std::vector<uint8_t> physical(const ggml_tensor*t,const ggml_tensor*b,const void*src,size_t elem) {
    std::vector<uint8_t> out(ggml_nbytes(b),0xa5);
    for(int64_t h=0;h<t->ne[2];++h)for(int64_t n=0;n<t->ne[1];++n)for(int64_t d=0;d<t->ne[0];++d){
        const size_t off=h*t->nb[2]+n*t->nb[1]+d*t->nb[0],i=(h*t->ne[1]+n)*t->ne[0]+d;
        if(off+elem>out.size())std::abort();
        std::memcpy(out.data()+off,static_cast<const uint8_t*>(src)+i*elem,elem);
    }
    return out;
}
bool unchanged(const ggml_tensor*t,const std::vector<uint8_t>&wanted) {
    std::vector<uint8_t> got(wanted.size());ggml_backend_tensor_get(t,got.data(),0,got.size());return got==wanted;
}
void set(const char*name,const char*value){if(setenv(name,value,1))std::abort();}
bool configure(const char*gate,const char*mode) {
    const std::pair<const char*,const char*> flags[]={
        {"GGML_SPACEMIT_BACKEND","1"},{"GGML_SPACEMIT_WORKERS","6"},{"GGML_SPACEMIT_ALLOW_EIGHT_WORKERS","0"},
        {"SPACEMIT_PERFER_CORE_ID","8,9,10,11,12,13"},{"GGML_SPACEMIT_SHARED_MODE","PINNED_TCM"},
        {"GGML_SPACEMIT_STRICT_AI_CORES","1"},{"SPERT_SHARED_MODE","TCM"},{"SPERT_BACKEND","spacemit-k3"},
        {"SPERT_RT_PRIO","0"},{"GGML_SPACEMIT_OCR_FA_PACK","1"},{"GGML_SPACEMIT_OCR_FA_COMPUTE","3"},
        {"GGML_SPACEMIT_OCR_PV_IME2","2"},{"GGML_SPACEMIT_OCR_FA_DYNAMIC_ROWS","512"},
        {"GGML_SPACEMIT_OCR_FA_FULL_KV","2"},{"GGML_SPACEMIT_OCR_FA_DIRECT","0"},
        {"GGML_SPACEMIT_OCR_FA_LAZY_INIT","0"},{"GGML_SPACEMIT_OCR_FA_TRUST_BOUNDS","0"},
        {"GGML_SPACEMIT_OCR_FA_HEAD_K_TCM","0"},{"GGML_SPACEMIT_OCR_FA_ALL_K_TCM","0"},
        {"GGML_SPACEMIT_OCR_FA_BATCH_CHECK","0"},{"GGML_SPACEMIT_OCR_FA_SEPARATE_BARRIER","0"},
        {"GGML_SPACEMIT_OCR_QK_PIPELINE","0"},{"GGML_SPACEMIT_OCR_PV_PIPELINE","0"},
        {"GGML_SPACEMIT_OCR_PV_TAIL_PIPELINE","0"},{"GGML_SPACEMIT_OCR_STRIDED_CAST","0"},
        {"GGML_SPACEMIT_PADDLE_CAST_KV_FUSE","0"},
        {"GGML_SPACEMIT_OCR_FA_FULL_PACK","1"},{"GGML_SPACEMIT_OCR_FA_LEAN_PREPARE","1"},
        {"GGML_SPACEMIT_OCR_FA_FUSED_QUERY","1"},{"GGML_SPACEMIT_OCR_FA_HALF_PACK","1"},
        {"GGML_SPACEMIT_OCR_FA_SOFTMAX_PACK","1"},{"GGML_SPACEMIT_OCR_FA_COMPACT","1"},
        {"GGML_SPACEMIT_OCR_FA_HEAD_K_TCM_2PASS","1"},{"GGML_SPACEMIT_OCR_FA_Q32","1"},
        {"GGML_SPACEMIT_OCR_FA_FIXED_INDEX","1"},{"GGML_SPACEMIT_OCR_FA_P_STORE64","1"},
        {"GGML_SPACEMIT_OCR_FA_QK_SCATTER","1"},
    };
    for(const auto&f:flags)set(f.first,f.second);
    set("GGML_SPACEMIT_OCR_FA_SERIAL_P2",mode);
    if(same(gate,"compact0")){set("GGML_SPACEMIT_OCR_FA_COMPACT","0");set("GGML_SPACEMIT_OCR_FA_Q32","0");}
    if(same(gate,"direct0"))set("GGML_SPACEMIT_OCR_FA_FULL_PACK","0");
    if(same(gate,"softmax0"))set("GGML_SPACEMIT_OCR_FA_SOFTMAX_PACK","0");
    if(same(gate,"softmax2"))set("GGML_SPACEMIT_OCR_FA_SOFTMAX_PACK","2");
    if(same(gate,"fullkv1"))set("GGML_SPACEMIT_OCR_FA_FULL_KV","1");
    const char*path=std::getenv("GGML_SPACEMIT_OCR_FA_PHASES");struct stat st{};
    return path&&*path&&!stat(path,&st)&&S_ISDIR(st.st_mode);
}
} // namespace

int main(int argc,char**argv) {
    if(argc!=7||(!same(argv[1],"--write-reference")&&!same(argv[1],"--compare-reference"))||
       !same(argv[3],"--gate")||!same(argv[5],"--serial")){
        std::fprintf(stderr,"Usage: %s --write-reference NEW_FILE|--compare-reference FILE --gate NAME --serial 0|1|2\n",argv[0]);return 2;
    }
    const bool write=same(argv[1],"--write-reference");const char*gate=argv[4],*mode=argv[6];
    const auto at=std::find_if(std::begin(gates),std::end(gates),[&](const char*p){return same(p,gate);});
    if(at==std::end(gates)||(!same(mode,"0")&&!same(mode,"1")&&!same(mode,"2"))||
       (write&&!same(mode,"0"))||(!write&&same(mode,"0")))return 2;
    const uint64_t gate_id=uint64_t(at-std::begin(gates));
    if(!configure(gate,mode)){std::fprintf(stderr,"Require existing fresh GGML_SPACEMIT_OCR_FA_PHASES directory\n");return 2;}
    const int fd=write?open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0644):open(argv[2],O_RDONLY);
    if(fd<0){std::perror(argv[2]);return 2;}
    std::unique_ptr<FILE,decltype(&std::fclose)> file(fdopen(fd,write?"wb":"rb"),&std::fclose);
    if(!file){close(fd);return 2;}
    std::array<uint64_t,9> header={MAGIC,VERSION,0x01020304,sizeof(float),sizeof(ggml_fp16_t),D,H,gate_id,8},want=header;
    if(!io(file.get(),write,header.data(),sizeof(header))||header!=want)return 2;
    ggml_backend_load_all();const auto dev=ggml_backend_dev_by_name("SPACEMIT0");if(!dev)return 3;
    const auto backend=ggml_backend_dev_init(dev,nullptr);if(!backend)return 3;
    bool ok=true;uint64_t records=0,rolling=FNV;
    const bool nonfinite=same(gate,"k_inf")||same(gate,"v_inf")||same(gate,"q_nan");
    for(int n:{848,1500})for(int layout:{0,1}){
        auto*ctx=ggml_init({32*1024*1024,nullptr,true});if(!ctx)return 4;
        auto*q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,n,H);
        auto*kb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,layout?H:n,layout?n:H);
        auto*vb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,layout?H:n,layout?n:H);
        auto*k=layout?ggml_permute(ctx,kb,0,2,1,3):kb;auto*v=layout?ggml_permute(ctx,vb,0,2,1,3):vb;
        auto*mask=same(gate,"mask")?ggml_new_tensor_2d(ctx,GGML_TYPE_F16,n,n):nullptr;
        auto*sink=same(gate,"sinks")?ggml_new_tensor_1d(ctx,GGML_TYPE_F32,H):nullptr;
        auto*y=ggml_flash_attn_ext(ctx,q,k,v,mask,1.f/std::sqrt(float(D)),0,same(gate,"softcap")?2.f:0.f);
        if(sink){ggml_flash_attn_ext_add_sinks(y,sink);ggml_set_input(sink);}
        ggml_set_input(q);ggml_set_input(kb);ggml_set_input(vb);if(mask)ggml_set_input(mask);ggml_set_output(y);
        char name[64];std::snprintf(name,sizeof(name),"serial_excl_%s_n%d_l%d",gate,n,layout);ggml_set_name(y,name);
        if(!ggml_backend_supports_op(backend,y))return 4;
        auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,y);
        auto*buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buffer)return 4;
        const std::array<const void*,5> pointers={q->data,kb->data,vb->data,y->data,sink?sink->data:nullptr};
        std::vector<float> qs(size_t(D)*n*H),out(qs.size()),previous,sinks(H);
        std::vector<ggml_fp16_t> ks(qs.size()),vs(qs.size()),ms(mask?size_t(n)*n:0);
        std::vector<uint8_t> sink_bytes;
        if(sink){for(int h=0;h<H;++h)sinks[h]=float(h-8)*0.03125f;
            sink_bytes.resize(H*sizeof(float));std::memcpy(sink_bytes.data(),sinks.data(),sink_bytes.size());
            ggml_backend_tensor_set(sink,sinks.data(),0,sink_bytes.size());}
        uint64_t last_input=0;
        for(int step=0;step<STEPS;++step){
            uint32_t seed=0x1765e321u^uint32_t(n*29+layout*79+step*317+gate_id*911);
            for(size_t i=0;i<qs.size();++i){qs[i]=float(int(rnd(seed)%65537)-32768)*0x1p-15f;
                ks[i]=ggml_fp32_to_fp16(float(int(rnd(seed)%65537)-32768)*0x1p-15f);
                vs[i]=ggml_fp32_to_fp16(float(int(rnd(seed)%65537)-32768)*0x1p-15f);}
            // Poison every head (or every Q row), so no six-worker range may
            // finish the serial route. Do not call nonfinite output finite.
            for(int h=0;h<H;++h){const size_t last=(size_t(h)*n+n-1)*D+D-1;
                if(same(gate,"k_inf"))ks[last]=ggml_fp32_to_fp16(INFINITY);
                if(same(gate,"v_inf"))vs[last]=ggml_fp32_to_fp16(INFINITY);
                if(same(gate,"q_nan"))for(int row=0;row<n;++row){uint32_t bits=0x7fc01234u;
                    std::memcpy(&qs[(size_t(h)*n+row)*D],&bits,sizeof(bits));}}
            if(mask)for(int row=0;row<n;++row)for(int token=0;token<n;++token)
                ms[size_t(row)*n+token]=ggml_fp32_to_fp16(token<=row?-(token%7)*0.03125f:-INFINITY);
            const auto qp=physical(q,q,qs.data(),sizeof(float)),kp=physical(k,kb,ks.data(),sizeof(ggml_fp16_t)),
                vp=physical(v,vb,vs.data(),sizeof(ggml_fp16_t));
            std::vector<uint8_t> mp(ms.size()*sizeof(ggml_fp16_t));if(mask)std::memcpy(mp.data(),ms.data(),mp.size());
            uint64_t ih=hash(hash(hash(FNV,qp.data(),qp.size()),kp.data(),kp.size()),vp.data(),vp.size());
            if(mask)ih=hash(ih,mp.data(),mp.size());
            if(sink)ih=hash(ih,sink_bytes.data(),sink_bytes.size());
            ggml_backend_tensor_set(q,qp.data(),0,qp.size());ggml_backend_tensor_set(kb,kp.data(),0,kp.size());
            ggml_backend_tensor_set(vb,vp.data(),0,vp.size());if(mask)ggml_backend_tensor_set(mask,mp.data(),0,mp.size());
            const uint32_t poison=0x7fc0176au;for(auto&x:out)std::memcpy(&x,&poison,sizeof(poison));
            ggml_backend_tensor_set(y,out.data(),0,out.size()*sizeof(float));
            if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 5;
            ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float));
            bool written=true,finite=true;for(float x:out){uint32_t bits;std::memcpy(&bits,&x,sizeof(bits));
                written=written&&bits!=poison;finite=finite&&std::isfinite(x);}
            const bool inputs=unchanged(q,qp)&&unchanged(kb,kp)&&unchanged(vb,vp)&&
                (!mask||unchanged(mask,mp))&&(!sink||unchanged(sink,sink_bytes));
            const std::array<const void*,5> now={q->data,kb->data,vb->data,y->data,sink?sink->data:nullptr};
            const bool mutation=!step||(ih!=last_input&&(nonfinite||std::memcmp(previous.data(),out.data(),out.size()*sizeof(float))));
            std::array<uint64_t,10> meta={MAGIC,gate_id,uint64_t(n),uint64_t(layout),uint64_t(step),ih,
                out.size(),k->nb[1],k->nb[2],uint64_t(nonfinite)},expected=meta;
            bool equal=io(file.get(),write,meta.data(),sizeof(meta))&&meta==expected;
            if(write)equal=equal&&io(file.get(),true,out.data(),out.size()*sizeof(float));
            else {std::vector<float>wanted(out.size());equal=equal&&io(file.get(),false,wanted.data(),wanted.size()*sizeof(float))&&
                !std::memcmp(wanted.data(),out.data(),out.size()*sizeof(float));}
            const bool pass=written&&(nonfinite||finite)&&inputs&&now==pointers&&mutation&&equal;
            ok=ok&&pass;rolling=hash(hash(rolling,meta.data(),sizeof(meta)),out.data(),out.size()*sizeof(float));++records;
            std::printf("exclusion gate=%s N=%d layout=%d step=%d serial=%s input_hash=%016llx output_written=%d finite=%d nonfinite_case=%d unchanged=%d same_address=%d mutation=%d bitwise_fixture=%d %s\n",
                gate,n,layout,step,mode,(unsigned long long)ih,int(written),int(finite),int(nonfinite),int(inputs),int(now==pointers),int(mutation),int(equal),pass?"PASS":"FAIL");
            std::fflush(stdout);previous=out;last_input=ih;
        }
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
    ggml_backend_free(backend);
    if(!ok)return 1; // Never make a failed reference valid.
    std::array<uint64_t,4> footer={MAGIC^UINT64_MAX,VERSION,records,rolling},wanted=footer;
    if(!io(file.get(),write,footer.data(),sizeof(footer))||footer!=wanted)return 1;
    if(!write&&std::fgetc(file.get())!=EOF)return 1;
    if(write&&(std::fflush(file.get())||fsync(fileno(file.get()))))return 1;
    std::printf("exclusion gate=%s serial=%s records=%llu bitwise_complete=1 PASS\n",gate,mode,(unsigned long long)records);
    return 0;
}
