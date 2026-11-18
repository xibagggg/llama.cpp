#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

// Separate CTest processes freeze each static flag before backend creation.
// This checks all output bits against the same-binary original dispatch and uses
// independent sampled FP64 values as an accuracy guard. No CPU implementation.
namespace {
constexpr int HQ=16, HK=2, STEPS=2;
constexpr uint64_t MAGIC=UINT64_C(0x5047514138455831);
constexpr uint64_t FORMAT_VERSION=3;
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,
              "Reference v3 requires IEEE754 binary64 FP64 error records");
struct shape { int d,nq,nk,layout,mask,softcap,sink; };
std::vector<shape> cases() {
    std::vector<shape> r;
    // Exact group path, both K/V layouts and partial KV/tokens/work units.
    for(int n:{5,7,8,9,15,16,31,63}) {
        r.push_back({128,n,129,0,1,0,0});
        r.push_back({128,n,383,1,2,0,0});
    }
    r.push_back({128,8,513,0,0,0,0}); // no mask
    r.push_back({128,9,257,1,3,1,1}); // finite additive mask, softcap, sinks
    r.push_back({128,7,385,0,2,0,1}); // all-masked row with sinks
    // Preserve original split-KV, tiled and other-D fallbacks.
    r.push_back({128,1,512,0,1,0,0});
    r.push_back({128,4,512,1,2,0,0});
    r.push_back({128,64,256,0,1,0,0});
    r.push_back({72,9,256,1,1,0,0});
    r.push_back({64,7,256,0,1,0,0});
    return r;
}
uint32_t rnd(uint32_t &s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
uint64_t hash(uint64_t h,const void*p,size_t n) {
    const auto*b=static_cast<const uint8_t*>(p);
    for(size_t i=0;i<n;++i)h=(h^b[i])*UINT64_C(1099511628211);
    return h;
}
std::vector<uint8_t> physical(const ggml_tensor*t,const ggml_tensor*b,const void*src,size_t elem) {
    std::vector<uint8_t> out(ggml_nbytes(b),0xa5);
    for(int64_t h=0;h<t->ne[2];++h)for(int64_t n=0;n<t->ne[1];++n)for(int64_t d=0;d<t->ne[0];++d) {
        size_t off=h*t->nb[2]+n*t->nb[1]+d*t->nb[0],i=(h*t->ne[1]+n)*t->ne[0]+d;
        if(off+elem>out.size())std::abort();
        std::memcpy(out.data()+off,static_cast<const uint8_t*>(src)+i*elem,elem);
    }
    return out;
}
bool unchanged(const ggml_tensor*t,const std::vector<uint8_t>&b) {
    std::vector<uint8_t> now(b.size());ggml_backend_tensor_get(t,now.data(),0,now.size());return now==b;
}
bool visible(shape c,int row,int token) {
    if(c.mask==0)return true;
    if(c.mask==2&&row%13==0)return false;
    const int prefix=std::min(c.nk,1+row*c.nk/std::max(c.nq,1));
    return token<prefix && !(c.mask==2 && token%11==5);
}
bool accuracy(shape c,const std::vector<float>&q,const std::vector<ggml_fp16_t>&k,
              const std::vector<ggml_fp16_t>&v,const std::vector<float>&sinks,const std::vector<float>&out,double &rel,double &maxabs) {
    double err=0,den=0;maxabs=0;
    for(int h=0;h<HQ;++h)for(int row:{0,c.nq/2,c.nq-1}) {
        std::vector<double> logits(c.nk,-INFINITY),p(c.nk,0.);
        const int kh=h/(HQ/HK);double mx=-INFINITY;
        for(int t=0;t<c.nk;++t)if(visible(c,row,t)) {
            double dot=0;for(int d=0;d<c.d;++d)dot+=double(q[(size_t(h)*c.nq+row)*c.d+d])*ggml_fp16_to_fp32(k[(size_t(kh)*c.nk+t)*c.d+d]);
            logits[t]=dot*(1.f/std::sqrt(float(c.d)));
            if(c.softcap)logits[t]=2.0*std::tanh(logits[t]/2.0);
            if(c.mask==3)logits[t]-=0.03125*(t%7);
            mx=std::max(mx,logits[t]);
        }
        if(c.sink)mx=std::max(mx,double(sinks[h]));
        double sum=c.sink?std::exp(double(sinks[h])-mx):0.; if(std::isfinite(mx))for(int t=0;t<c.nk;++t)if(std::isfinite(logits[t]))sum+=(p[t]=std::exp(logits[t]-mx));
        for(int d=0;d<c.d;++d) {
            double ref=0;if(sum)for(int t=0;t<c.nk;++t)ref+=p[t]/sum*ggml_fp16_to_fp32(v[(size_t(kh)*c.nk+t)*c.d+d]);
            double e=out[(size_t(row)*HQ+h)*c.d+d]-ref;err+=e*e;den+=ref*ref;maxabs=std::max(maxabs,std::abs(e));
        }
    }
    rel=std::sqrt(err/std::max(den,1e-30));return std::isfinite(rel)&&std::isfinite(maxabs)&&rel<0.003&&maxabs<0.003;
}
bool io(FILE*f,bool write,void*p,size_t n) {return write?std::fwrite(p,1,n,f)==n:std::fread(p,1,n,f)==n;}
}
int main(int argc,char**argv) {
    if(argc!=3||(std::strcmp(argv[1],"--write-reference")&&std::strcmp(argv[1],"--compare-reference")))return 2;
    const bool write=!std::strcmp(argv[1],"--write-reference");
    const char*flag=std::getenv("GGML_SPACEMIT_PADDLE_FA_GQA8_EXACT");
    if(!flag||std::strcmp(flag,write?"0":"1")){std::fprintf(stderr,"Explicit matching GQA8 exact flag required\n");return 2;}
    const int fd=write?open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0644):open(argv[2],O_RDONLY);
    if(fd<0){std::perror(argv[2]);return 2;}
    std::unique_ptr<FILE,decltype(&std::fclose)> f(fdopen(fd,write?"wb":"rb"),&std::fclose);if(!f){close(fd);return 2;}
    const auto cs=cases();
    // v3 adds exact FP64 error bytes and an input-bound integrity hash per case.
    // The original relative threshold remains visible as legacy_relative_pass;
    // maxabs<0.003 stays hard, and candidate errors may never exceed reference.
    std::array<uint64_t,10> hdr={MAGIC,FORMAT_VERSION,0x01020304,sizeof(float),sizeof(ggml_fp16_t),sizeof(double),HQ,HK,cs.size(),2},wanted=hdr;
    if(!io(f.get(),write,hdr.data(),sizeof(hdr))||hdr!=wanted)return 2;
    const char * workers=std::getenv("GGML_SPACEMIT_WORKERS");
    const char * allow=std::getenv("GGML_SPACEMIT_ALLOW_EIGHT_WORKERS");
    if(!workers||std::strcmp(workers,"6")||!allow||std::strcmp(allow,"0")) {
        std::fprintf(stderr,"Explicit workers=6 and ALLOW_EIGHT_WORKERS=0 required\n");return 2;
    }
    ggml_backend_load_all();const auto dev=ggml_backend_dev_by_name("SPACEMIT0");
    if(!dev)return 3;const auto backend=ggml_backend_dev_init(dev,nullptr);if(!backend)return 3;
    uint64_t values=0;bool ok=true;
    for(size_t id=0;id<cs.size();++id) {
        const auto c=cs[id];auto*ctx=ggml_init({32*1024*1024,nullptr,true});if(!ctx)return 4;
        auto*q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,c.d,c.nq,HQ);
        auto*kb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,c.d,c.layout?HK:c.nk,c.layout?c.nk:HK);
        auto*vb=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,c.d,c.layout?HK:c.nk,c.layout?c.nk:HK);
        auto*k=c.layout?ggml_permute(ctx,kb,0,2,1,3):kb;auto*v=c.layout?ggml_permute(ctx,vb,0,2,1,3):vb;
        const int padded_q=c.nq;
        auto*mask=c.mask?ggml_new_tensor_2d(ctx,GGML_TYPE_F16,c.nk,padded_q):nullptr;
        auto*sink=c.sink?ggml_new_tensor_1d(ctx,GGML_TYPE_F32,HQ):nullptr;
        auto*y=ggml_flash_attn_ext(ctx,q,k,v,mask,1.f/std::sqrt(float(c.d)),0,c.softcap?2.f:0.f);
        if(sink){ggml_flash_attn_ext_add_sinks(y,sink);ggml_set_input(sink);}
        ggml_set_input(q);ggml_set_input(kb);ggml_set_input(vb);if(mask)ggml_set_input(mask);ggml_set_output(y);
        char name[64];std::snprintf(name,sizeof(name),"paddle_gqa8_exact_c%zu_d%d_q%d_k%d",id,c.d,c.nq,c.nk);ggml_set_name(y,name);
        if(!ggml_backend_supports_op(backend,y))return 4;
        auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,y);auto*buf=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buf)return 4;
        std::vector<float> qs(size_t(c.d)*c.nq*HQ),out(qs.size()),previous,sinks(HQ);
        for(int h=0;h<HQ;++h)sinks[h]=float(h-8)*0.03125f;
        if(sink)ggml_backend_tensor_set(sink,sinks.data(),0,sinks.size()*sizeof(float));
        std::vector<ggml_fp16_t> ks(size_t(c.d)*c.nk*HK),vs(ks.size()),ms(size_t(c.nk)*padded_q,ggml_fp32_to_fp16(-INFINITY));
        const std::array<const void*,5> pointers={q->data,kb->data,vb->data,y->data,sink?sink->data:nullptr};
        for(int step=0;step<STEPS;++step) {
            uint32_t rng=0x1769123u^uint32_t(id*911+step*79);
            for(auto&x:qs)x=float(int(rnd(rng)%65537)-32768)*0x1p-15f;
            for(auto&x:ks)x=ggml_fp32_to_fp16(float(int(rnd(rng)%65537)-32768)*0x1p-15f);
            for(auto&x:vs)x=ggml_fp32_to_fp16(float(int(rnd(rng)%65537)-32768)*0x1p-15f);
            const uint16_t edges[]={0,0x8000,1,0x8001,0x3ff,0x83ff,0x400,0x8400};
            for(size_t i=0;i<qs.size();i+=97)qs[i]=ggml_fp16_to_fp32(edges[(i/97+step)%8]);
            for(size_t i=0;i<ks.size();i+=97)ks[i]=vs[i]=edges[(i/97+step)%8];
            for(int row=0;row<c.nq;++row)for(int t=0;t<c.nk;++t)ms[size_t(row)*c.nk+t]=ggml_fp32_to_fp16(visible(c,row,t)?(c.mask==3?-0.03125f*(t%7):0.f):-INFINITY);
            const auto qp=physical(q,q,qs.data(),sizeof(float)),kp=physical(k,kb,ks.data(),sizeof(ggml_fp16_t)),vp=physical(v,vb,vs.data(),sizeof(ggml_fp16_t));
            uint64_t ih=hash(UINT64_C(14695981039346656037),qp.data(),qp.size());ih=hash(ih,kp.data(),kp.size());ih=hash(ih,vp.data(),vp.size());if(mask)ih=hash(ih,ms.data(),ms.size()*sizeof(ggml_fp16_t));if(sink)ih=hash(ih,sinks.data(),sinks.size()*sizeof(float));
            ggml_backend_tensor_set(q,qp.data(),0,qp.size());ggml_backend_tensor_set(kb,kp.data(),0,kp.size());ggml_backend_tensor_set(vb,vp.data(),0,vp.size());
            if(mask)ggml_backend_tensor_set(mask,ms.data(),0,ms.size()*sizeof(ggml_fp16_t));
            std::fill(out.begin(),out.end(),std::nanf("176"));ggml_backend_tensor_set(y,out.data(),0,out.size()*sizeof(float));
            const auto begin=std::chrono::steady_clock::now();
            if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 5;
            const auto graph_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
            ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float));
            bool finite=std::all_of(out.begin(),out.end(),[](float x){return std::isfinite(x);});
            double rel=0,ma=0;
            const bool legacy_accuracy_pass=accuracy(c,qs,ks,vs,sinks,out,rel,ma);
            const bool legacy_relative_pass=std::isfinite(rel)&&rel<0.003;
            // Original m1 accumulates V in F16. Its measured FP64 error is the
            // reference floor, never a candidate-induced error or a precision
            // improvement claim. The independent absolute bound stays hard.
            const bool absolute_accurate=std::isfinite(rel)&&std::isfinite(ma)&&ma<0.003;
            bool sameinputs=unchanged(q,qp)&&unchanged(kb,kp)&&unchanged(vb,vp);
            if(mask){std::vector<uint8_t> mb(ms.size()*sizeof(ggml_fp16_t));std::memcpy(mb.data(),ms.data(),mb.size());sameinputs=sameinputs&&unchanged(mask,mb);}
            if(sink){std::vector<uint8_t> sb(sinks.size()*sizeof(float));std::memcpy(sb.data(),sinks.data(),sb.size());sameinputs=sameinputs&&unchanged(sink,sb);}
            const std::array<const void*,5> now={q->data,kb->data,vb->data,y->data,sink?sink->data:nullptr};
            bool changed=!step||std::memcmp(out.data(),previous.data(),out.size()*sizeof(float));
            std::array<uint64_t,14> meta={MAGIC,id,uint64_t(step),uint64_t(c.d),uint64_t(c.nq),uint64_t(c.nk),uint64_t(c.layout),uint64_t(c.mask),ih,out.size(),k->nb[1],k->nb[2],uint64_t(c.softcap),uint64_t(c.sink)},expected=meta;
            bool match=io(f.get(),write,meta.data(),sizeof(meta))&&meta==expected;
            std::array<double,2> reference_error={rel,ma};
            std::array<uint64_t,2> reference_error_meta={hash(ih,reference_error.data(),sizeof(reference_error)),uint64_t(legacy_relative_pass)};
            bool reference_error_verified=true,fp64_no_worse=true;
            if(write) {
                match=match&&io(f.get(),true,reference_error.data(),sizeof(reference_error))&&
                    io(f.get(),true,reference_error_meta.data(),sizeof(reference_error_meta))&&
                    io(f.get(),true,out.data(),out.size()*sizeof(float));
            } else {
                std::vector<float> ref(out.size());
                match=match&&io(f.get(),false,reference_error.data(),sizeof(reference_error))&&
                    io(f.get(),false,reference_error_meta.data(),sizeof(reference_error_meta))&&
                    io(f.get(),false,ref.data(),ref.size()*sizeof(float));
                double reference_rel=0,reference_maxabs=0;
                accuracy(c,qs,ks,vs,sinks,ref,reference_rel,reference_maxabs);
                const std::array<double,2> recomputed={reference_rel,reference_maxabs};
                reference_error_verified=match&&reference_error_meta[0]==hash(ih,reference_error.data(),sizeof(reference_error))&&
                    !std::memcmp(recomputed.data(),reference_error.data(),sizeof(reference_error))&&
                    reference_error_meta[1]==uint64_t(std::isfinite(reference_rel)&&reference_rel<0.003)&&
                    std::isfinite(reference_rel)&&std::isfinite(reference_maxabs)&&reference_maxabs<0.003;
                fp64_no_worse=reference_error_verified&&rel<=reference_error[0]&&ma<=reference_error[1];
                // Exact F32 output equality remains an independent hard gate.
                match=match&&!std::memcmp(ref.data(),out.data(),out.size()*sizeof(float));
            }
            bool pass=finite&&absolute_accurate&&sameinputs&&changed&&now==pointers&&match&&reference_error_verified&&fp64_no_worse;ok=ok&&pass;
            std::printf("case=%zu D=%d Q=%d KV=%d layout=%d mask=%d step=%d input_hash=%016llx rel=%.17g maxabs=%.17g finite=%d unchanged=%d changed=%d fixture=%d softcap=%d sink=%d graph_ns=%lld legacy_relative_pass=%d legacy_accuracy_pass=%d fp64_absolute_pass=%d reference_rel=%.17g reference_maxabs=%.17g reference_error_hash=%016llx reference_error_verified=%d fp64_no_worse=%d format_version=%llu %s\n",id,c.d,c.nq,c.nk,c.layout,c.mask,step,(unsigned long long)ih,rel,ma,int(finite),int(sameinputs),int(changed),int(match),c.softcap,c.sink,(long long)graph_ns,int(legacy_relative_pass),int(legacy_accuracy_pass),int(absolute_accurate),reference_error[0],reference_error[1],(unsigned long long)reference_error_meta[0],int(reference_error_verified),int(fp64_no_worse),(unsigned long long)FORMAT_VERSION,pass?"PASS":"FAIL");std::fflush(stdout);
            previous=out;values+=out.size();
        }
        ggml_backend_buffer_free(buf);ggml_free(ctx);
    }
    std::array<uint64_t,3> footer={MAGIC,cs.size()*STEPS,values},expected=footer;
    bool final=ok&&io(f.get(),write,footer.data(),sizeof(footer))&&footer==expected;
    if(write)final=final&&!std::fflush(f.get());else final=final&&std::fgetc(f.get())==EOF&&!std::ferror(f.get());
    ggml_backend_free(backend);std::printf("suite executions=%zu values=%llu %s\n",cs.size()*STEPS,(unsigned long long)values,final?"PASS":"FAIL");return final?0:1;
}
