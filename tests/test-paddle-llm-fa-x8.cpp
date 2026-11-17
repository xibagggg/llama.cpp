#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

// Separate CTest processes freeze each static flag before backend creation.
// This checks all output bits against the same-binary x4 control and uses
// independent sampled FP64 values as an accuracy guard. No CPU implementation.
namespace {
constexpr int HQ=16, HK=2, STEPS=2;
constexpr uint64_t MAGIC=UINT64_C(0x504c4c4d58463831);
struct shape { int d,nq,nk,layout,mask; };
std::vector<shape> cases() {
    std::vector<shape> r;
    for(int n:{8,9,63,64,65,212,512}) {
        r.push_back({128,n,256,0,1});
        r.push_back({128,n,512,1,2});
    }
    r.push_back({128,65,1024,0,0});
    r.push_back({128,212,1024,1,1});
    r.push_back({128,1,256,0,1}); // non-tiled decode fallback
    r.push_back({128,7,256,1,2}); // non-tiled fallback / all-masked row
    r.push_back({72,65,256,0,0}); // existing vision dimension remains unchanged
    r.push_back({64,65,256,1,1}); // unsupported candidate dimension fallback
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
              const std::vector<ggml_fp16_t>&v,const std::vector<float>&out,double &rel,double &maxabs) {
    double err=0,den=0;maxabs=0;
    for(int h=0;h<HQ;++h)for(int row:{0,c.nq/2,c.nq-1}) {
        std::vector<double> logits(c.nk,-INFINITY),p(c.nk,0.);
        const int kh=h/(HQ/HK);double mx=-INFINITY;
        for(int t=0;t<c.nk;++t)if(visible(c,row,t)) {
            double dot=0;for(int d=0;d<c.d;++d)dot+=double(q[(size_t(h)*c.nq+row)*c.d+d])*ggml_fp16_to_fp32(k[(size_t(kh)*c.nk+t)*c.d+d]);
            logits[t]=dot*(1.f/std::sqrt(float(c.d)));mx=std::max(mx,logits[t]);
        }
        double sum=0; if(std::isfinite(mx))for(int t=0;t<c.nk;++t)if(std::isfinite(logits[t]))sum+=(p[t]=std::exp(logits[t]-mx));
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
    const char*flag=std::getenv("GGML_SPACEMIT_PADDLE_LLM_FA_X8");
    if(!flag||std::strcmp(flag,write?"0":"1")){std::fprintf(stderr,"Explicit matching LLM x8 flag required\n");return 2;}
    const int fd=write?open(argv[2],O_WRONLY|O_CREAT|O_EXCL,0644):open(argv[2],O_RDONLY);
    if(fd<0){std::perror(argv[2]);return 2;}
    std::unique_ptr<FILE,decltype(&std::fclose)> f(fdopen(fd,write?"wb":"rb"),&std::fclose);if(!f){close(fd);return 2;}
    const auto cs=cases();
    std::array<uint64_t,8> hdr={MAGIC,1,0x01020304,sizeof(float),sizeof(ggml_fp16_t),HQ,HK,cs.size()},wanted=hdr;
    if(!io(f.get(),write,hdr.data(),sizeof(hdr))||hdr!=wanted)return 2;
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
        auto*y=ggml_flash_attn_ext(ctx,q,k,v,mask,1.f/std::sqrt(float(c.d)),0,0);
        ggml_set_input(q);ggml_set_input(kb);ggml_set_input(vb);if(mask)ggml_set_input(mask);ggml_set_output(y);
        char name[64];std::snprintf(name,sizeof(name),"llm_x8_c%zu_d%d_q%d_k%d",id,c.d,c.nq,c.nk);ggml_set_name(y,name);
        if(!ggml_backend_supports_op(backend,y))return 4;
        auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,y);auto*buf=ggml_backend_alloc_ctx_tensors(ctx,backend);if(!buf)return 4;
        std::vector<float> qs(size_t(c.d)*c.nq*HQ),out(qs.size()),previous;
        std::vector<ggml_fp16_t> ks(size_t(c.d)*c.nk*HK),vs(ks.size()),ms(size_t(c.nk)*padded_q,ggml_fp32_to_fp16(-INFINITY));
        const std::array<const void*,4> pointers={q->data,kb->data,vb->data,y->data};
        for(int step=0;step<STEPS;++step) {
            uint32_t rng=0x1769123u^uint32_t(id*911+step*79);
            for(auto&x:qs)x=float(int(rnd(rng)%65537)-32768)*0x1p-15f;
            for(auto&x:ks)x=ggml_fp32_to_fp16(float(int(rnd(rng)%65537)-32768)*0x1p-15f);
            for(auto&x:vs)x=ggml_fp32_to_fp16(float(int(rnd(rng)%65537)-32768)*0x1p-15f);
            const uint16_t edges[]={0,0x8000,1,0x8001,0x3ff,0x83ff,0x400,0x8400};
            for(size_t i=0;i<qs.size();i+=97)qs[i]=ggml_fp16_to_fp32(edges[(i/97+step)%8]);
            for(size_t i=0;i<ks.size();i+=97)ks[i]=vs[i]=edges[(i/97+step)%8];
            for(int row=0;row<c.nq;++row)for(int t=0;t<c.nk;++t)ms[size_t(row)*c.nk+t]=ggml_fp32_to_fp16(visible(c,row,t)?0.f:-INFINITY);
            const auto qp=physical(q,q,qs.data(),sizeof(float)),kp=physical(k,kb,ks.data(),sizeof(ggml_fp16_t)),vp=physical(v,vb,vs.data(),sizeof(ggml_fp16_t));
            uint64_t ih=hash(UINT64_C(14695981039346656037),qp.data(),qp.size());ih=hash(ih,kp.data(),kp.size());ih=hash(ih,vp.data(),vp.size());if(mask)ih=hash(ih,ms.data(),ms.size()*sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(q,qp.data(),0,qp.size());ggml_backend_tensor_set(kb,kp.data(),0,kp.size());ggml_backend_tensor_set(vb,vp.data(),0,vp.size());
            if(mask)ggml_backend_tensor_set(mask,ms.data(),0,ms.size()*sizeof(ggml_fp16_t));
            std::fill(out.begin(),out.end(),std::nanf("176"));ggml_backend_tensor_set(y,out.data(),0,out.size()*sizeof(float));
            if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 5;
            ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float));
            bool finite=std::all_of(out.begin(),out.end(),[](float x){return std::isfinite(x);});
            double rel=0,ma=0;bool accurate=accuracy(c,qs,ks,vs,out,rel,ma);
            bool sameinputs=unchanged(q,qp)&&unchanged(kb,kp)&&unchanged(vb,vp);
            if(mask){std::vector<uint8_t> mb(ms.size()*sizeof(ggml_fp16_t));std::memcpy(mb.data(),ms.data(),mb.size());sameinputs=sameinputs&&unchanged(mask,mb);}
            const std::array<const void*,4> now={q->data,kb->data,vb->data,y->data};
            bool changed=!step||std::memcmp(out.data(),previous.data(),out.size()*sizeof(float));
            std::array<uint64_t,12> meta={MAGIC,id,uint64_t(step),uint64_t(c.d),uint64_t(c.nq),uint64_t(c.nk),uint64_t(c.layout),uint64_t(c.mask),ih,out.size(),k->nb[1],k->nb[2]},expected=meta;
            bool match=io(f.get(),write,meta.data(),sizeof(meta))&&meta==expected;
            if(write)match=match&&io(f.get(),true,out.data(),out.size()*sizeof(float));
            else {std::vector<float> ref(out.size());match=match&&io(f.get(),false,ref.data(),ref.size()*sizeof(float))&&!std::memcmp(ref.data(),out.data(),out.size()*sizeof(float));}
            bool pass=finite&&accurate&&sameinputs&&changed&&now==pointers&&match;ok=ok&&pass;
            std::printf("case=%zu D=%d Q=%d KV=%d layout=%d mask=%d step=%d input_hash=%016llx rel=%.9g maxabs=%.9g finite=%d unchanged=%d changed=%d fixture=%d %s\n",id,c.d,c.nq,c.nk,c.layout,c.mask,step,(unsigned long long)ih,rel,ma,int(finite),int(sameinputs),int(changed),int(match),pass?"PASS":"FAIL");std::fflush(stdout);
            previous=out;values+=out.size();
        }
        ggml_backend_buffer_free(buf);ggml_free(ctx);
    }
    std::array<uint64_t,3> footer={MAGIC,cs.size()*STEPS,values},expected=footer;
    bool final=ok&&io(f.get(),write,footer.data(),sizeof(footer))&&footer==expected;
    if(write)final=final&&!std::fflush(f.get());else final=final&&std::fgetc(f.get())==EOF&&!std::ferror(f.get());
    ggml_backend_free(backend);std::printf("suite executions=%zu values=%llu %s\n",cs.size()*STEPS,(unsigned long long)values,final?"PASS":"FAIL");return final?0:1;
}
