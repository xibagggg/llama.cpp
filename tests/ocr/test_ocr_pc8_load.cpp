#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ocr-pc8-load.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>

extern "C" void ggml_abort(const char * file,int line,const char * fmt,...) {
    std::fprintf(stderr,"ggml assertion at %s:%d %s\n",file,line,fmt);std::abort();
}

static void reference(const block_q4_0 * q,void * output,int K,int N) {
    const int NP=(N+31)/32*32;
    std::memset(output,0,128+size_t(NP)*(4+K));
    auto * scales=reinterpret_cast<float*>(static_cast<char*>(output)+128);
    auto * packed=reinterpret_cast<int8_t*>(scales+NP);
    std::vector<float> row(K);
    for(int r=0;r<N;++r) {
        float maximum=0;
        for(int k=0;k<K;k+=32) {
            const auto & b=q[size_t(r)*(K/32)+k/32];
            const float d=ggml_compute_fp16_to_fp32(b.d);
            for(int j=0;j<32;++j) {
                const int v=(j<16?b.qs[j]&15:b.qs[j-16]>>4)-8;
                row[k+j]=d*v;maximum=std::max(maximum,std::fabs(row[k+j]));
            }
        }
        const float scale=maximum/127;scales[r]=scale;
        const float inv=scale?1/scale:0;
        for(int k=0;k<K;k+=32)for(int j=0;j<32;++j)
            packed[size_t(r/32)*32*K+(k/32)*1024+(r%32)*32+j]=
                int8_t(std::max(-127.0f,std::min(127.0f,std::nearbyint(row[k+j]*inv))));
    }
}
int main() {
    int cases=0;
    for(const auto shape:{std::pair<int,int>{1152,1152},{1152,4304},{4352,1152},{64,33}}) {
        const auto K=shape.first,N=shape.second;
        std::vector<block_q4_0> input(size_t(N)*(K/32));
        for(size_t i=0;i<input.size();++i) {
            auto & b=input[i];
            // Include zero, subnormal, signed scales, all nibbles and rows
            // with a different absolute maximum. Keep values finite.
            b.d=i%(K/32)==0?0:i%19==0?1:uint16_t((i*31)%0x7000);
            if(i%11==0)b.d|=0x8000;
            for(int j=0;j<16;++j)b.qs[j]=uint8_t(i*17+j*31);
        }
        const auto original=input;
        const size_t bytes=128+size_t((N+31)/32*32)*(4+K);
        std::vector<uint64_t> golden((bytes+7)/8),actual((bytes+7)/8+16);
        for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
            std::fesetround(rounding);
            reference(input.data(),golden.data(),K,N);
            for(int threads:{1,2,4,8}) {
                std::fill(actual.begin(),actual.end(),UINT64_C(0xa5a5a5a5a5a5a5a5));
                const auto start=std::chrono::steady_clock::now();
                ocr_pc8_load::repack(input.data(),actual.data(),K,N,threads,
                    [](ggml_half h){return ggml_compute_fp16_to_fp32(h);});
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                if(std::memcmp(actual.data(),golden.data(),bytes))return 1;
                if(std::memcmp(input.data(),original.data(),input.size()*sizeof(block_q4_0)))return 2;
                const auto * tail=reinterpret_cast<const uint8_t*>(actual.data())+bytes;
                for(size_t i=0;i<128;++i)if(tail[i]!=0xa5)return 3;
                if(std::fegetround()!=rounding)return 4;
                ++cases;
                if(rounding==FE_TONEAREST)std::printf("K=%d N=%d threads=%d seconds=%.6f exact=true\n",K,N,threads,seconds);
            }
        }
    }
    std::printf("PASS cases=%d output=byte-exact inputs=unchanged guard=preserved rounding=preserved\n",cases);
}
