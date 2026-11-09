#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <spert.hpp>
#include <riscv_vector.h>
#include "../../ggml/src/ggml-spacemit/ocr-qk-ime2-v53.h"
#include "../../ggml/src/ggml-spacemit/ocr-pv-ime2.h"

int main() {
    spert::StreamConfig config; config.n_cores=1; config.core_ids={8};
    spert::Stream stream(config);
    if(!stream.valid())return 1;
    int cases=0;
    double worst_relative[2]{},worst_absolute[2]{};
    for(int rows:{1,7,8,9,16,17,63,64})
    for(int count:{1,7,8,63,64})
    for(int passes:{1,2}) {
        constexpr int stride=67,guard=32;
        std::vector<float> p(rows*stride,-123.f),out(rows*72+guard,777.f),initial(rows*72);
        std::vector<_Float16> v(count*72);
        for(int i=0;i<rows;++i)for(int k=0;k<count;++k)
            p[i*stride+k]=std::exp(-float((i*131+k*97)%1000)/41.f);
        for(size_t i=0;i<v.size();++i) {
            float value=std::sin(float(i)*.13f)*4.f;
            if(i%17==0)value=std::ldexp(1.f,-24);
            v[i]=(_Float16)value;
        }
        for(size_t i=0;i<initial.size();++i)out[i]=initial[i]=std::cos(float(i)*.17f)*.3f;
        const auto saved_p=p;const auto saved_v=v;
        bool ok=false,replay=true;
        const auto future=stream.launch(spert::Grid{1},[&](spert::Context * context) {
            const auto scratch=context->shared_buffer();
            if(scratch.size<ocr_pv_ime2::scratch_bytes)return;
            ok=ocr_pv_ime2::forward(out.data(),p.data(),v.data(),rows,count,stride,scratch.data,passes);
            const auto first=out;
            for(int repeat=0;repeat<3;++repeat) {
                std::copy(initial.begin(),initial.end(),out.begin());
                ok=ok&&ocr_pv_ime2::forward(out.data(),p.data(),v.data(),rows,count,stride,scratch.data,passes);
                replay=replay&&std::memcmp(first.data(),out.data(),out.size()*sizeof(float))==0;
            }
        });
        if(future.sync()!=spert::Status::Ok||!ok||!replay)return 2;
        if(std::memcmp(p.data(),saved_p.data(),p.size()*4)||
           std::memcmp(v.data(),saved_v.data(),v.size()*2))return 3;
        for(size_t i=initial.size();i<out.size();++i)if(out[i]!=777.f)return 4;
        double error=0.,energy=0.,max_error=0.;
        for(int r=0;r<rows;++r)for(int d=0;d<72;++d) {
            double reference=initial[r*72+d];
            for(int k=0;k<count;++k)reference+=double(p[r*stride+k])*float(v[k*72+d]);
            const double e=out[r*72+d]-reference;
            if(!std::isfinite(out[r*72+d]))return 5;
            error+=e*e;energy+=reference*reference;max_error=std::max(max_error,std::abs(e));
        }
        const double relative=std::sqrt(error/std::max(energy,1.e-30));
        worst_relative[passes-1]=std::max(worst_relative[passes-1],relative);
        worst_absolute[passes-1]=std::max(worst_absolute[passes-1],max_error);
        if(relative>(passes==2?2.e-6:2.e-3)) {
            std::printf("FAIL rows=%d count=%d passes=%d rel=%g max=%g\n",rows,count,passes,relative,max_error);
            return 6;
        }
        ++cases;
    }
    // Out-of-range V must fall back before modifying any destination value.
    for(float invalid:{64.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
        std::vector<float> out(72,123.f),p(64,1.f);
        std::vector<_Float16> v(64*72,(_Float16)1.f);v[0]=(_Float16)invalid;
        bool accepted=true;
        auto future=stream.launch(spert::Grid{1},[&](spert::Context *context){
            accepted=ocr_pv_ime2::forward(out.data(),p.data(),v.data(),1,64,64,context->shared_buffer().data,2);
        });
        if(future.sync()!=spert::Status::Ok||accepted)return 7;
        for(float value:out)if(value!=123.f)return 8;
    }
    std::printf("PASS cases=%d replay=exact inputs=preserved guards=preserved fallback=3\n",cases);
    for(int i=0;i<2;++i)std::printf("passes=%d max_relative_l2=%.12g max_abs=%.12g\n",i+1,worst_relative[i],worst_absolute[i]);
}
