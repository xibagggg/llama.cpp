#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-spacemit.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
static unsigned mix(unsigned x){x^=x>>16;x*=0x7feb352d;x^=x>>15;return x*0x846ca68b;}
int main(){
 auto backend=ggml_backend_spacemit_init();if(!backend)return 1;
 for(int width:{1,7,63,65,72,129})for(int outer:{1,3}){
  const int rows=width==72?1008:13,heads=outer==1?1:16;
  ggml_init_params p{};p.no_alloc=true;p.mem_size=2*1024*1024;auto ctx=ggml_init(p);
  auto x=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,width,heads,rows,outer);
  ggml_set_input(x);ggml_set_output(x);
  auto view=ggml_permute(ctx,x,0,2,1,3);auto y=ggml_cast(ctx,view,GGML_TYPE_F16);ggml_set_output(y);
  if(!ggml_backend_supports_op(backend,y))return 2;
  auto g=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(g,y);
  auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!ggml_gallocr_alloc_graph(alloc,g))return 3;
  std::vector<float>a(ggml_nelements(x)),b,check(a.size());
  std::vector<ggml_fp16_t>actual(a.size()),reference(a.size());
  for(size_t i=0;i<a.size();++i){
   a[i]=(int(mix(i+91)%65536)-32768)*.000731f;
   if(i%13==0)a[i]=std::ldexp((i%2?-1.f:1.f),-24);
   if(i%17==0)a[i]=std::ldexp((i%2?-1.f:1.f),-14);
   if(i%19==0)a[i]=std::copysign(0.f,i%2?-1.f:1.f);
   if(i%23==0)a[i]=(1.f+std::ldexp(1.f,-11));
  }
  b=a;for(auto&f:b)f=f*.5f+.125f;
  auto expected=[&](const std::vector<float>&input){
   for(int batch=0;batch<outer;++batch)for(int head=0;head<heads;++head)
    for(int row=0;row<rows;++row)for(int col=0;col<width;++col)
     reference[(((batch*heads+head)*rows+row)*width+col)]=
      ggml_fp32_to_fp16(input[(((batch*rows+row)*heads+head)*width+col)]);
  };
  for(int rep=-1;rep<5;++rep){
   const auto&input=rep==3?b:a;expected(input);ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
   const auto start=std::chrono::steady_clock::now();
   if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS)return 4;
   const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
   ggml_backend_tensor_get(y,actual.data(),0,actual.size()*2);
   if(std::memcmp(actual.data(),reference.data(),actual.size()*2))return 5;
   ggml_backend_tensor_get(x,check.data(),0,check.size()*4);
   if(std::memcmp(check.data(),input.data(),check.size()*4))return 6;
   if(rep==0)std::printf("PASS width=%d rows=%d heads=%d batch=%d ms=%.6f\n",width,rows,heads,outer,ms);
  }
  ggml_gallocr_free(alloc);ggml_free(ctx);
 }
 ggml_backend_free(backend);std::puts("PASS 12 shapes, FP16 rounding, A_B_A, immutable input");
}
