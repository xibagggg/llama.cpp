#pragma once
// Experimental D72 QK matrix product. Inputs are still FP16. Scaling K by
// 2^10 preserves its FP16 subnormals on IME2; undo scaling in FP32 afterward.
namespace ocr_qk_ime2_v53 {
constexpr int D=72;
constexpr size_t scratch_bytes=64*D*2+64*D*2+256*4;
static bool pack_q(const _Float16*q,_Float16*p,int rows){
 for(int g=0;g<(rows+7)/8;++g)for(int d=0;d<D;d+=8)for(int r=0;r<8;++r){
  auto*dst=p+g*8*D+d*8+r*8;
  if(g*8+r<rows){for(int j=0;j<8;++j){const float a=float(q[(g*8+r)*D+d+j]);if(!std::isfinite(a)||std::fabs(a)>63.f)return false;dst[j]=(_Float16)(a*1024.f);}}
  else std::memset(dst,0,16);
 }
 return true;
}
static bool pack_k(const _Float16*k,_Float16*p,int stride,int count){
 for(int g=0;g<8;++g)for(int d=0;d<D;d+=8)for(int r=0;r<8;++r){
  const size_t offset=g*8*D+d*8+r*8;
  for(int j=0;j<8;++j){const float a=g*8+r<count?float(k[(d+j)*stride+g*8+r]):0.f;if(!std::isfinite(a)||std::fabs(a)>63.f)return false;p[offset+j]=(_Float16)(a*1024.f);}
 }
 return true;
}
__attribute__((noinline)) static void core(const _Float16*x,const _Float16*w,float*c){
 const _Float16*w1=w+8*D,*w2=w+16*D,*w3=w+24*D;int blocks=D/8;
 asm volatile(
 "vsetvli t0,zero,e32,m2,ta,ma\n\t"
 "vmv.v.i v16,0\n\tvmv.v.i v18,0\n\tvmv.v.i v20,0\n\tvmv.v.i v22,0\n\t"
 "vsetvli t0,zero,e16,m1,ta,ma\n\t1:\n\t"
 "vle16.v v2,(%[x])\n\tvle16.v v8,(%[w])\n\t"
 "vle16.v v9,(%[w1])\n\tvle16.v v10,(%[w2])\n\tvle16.v v11,(%[w3])\n\t"
 "smt.vfwmadot v16,v2,v8\n\tsmt.vfwmadot v18,v2,v9\n\t"
 "smt.vfwmadot v20,v2,v10\n\tsmt.vfwmadot v22,v2,v11\n\t"
 "addi %[x],%[x],128\n\taddi %[w],%[w],128\n\taddi %[w1],%[w1],128\n\t"
 "addi %[w2],%[w2],128\n\taddi %[w3],%[w3],128\n\t"
 "addi %[blocks],%[blocks],-1\n\tbnez %[blocks],1b\n\t"
 "vsetvli t0,zero,e32,m2,ta,ma\n\tvse32.v v16,(%[c])\n\t"
 "addi t0,%[c],256\n\tvse32.v v18,(t0)\n\taddi t0,%[c],512\n\tvse32.v v20,(t0)\n\taddi t0,%[c],768\n\tvse32.v v22,(t0)\n\t"
 :[x]"+&r"(x),[w]"+&r"(w),[w1]"+&r"(w1),[w2]"+&r"(w2),[w3]"+&r"(w3),[blocks]"+&r"(blocks)
 :[c]"r"(c):"t0","v2","v8","v9","v10","v11","v16","v17","v18","v19","v20","v21","v22","v23","memory");
}
static bool qk(float*dst,const _Float16*packed_q,const _Float16*k,int stride,int rows,int count,void*scratch){
 auto*pk=static_cast<_Float16*>(scratch);auto*tmp=reinterpret_cast<float*>(pk+64*D);
 if(!pack_k(k,pk,stride,count))return false;
 for(int q=0;q<rows;q+=8)for(int key=0;key<count;key+=32){
  core(packed_q+q*D,pk+key*D,tmp);
  for(int g=0;g<4;++g)for(int r=0;r<8&&q+r<rows;++r)for(int j=0;j<8&&key+g*8+j<count;++j)
   dst[(q+r)*stride+key+g*8+j]=tmp[g*64+r*8+j]*(1.f/1048576.f);
 }
 return true;
}
}
