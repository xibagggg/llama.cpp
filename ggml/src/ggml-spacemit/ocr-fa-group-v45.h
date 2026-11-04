#pragma once
namespace ocr_fa_group_v45 {
inline int mode(){static const int v=[] {const char*p=std::getenv("GGML_SPACEMIT_OCR_FA_GROUP");
 return p&&p[0]>='1'&&p[0]<='6'&&!p[1]?int(p[0]-'0'):0;}();return v;}
inline int qk_group(){const int m=mode();return m==1||m==5?10:m==2||m==6?12:8;}
inline int pv_group(){const int m=mode();return m==3||m==5?10:m==4||m==6?12:8;}
inline void qk10(float*dst,const _Float16*q,const _Float16*k,int dk,int stride,int count){
 const size_t vl=__riscv_vsetvl_e16m1(count);
 auto a0=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a1=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a2=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a3=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a4=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a5=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a6=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a7=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a8=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a9=__riscv_vfmv_v_f_f32m2(0,vl);
 for(int d=0;d<dk;++d){const auto kv=__riscv_vle16_v_f16m1(k+d*stride,vl);
  a0=__riscv_vfwmacc_vf_f32m2(a0,q[0*dk+d],kv,vl);
  a1=__riscv_vfwmacc_vf_f32m2(a1,q[1*dk+d],kv,vl);
  a2=__riscv_vfwmacc_vf_f32m2(a2,q[2*dk+d],kv,vl);
  a3=__riscv_vfwmacc_vf_f32m2(a3,q[3*dk+d],kv,vl);
  a4=__riscv_vfwmacc_vf_f32m2(a4,q[4*dk+d],kv,vl);
  a5=__riscv_vfwmacc_vf_f32m2(a5,q[5*dk+d],kv,vl);
  a6=__riscv_vfwmacc_vf_f32m2(a6,q[6*dk+d],kv,vl);
  a7=__riscv_vfwmacc_vf_f32m2(a7,q[7*dk+d],kv,vl);
  a8=__riscv_vfwmacc_vf_f32m2(a8,q[8*dk+d],kv,vl);
  a9=__riscv_vfwmacc_vf_f32m2(a9,q[9*dk+d],kv,vl);
 }
 __riscv_vse32_v_f32m2(dst+0*stride,a0,vl);
 __riscv_vse32_v_f32m2(dst+1*stride,a1,vl);
 __riscv_vse32_v_f32m2(dst+2*stride,a2,vl);
 __riscv_vse32_v_f32m2(dst+3*stride,a3,vl);
 __riscv_vse32_v_f32m2(dst+4*stride,a4,vl);
 __riscv_vse32_v_f32m2(dst+5*stride,a5,vl);
 __riscv_vse32_v_f32m2(dst+6*stride,a6,vl);
 __riscv_vse32_v_f32m2(dst+7*stride,a7,vl);
 __riscv_vse32_v_f32m2(dst+8*stride,a8,vl);
 __riscv_vse32_v_f32m2(dst+9*stride,a9,vl);
}
inline void pv10(float*dst,const float*p,const _Float16*v,int dv,int stride,int count){
 for(int off=0;off<dv;){const size_t vl=__riscv_vsetvl_e16m1(dv-off);
 auto a0=__riscv_vle32_v_f32m2(dst+0*dv+off,vl);
 auto a1=__riscv_vle32_v_f32m2(dst+1*dv+off,vl);
 auto a2=__riscv_vle32_v_f32m2(dst+2*dv+off,vl);
 auto a3=__riscv_vle32_v_f32m2(dst+3*dv+off,vl);
 auto a4=__riscv_vle32_v_f32m2(dst+4*dv+off,vl);
 auto a5=__riscv_vle32_v_f32m2(dst+5*dv+off,vl);
 auto a6=__riscv_vle32_v_f32m2(dst+6*dv+off,vl);
 auto a7=__riscv_vle32_v_f32m2(dst+7*dv+off,vl);
 auto a8=__riscv_vle32_v_f32m2(dst+8*dv+off,vl);
 auto a9=__riscv_vle32_v_f32m2(dst+9*dv+off,vl);
 for(int t=0;t<count;++t){const auto vv=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+t*dv+off,vl),vl);
  a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t],vv,vl);
  a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t],vv,vl);
  a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t],vv,vl);
  a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t],vv,vl);
  a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t],vv,vl);
  a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t],vv,vl);
  a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t],vv,vl);
  a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t],vv,vl);
  a8=__riscv_vfmacc_vf_f32m2(a8,p[8*stride+t],vv,vl);
  a9=__riscv_vfmacc_vf_f32m2(a9,p[9*stride+t],vv,vl);
 }
 __riscv_vse32_v_f32m2(dst+0*dv+off,a0,vl);
 __riscv_vse32_v_f32m2(dst+1*dv+off,a1,vl);
 __riscv_vse32_v_f32m2(dst+2*dv+off,a2,vl);
 __riscv_vse32_v_f32m2(dst+3*dv+off,a3,vl);
 __riscv_vse32_v_f32m2(dst+4*dv+off,a4,vl);
 __riscv_vse32_v_f32m2(dst+5*dv+off,a5,vl);
 __riscv_vse32_v_f32m2(dst+6*dv+off,a6,vl);
 __riscv_vse32_v_f32m2(dst+7*dv+off,a7,vl);
 __riscv_vse32_v_f32m2(dst+8*dv+off,a8,vl);
 __riscv_vse32_v_f32m2(dst+9*dv+off,a9,vl);
 off+=vl;
 }
}
inline void qk12(float*dst,const _Float16*q,const _Float16*k,int dk,int stride,int count){
 const size_t vl=__riscv_vsetvl_e16m1(count);
 auto a0=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a1=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a2=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a3=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a4=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a5=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a6=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a7=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a8=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a9=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a10=__riscv_vfmv_v_f_f32m2(0,vl);
 auto a11=__riscv_vfmv_v_f_f32m2(0,vl);
 for(int d=0;d<dk;++d){const auto kv=__riscv_vle16_v_f16m1(k+d*stride,vl);
  a0=__riscv_vfwmacc_vf_f32m2(a0,q[0*dk+d],kv,vl);
  a1=__riscv_vfwmacc_vf_f32m2(a1,q[1*dk+d],kv,vl);
  a2=__riscv_vfwmacc_vf_f32m2(a2,q[2*dk+d],kv,vl);
  a3=__riscv_vfwmacc_vf_f32m2(a3,q[3*dk+d],kv,vl);
  a4=__riscv_vfwmacc_vf_f32m2(a4,q[4*dk+d],kv,vl);
  a5=__riscv_vfwmacc_vf_f32m2(a5,q[5*dk+d],kv,vl);
  a6=__riscv_vfwmacc_vf_f32m2(a6,q[6*dk+d],kv,vl);
  a7=__riscv_vfwmacc_vf_f32m2(a7,q[7*dk+d],kv,vl);
  a8=__riscv_vfwmacc_vf_f32m2(a8,q[8*dk+d],kv,vl);
  a9=__riscv_vfwmacc_vf_f32m2(a9,q[9*dk+d],kv,vl);
  a10=__riscv_vfwmacc_vf_f32m2(a10,q[10*dk+d],kv,vl);
  a11=__riscv_vfwmacc_vf_f32m2(a11,q[11*dk+d],kv,vl);
 }
 __riscv_vse32_v_f32m2(dst+0*stride,a0,vl);
 __riscv_vse32_v_f32m2(dst+1*stride,a1,vl);
 __riscv_vse32_v_f32m2(dst+2*stride,a2,vl);
 __riscv_vse32_v_f32m2(dst+3*stride,a3,vl);
 __riscv_vse32_v_f32m2(dst+4*stride,a4,vl);
 __riscv_vse32_v_f32m2(dst+5*stride,a5,vl);
 __riscv_vse32_v_f32m2(dst+6*stride,a6,vl);
 __riscv_vse32_v_f32m2(dst+7*stride,a7,vl);
 __riscv_vse32_v_f32m2(dst+8*stride,a8,vl);
 __riscv_vse32_v_f32m2(dst+9*stride,a9,vl);
 __riscv_vse32_v_f32m2(dst+10*stride,a10,vl);
 __riscv_vse32_v_f32m2(dst+11*stride,a11,vl);
}
inline void pv12(float*dst,const float*p,const _Float16*v,int dv,int stride,int count){
 for(int off=0;off<dv;){const size_t vl=__riscv_vsetvl_e16m1(dv-off);
 auto a0=__riscv_vle32_v_f32m2(dst+0*dv+off,vl);
 auto a1=__riscv_vle32_v_f32m2(dst+1*dv+off,vl);
 auto a2=__riscv_vle32_v_f32m2(dst+2*dv+off,vl);
 auto a3=__riscv_vle32_v_f32m2(dst+3*dv+off,vl);
 auto a4=__riscv_vle32_v_f32m2(dst+4*dv+off,vl);
 auto a5=__riscv_vle32_v_f32m2(dst+5*dv+off,vl);
 auto a6=__riscv_vle32_v_f32m2(dst+6*dv+off,vl);
 auto a7=__riscv_vle32_v_f32m2(dst+7*dv+off,vl);
 auto a8=__riscv_vle32_v_f32m2(dst+8*dv+off,vl);
 auto a9=__riscv_vle32_v_f32m2(dst+9*dv+off,vl);
 auto a10=__riscv_vle32_v_f32m2(dst+10*dv+off,vl);
 auto a11=__riscv_vle32_v_f32m2(dst+11*dv+off,vl);
 for(int t=0;t<count;++t){const auto vv=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(v+t*dv+off,vl),vl);
  a0=__riscv_vfmacc_vf_f32m2(a0,p[0*stride+t],vv,vl);
  a1=__riscv_vfmacc_vf_f32m2(a1,p[1*stride+t],vv,vl);
  a2=__riscv_vfmacc_vf_f32m2(a2,p[2*stride+t],vv,vl);
  a3=__riscv_vfmacc_vf_f32m2(a3,p[3*stride+t],vv,vl);
  a4=__riscv_vfmacc_vf_f32m2(a4,p[4*stride+t],vv,vl);
  a5=__riscv_vfmacc_vf_f32m2(a5,p[5*stride+t],vv,vl);
  a6=__riscv_vfmacc_vf_f32m2(a6,p[6*stride+t],vv,vl);
  a7=__riscv_vfmacc_vf_f32m2(a7,p[7*stride+t],vv,vl);
  a8=__riscv_vfmacc_vf_f32m2(a8,p[8*stride+t],vv,vl);
  a9=__riscv_vfmacc_vf_f32m2(a9,p[9*stride+t],vv,vl);
  a10=__riscv_vfmacc_vf_f32m2(a10,p[10*stride+t],vv,vl);
  a11=__riscv_vfmacc_vf_f32m2(a11,p[11*stride+t],vv,vl);
 }
 __riscv_vse32_v_f32m2(dst+0*dv+off,a0,vl);
 __riscv_vse32_v_f32m2(dst+1*dv+off,a1,vl);
 __riscv_vse32_v_f32m2(dst+2*dv+off,a2,vl);
 __riscv_vse32_v_f32m2(dst+3*dv+off,a3,vl);
 __riscv_vse32_v_f32m2(dst+4*dv+off,a4,vl);
 __riscv_vse32_v_f32m2(dst+5*dv+off,a5,vl);
 __riscv_vse32_v_f32m2(dst+6*dv+off,a6,vl);
 __riscv_vse32_v_f32m2(dst+7*dv+off,a7,vl);
 __riscv_vse32_v_f32m2(dst+8*dv+off,a8,vl);
 __riscv_vse32_v_f32m2(dst+9*dv+off,a9,vl);
 __riscv_vse32_v_f32m2(dst+10*dv+off,a10,vl);
 __riscv_vse32_v_f32m2(dst+11*dv+off,a11,vl);
 off+=vl;
 }
}
}
