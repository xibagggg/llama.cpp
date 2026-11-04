from pathlib import Path
import os,paramiko
ROOT=Path(__file__).resolve().parents[1];SRC=Path('Y:/llama.cpp-tool-spacemit-mini-CPM-OCR/ocr-adaptation/managed-source')
c=paramiko.SSHClient();c.load_host_keys(str(Path.home()/'.ssh/known_hosts'));c.connect('10.0.50.15',username='huangjingtao',password=os.environ['OCR_BUILD_HOST_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=15)
try:
 for name in ['ocr-fa-pack-v40.h','rvv_kernels.cpp']:
  cmd='cd /home/huangjingtao/llama.cpp-tool-spacemit-mini-CPM-OCR/ocr-adaptation/managed-source && git show 03fe87e:ggml/src/ggml-spacemit/'+name
  _,out,err=c.exec_command(cmd);s=out.read().decode();assert out.channel.recv_exit_status()==0,err.read().decode()
  (SRC/'ggml/src/ggml-spacemit'/name).write_text(s,encoding='utf8',newline='\n')
finally:c.close()
p=SRC/'ggml/src/ggml-spacemit/ocr-qk-ime2-v53.h';s=p.read_text(encoding='utf8')
s=s.replace('64*D*2+64*D*2+256*4','64*D*2+64*D*2+512*4')
pos=s.index('static bool qk(')
code='''// Two query panels reuse the four K panels in each K8 iteration.
__attribute__((noinline)) static void core16(const _Float16*x,const _Float16*w,float*c){
 const _Float16*x1=x+8*D,*w1=w+8*D,*w2=w+16*D,*w3=w+24*D;int blocks=D/8;
 asm volatile(
 "vsetvli t0,zero,e32,m2,ta,ma\\n\\t"
'''
for reg in range(16,32,2):code+=f' "vmv.v.i v{reg},0\\n\\t"\n'
code+=' "vsetvli t0,zero,e16,m1,ta,ma\\n\\t1:\\n\\t"\n'
for reg,var in [(2,'x'),(3,'x1'),(8,'w'),(9,'w1'),(10,'w2'),(11,'w3')]:code+=f' "vle16.v v{reg},(%[{var}])\\n\\t"\n'
for qr,qreg in [(0,2),(1,3)]:
 for g in range(4):code+=f' "smt.vfwmadot v{16+qr*8+g*2},v{qreg},v{8+g}\\n\\t"\n'
for var in ['x','x1','w','w1','w2','w3']:code+=f' "addi %[{var}],%[{var}],128\\n\\t"\n'
code+=' "addi %[blocks],%[blocks],-1\\n\\tbnez %[blocks],1b\\n\\t"\n "vsetvli t0,zero,e32,m2,ta,ma\\n\\t"\n'
for g in range(8):
 code+=f' "addi t0,%[c],{g*256}\\n\\tvse32.v v{16+g*2},(t0)\\n\\t"\n'
code+=''' :[x]"+&r"(x),[x1]"+&r"(x1),[w]"+&r"(w),[w1]"+&r"(w1),[w2]"+&r"(w2),[w3]"+&r"(w3),[blocks]"+&r"(blocks)
 :[c]"r"(c):"t0","v2","v3","v8","v9","v10","v11","v16","v17","v18","v19","v20","v21","v22","v23","v24","v25","v26","v27","v28","v29","v30","v31","memory");
}
'''
s=s[:pos]+code+s[pos:]
s=s.replace('for(int q=0;q<rows;q+=8)for(int key=0;key<count;key+=32){\n  core(packed_q+q*D,pk+key*D,tmp);',
 '''for(int q=0;q<rows;q+=16)for(int key=0;key<count;key+=32){
  const int qr=std::min(16,rows-q);
  if(qr>8)core16(packed_q+q*D,pk+key*D,tmp);
  else core(packed_q+q*D,pk+key*D,tmp);''')
s=s.replace('for(int r=0;r<8&&q+r<rows;++r){\n   auto values=__riscv_vluxei32_v_f32m1(tmp+r*8,idx,columns);',
 '''for(int r=0;r<qr;++r){
   auto values=__riscv_vluxei32_v_f32m1(tmp+(r/8)*256+(r%8)*8,idx,columns);''')
assert 'core16(packed_q' in s and 'r<qr' in s
p.write_text(s,encoding='utf8',newline='\n')
base=ROOT/'scripts';t=(base/'build_attention_v59.sh').read_text().replace('v59','v61').replace('isolate vectorized IME2 QK on v50 with shape and mask fallback','reuse IME2 key panels across sixteen attention queries')
(base/'build_attention_v61.sh').write_text(t,encoding='utf8',newline='\n')
for name in ['deploy_fa_compute_v59.py','run_qk_ime2_v59_micro_board.sh']:
 t=(base/name).read_text().replace('v59','v61');(base/name.replace('v59','v61')).write_text(t,encoding='utf8',newline='\n')
