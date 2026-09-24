from pathlib import Path
import subprocess,json,itertools,random,math,hashlib,struct,tempfile,os,collections
inc=Path(__file__).resolve().parent.parent/'gpu_epochs'
TEMP=tempfile.TemporaryDirectory(prefix='qsb-coverage-')
root=Path(TEMP.name)
schedule=(inc/'window_schedule_shared.cuh').read_text().split('/* First-block states')[0]
(root/'schedule_host.cuh').write_text(schedule)
source=r'''
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define __device__
#define __constant__
#define QSB_SE_WINDOWS 128
#define QSB_SE_PER_EPOCH 128
#define cudaSuccess 0
#define QSB_TO_SYMBOL(dst,src,n) (memcpy(&(dst),src,n),0)
#define QSB_FROM_SYMBOL(dst,src,n) (memcpy(dst,&(src),n),0)
uint32_t K[64]={};
static uint32_t qsb_host_rotr(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
#include "schedule_host.cuh"
int main(){
 (void)qsb_window_first_key; (void)qsb_window_second_key;
 uint8_t w[256][3];
 if(!qsb_select_window_family(w,127,0)||!qsb_select_window_family(w,128,-1)||!qsb_select_window_family(w,128,2)||!qsb_select_window_family(w,256,1)||!qsb_select_window_family(NULL,128,0))return 9;
 for(int mode=0;mode<3;mode++){
  int count=mode==2?256:128,phase=mode==1;
  if(qsb_select_window_family(w,count,phase))return 8;
  for(int i=0;i<count;i++)printf("W %d %u %u %u\n",mode,w[i][0],w[i][1],w[i][2]);
 }
 uint8_t rows[1500];uint32_t constants[5];
 for(unsigned seed=0;seed<3;seed++){
  uint32_t r=seed+17;
  for(int i=0;i<1500;i++){r=1664525u*r+1013904223u;rows[i]=r>>24;}
  for(int i=0;i<5;i++){r=1664525u*r+1013904223u;constants[i]=r;}
  for(int phase: {0,1,0}){
   if(qsb_select_window_family(w,128,phase))return 7;
   // No poisoning/reset between families: detect incomplete replacement of live values.
   if(qsb_prepare_window_schedule(rows,w,constants))return 6;
   for(int lane=0;lane<128;lane++){
    printf("S %u %d %d %u %u",seed,phase,lane,QSB_FIRST_CLASS[lane],QSB_WINDOW_CLASS[lane]);
    for(int i=0;i<14;i++)printf(" %u",QSB_FIRST_UNIQUE[i][QSB_FIRST_CLASS[lane]]);
    for(int i=0;i<64;i++)printf(" %u",QSB_WINDOW_SECOND[i][QSB_WINDOW_CLASS[lane]]);
    printf("\n");
   }
  }
 }
}
'''.replace('#include <stdint.h>','#include <stdint.h>\n#include <initializer_list>')
(root/'host.cpp').write_text(source)
subprocess.run([os.environ.get('CXX','c++'),'-std=c++11','-O2','-Wall','-Wextra','-Werror','-I',str(inc),str(root/'host.cpp'),'-o',str(root/'host')],check=True)
lines=subprocess.check_output([str(root/'host')],text=True).splitlines()
actual=[[],[],[]]
for line in lines:
 if line.startswith('W '):
  mode,*triple=map(int,line.split()[1:]);actual[mode].append(tuple(v-137 for v in triple))
def fk(w):return tuple(i for i in range(13) if i not in w)[:6]
def sk(w):return tuple(i for i in range(12,-1,-1) if i not in w)[:5]
pool=list(itertools.combinations(range(13),3))
A=[w for w in pool if w[0]>=6 or (w[0]<=5 and w[1]>=7) or (w[0]==0 and w[1]==1 and 8<=w[2]<=10)]
groups=collections.defaultdict(list)
for w in pool:
 if w not in A:groups[fk(w)].append(w)
B=[w for group in sorted(groups.values(),key=lambda g:(-len(g),fk(g[0]))) for w in group][:128]
assert actual[0]==sorted(A,key=lambda w:(sk(w),fk(w)))
assert actual[1]==sorted(B,key=lambda w:(sk(w),fk(w)))
old=[w for w in itertools.combinations(range(13),3) if not(w[0]>=1 and w[2]<=7 and w[:2]!=(1,2))]
assert actual[2]==sorted(old,key=lambda w:(sk(w),fk(w)))
assert not set(actual[0])&set(actual[1])
counts=[len({fk(w) for w in ws}) for ws in actual];assert counts==[8,47,54]
def rotr(x,n):return ((x>>n)|(x<<(32-n)))&0xffffffff
n_schedule=0
for line in lines:
 if not line.startswith('S '):continue
 seed,phase,lane,fc,sc,*words=map(int,line.split()[1:]);r=seed+17;rows=[]
 for i in range(1500):r=(1664525*r+1013904223)&0xffffffff;rows.append(r>>24)
 const=[]
 for i in range(5):r=(1664525*r+1013904223)&0xffffffff;const.append(r)
 w=actual[phase][lane];pending=b'\x00'*8+b''.join(bytes(rows[10*(137+i):10*(138+i)]) for i in range(13) if i not in w)+struct.pack('>5I',*const)
 expected=list(struct.unpack('>32I',pending));expanded=expected[16:]
 for i in range(16,64):
  x,y=expanded[i-15],expanded[i-2]
  expanded.append((expanded[i-16]+(rotr(x,7)^rotr(x,18)^(x>>3))+expanded[i-7]+(rotr(y,17)^rotr(y,19)^(y>>10)))&0xffffffff)
 assert words==expected[2:16]+expanded
 assert 0<=fc<counts[phase] and 0<=sc<128
 n_schedule+=1
# Exhaustive reduced domains and varied capacities, including one/odd tails.
# Mirrors only loop/index arithmetic; the source review establishes its binding to kernels.
tested=0
for nepoch in list(range(1,34))+[63,64,65,129]:
 for cap in [1,2,3,7,8,16,32,64]:
  seen=set();published=0
  for phase in [0,1]:
   base=0
   while base<nepoch:
    batch=min(cap,nepoch-base);pair_mul=4;nblk=(batch+pair_mul-1)//pair_mul
    tentative=[]
    for block in range(nblk):
     for tid in range(256):
      lane=tid&127;eA=pair_mul*block+2*(tid//128)
      for side in [0,1]:
       e=eA+side
       if eA>=batch or e>=batch:continue
       publish_epoch=pair_mul*block+2*(tid//128)+side
       tag=publish_epoch*128+lane
       assert tag//128==e and tag%128==lane
       identity=(base+e,actual[phase][lane])
       assert identity not in seen;seen.add(identity);tentative.append(identity)
    # Replay+publish completes before any family replacement; no old records remain live.
    published+=len(tentative);base+=batch
   assert base==nepoch
  assert published==len(seen)==nepoch*256;tested+=1
n=math.comb(137,6);capacity=1048576
result={'status':'PASS','gpu_executed':False,'selector_triples':sum(map(len,actual)),'invalid_cases':5,'first_classes':counts,'actual_host_schedule_lane_checks':n_schedule,'transition_tail_models':tested,'real_epochs':n,'real_capacity':capacity,'real_tail':n%capacity,'total_disjoint_candidates':n*256,'selector_sha256':hashlib.sha256((inc/'window_families.h').read_bytes()).hexdigest()}
print(json.dumps(result,indent=2));TEMP.cleanup()
