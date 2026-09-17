#!/usr/bin/env python3
"""Execute immutable HM43 C++ with32 cooperative CPU lanes; no CUDA execution."""
import argparse,hashlib,itertools,json,random,re,subprocess,sys,tempfile,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
DONOR=ROOT/'research/pending_sep17/pr189/head/tests/gpu_epochs/hm43_warp_inverse.cuh'
MATH=ROOT/'research/weak_field/ordinary_region/candidate/GPUMath.h'
P=2**256-2**32-977;U=2**64;MASK=U-1;MOD=2**384
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
words=lambda x,n:[(x>>(64*j))&MASK for j in range(n)]
CPP=r'''
#include "host_support.hpp"
#include "hm43_warp_inverse.cuh"
#include <iostream>
#include <vector>
struct Case { int kind; uint64_t a[32],b[32],expected[32][5]; };
int main(){
 size_t count;if(!(std::cin>>count))return 4;std::vector<Case> cases(count);
 for(auto& c:cases){std::cin>>c.kind;for(auto& v:c.a)std::cin>>v;for(auto& v:c.b)std::cin>>v;for(auto& row:c.expected)for(auto& v:row)std::cin>>v;}
 if(!std::cin)return 5;
 qsb_hm43_host::Warp warp;std::atomic<bool> failed{false};std::atomic<unsigned long long> checked{0};
 std::vector<std::thread> threads;
 for(int lane=0;lane<32;lane++)threads.emplace_back([&,lane]{
  qsb_hm43_host::current_warp=&warp;qsb_hm43_host::current_lane=lane;
  for(size_t ci=0;ci<count;ci++){
   const Case& c=cases[ci];warp.join(lane);uint64_t out[5]={0,0,0,0,0};
   if(c.kind==0)out[0]=hm43_add(c.a[lane],c.b[lane],lane);
   if(c.kind==1)out[0]=hm43_negate(c.a[lane],lane);
   if(c.kind==2)out[0]=hm43_signed_mul(c.a[lane],(int64_t)c.b[lane],lane);
   if(c.kind==3)out[0]=hm43_multiple_p(c.b[lane],lane&7);
   if(c.kind==4){
    for(int j=0;j<5;j++)out[j]=lane==0?c.a[j]:(0xdeadbeefULL+(uint64_t)lane*31+j);
    hm43_warp_inverse(out,lane);
   }
   const bool inspect=c.kind>=3 || (lane&7)<6;
   if(inspect)for(int j=0;j<(c.kind==4?5:1);j++){
    if(out[j]!=c.expected[lane][j]){
     if(!failed.exchange(true))fprintf(stderr,"NUMERIC_MISMATCH case=%zu kind=%d lane=%d word=%d expected=%016llx actual=%016llx\n",ci,c.kind,lane,j,(unsigned long long)c.expected[lane][j],(unsigned long long)out[j]);
    }else checked++;
   }
   warp.join(lane);
   if(lane==0 && (ci%64==0 || ci+1==count)){fprintf(stderr,"PROGRESS completed=%zu/%zu collectives=%llu\n",ci+1,count,(unsigned long long)warp.generation);fflush(stderr);}
   if(failed.load())break;
  }
 });
 for(auto& t:threads)t.join();
 if(failed.load())return 2;
 for(int lane=0;lane<32;lane++)if(warp.lane_calls[lane]!=warp.generation)return 6;
 printf("{\"status\":\"PASS_ACTUAL_COOPERATIVE_HOST_HM43\",\"cases\":%zu,\"checked_words\":%llu,\"collectives\":%llu,\"exchanges\":%llu,\"ballots\":%llu,\"harness_joins\":%llu,\"participation_mask\":\"%08x\",\"all32_lane_counts_equal\":true}\n",count,checked.load(),(unsigned long long)warp.generation,(unsigned long long)warp.exchanges,(unsigned long long)warp.ballots,(unsigned long long)warp.joins,warp.observed_mask);
}
'''
def fixtures():
 rng=random.Random(0x43189);cases=[];counts={str(i):0 for i in range(5)}
 def append(kind,a,b,expect):cases.append((kind,a,b,expect));counts[str(kind)]+=1
 def unit(kind,groups,bs):
  a=[];b=[];exp=[]
  for group,x in enumerate(groups):
   aw=words(x,6)+[MASK,MASK];v=bs[group]
   a+=aw;b+=[v&MASK]*8
   if kind==0:e=(x+v)%MOD
   elif kind==1:e=(-x)%MOD
   elif kind==2:e=x*v%MOD
   else:e=v*P
   exp+=words(e,6)+[0,0]
  append(kind,a,b,[[v,0,0,0,0] for v in exp])
 # Exhaust all6word generate/propagate/kill patterns. Guards intentionally
 # contain allones: they must never transmit carry between live groups.
 for pattern in itertools.product(range(3),repeat=6):
  a=[];b=[];e=[]
  for g in range(4):
   aa=[];bb=[]
   for t in pattern:
    x,y=[(0,0),(MASK,0),(MASK,1)][t];aa.append(x);bb.append(y)
   a+=aa+[MASK,MASK];b+=bb+[0,0]
   v=sum(x<<(64*j) for j,x in enumerate(aa))+sum(x<<(64*j) for j,x in enumerate(bb))
   e+=words(v%MOD,6)+[0,0]
  append(0,a,b,[[x,0,0,0,0] for x in e])
 for _ in range(128):
  xs=[rng.getrandbits(384) for g in range(4)];bs=[rng.getrandbits(384) for g in range(4)]
  a=sum([words(x,6)+[MASK,MASK] for x in xs],[]);b=sum([words(x,6)+[0,0] for x in bs],[])
  e=sum([words((x+y)%MOD,6)+[0,0] for x,y in zip(xs,bs)],[])
  append(0,a,b,[[v,0,0,0,0] for v in e])
 for i in range(160):
  xs=[rng.getrandbits(384) for _ in range(4)]
  if i<6:xs=[0,1,1<<(64*i),MOD-1]
  unit(1,xs,[0]*4)
  signed_values=[rng.randrange(-2**319,2**319) for _ in range(4)]
  coefs=[rng.randrange(-2**62,2**62) for _ in range(4)]
  if i<4:coefs=[0,1,-1,-(2**62-1)]
  unit(2,[x%MOD for x in signed_values],coefs)
  unit(3,[0]*4,[0,1,2**62-1,rng.getrandbits(62)] if i<4 else [rng.getrandbits(62) for _ in range(4)])
 roots=[0,1,2,P-1,P,P+1,2**256-1]+[1<<i for i in [32,63,64,127,128,191,192,255]]+[P-d for d in [65537,2**32,2**64,2**128]]+[rng.randrange(1,P) for _ in range(96)]
 for root in roots:
  inv=pow(root%P,-1,P) if root%P else 0
  append(4,words(root,5)+[0]*27,[0]*32,[words(inv,5) for _ in range(32)])
 text=str(len(cases))+'\n'+'\n'.join(' '.join(map(str,[kind]+a+b+sum(e,[]))) for kind,a,b,e in cases)+'\n'
 return text,counts

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=HERE/'helper-results.json');args=parser.parse_args()
 assert sha(DONOR)=='940f4c6ad4272a71d38533123a809c8f1affc7a9e0beca959db48380f24cd1fc'
 donor=DONOR.read_text();math=MATH.read_text();blocks=[]
 for name in ['MM64','MSK62','_IsNegative(x)','AddP(r)','SubP(r)']:
  m=re.search(r'^#define '+re.escape(name)+r'[^\n]*(?:\\\n[^\n]*)*',math,re.M)
  # Explicit continuation walk keeps actual complete macro definitions.
  start=math.index('#define '+name+' ');lines=math[start:].splitlines();block=[]
  for line in lines:
   block.append(line)
   if not line.endswith('\\'):break
  blocks.append('\n'.join(block))
 projection='// Exact macros extracted from corrected3ac GPUMath; scalar PTX ops projected in host_support.hpp.\n'+'\n'.join(blocks)+'\n'
 (HERE/'field_projection.hpp').write_text(projection)
 data,counts=fixtures();reports={}
 mutants={'positive':donor,'carry_dropped':donor.replace('return sum+((carry>>lane)&1u);','return sum;'),
  'negate_word5_omitted':donor.replace('return ~x+(uint64_t)((zeros&lower)==lower);','return (lane&7)==5 ? x : ~x+(uint64_t)((zeros&lower)==lower);'),
  'guard_propagation_leak':donor.replace('word<5 && sum<a','word<8 && sum<a').replace('word<6 && sum==~0ULL','word<8 && sum==~0ULL')}
 assert len(set(mutants.values()))==4
 logs=HERE/'helper-artifacts';logs.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='qsb-hm43-host-') as td:
  td=Path(td)
  for name,header in mutants.items():
   folder=td/name;folder.mkdir();(folder/'hm43_warp_inverse.cuh').write_text(header);(folder/'check.cpp').write_text(CPP)
   executable=folder/'check';cmd=['c++','-std=c++17','-O1','-pthread','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I',str(HERE),str(folder/'check.cpp'),'-o',str(executable)]
   built=subprocess.run(cmd,text=True,capture_output=True,timeout=60)
   assert built.returncode==0,(name,built.stderr)
   started=time.monotonic()
   with (logs/(name+'.stdout')).open('w') as out,(logs/(name+'.stderr')).open('w') as err:
    try:
     proc=subprocess.run([str(executable)],input=data,text=True,stdout=out,stderr=err,timeout=180)
     rc=proc.returncode;timed=False
    except subprocess.TimeoutExpired:rc=None;timed=True
   stderr=(logs/(name+'.stderr')).read_text();stdout=(logs/(name+'.stdout')).read_text()
   if name=='positive':assert rc==0,(name,rc,timed,stderr[-3000:]);result=json.loads(stdout)
   else:assert rc==2 and 'NUMERIC_MISMATCH' in stderr,(name,rc,timed,stderr[-3000:]);result=None
   assert 'runtime error:' not in stderr and 'ERROR: AddressSanitizer' not in stderr
   reports[name]={'compiled':True,'exit_code':rc,'timeout':timed,'elapsed_host_seconds_not_performance_proxy':round(time.monotonic()-started,3),'source_sha256':hashlib.sha256(header.encode()).hexdigest(),'result':result,'diagnostic':next((line for line in stderr.splitlines() if 'MISMATCH' in line),None),'stdout_sha256':sha(logs/(name+'.stdout')),'stderr_sha256':sha(logs/(name+'.stderr'))}
   print(name,rc,result or reports[name]['diagnostic'],flush=True)
 result={'status':'PASS_ACTUAL_COOPERATIVE_HOST_HM43_AND_COMPILED_MUTATIONS','header_sha256':sha(DONOR),'corrected_dependency_source_sha256':sha(MATH),'field_projection_sha256':sha(HERE/'field_projection.hpp'),'host_warp_sha256':sha(HERE/'host_warp.hpp'),'host_support_sha256':sha(HERE/'host_support.hpp'),'checker_sha256':sha(Path(__file__)),'generated_cpp_sha256':hashlib.sha256(CPP.encode()).hexdigest(),'fixture_sha256':hashlib.sha256(data.encode()).hexdigest(),'case_counts':dict(zip(['six_word_add','six_word_negate','signed_coefficient_product','sparse_m_times_p','complete_inverse'],[counts[str(i)] for i in range(5)])),'inverse_outputs_checked':32*counts['4'],'reports':reports,'sanitizers':'AddressSanitizer and UndefinedBehaviorSanitizer, all4 successfully compiled executables','collective_contract':'32hostthreads; full participation and identical collective kind per ordinal; every lane has same total count;10s per-rendezvous timeout includes epoch/arrival mask,180s process limit and persistent progress logs. Guard lanes participate.','arithmetic_projection':'Actual immutable HM43 header uses HM43_HOST_ORACLE __uint128 mulhi; actual AddP/SubP/macros from corrected3ac, their UADD/USUB primitives and CTZ/clz are scalar semantic replacements. Expected results from independent Python bigint.','limits':['No CUDA/PTX execution or device scheduling proof','Finite cases do not prove universal delayed-divstep signed bounds','No GPU throughput or native resource qualification','Helper only; EC192 tree integration belongs to separate check'],'candidate_modified':False,'gpu_executed':False,'native_cuda_built':False}
 args.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
