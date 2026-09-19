#!/usr/bin/env python3
"""Pure Python audit of GLV integer representation, actual PTX and SHA schedule.
No native compilation, device execution or performance evidence.
"""
from pathlib import Path
import re,random,json,hashlib
from ptx_field_model import Program,function,extract_ptx,check_semantics
HERE=Path(__file__).resolve().parent
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
LAM=0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
P=(1<<256)-(1<<32)-977;MASK=(1<<32)-1

def constants(text,name,bits):
 m=re.search(r'const uint(?:32|64)_t '+name+r'\[\d+\]=\{([^}]+)\}',text)
 v=[int(re.sub(r'[uUlL]+$','',x.strip()),0) for x in m[1].split(',')]
 return sum(x<<(bits*i) for i,x in enumerate(v))

def small(a,b,W):
 aa=[(a>>(32*i))&MASK for i in range(4)];bb=[(b>>(32*i))&MASK for i in range(W)];rr=[0]*9
 for i in range(4):
  carry=0
  for j in range(W):
   t=aa[i]*bb[j]+rr[i+j]+carry;rr[i+j]=t&MASK;carry=t>>32
  rr[i+W]=carry
 return sum(rr[i]<<(32*i) for i in range(8))

def code_sum(x):
 mag=abs(x);sign=-1 if x<0 else 1;ordinary=0;top=0
 for ch in range(8):
  bits=9 if ch==7 else 17;f=(mag>>(17*ch))&((1<<bits)-1)
  if ch==0:ordinary+=sign*((1<<127)-(1<<16)+f)
  else:
   neg=1-(f>>(bits-1));idx=(f^-neg)&((1<<(bits-1))-1)
   value=sign*(-1 if neg else 1)*(2*idx+1)*(1<<(17*ch-1))
   if ch==7:top=value
   else:ordinary+=value
 assert ordinary+top==x
 return ordinary,top

def glv(rng):
 text=(HERE/'GLVScalar.cuh').read_text();g1=constants(text,'g1',64);g2=constants(text,'g2',64)
 a1=constants(text,'a1',32);a2=constants(text,'a2',32);b1=constants(text,'b1',32)
 assert a1*a1+a2*b1==N and (a1-LAM*b1)%N==0 and (a2+LAM*a1)%N==0
 q=Program(extract_ptx(function(text,'void q9_wide(')))
 limbs=lambda x:[(x>>(64*i))&((1<<64)-1) for i in range(4)]
 vectors=[0,1,2,N-1,N,(1<<256)-1]+[rng.getrandbits(256) for _ in range(1500)]
 for x in vectors:
  y=rng.getrandbits(256);v=q.run(limbs(x)+limbs(y),input_base=8,outputs=8)
  assert sum(x<<(64*i) for i,x in enumerate(v))==x*y
 edge=[0,1,2,N-1,N,N+1,(1<<256)-1,LAM,LAM-1,LAM+1]
 for i in range(256):edge.extend([(1<<i)-1,1<<i,(1<<i)+1])
 vals=edge+[rng.getrandbits(256) for _ in range(100000)]
 for raw in vals:
  k=raw%N;c1=(k*g1+(1<<383))>>384;c2=(k*g2+(1<<383))>>384
  for c,a,W in [(c1,a1,4),(c2,a2,5),(c1,b1,4),(c2,a1,4)]:assert small(c,a,W)==c*a% (1<<256)
  x=k-c1*a1-c2*a2;y=c1*b1-c2*a1
  assert abs(x)<1<<128 and abs(y)<1<<128 and (x+LAM*y-k)%N==0
  px,jx=code_sum(x);py,jy=code_sum(y);assert (px+LAM*py+jx+LAM*jy-k)%N==0
 return dict(wide_product_ptx_cases=len(vectors),split_and_table_reconstruction_cases=len(vals))

ROR=lambda x,n:((x>>n)|(x<<(32-n)))&MASK
s0=lambda x:ROR(x,7)^ROR(x,18)^(x>>3)
s1=lambda x:ROR(x,17)^ROR(x,19)^(x>>10)
S0=lambda x:ROR(x,2)^ROR(x,13)^ROR(x,22)
S1=lambda x:ROR(x,6)^ROR(x,11)^ROR(x,25)

def sha_model():
 src=(HERE/'pinning.cu').read_text();a=src.index('__device__ __forceinline__ void _SHA256TransformDigest32(');body=src[a:src.index('\n}',a)+2]
 body=re.sub(r'/\*[\s\S]*?\*/|//[^\n]*','',body)
 body=re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b',r'\1',body)
 gpu=(HERE/'GPUHash.h').read_text();K=[int(x,16) for x in re.findall(r'0x[0-9a-fA-F]+',gpu[gpu.index('uint32_t K[]'):gpu.index('};',gpu.index('uint32_t K[]'))])]
 assert len(K)==64
 rounds=re.findall(r'S2Round\(([^;]+)\);',body);assert len(rounds)==16
 assigns=re.findall(r'w\[(\d+)\]\s*(\+=|=)\s*([^;]+);',body);assert len(assigns)==16
 macro=(HERE/'sha_schedule_interleaved.cuh').read_text();macro=macro[macro.index('#define QSB_SHA_INTERLEAVED_16'):]
 rotations=re.findall(r'QSB_SHA_SCHEDULE_STEP\((\d+),([a-h,]+),base\)',macro);assert len(rotations)==16
 assert [int(j) for j,_ in rotations]==list(range(16))
 assert body.count('QSB_SHA_INTERLEAVED_16(')==2
 assert 'SHA256_RND(16)' in body
 init={a:int(b,0) for a,b in re.findall(r'uint32_t ([a-h]) = (0x[0-9a-f]+);',body)}
 assert len(init)==8
 outputs=re.findall(r'out\[(\d+)\] = (0x[0-9a-f]+) \+ ([a-h]);',body);assert len(outputs)==8
 def run(msg):
  env=dict(init);env.update(w=[None]*16,K=K,s0=s0,s1=s1)
  env['w'][:8]=[int.from_bytes(msg[i:i+4],'big') for i in range(0,32,4)]
  ev=lambda x:eval(x,{'__builtins__':{}},env)
  def step(names,k,w):
   a,b,c,d,e,f,g,h=names;av,bv,cv,dv,evv,fv,gv,hv=[env[x] for x in names]
   t1=(hv+S1(evv)+(gv^(evv&(fv^gv)))+k+w)&MASK;t2=(S0(av)+((av&bv)|(cv&(av|bv))))&MASK
   env[d]=(dv+t1)&MASK;env[h]=(t1+t2)&MASK
  for call in rounds:
   args=[x.strip() for x in call.split(',')];step(args[:8],ev(args[8]),ev(args[9]))
  for j,op,expr in assigns:
   j=int(j);env['w'][j]=((env['w'][j] if op=='+=' else 0)+ev(expr))&MASK
  for j,names in rotations:step(names.split(','),K[16+int(j)],env['w'][int(j)])
  for base in (32,48):
   for j,names in rotations:
    j=int(j);w=env['w'];w[j]=(w[j]+s1(w[(j+14)&15])+w[(j+9)&15]+s0(w[(j+1)&15]))&MASK
    step(names.split(','),K[base+j],w[j])
  out=[0]*8
  for i,iv,v in outputs:out[int(i)]=(int(iv,0)+env[v])&MASK
  return b''.join(x.to_bytes(4,'big') for x in out)
 return run

def boundary(rng):
 B=1<<256
 rawvals=[0,1,P-1,P,P+1,B-1];avals=[0,1,P-1,P-2,B-(1<<192)-1,B-(1<<192),B-2*(B-P)-1]
 pairs=[(x,a) for x in rawvals for a in avals if 0<=a<P]+[(rng.randrange(B),rng.randrange(P)) for _ in range(100000)]
 for x,a in pairs:
  raw=x%P if a>>192==(1<<64)-1 else x
  total=raw+a;assert total<2*P
  candidate=total-P if total>=P else total
  assert candidate==(x%P+a)%P
 return len(pairs)

def main():
 rng=random.Random(541543534);check_semantics();result=glv(rng)
 fn=sha_model();msgs=[bytes(32),bytes([255])*32]+[(1<<i).to_bytes(32,'big') for i in range(256)]+[rng.randbytes(32) for _ in range(1024)]
 for m in msgs:assert fn(m)==hashlib.sha256(m).digest()
 result.update(sha_messages=len(msgs),boundary_add_cases=boundary(rng),status='PASS',native_compilation=False,gpu_measured=False)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
