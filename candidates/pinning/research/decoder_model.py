from pathlib import Path
import random,json
W=Path(__file__).resolve().parent;B=1<<256;mask32=(1<<32)-1;mask64=(1<<64)-1
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
rng=random.Random(202609202115)
def oldpack(f,bits,last,negative):
 tm=-negative if last else (f>>(bits-1))-1
 return ((f^tm)&((1<<(bits-1))-1))|((tm<0)<<31)
def pack(f,bits,last,negative):
 if last:
  tm=-negative;sign=negative<<31
 else:
  z=(~((f<<(32-bits))&mask32))&mask32
  tm=-(z>>31);sign=z&0x80000000
 return ((f^tm)&((1<<(bits-1))-1))|sign
checks=0
for bits,last in [(18,False),(17,False),(17,True)]:
 for f in range(1<<(16 if last else bits)):
  for neg in (range(2) if last else [0]):
   assert oldpack(f,bits,last,neg)==pack(f,bits,last,neg);checks+=1

def decode(M,negative,new):
 limbs=[M>>(64*i)&mask64 for i in range(4)];words=[M>>(32*i)&mask32 for i in range(8)];out=[]
 for c in range(15):
  pos=1 if c==0 else 17*c+2;bits=18 if c==0 else 17
  if new:
   wi,ws=divmod(pos,32);hi=words[(wi+1)&7] if wi<7 else 0
   f=((words[wi]>>ws)|(hi<<(32-ws)))&((1<<bits)-1)
  else:
   j,sh=divmod(pos,64);value=limbs[j]>>sh
   if j<3 and sh>46:value|=(limbs[j+1]<<(64-sh))&mask64
   f=value&((1<<bits)-1)
  out.append((pack if new else oldpack)(f,bits,c==14,negative))
 return out
scalars=[0,1,N//2,N//2+1,N-1,N,N+1,B-1]+[rng.getrandbits(256) for _ in range(4000)]
for k in scalars:
 t=k%N;D=2*t-N;M=D%(1<<256);negative=int(D<0)
 assert decode(M,negative,False)==decode(M,negative,True)
 # Registers for first2 and original offsets for remainder preserve all15 codes.
 codes=decode(M,negative,True);seed=codes[:2];shared={c:codes[c] for c in range(2,15)}
 assert seed+[shared[c] for c in range(2,15)]==decode(M,negative,False)
# Slices are bitwise linear; every basis bit covers the complete extraction map.
for M in [0,B-1]+[1<<i for i in range(256)]:
 for negative in [0,1]:assert decode(M,negative,False)==decode(M,negative,True)
r={'status':'PASS','exhaustive_packed_field_cases':checks,'raw_scalar_cases':len(scalars),'digit_comparisons':15*len(scalars),'basis_vectors_and_negatives':516,'shared_bytes_removed_per_candidate':16,'scope':'Independent proof/model of public6fe3a56 description only; no production edits, no donor source fetched, no native compile. Compiler may already fold shifts/signs; register lifetime from two seeds requires remote evidence. Hold small decoder component for meaningful later integration.'}
(W/'decoder-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
