import sys, random, hashlib, struct, inspect, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'harness'))
import crypto as C
rng=random.Random(20260916)
# Reuse reference SHA compression with an explicit starting state.
src=inspect.getsource(C.sha256_midstate).replace('def sha256_midstate(data: bytes):','def resume(data, initial):').replace('h = list(_H0)','h = list(initial)')
ns=dict(vars(C));exec(src,ns);resume=ns['resume']
sha_cases=0
for seed in range(4):
 prefix=bytes(rng.getrandbits(8) for _ in range(9920))
 suffix=bytes(rng.getrandbits(8) for _ in range(75))
 mid=C.sha256_midstate(prefix)
 for seq in [0,1,0x80000000,0xffffffff,rng.getrandbits(32)]:
  block=bytearray(suffix[:64]);block[31:35]=seq.to_bytes(4,'little')
  state=resume(bytes(block),mid)
  for lt in [0,1,255,256,0xffffffff,500000000,1744599999]+[rng.getrandbits(32) for _ in range(10)]:
   tail=bytearray(64);tail[:11]=suffix[64:];tail[11]=128;tail[56:]=(9995*8).to_bytes(8,'big')
   w=list(struct.unpack('>16I',tail));be=int.from_bytes(lt.to_bytes(4,'little'),'big')
   w[0]=(w[0]&0xffffff00)|(be>>24);w[1]=(w[1]&255)|((be<<8)&0xffffffff)
   first=resume(struct.pack('>16I',*w),state)
   b=struct.pack('>8I',*first)+b'\x80'+bytes(23)+(256).to_bytes(8,'big')
   final=resume(b,C._H0); got=struct.pack('>8I',*final)
   actual=bytearray(suffix);actual[31:35]=seq.to_bytes(4,'little');actual[67:71]=lt.to_bytes(4,'little')
   assert got==C.sha256d(prefix+actual)
   z=[(final[6-2*i]<<32)|final[7-2*i] for i in range(4)]
   assert sum(x<<(64*i) for i,x in enumerate(z))==int.from_bytes(got,'big')
   sha_cases+=1

def mixed(p,q):
 X,Y,Z=p;x,y=q
 u=(y*Z-Y)%C.P;v=(x*Z-X)%C.P
 v2=v*v%C.P;v3=v2*v%C.P;vx=v2*X%C.P
 a=(u*u*Z-v3-2*vx)%C.P
 return (v*a%C.P,(u*(vx-a)-v3*Y)%C.P,v3*Z%C.P)
def affine(p):
 X,Y,Z=p;assert Z
 inv=pow(Z,-1,C.P);return X*inv%C.P,Y*inv%C.P

def digit(z,j):
 limbs=[(z>>(64*i))&((1<<64)-1) for i in range(4)]
 bit=j*14;word=bit>>6;shift=bit&63
 bits=limbs[word]>>shift
 if shift>50 and word<3: bits|=limbs[word+1]<<(64-shift)
 return bits&16383
scalar_cases=[0,1,C.N-1,C.N,(1<<256)-1]+[1<<i for i in range(256)]+[rng.getrandbits(256) for _ in range(10000)]
for z in scalar_cases:
 assert sum(digit(z,j)<<(14*j) for j in range(19))==z
 assert digit(z,18)<16
# Sample actual table values, independently built with the affine reference.
ec_cases=0
for case in range(12):
 a=rng.randrange(1,C.N);h=C.point_mul(a);R=C.point_mul(rng.randrange(1,C.N));neg2=C.point_mul(C.N-2,R)
 z=scalar_cases[case] if case<5 else rng.getrandbits(256)
 p=C.point_add(R,C.point_mul(digit(z,0),h));q=(p[0],p[1],1)
 base=h
 for j in range(1,19):
  base=C.point_mul(1<<14,base)
  if digit(z,j):q=mixed(q,C.point_mul(digit(z,j),base))
 q2=mixed(q,neg2)
 prod_inv=pow(q[2]*q2[2]%C.P,-1,C.P)
 got1=(q[0]*prod_inv*q2[2]%C.P,q[1]*prod_inv*q2[2]%C.P)
 got2=(q2[0]*prod_inv*q[2]%C.P,q2[1]*prod_inv*q[2]%C.P)
 P=C.point_mul(z*a%C.N)
 assert got1==C.point_add(P,R)
 assert got2==C.point_add(P,(R[0],-R[1]%C.P))
 ec_cases+=1
# Compressed public-key words and padding exactly match the byte definition.
for _ in range(1000):
 x=rng.getrandbits(256); y=rng.getrandbits(256);limbs=[(x>>(64*i))&((1<<64)-1) for i in range(4)]
 pb=[((2+(y&1))<<24)|(limbs[3]>>40)]
 for i in range(1,8):
  bit=264-32*(i+1);limb=bit//64;shift=bit%64;v=limbs[limb]>>shift
  if shift>32:v|=limbs[limb+1]<<(64-shift)
  pb.append(v&0xffffffff)
 pb.append(((limbs[0]<<24)&0xffffffff)|0x00800000);pb += [0]*6+[264]
 raw=bytes([2+(y&1)])+x.to_bytes(32,'big')
 assert struct.pack('>16I',*pb)==raw+b'\x80'+bytes(22)+(264).to_bytes(8,'big')
 # Direct word gate vs the verifier for every bit difficulty.
 hs=list(struct.unpack('>8I',hashlib.sha256(raw).digest()))
 for n in [0,1,8,24,31,32,33,63,64,128,255,256]:
  hit=all(v==0 for v in hs[:n//32]) and (n%32==0 or hs[n//32]>>(32-n%32)==0)
  assert hit==C.is_hit(hashlib.sha256(raw).digest(),n)
result={'sha256d_cases':sha_cases,'window_decompositions':len(scalar_cases),'full_recovery_cases':ec_cases,'pubkey_encodings':1000,'gate_comparisons':12000,'status':'PASS','limitation':'Mathematical reference checks only; CUDA compilation and GPU execution delegated to official runner.'}
print(json.dumps(result,indent=2))

