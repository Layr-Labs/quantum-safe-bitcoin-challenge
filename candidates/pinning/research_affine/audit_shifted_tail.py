#!/usr/bin/env python3
"""Execute actual shifted-tail C++ header over exact field shims + OpenSSL.

The independent oracle uses affine Python secp256k1 operations and hashlib.
No CUDA execution or performance measurement is claimed.
"""
from pathlib import Path
import ctypes,hashlib,json,random,subprocess,sys
ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'_build_shifted_audit';BUILD.mkdir(exist_ok=True)
CPP=r'''
#include <boost/multiprecision/cpp_int.hpp>
#include "../shifted_tail.cuh"
using boost::multiprecision::cpp_int;
static const cpp_int fp=(cpp_int(1)<<256)-(cpp_int(1)<<32)-977;
static cpp_int load(const uint64_t *x){cpp_int v=0;for(int i=3;i>=0;i--){v<<=64;v+=x[i];}return v;}
static void save(uint64_t *o,cpp_int v){v%=fp;if(v<0)v+=fp;for(int i=0;i<4;i++){o[i]=(uint64_t)(v&((cpp_int(1)<<64)-1));v>>=64;}}
struct Field {
 static void sub(uint64_t*o,const uint64_t*a,const uint64_t*b){save(o,load(a)-load(b));}
 static void mul(uint64_t*o,const uint64_t*a,const uint64_t*b){save(o,load(a)*load(b));}
 static void sqr(uint64_t*o,const uint64_t*a){save(o,load(a)*load(a));}
 static void neg(uint64_t*o,const uint64_t*a){save(o,-load(a));}
 static uint32_t parity_product(const uint64_t*a,const uint64_t*b,const uint64_t*c,uint32_t neg){
  cpp_int v=(load(a)*load(b)+load(c))%fp;if(neg&&v!=0)v=fp-v;return (uint32_t)(v&1);
 }
};
extern "C" int build(QsbShiftedTailRecord *o,const uint8_t *nri,const uint8_t *rx,const uint8_t *ry,unsigned first,unsigned count){return qsb_build_shifted_tail(o,nri,rx,ry,first,count);}
extern "C" int evaluate(uint64_t *xs,uint32_t *par,const QsbShiftedTailRecord*record,const QsbShiftedTailPoint*prefix,unsigned negative){
 QsbShiftedTailPoint tail[2];qsb_shifted_tail_load<Field>(tail,record,0,negative);
 uint64_t den[4],delta[2][4],num[2][4],inverse[4];
 if(!qsb_shifted_tail_prepare<Field>(den,delta,num,*prefix,tail))return 0;
 save(inverse,boost::multiprecision::powm(load(den),fp-2,fp));
 *par=qsb_shifted_tail_finish<Field>((uint64_t(*)[4])xs,inverse,delta,num,*prefix,tail);
 return 1;
}
extern "C" void load_signed(QsbShiftedTailPoint *o,const QsbShiftedTailRecord*r,unsigned neg){qsb_shifted_tail_load<Field>(o,r,0,neg);}
'''
(BUILD/'audit.cpp').write_text(CPP)
subprocess.run(['rtk','proxy','g++','-O2','-shared','-fPIC','-Wno-deprecated-declarations',
 '-I/tmp/qsb-boost-audit/usr/include',str(BUILD/'audit.cpp'),'-lcrypto','-o',str(BUILD/'audit.so')],check=True)
lib=ctypes.CDLL(str(BUILD/'audit.so'))
U64=ctypes.c_uint64;U32=ctypes.c_uint32;U8=ctypes.c_uint8
lib.build.argtypes=[ctypes.POINTER(U64),ctypes.POINTER(U8),ctypes.POINTER(U8),ctypes.POINTER(U8),U32,U32]
lib.evaluate.argtypes=[ctypes.POINTER(U64),ctypes.POINTER(U32),ctypes.POINTER(U64),ctypes.POINTER(U64),U32]
lib.load_signed.argtypes=[ctypes.POINTER(U64),ctypes.POINTER(U64),U32]
P=2**256-2**32-977
N=0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
G=(0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798,
   0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8)
def add(a,b):
 if a is None:return b
 if b is None:return a
 x,y=a;u,v=b
 if x==u:
  if (y+v)%P==0:return None
  slope=3*x*x*pow(2*y,-1,P)%P
 else:slope=(v-y)*pow(u-x,-1,P)%P
 nx=(slope*slope-x-u)%P
 return nx,(slope*(x-nx)-y)%P
def neg(a):return None if a is None else (a[0],(-a[1])%P)
def mul(k,a=G):
 k%=N;r=None
 while k:
  if k&1:r=add(r,a)
  a=add(a,a);k>>=1
 return r
def words(n):return [(n>>(64*i))&((1<<64)-1) for i in range(4)]
def pack(q):return (U64*8)(*([0]*8 if q is None else words(q[0])+words(q[1])))
def unpack(a,start=0):
 x=sum(int(a[start+i])<<(64*i) for i in range(4));y=sum(int(a[start+4+i])<<(64*i) for i in range(4))
 return None if x==0 and y==0 else (x,y)
def bytes32(n):return (U8*32).from_buffer_copy(n.to_bytes(32,'little'))
def build(nri,r,first,count):
 a=(U64*(16*count))();assert lib.build(a,bytes32(nri),bytes32(r[0]),bytes32(r[1]),first,count)
 return a
def record(a,j):return (U64*16)(*a[16*j:16*j+16])
def digest(q):return hashlib.sha256(bytes([2+(q[1]&1)])+q[0].to_bytes(32,'big')).digest()
def recode(k):
 d=2*(k%N)-N;digits=[];shift=0
 for c in range(14):
  width=18 if c==0 else 17
  v=d%(1<<(width+1))-(1<<width);digits.append((v,shift));d=(d-v)>>width;shift+=width
 digits.append((d,shift));assert shift==239
 assert sum(v*(1<<s) for v,s in digits)==2*(k%N)-N
 assert all(v&1 for v,_ in digits)
 return digits

rng=random.Random(0x5348494654454454)
counts={'table_points':0,'signed_loads':0,'key_pairs':0,'hashes':0,'recoder_cases':0,
        'singular_declines':0,'infinity_declines':0,'priority_checks':0,'builder_chunk_crossings':0}
pool=[G]
for _ in range(255):pool.append(add(pool[-1],G))
def check(prefix,l,r,rec,negative):
 loaded=(U64*16)();lib.load_signed(loaded,rec,negative)
 signed_l=neg(l) if negative else l
 expected_tail=[add(signed_l,r),add(signed_l,neg(r))]
 assert [unpack(loaded,0),unpack(loaded,8)]==expected_tail
 counts['signed_loads']+=1
 xs=(U64*8)();par=U32()
 ok=lib.evaluate(xs,ctypes.byref(par),rec,pack(prefix),negative)
 singular=prefix is None or any(t is None or t[0]==prefix[0] for t in expected_tail)
 if singular:
  assert not ok;counts['singular_declines']+=1;return
 assert ok
 direct=[add(add(prefix,signed_l),r),add(add(prefix,signed_l),neg(r))]
 got_hashes=[];want_hashes=[]
 for arm in range(2):
  x=sum(int(xs[4*arm+i])<<(64*i) for i in range(4));parity=(par.value>>arm)&1
  assert x==direct[arm][0] and parity==(direct[arm][1]&1)
  got=hashlib.sha256(bytes([2+parity])+x.to_bytes(32,'big')).digest()
  want=digest(direct[arm]);assert got==want
  got_hashes.append(got);want_hashes.append(want);counts['hashes']+=1
 # Easier gate exercises first-success recid priority without changing the
 # actual full digests used above; production N is not modified by this audit.
 for mask in (1,3,7,15):
  got=next((i for i,h in enumerate(got_hashes) if h[0]&mask==0),None)
  want=next((i for i,h in enumerate(want_hashes) if h[0]&mask==0),None)
  assert got==want;counts['priority_checks']+=1
 counts['key_pairs']+=1

for trial in range(3):
 nri=rng.randrange(1,N);r=mul(rng.randrange(1,N));b=mul(nri*pow(2,238,N)%N)
 for first,count in ((0,17),(32760,17),(65519,17)):
  table=build(nri,r,first,count);l=mul(2*first+1,b);step=add(b,b)
  for j in range(count):
   rec=record(table,j)
   assert unpack(rec)==add(l,r) and unpack(rec,8)==add(l,neg(r));counts['table_points']+=2
   for negative in (0,1):check(pool[rng.randrange(len(pool))],l,r,rec,negative)
   l=add(l,step)
 # Full scalar reconstruction: prefix derives from the first14 signeddigits,
 # while independent expected recovery uses the full scalar z*nri.
 edge=[0,1,2,N//2-1,N//2,N//2+1,N-2,N-1,N,N+1,2**256-1]
 for k in edge+[rng.getrandbits(256) for _ in range(64)]:
  digits=recode(k);last=digits[-1][0];index=(abs(last)-1)//2;negative=int(last<0)
  rec=record(build(nri,r,index,1),0);l=mul((2*index+1),b)
  prefix_scalar=sum(v*(1<<s) for v,s in digits[:-1])*pow(2,-1,N)*nri%N
  prefix=mul(prefix_scalar);assert add(prefix,neg(l) if negative else l)==mul(k*nri)
  check(prefix,l,r,rec,negative);counts['recoder_cases']+=1
 # All zero denominator cases must leave the fast path for an exact fallback.
 rec=record(build(nri,r,0,1),0)
 for prefix in (None,unpack(rec),neg(unpack(rec,8))):check(prefix,b,r,rec,0)

# Exercise actual point batching across the builder's 1024-entry boundary.
# A nonzero start also checks continuation is relative to the requested slice.
nri=rng.randrange(1,N);r=mul(rng.randrange(1,N));b=mul(nri*pow(2,238,N)%N)
first,count=83,1031;table=build(nri,r,first,count);l=mul(2*first+1,b);step=add(b,b)
for j in range(count):
 rec=record(table,j)
 assert unpack(rec)==add(l,r) and unpack(rec,8)==add(l,neg(r));counts['table_points']+=2
 if j in (0,1022,1023,1024,1025,1030):
  for negative in (0,1):check(pool[rng.randrange(len(pool))],l,r,rec,negative)
 l=add(l,step)
counts['builder_chunk_crossings']+=1

# Build a real infinity table record deliberately: L_0-R=O.
nri=17;b=mul(nri*pow(2,238,N)%N);rec=record(build(nri,b,0,1),0)
assert unpack(rec,8) is None
for negative in (0,1):
 xs=(U64*8)();par=U32();assert not lib.evaluate(xs,ctypes.byref(par),rec,pack(G),negative)
 counts['infinity_declines']+=1
result={'header_sha256':hashlib.sha256((ROOT/'shifted_tail.cuh').read_bytes()).hexdigest(),
 'checks':counts,'mismatches':0,'GPU_execution':False,
 'scope':'Actual host table builder, signed loader, denominator and finish C++ bodies; exact field-policy shims; independent affine Python recovery and SHA256.'}
(ROOT/'shifted_tail_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
