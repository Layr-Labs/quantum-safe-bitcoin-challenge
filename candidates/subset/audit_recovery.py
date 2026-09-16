#!/usr/bin/env python3
"""Pure Python integration/edge models; no compiler, GPU or OpenSSL calls."""
from pathlib import Path
import hashlib,random,re
from audit_deferred_chain import P,G,affine_add,scalar_mult
from ptx_field_model import Program,extract_ptx,function,check_semantics
R=Path(__file__).resolve().parent
MASK=(1<<256)-1
def inverse_model(values):
    """Mirror the composed shuffle + fused shared schedule, modulo p.

    A phase is separated by each CUDA barrier. Ownership is checked in each
    fused phase: all source leaves belong to the same unique owner.
    """
    size=len(values)
    assert size in (32,64,128,256)
    n=size//4
    tree=[None]*(2*n)
    ops=barriers=0
    def mul(a,b):
        nonlocal ops
        ops+=1
        return a*b%P
    sibling=[values[i^1] for i in range(size)]
    pair=[0]*size
    for i in range(0,size,2):pair[i]=mul(values[i],sibling[i])
    other=[pair[(i&~1)^2] for i in range(size)]
    for i in range(0,size,4):tree[n+i//4]=mul(pair[i],other[i])
    barriers+=1
    for tid in range(n//4):
        node=n//4+tid;leaf=4*node
        a=mul(tree[leaf],tree[leaf+1])
        b=mul(tree[leaf+2],tree[leaf+3])
        tree[2*node]=a;tree[2*node+1]=b
        tree[node]=mul(a,b)
    barriers+=1
    width=n//8
    while width:
        for tid in range(width):
            node=width+tid
            tree[node]=mul(tree[2*node],tree[2*node+1])
        barriers+=1;width//=2
    tree[1]=pow(tree[1],-1,P);barriers+=1
    width=1
    while width<n//4:
        for tid in range(width):
            node=width+tid
            parent,left,right=tree[node],tree[2*node],tree[2*node+1]
            tree[2*node]=mul(parent,right)
            tree[2*node+1]=mul(parent,left)
        barriers+=1;width*=2
    for tid in range(n//4):
        node=n//4+tid;leaf=4*node
        parent,left,right=tree[node],tree[2*node],tree[2*node+1]
        right,left=mul(parent,right),mul(parent,left)
        a,b,c,d=tree[leaf:leaf+4]
        tree[leaf]=mul(right,b);tree[leaf+1]=mul(right,a)
        tree[leaf+2]=mul(left,d);tree[leaf+3]=mul(left,c)
    barriers+=1
    for i in range(0,size,2):pair[i]=mul(tree[n+i//4],other[i])
    result=[mul(pair[i&~1],sibling[i]) for i in range(size)]
    assert ops==3*size-3,(size,ops)
    assert barriers==2*(size.bit_length()-1)-4,(size,barriers)
    return result



def limbs(a): return [(a>>(64*i))&((1<<64)-1) for i in range(4)]
def packed(a): return sum(x<<(64*i) for i,x in enumerate(a))
def finish(X,Y,A,B,xR,yR):
 d=(xR*A-X)%P; W=A*A*d%P
 if W==0:return None
 C=A*d*d%P;inv=pow(W,-1,P);h=B*inv%P
 xs=(2*xR-C*inv)%P
 m=(yR*B-Y)*h%P;x1=(m*m-xs)%P;y1=(m*(xR-x1)-yR)%P
 m=(yR*B+Y)*h%P;x2=(m*m-xs)%P;y2=(yR-m*(xR-x2))%P
 return (x1,y1),(x2,y2)
def perm(a,b,s):
 data=a.to_bytes(4,'little')+b.to_bytes(4,'little')
 return sum(data[(s>>(4*i))&7]<<(8*i) for i in range(4))
def keyblock(x,parity):
 w=[(x>>(32*i))&0xffffffff for i in range(8)]
 out=[perm(w[7],2+parity,0x4321)]
 out += [perm(w[i],w[i-1],0x0765) for i in range(7,0,-1)]
 out += [perm(w[0],0x80,0x0456)]+[0]*6+[264]
 return b''.join(v.to_bytes(4,'big') for v in out)
def main():
 rng=random.Random(625363);s=(R/'tests/gpu_epochs/tree.cu').read_text();m=(R/'GPUMath.h').read_text()
 check_semantics()
 mul=Program(extract_ptx(function(m,'void _ModMultCore(')))
 sqr=Program(extract_ptx(function((R/'square32.cuh').read_text(),'void qsb_square32(')))
 cases=[0,1,2,P-1,P-2,P-65537,(1<<256)-1]+[P-i for i in range(1,2049)]+[rng.getrandbits(256) for _ in range(4000)]
 for a in cases:
  for b in (a,P-1,rng.getrandbits(256)):
   raw=packed(mul.run(limbs(a)+limbs(b)));assert raw%P==a*b%P
  assert packed(sqr.run(limbs(a)))%P==a*a%P
 assert 'addc.cc.u32 z7, z7, 0;' in extract_ptx(function(m,'void _ModMultCore('))
 # Removing the new final fold must reproduce the old boundary defect.
 text=extract_ptx(function(m,'void _ModMultCore('))
 text=text[:text.index('.reg .u32 cf;')]+text[text.index('mov.b64 %0, {z0,z1};'):]
 a=P-65537
 assert packed(Program(text).run(limbs(a)*2))%P!=a*a%P
 # Exact regular curve pairs rescaled into valid XYZZ, both recovery IDs.
 points=[scalar_mult(i+1) for i in range(128)]
 for i in range(2500):
  Q=points[rng.randrange(128)];T=points[rng.randrange(128)];z=rng.randrange(1,P);A=z*z%P;B=A*z%P
  got=finish(Q[0]*A%P,Q[1]*B%P,A,B,*T)
  if Q[0]==T[0]:assert got is None
  else:assert got==(affine_add(Q,T),affine_add(Q,(T[0],-T[1]%P)))
 # Parity includes zero, whose negation must not flip bit.
 for a in cases:
  c=a%P;assert ((-c)%P)&1 == (0 if c==0 else (c&1)^1)
 for size in (32,64,128,256):
  for _ in range(20):
   vals=[1 if rng.randrange(5)==0 else rng.randrange(1,P) for i in range(size)]
   assert inverse_model(vals)==[pow(x,-1,P) for x in vals]
 for _ in range(5000):
  x=rng.randrange(P);bit=rng.randrange(2);block=keyblock(x,bit)
  assert block==bytes([2+bit])+x.to_bytes(32,'big')+b'\x80'+bytes(22)+(264).to_bytes(8,'big')
 for y in cases:
  y=y%(P-1)+1
  for neg in (0,1):
   mask=MASK if neg else 0
   assert ((y^mask)+(mask&(P+1)))&MASK==(P-y if neg else y)
 # Source-shape checks of dispatch, liveness and finite normalization batches.
 assert s.count('kernel_digest<true><<<')==1 and s.count('kernel_digest<false><<<')==2
 assert s.count('_PointAddXYZZ<true>')==s.count('_PointAddXYZZ<false>')==1
 assert '&& !calibrate && single_hash' in s and 'bool field_zero' in s
 assert s.index('qsb_block_inverse_tree(prod);')<s.index('if(!usable)return;')
 assert '*neg = (uint64_t)(int64_t)(ec >> 31);' in s and 'uint64_t m=neg;' in s
 assert 'EC_POINTs_make_affine' in s and 'out+(size_t)(i+1)*8' in s
 assert 'uint32_t x32[8]' not in s and 'pts_x[' not in s
 assert 'uint32_t negative_parity = (s[0]|s[1]|s[2]|s[3]) ? ((uint32_t)(s[0]&1ULL)^1u) : 0u;' in function(s,'uint32_t qsb_xyzz_finish_precomputed(')
 N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
 for ch in range(15):
  base=pow(2,0 if ch==0 else 17*ch+1,N)
  for step,count in ((base,255),(base*256%N,1023 if ch==0 else 511)):
   old=step;points=[]
   for i in range(count):
    points.append(step if i==0 else (points[-1]+step)%N)
    assert points[i]==old;old=(old+step)%N
 seen=set()
 def include(p):
  p=p.resolve();assert p.is_relative_to(R)
  if p in seen:return
  seen.add(p)
  for name in re.findall(r'^\s*#include\s+"([^"]+)"',p.read_text(),re.M):include(p.parent/name)
 include(R/'subset.cu')
 print('PASS:',len(cases)*3,'PTX multiply;',len(cases),'PTX square; mutant rejected; 2500 recovery; 9600 inverse; 5000 key blocks;',len(cases)*2,'sign cases;12002 ladder coefficients;',len(seen),'includes')
if __name__=='__main__':main()
