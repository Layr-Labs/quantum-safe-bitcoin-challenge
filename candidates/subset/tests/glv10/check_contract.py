from pathlib import Path
from fractions import Fraction as F
import ctypes, subprocess, random, json, hashlib, tempfile, os, shutil
SOURCE=Path(__file__).resolve().parents[2]
TEMP=tempfile.TemporaryDirectory(prefix="qsb-glv10-contract-")
root=Path(TEMP.name)
shutil.copy(SOURCE/"glv10_geometry.h",root/"glv10_geometry.h")
N=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
P=2**256-2**32-977
A=0x3086D221A7D46BCDE86C90E49284EB15
B=0xE4437ED6010E88286F547FA90ABFE4C3
C=A+B
LAM=0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
BETA=0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
G1=0x3086D221A7D46BCDE86C90E49284EB153DAA8A1471E8CA7FE893209A45DBB031
G2=0xE4437ED6010E88286F547FA90ABFE4C4221208AC9DF506C61571B4AE8AC47F71
BOUND=0xA2A8918CA85BAFE22016D0B917E4DD77
D=42639943; S=102
shifts=[0,24,50,76,102]; sizes=[16777216,33554432,33554432,33554432,21319972]
offsets=[sum(sizes[:i]) for i in range(5)]; bias=((D+1)<<101)-(1<<23)
# Independent integer dynamic programming, no balanced-exponent assumption.
dp={0:(0,[])}
for c in range(4):
    nxt={}
    for total,(cost,widths) in dp.items():
        for w in range(1,128-total):
            candidate=(cost+(1<<(w if c==0 else w-1)),widths+[w])
            if total+w not in nxt or candidate[0]<nxt[total+w][0]: nxt[total+w]=candidate
    dp=nxt
best=min((cost+(((BOUND>>s)|1)+1)//2,s,widths) for s,(cost,widths) in dp.items())
assert best[0]==sum(sizes) and best[1]==102
# Exact interval/lattice assertions, covering all legal magnitudes.
e1=F(abs(G1*N-A*2**384),N*2**128); e2=F(abs(G2*N-B*2**384),N*2**128)
r1=A*(F(1,2)+e1)+C*(F(1,2)+e2); r2=B*(F(1,2)+e1)+A*(F(1,2)+e2)
assert A*A+B*C==N and (A-LAM*B)%N==0 and (C+LAM*A)%N==0
assert r1<BOUND+1 and r2<BOUND+1
lm=(A*2**128+C*r2)/N; ln=(B*2**128+A*r2)/N
assert lm<1 and ln<1
assert 0<(D-1)*2**101 < (D+3)*2**101 < 2**128
assert (D+1)*2**102 < 2**128
qzero=F(N,B)*(F(1,2)+e2)
assert r2 < D*2**102 and qzero < D*2**102
# Compile literal proposed helper and challenge all bit/carry boundaries with both signs.
(root/'export.cpp').write_text('#include "glv10_geometry.h"\nextern "C" unsigned code(uint64_t a,uint64_t b,unsigned sign,int c){uint64_t m[2]={a,b};return glv10_code(m,sign,c);}\n')
subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++11','-shared',str(root/'export.cpp'),'-o',str(root/'helper.dylib')],check=True)
f=ctypes.CDLL(str(root/'helper.dylib')).code
f.argtypes=[ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint,ctypes.c_int]; f.restype=ctypes.c_uint
rng=random.Random(409024)
values={0,1,BOUND-1,BOUND}
for bit in range(129):
    for center in [1<<bit,BOUND-((1<<bit)-1), (BOUND>>bit)<<bit]:
        for delta in [-2,-1,0,1,2]:
            if 0<=center+delta<=BOUND: values.add(center+delta)
values.update(rng.randrange(BOUND+1) for _ in range(30000))
def terms(m,negative):
    out=[]
    for i,s in enumerate(shifts):
        code=f(m&((1<<64)-1),m>>64,negative,i)
        index=(code&0x7fffffff)-offsets[i]
        assert 0<=index<sizes[i]
        t=bias+index if i==0 else (2*index+1)<<(s-1)
        out.append(-t if code>>31 else t)
    assert sum(out)==(-m if negative else m)
    prefix=out[0]
    for t in out[1:]:
        assert abs(prefix+t)<2**128 and abs(prefix-t)<2**128
        prefix+=t
    return out
for m in sorted(values):
    for sign in [0,1]: terms(m,sign)
# Complete affine independent group oracle; check recid +/- C outputs for new geometry.
def add(p,q):
    if p is None:return q
    if q is None:return p
    x,y=p; xx,yy=q
    if x==xx:
        if (y+yy)%P==0:return None
        slope=3*x*x*pow(2*y,-1,P)%P
    else:slope=(yy-y)*pow(xx-x,-1,P)%P
    X=(slope*slope-x-xx)%P
    return X,(slope*(x-X)-y)%P
G=(0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
def mul(k):
    k%=N; p=None; q=G
    while k:
        if k&1:p=add(p,q)
        q=add(q,q);k>>=1
    return p
def chain(r,initial=None):
    acc=initial
    for t in terms(abs(r),int(r<0)):
        point=mul(t)
        if acc is not None:assert acc[0]!=point[0], 'incomplete-add exception'
        acc=add(acc,point)
    return acc
scalarcases=[0,1,2,N-1,N,N+1,A,B,C]+[(p+LAM*q)%N for p in [-1,0,1] for q in [-1,0,1]]+[rng.randrange(N) for _ in range(40)]
split_classes=set()
for z in scalarcases:
    z%=N; c1=(z*G1+2**383)>>384;c2=(z*G2+2**383)>>384
    p=z-c1*A-c2*C;q=c1*B-c2*A
    split_classes.add(((p>0)-(p<0),(q>0)-(q<0)))
    if not z:actual=None
    elif q:
        acc=chain(q); acc=(acc[0]*BETA%P,acc[1]); actual=chain(p,acc)
    else:actual=chain(p)
    expected=mul(z);assert actual==expected
    fixed=mul(rng.randrange(1,N))
    for recid in [0,1]:
        offset=(fixed[0],(-fixed[1])%P) if recid else fixed
        assert add(actual,offset)==add(expected,offset)
assert len(split_classes)==9
GiB=2**30
mem={}
for name,table in [('minimum',sum(sizes)*64),('dense19',18376368960)]:
    for stack in [32768,8192,1024]:
        mem[f'{name}_stack{stack}_phaseA_GiB']=(table+972*2**20+128*1536*stack)/GiB
        mem[f'{name}_stack{stack}_phaseAB_GiB']=(table+972*2**20+1536*2**20+128*1536*stack)/GiB
result={'status':'PASS','gpu_executed':False,'minimum_records':best[0],'minimum_widths':best[2],'header_sha256':hashlib.sha256((root/'glv10_geometry.h').read_bytes()).hexdigest(),'split_sign_classes':len(split_classes),'compiled_header_codes':len(values)*10,'point_cases':len(scalarcases),'recid_checks':2*len(scalarcases),'lattice_bounds':[float(lm),float(ln)],'qzero_bound':float(qzero/2**128),'memory_models':mem}
print(json.dumps(result,indent=2));TEMP.cleanup()
