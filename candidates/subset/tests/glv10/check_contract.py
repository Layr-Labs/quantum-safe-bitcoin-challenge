from pathlib import Path
from fractions import Fraction as F
import ctypes, subprocess, random, json, hashlib, tempfile, os, shutil, argparse, re
SOURCE=Path(__file__).resolve().parents[2]
args=argparse.ArgumentParser(description='Independent six-bank GLV12 contract against the literal header')
args.add_argument('--geometry-header',type=Path,default=SOURCE/'glv10_geometry.h')
args=args.parse_args()
TEMP=tempfile.TemporaryDirectory(prefix="qsb-glv10-contract-")
root=Path(TEMP.name)
shutil.copy(args.geometry_header,root/"glv10_geometry.h")
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
D=10659985; S=104
shifts=[0,18,37,56,80,104]; sizes=[262144,262144,262144,8388608,8388608,5329993]
offsets=[0,262144,524288,6116425,14505033,786432]; bias=((D+1)<<103)-(1<<17)
physical=[0,1,2,5,3,4]
spans=[(offsets[c],offsets[c]+sizes[c],c) for c in physical]
assert spans[0][0]==0 and spans[-1][1]==sum(sizes)==22893641
assert all(spans[i][1]==spans[i+1][0] for i in range(5))
assert sum(sizes[:3])*64==48*2**20
# Exact interval/lattice assertions, covering all legal magnitudes.
e1=F(abs(G1*N-A*2**384),N*2**128); e2=F(abs(G2*N-B*2**384),N*2**128)
r1=A*(F(1,2)+e1)+C*(F(1,2)+e2); r2=B*(F(1,2)+e1)+A*(F(1,2)+e2)
assert A*A+B*C==N and (A-LAM*B)%N==0 and (C+LAM*A)%N==0
assert r1<BOUND+1 and r2<BOUND+1
lm=(A*2**128+C*r2)/N; ln=(B*2**128+A*r2)/N
assert lm<1 and ln<1
assert 0<(D-1)*2**103 < (D+3)*2**103 < 2**128
assert (D+1)*2**104 < 2**128
qzero=F(N,B)*(F(1,2)+e2)
assert r2 < D*2**104 and qzero < D*2**104
# Compile literal proposed helper and challenge all bit/carry boundaries with both signs.
(root/'export.cpp').write_text(r'''#include "glv10_geometry.h"
extern "C" unsigned code(uint64_t a,uint64_t b,unsigned sign,int c){uint64_t m[2]={a,b};return glv10_code(m,sign,c);}
extern "C" uint64_t geometry(int c,int part){return part==0?glv10_entries(c):part==1?glv10_offset(c):glv10_shift(c);}
extern "C" int bank(uint64_t record){return glv10_bank_for_record(record);}
extern "C" uint64_t constant(int i){const uint64_t values[]={GLV10_CHUNKS,GLV10_TERMS,GLV10_LOW_BITS,GLV10_LADDER,GLV10_RECORDS,GLV10_BYTES};return values[i];}
''')
subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++14','-shared',str(root/'export.cpp'),'-o',str(root/'helper.dylib')],check=True)
lib=ctypes.CDLL(str(root/'helper.dylib'));f=lib.code
lib.geometry.argtypes=[ctypes.c_int,ctypes.c_int];lib.geometry.restype=ctypes.c_uint64
lib.constant.argtypes=[ctypes.c_int];lib.constant.restype=ctypes.c_uint64
for c in range(6):
    assert [lib.geometry(c,i) for i in range(3)]==[sizes[c],offsets[c],shifts[c]]
expected_constants=[6,12,12,4096,22893641,1465193024]
assert [lib.constant(i) for i in range(len(expected_constants))]==expected_constants
lib.bank.argtypes=[ctypes.c_uint64];lib.bank.restype=ctypes.c_int
bank_boundaries=sorted({r for start,end,c in spans for edge in [start,end] for r in [edge-1,edge,edge+1] if 0<=r<sum(sizes)})
for r in bank_boundaries:
    expected=next(c for c in range(6) if offsets[c]<=r<offsets[c]+sizes[c])
    assert lib.bank(r)==expected,(r,lib.bank(r),expected)
for invalid in [sum(sizes),sum(sizes)+1,2**32-1,2**64-1]: assert lib.bank(invalid)==-1
assert sum(sizes)%256==73
assert [m>>12 for m in [sizes[0]-1]+[2*x-1 for x in sizes[1:]]]==[63,127,127,4095,4095,2602]
assert 0<bias-(2**18-4096) and bias+4095+(2**18-4096)<N
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
fixture_text=(SOURCE/'glv10_audit_scalars.h').read_text()
fixture_limbs=[int(x,16) for x in re.findall(r'UINT64_C\(0x([0-9a-fA-F]+)\)',fixture_text)]
assert len(fixture_limbs)==87*4
fixture_scalars=[sum(fixture_limbs[i+j]<<(64*j) for j in range(4)) for i in range(0,len(fixture_limbs),4)]
assert len(set(fixture_scalars))==87
fallback_counts=[]
for reciprocal,threshold in [(G1,0x7ffffffc),(G2,0x7ffffffd)]:
    hits=0
    for z in fixture_scalars:
        k=z%N;aa=[(k>>(32*i))&0xffffffff for i in range(8)];bb=[(reciprocal>>(32*i))&0xffffffff for i in range(8)]
        kept=sum(aa[i]*bb[j]<<(32*(i+j)) for i in range(8) for j in range(8) if i+j>=10)
        word=(kept>>352)&0xffffffff
        hits+=threshold<=word<0x80000000
    assert hits>=6
    fallback_counts.append(hits)
scalarcases+=fixture_scalars
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
result={'status':'PASS','gpu_executed':False,'records':sum(sizes),'physical_bank_order':physical,'literal_constants_checked':len(expected_constants),'record_bank_boundaries_checked':len(bank_boundaries),'full_table_tail_live':73,'full_table_tail_pad':183,'header_sha256':hashlib.sha256((root/'glv10_geometry.h').read_bytes()).hexdigest(),'split_sign_classes':len(split_classes),'production_scalar_fixtures':len(fixture_scalars),'forced_fallback_counts':fallback_counts,'compiled_header_codes':len(values)*12,'point_cases':len(scalarcases),'recid_checks':2*len(scalarcases),'lattice_bounds':[float(lm),float(ln)],'qzero_bound':float(qzero/2**128)}
print(json.dumps(result,indent=2));TEMP.cleanup()
