#!/usr/bin/env python3
"""Compile the production recoder on CPU; compare independent integer scalars.

Does not execute CUDA, table point arithmetic, or measure GPU throughput.
"""
from fractions import Fraction
import json
from pathlib import Path
import random
import re
import struct
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
source=(HERE/'GLVScalar.cuh').read_text()
# Exact production helper block, including both geometry controls.
helper=source[:source.index('// QSB/VanitySearch')]
helper=helper.replace('#pragma once','')
cpp='''#include <cstdio>
#define __host__
#define __device__
#define __forceinline__ inline
'''+helper+r'''
int main(){
    uint64_t m[2];uint32_t sign;
    while(std::fread(m,sizeof(m),1,stdin)==1){
        if(std::fread(&sign,sizeof(sign),1,stdin)!=1)return 2;
        uint32_t out[6];for(int c=0;c<6;c++)out[c]=q9_bigtbl_code(m,sign,c);
        if(std::fwrite(out,sizeof(out),1,stdout)!=1)return 3;
    }
}
'''
N=0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
A=0x3086d221a7d46bcde86c90e49284eb15
B=0xe4437ed6010e88286f547fa90abfe4c3
C=A+B
assert A*A+B*C==N

def const(name):
    words=re.search(r'const uint64_t '+name+r'\[4\]=\{([^}]+)\}',source).group(1)
    return sum(int(v.strip().removesuffix('ULL'),16)<<(64*i) for i,v in enumerate(words.split(',')))
G1,G2=const('g1'),const('g2')
D=1<<384
bound1=abs(Fraction(D-G1*A-G2*C,D))*(N-1)+Fraction(A+C,2)
bound2=abs(Fraction(G1*B-G2*A,D))*(N-1)+Fraction(A+B,2)
bound=max(bound1,bound2).__ceil__()
assert bound==0xa2a8918ca85bafe22016d0b917e4dd77,hex(bound)

def split(k):
    k%=N;c1=(k*G1+(D>>1))//D;c2=(k*G2+(D>>1))//D
    r1=k-c1*A-c2*C;r2=c1*B-c2*A
    assert max(abs(r1),abs(r2))<=bound
    lam=(A*pow(B,-1,N))%N
    assert (r1+lam*r2-k)%N==0
    return r1,r2

rng=random.Random(0x4CA_C4E)
values={0,1,2,bound,bound-1,bound-2}
# Chunk boundaries, centered sign transitions, and all-bit carry patterns.
for i in range(128):
    for delta in (-2,-1,0,1,2):
        for m in [(1<<i)+delta,bound-(1<<i)+delta]:
            if 0<=m<=bound:values.add(m)
for shift,center in [(100,170559769),(104,10659985)]:
    for f in [0,1,center//2,center//2+1,center-1,center]:
        for low in [0,1,(1<<shift)-2,(1<<shift)-1]:
            m=(f<<shift)+low
            if m<=bound:values.add(m)
for _ in range(50000):
    for r in split(rng.randrange(N)):values.add(abs(r))
values.update(rng.randrange(bound+1) for _ in range(100000))
cases=[(m,s) for m in sorted(values) for s in [0,1]]
blob=b''.join(struct.pack('<QQI',m&((1<<64)-1),m>>64,s) for m,s in cases)
results=[]
with tempfile.TemporaryDirectory(prefix='qsb-fourhot-') as tmp:
    tmp=Path(tmp);p=tmp/'check.cpp';p.write_text(cpp)
    for mode in [0,1]:
        exe=tmp/f'check{mode}'
        subprocess.run(['g++','-O2','-std=c++17','-fsanitize=undefined','-fno-sanitize-recover=all',f'-DQSB_FOUR_HOT={mode}',str(p),'-o',str(exe)],check=True)
        out=subprocess.run([str(exe)],input=blob,stdout=subprocess.PIPE,check=True).stdout
        assert len(out)==24*len(cases)
        if mode:
            entries=[262144,262144,131072,131072,67108864,85279885]
            offsets=[0,262144,524288,655360,786432,67895296]
            shifts=[0,18,37,55,73,100];center=170559769;radix=16384
        else:
            entries=[262144,262144,262144,8388608,8388608,5329993]
            offsets=[0,262144,524288,6116425,14505033,786432]
            shifts=[0,18,37,56,80,104];center=10659985;radix=4096
        assert ((bound>>shifts[-1])|1)==center
        intervals=sorted((o,o+e) for o,e in zip(offsets,entries))
        assert intervals[0][0]==0 and all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
        assert intervals[-1][1]==sum(entries)<1<<31
        bias=((center+1)<<(shifts[-1]-1))-(1<<17)
        max_high=0
        for c,e in enumerate(entries):
            for i in [0,1,2,e-1]:
                mult=i if c==0 else 2*i+1
                hi,lo=divmod(mult,radix)
                assert hi<radix and lo<radix
                max_high=max(max_high,hi)
                coeff=bias+mult if c==0 else mult<<(shifts[c]-1)
                rebuilt=(bias+lo+hi*radix) if c==0 else (lo+hi*radix)<<(shifts[c]-1)
                assert coeff==rebuilt
        for (m,sign),codes in zip(cases,struct.iter_unpack('<6I',out)):
            total=0
            for c,code in enumerate(codes):
                index=(code&0x7fffffff)-offsets[c]
                assert 0<=index<entries[c],(mode,m,c,index)
                coeff=bias+index if c==0 else (2*index+1)<<(shifts[c]-1)
                total+=-coeff if code>>31 else coeff
            assert total==(-m if sign else m),(mode,m,sign,total)
        results.append({'four_hot':mode,'signed_components':len(cases),'decoded_digits':len(cases)*6,'table_bytes':sum(entries)*64,'max_ladder_high':max_high,'ubsan':'pass'})
print(json.dumps({'bound':hex(bound),'source_compiled':True,'gpu_executed':False,'results':results},indent=2))
