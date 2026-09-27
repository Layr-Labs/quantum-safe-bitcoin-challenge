#!/usr/bin/env python3
"""Audit the actual coefficient screen against exact big-integer rounding.

CUDA integer instructions used by the imported primitives have CPU equivalents
here. This executes the source control flow, not GPU machine code.
"""
import ctypes
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

source=Path(__file__).with_name('GLVScalar.cuh').read_text()

def extract(name):
    start=source.index('__device__ __forceinline__',source.index(name)-55)
    brace=source.index('{',start);end=brace+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end].replace('__device__','').replace('__forceinline__','inline')

# Constants come from the production fallback, not a second hand-copied list.
constants=[]
for name in ['g1','g2']:
    words=re.search(r'const uint64_t '+name+r'\[4\]=\{([^}]+)\}',source).group(1)
    limbs=[int(v.removesuffix('ULL'),16) for v in words.split(',')]
    constants.append(sum(v<<(64*i) for i,v in enumerate(limbs)))
B=1<<32
bounds=[]
for which,g in enumerate(constants):
    d=[(g>>(32*i))&(B-1) for i in range(8)]
    assert d[7]+d[6]<B
    omitted=sum((B-1)*d[j]*B**(i+j) for i in range(8) for j in range(8) if i+j<10)
    omitted+=5*(B-1)*B**10
    ceiling=(omitted+B**11-1)//B**11
    assert ceiling==[9,8][which]
    bounds.append(ceiling)

cpp=r"""
#include <stdint.h>
struct ulonglong2 {uint64_t x,y;};
static uint64_t fallback_calls=0;
static uint64_t gvalues[2][4];
inline uint64_t q9_mulw(uint32_t a,uint32_t b){return (uint64_t)a*b;}
inline uint64_t q9_madw(uint32_t a,uint32_t b,uint64_t c){return (uint64_t)a*b+c;}
inline void q9_high15_add(uint64_t *a,uint32_t *o,uint64_t p){
    unsigned __int128 sum=(unsigned __int128)*a+p;*a=(uint64_t)sum;*o+=(uint32_t)(sum>>64);
}
template<int WHICH> inline ulonglong2 q9_coeff_fallback(uint64_t k0,uint64_t k1,uint64_t k2,uint64_t k3){
    fallback_calls++;
    uint64_t kk[4]={k0,k1,k2,k3};uint32_t a[8],b[8],p[16]={};
    for(int i=0;i<8;i++) {a[i]=(uint32_t)(kk[i/2]>>(32*(i%2)));b[i]=(uint32_t)(gvalues[WHICH-1][i/2]>>(32*(i%2)));}
    for(int i=0;i<8;i++){
        uint64_t carry=0;
        for(int j=0;j<8;j++){
            uint64_t t=(uint64_t)a[i]*b[j]+p[i+j]+carry;
            p[i+j]=(uint32_t)t;carry=t>>32;
        }
        p[i+8]=(uint32_t)carry;
    }
    unsigned __int128 lo=(unsigned __int128)((uint64_t)p[12]|((uint64_t)p[13]<<32))+(p[11]>>31);
    return {(uint64_t)lo,((uint64_t)p[14]|((uint64_t)p[15]<<32))+(uint64_t)(lo>>64)};
}
#define QSB_GLV_LEAN 1
#define QSB_GLV_COEFF_BOUNDS 1
"""
cpp+=extract('q9_high15_begin(')+'\n'+extract('q9_mulhi32(')+'\n'
cpp+=extract('q9_round_coeff(')+'\n'
cpp+='template<int WHICH,uint32_t FALLBACK_WORD>\n'+extract('q9_coeff_high15(')
cpp+=r"""
extern "C" uint64_t evaluate(int which,const uint64_t *k,const uint64_t *g,uint64_t *out){
    for(int i=0;i<4;i++)gvalues[which][i]=g[i];
    fallback_calls=0;
    if(which==0)q9_coeff_high15<1,0x7ffffffcU>(out,k,g);
    else q9_coeff_high15<2,0x7ffffffdU>(out,k,g);
    return fallback_calls;
}
"""
# Validate the actual production guard rather than supplying one in the wrapper.
assert '0x7ffffff7U' in cpp and '0x7ffffff8U' in cpp
rng=random.Random(0x10C0EFF)
counts=[0,0];fallbacks=[[0,0],[0,0]]
with tempfile.TemporaryDirectory() as td:
    td=Path(td);(td/'test.cpp').write_text(cpp);calls=[]
    for mode in [0,1]:
        output=td/f'audit{mode}.so'
        subprocess.run(['g++','-O3','-shared','-fPIC',f'-DQSB_GLV_HIGH10_HI={mode}',str(td/'test.cpp'),'-o',str(output)],check=True)
        call=ctypes.CDLL(str(output)).evaluate
        call.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64)]
        call.restype=ctypes.c_uint64;calls.append(call)
    def check(which,k):
        if not 0<=k<1<<256:return
        g=constants[which];expected=(k*g+(1<<383))>>384
        kk=(ctypes.c_uint64*4)(*[(k>>(64*i))&((1<<64)-1) for i in range(4)])
        gg=(ctypes.c_uint64*4)(*[(g>>(64*i))&((1<<64)-1) for i in range(4)])
        for mode,call in enumerate(calls):
            out=(ctypes.c_uint64*2)();fallbacks[mode][which]+=call(which,kk,gg,out)
            assert out[0]+(out[1]<<64)==expected,(mode,which,hex(k))
        counts[which]+=1
    for which,g in enumerate(constants):
        for k in [0,1,(1<<256)-1]:check(which,k)
        for bit in range(256):
            for delta in [-2,-1,0,1,2]:check(which,(1<<bit)+delta)
        for _ in range(100000):check(which,rng.getrandbits(256))
        # Construct values on both sides of the actual coefficient half threshold.
        for _ in range(12000):
            q=rng.randrange(g>>128)
            boundary=((2*q+1)*(1<<383))//g
            for delta in [-3,-2,-1,0,1,2,3]:check(which,boundary+delta)
        # Enter and leave both guards, and cross the word-11 wrap boundary.
        # A unit is one word-11 increment expressed in input-scalar units.
        unit=(1<<352)//g
        for _ in range(1000):
            q=rng.randrange(g>>128)
            for origin in [((2*q+1)*(1<<383))//g,(q*(1<<384))//g]:
                for offset in range(-14,5):
                    for delta in [-1,0,1]:check(which,origin+offset*unit+delta)
        # All-high and sparse 32-bit limbs stress the bounded carry additions.
        for word in [0,1,0x7fffffff,0x80000000,0xfffffffe,0xffffffff]:
            for mask in range(256):
                k=sum((word if mask&(1<<i) else 0)<<(32*i) for i in range(8))
                check(which,k)
assert all(x>1000 for row in fallbacks for x in row),fallbacks
print(json.dumps({'passed':True,'coefficient_cases':counts,'fallback_calls_off_on':fallbacks,'error_bound_units_2pow352':bounds,'gpu_executed':False},indent=2))
