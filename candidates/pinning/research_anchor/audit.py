#!/usr/bin/env python3
"""Independent affine check of extracted native C++ point-body execution."""
from pathlib import Path
import ctypes,json,random,subprocess
ROOT=Path(__file__).resolve().parent
P=2**256-2**32-977
G=(0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798,
   0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8)
OFF=0x800001e8

def plus(a,b):
    if a is None:return b
    if b is None:return a
    x,y=a;u,v=b
    if x==u:
        if (y+v)%P==0:return None
        slope=3*x*x*pow(2*y,-1,P)%P
    else:slope=(v-y)*pow(u-x,-1,P)%P
    nx=(slope*slope-x-u)%P
    return nx,(slope*(x-nx)-y)%P
def limbs(v):return [(v>>(64*i))&((1<<64)-1) for i in range(4)]
def pack(q):return (ctypes.c_uint64*8)(*(limbs(q[0])+limbs(q[1]+OFF)))
def val(s,j):return sum(int(s[j+i])<<(64*i) for i in range(4))

subprocess.run(['g++','-O2','-shared','-fPIC','-I/tmp/qsb-boost-audit/usr/include',
                str(ROOT/'audit.cpp'),'-o',str(ROOT/'audit.so')],check=True)
lib=ctypes.CDLL(str(ROOT/'audit.so'))
ptr=ctypes.POINTER(ctypes.c_uint64)
lib.seed.argtypes=[ptr,ptr,ptr]
lib.step.argtypes=[ptr,ptr,ptr,ctypes.c_int]

rng=random.Random(0x414e43484f52)
pool=[G]
for i in range(1023):pool.append(plus(pool[-1],G))
checks=0;chains=0;singular=0
for trial in range(2048):
    points=[pool[rng.randrange(len(pool))] for _ in range(15)]
    points=[(x,(-y)%P) if rng.getrandbits(1) else (x,y) for x,y in points]
    # Exclude singular ordinary-add cases; the production chain likewise is
    # incomplete there. Count exclusions rather than claiming completeness.
    a,b=points[:2]
    if a[0]==b[0]:singular+=1;continue
    expected=plus(a,b)
    state=(ctypes.c_uint64*16)()
    lib.seed(state,pack(a),pack(b))
    anchor=a
    for i in range(1,15):
        if i>=2:
            b=points[i]
            if expected is None or expected[0]==b[0]:singular+=1;break
            lib.step(state,pack(b),pack(anchor),i==14)
            expected=plus(expected,b)
            anchor=b
        D,N,U,V=[val(state,j) for j in (0,4,8,12)]
        assert pow(U,3,P)==pow(V,2,P)
        actual_x=D*pow(U,-1,P)%P
        if i!=14:actual_x=(actual_x+anchor[0])%P
        actual_y=(-N*pow(V,-1,P)-anchor[1])%P
        assert (actual_x,actual_y)==expected,(trial,i)
        checks+=1
    else:chains+=1
result={'exact_field_chain_cases':chains,'checked_point_states':checks,
        'singular_exclusions':singular,'mismatches':0,
        'scope':'Actual C++ seed and point bodies over exact field operations; not CUDA PTX semantics or device performance.'}
(ROOT/'audit_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
