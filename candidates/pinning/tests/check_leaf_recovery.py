#!/usr/bin/env python3
"""Execute production stages/collectives on CPU against independent affine EC.

This does not execute CUDA or emulate PTX arithmetic: OpenSSL supplies field
primitives. The actual CUDA stage source and cooperative barrier placement run
unchanged. Native compilation is a separate check.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile

TRACK = Path(__file__).resolve().parents[1]
P = 2**256-2**32-977
MASK = 2**64-1
U64 = C.c_uint64
MUTANTS = {
    'wrong_sibling': ('products[k][tid^(QSB_RECOVERY_N/2)]', 'products[k][tid^(QSB_RECOVERY_N/4)]'),
    'wrong_pair': ('inverses[k][tid&(QSB_RECOVERY_N/2-1)]', 'inverses[k][tid&(QSB_RECOVERY_N/4-1)]'),
    'missing_scale': ('qsb_recovery_mul(H,U,sibling);', 'Load256(H,sibling);'),
    'wrong_plane': ('saved[4u*stride+idx]', 'saved[3u*stride+idx]'),
}

def limbs(x):
    return [(x>>(64*i))&MASK for i in range(4)]

def unlimbs(a):
    return sum(int(a[i])<<(64*i) for i in range(4))

def point(rng):
    while True:
        x=rng.randrange(P); yy=(x*x*x+7)%P; y=pow(yy,(P+1)//4,P)
        if y*y%P==yy:
            return x,y if rng.randrange(2) else P-y

def affine_add(p,q):
    x,y=p;a,b=q
    if x==a:
        if (y+b)%P==0:return None
        slope=3*x*x*pow(2*y,-1,P)%P
    else:slope=(b-y)*pow(a-x,-1,P)%P
    rx=(slope*slope-x-a)%P
    return rx,(slope*(x-rx)-y)%P

def extract(source):
    begin=source.index('    /* Recover P+R and P-R together')
    end=source.index('    return;\n    } else {',begin)
    zero=source[begin:end]
    begin=source.index('    bool usable = false;',end)
    end=source.index('    /* Check both pubkeys',begin)
    two=source[begin:end]
    common='''
    int idx=(int)(blockIdx.x*QSB_RECOVERY_N+threadIdx.x);
    bool active=idx<batch_size;
    uint64_t qx[4]={},qy[4]={},qzz[4]={},qzzz[4]={},prod[5]={};
'''
    return 'static void stage_zero(){\n'+common+'''
    if(active){
        memcpy(qx,inputs+(size_t)idx*16,32);
        memcpy(qy,inputs+(size_t)idx*16+4,32);
        memcpy(qzz,inputs+(size_t)idx*16+8,32);
        memcpy(qzzz,inputs+(size_t)idx*16+12,32);
    }
'''+zero+'}\nstatic void stage_two(){\n'+common+two+'''
    assert(active);
    memcpy(results+(size_t)idx*9,q1x,32);
    memcpy(results+(size_t)idx*9+4,q2x,32);
    results[(size_t)idx*9+8]=4|y_parities;
}
'''

def build(tmp,mutant,tree_n):
    header=(TRACK/'LeafRecovery.cuh').read_text()
    if mutant:
        a,b=MUTANTS[mutant];assert header.count(a)==1
        header=header.replace(a,b)
    (tmp/'LeafRecovery.cuh').write_text(header)
    support=(TRACK/'tests/leaf_recovery_host.cpp').read_text()
    source=(TRACK/'pinning.cu').read_text()
    stages=extract(source)
    begin=source.index('__device__ __forceinline__ void qsb_block_inverse(')
    end=source.index('/* Shared-denominator recovery',begin)
    support=support.replace('// BEGIN_EXTRACTED_ROOTS\n// END_EXTRACTED_ROOTS',source[begin:end])
    support=support.replace('// BEGIN_EXTRACTED_STAGES\n// END_EXTRACTED_STAGES',stages)
    cpp=tmp/'audit.cpp';cpp.write_text(support)
    so=tmp/'audit.so'
    dev=Path(os.environ.get('QSB_OPENSSL_DEV','/usr'))
    includes=['-I'+str(TRACK)]
    if (dev/'usr/include/openssl').exists():
        includes+=['-I'+str(dev/'usr/include'),'-I'+str(dev/'usr/include/x86_64-linux-gnu'),'-L'+str(dev/'lib')]
    subprocess.run(['g++','-O2','-std=c++17','-fPIC','-shared','-Wno-unknown-pragmas',
                    '-DQSB_RECOVERY_N='+str(tree_n),'-fsanitize=undefined','-fno-sanitize-recover=all',*includes,
                    str(cpp),'-lcrypto','-o',str(so)],check=True)
    lib=C.CDLL(str(so))
    ptr=C.POINTER(U64)
    lib.pipeline.argtypes=[C.c_int,ptr,ptr,ptr,ptr,C.c_uint]
    lib.constant.argtypes=[ptr,C.c_char_p,C.c_char_p]
    lib.root_test.argtypes=[C.c_int,ptr,ptr,C.c_uint]
    return lib

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mutant',choices=MUTANTS)
    ap.add_argument('--fast',action='store_true');ap.add_argument('--tree',type=int,choices=[128,256],default=128);args=ap.parse_args()
    rng=random.Random(202609171057)
    cases=0;skipped=0;batches=0;root_cases=0
    with tempfile.TemporaryDirectory(prefix='qsb-leaf-audit-') as td:
        lib=build(Path(td),args.mutant,args.tree)
        for _ in range(128):
            a,b=point(rng);out=(U64*4)()
            assert lib.constant(out,a.to_bytes(32,'little'),b.to_bytes(32,'little'))==1
            assert unlimbs(out)==3*a*a*pow(2*b,-1,P)%P
        out=(U64*4)()
        assert lib.constant(out,(1).to_bytes(32,'little'),bytes(32))==0
        counts=[257] if args.fast else [1,2,31,32,33,127,128,129,255,256,257,511,512,777,1024]
        for n in counts:
            R=point(rng);a,b=R;coords=[];expected=[]
            for i in range(n):
                pt=point(rng)
                if i%37==0:pt=R
                elif i%37==1:pt=(a,P-b)
                z=[1,2,P-1,rng.randrange(1,P)][i%4]
                U=z*z%P;V=U*z%P
                X=pt[0]*U%P;Y=pt[1]*V%P
                if i%37==2:U=V=X=Y=0
                if U==0 or (a*U-X)%P==0:
                    expected.append(None);skipped+=1
                else:
                    expected.append((affine_add(pt,R),affine_add(pt,(a,P-b))))
                coords.extend(limbs(X)+limbs(Y)+limbs(U)+limbs(V))
            ins=(U64*len(coords))(*coords);aa=(U64*4)(*limbs(a));bb=(U64*4)(*limbs(b))
            for order in range(1 if args.fast else 3):
                got=(U64*(n*9))()
                assert lib.pipeline(n,ins,aa,bb,got,1234+order*9973)==1
                for i,ref in enumerate(expected):
                    actual=list(got[i*9:i*9+9])
                    if ref is None:
                        assert actual==[0]*9,('unusable',n,i,actual)
                        continue
                    p,q=ref
                    assert actual[8]&4,('missing',n,i)
                    assert unlimbs(actual[:4])==p[0],('x1',n,i)
                    assert unlimbs(actual[4:8])==q[0],('x2',n,i)
                    assert actual[8]&3==(p[1]&1)|((q[1]&1)<<1),('parity',n,i)
                batches+=1
            cases+=n
        if not args.fast:
            for n in (1,255,256,257,65535,65536,65537,131072):
                values=[rng.randrange(1,P) for _ in range(n)]
                ins=(U64*(4*n))(*(l for x in values for l in limbs(x)))
                out=(U64*(4*n))()
                assert lib.root_test(n,ins,out,4411+n)==1
                assert all(x*unlimbs(out[4*i:4*i+4])%P==1 for i,x in enumerate(values))
                root_cases+=n
    fingerprint=hashlib.sha256(b''.join((TRACK/f).read_bytes() for f in
        ('LeafRecovery.cuh','RecoveryConstant.h','pinning.cu'))).hexdigest()
    print(json.dumps(dict(passed=True,tree_leaves=args.tree,candidates=cases,unusable=skipped,
        shuffled_launch_cases=batches,constant_cases=129,actual_root_cases=root_cases,ubsan=True,
        source_fingerprint=fingerprint,gpu_executed=False)))

if __name__=='__main__':main()
