#!/usr/bin/env python3
"""Execute selected host primitives and model actual inline PTX; no CUDA claim."""
import ctypes as CT
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from preflight import source_identity
from ptx_field_model import Program,check_semantics,extract_ptx,function
P=(1<<256)-(1<<32)-977
U64=CT.c_uint64

def words(x):return [x>>(64*i)&((1<<64)-1) for i in range(4)]
def integer(x):return sum(int(v)<<(64*i) for i,v in enumerate(x))
def main():
    identity=source_identity();check_semantics()
    bodies=[function((ROOT/'GPUMath.h').read_text(),'__device__ __forceinline__ void _ModMultCore('),
            function((ROOT/'square32.cuh').read_text(),'__device__ __forceinline__ void qsb_square32(')]
    rng=random.Random(260916622)
    edges=[0,1,2,(1<<128)-1,1<<128,1<<255,P-65537,P-2,P-1,P,P+1,(1<<256)-1]
    cases=list(itertools.product(edges,repeat=2))
    cases += [(P-i,P-j) for i in (1,65535,65536,65537,100000,1<<31)
              for j in (1,65535,65536,65537,100000,1<<31)]
    cases += [(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(20000)]
    counts={'host_mul':0,'host_square':0,'ptx_mul':0,'ptx_square':0}
    B=1<<256;C=(1<<32)+977;overflow_cases=0
    assert C*C+C < (1<<96)
    with tempfile.TemporaryDirectory(prefix='qsb-selected-field-') as td:
        cpp,so=Path(td)/'field.cpp',Path(td)/'field.so'
        cpp.write_text('#include <stdint.h>\n#define __device__\n#define __forceinline__ inline\n'+'\n'.join(bodies)+
            '\nextern "C" void mul(uint64_t*r,const uint64_t*a,const uint64_t*b){_ModMultCore(r,a,b);}\n'+
            'extern "C" void square(uint64_t*r,const uint64_t*a){qsb_square32(r,a);}\n')
        subprocess.run(['c++','-std=c++17','-O2','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
        lib=CT.CDLL(str(so));lib.mul.argtypes=[CT.POINTER(U64)]*3;lib.square.argtypes=[CT.POINTER(U64)]*2
        programs=[Program(extract_ptx(b)) for b in bodies]
        for i,(a,b) in enumerate(cases):
            # The carry bit and low half are correlated, not independent inputs.
            product=a*b;first=product%B+C*(product//B)
            assert first//B<=C
            second=first%B+C*(first//B)
            assert second<=B-1+C*C
            if second>=B:
                overflow_cases+=1
                assert second-B<C*C and second-B+C<(1<<96)
            for square in (False,True):
                op='square' if square else 'mul';expected=a*(a if square else b)%P
                for alias in range(2 if square else 3):
                    aa,bb,out=(U64*4)(*words(a)),(U64*4)(*words(b)),(U64*4)()
                    if alias==1:out=aa
                    if alias==2:out=bb
                    if square:lib.square(out,aa)
                    else:lib.mul(out,aa,bb)
                    assert integer(out)==expected,(op,alias,a,b,integer(out),expected)
                    counts['host_'+op]+=1
                if i<2180:
                    raw=integer(programs[square].run(words(a)+([] if square else words(b))))
                    assert raw%P==expected,(op,a,b,raw,expected)
                    counts['ptx_'+op]+=1
        # A non-.cc high limb would leave a stale carry bit; preserve this witness.
        code=extract_ptx(bodies[0]);a,b=P-1,P-(1<<224)
        assert code.count('addc.cc.u32 z7, z7, 0;')==1
        mutant=code.replace('addc.cc.u32 z7, z7, 0;','addc.u32 z7, z7, 0;')
        assert integer(Program(mutant).run(words(a)+words(b)))%P!=a*b%P
        assert integer(Program(code).run(words(a)+words(b)))%P==a*b%P
    assert source_identity()==identity
    print(json.dumps({'status':'PASS','gpu_executed':False,'validation_level':'actual host primitives and PTX semantic model',
        'counts':counts,'bounded_fold_overflow_cases':overflow_cases,'carry_flag_mutation_rejected':True,'ptx_scope':'ASM residue before C++ canonical subtraction',
        **identity},indent=2))
if __name__=='__main__':main()
