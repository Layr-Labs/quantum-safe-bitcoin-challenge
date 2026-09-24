#!/usr/bin/env python3
"""Post-result audit: actual macros, complete MAC/reduction, both explicit flags."""
from pathlib import Path
import sys,re,subprocess,itertools,random,json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'research'))
from ptx_field_model import Program,extract_ptx
ptxs=[];programs=[]
assert '#define QSB_C31 1' in (ROOT/'pinning.cu').read_text()
for mode in [0,1]:
    run=subprocess.run(['g++','-E','-P','-x','c++','-I',str(ROOT),
        '-D__CUDA_ARCH__=890','-DQSB_C31=1','-DQSB_HOST_GATE=1',
        f'-DQSB_MAC_HALF_SEED={mode}','-'],input='#include "GPUMath.h"\n',
        text=True,capture_output=True,check=True)
    start=run.stdout.index('__device__ __forceinline__ void qsb_muladd_seed(')
    code=extract_ptx(run.stdout[start:])
    code=re.sub(r'/\*.*?\*/','',code,flags=re.S)
    assert 'QSB_' not in code
    ptxs.append(code)
    body=code.strip()[1:-1]
    body=re.sub(r'{\s*(?=\.reg)','',body)
    body=re.sub(r';\s*}', ';',body)
    programs.append(Program('{'+body+'}'))
marker='.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3'
assert ptxs[0][ptxs[0].index(marker):]==ptxs[1][ptxs[1].index(marker):]
for program in programs:
    index=next(i for i,(_,args) in enumerate(program.ops) if args[0]=='f0')
    assert program.ops[index][0]=='add.cc.u64'
B=1<<256;mask=(1<<64)-1
words=lambda v:[(v>>(64*i))&mask for i in range(4)]
edges=[0,1,2,B-1,B-2,B-(1<<32)-977]
cases=list(itertools.product(edges,repeat=3));rng=random.Random(336189582)
cases.extend(tuple(rng.getrandbits(256) for _ in range(3)) for _ in range(1000))
for a,b,c in cases:
    inputs=words(a)+words(b)+words(c)
    assert programs[0].run(inputs)==programs[1].run(inputs),(a,b,c)
result={'passed':True,'fully_expanded_mac_cases':len(cases),
    'reduction_tail_equal':True,'first_reduction_cc_op':'add.cc.u64 f0',
    'gpu_executed':False,'production_default_is_overridden':True,
    'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['GPUMath.h','negative_y_mac.cuh']}}
(HERE/'full-mac-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
