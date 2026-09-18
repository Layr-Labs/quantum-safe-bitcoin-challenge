#!/usr/bin/env python3
"""Actual inline PTX CPU semantic checks, not CUDA execution or timing."""
from collections import Counter
import hashlib,itertools,json,random,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from ptx_field_model import Program,check_semantics,extract_ptx,function
P=(1<<256)-(1<<32)-977;B=1<<256;C=(1<<32)+977
def words(v):return [v>>(64*i)&((1<<64)-1) for i in range(4)]
def integer(v):return sum(x<<(64*i) for i,x in enumerate(v))

def main():
    check_semantics()
    header=HERE/'candidate/GPUMath.h';data=header.read_bytes()
    ptx=extract_ptx(function(data.decode(),'__device__ __forceinline__ void _ModMultCore('))
    assert ptx.strip()==(HERE/'field.ptx').read_text().strip()
    raw=(HERE/'product.ptx').read_text();assert raw.startswith(ptx[:ptx.index('.reg .u64 r0,r1,r2,r3')])
    field=Program(ptx);product=Program(raw)
    edges={0,1,2,P-65537,P-2,P-1,P,P+1,B-2,B-1}
    for bit in range(32,256,32):edges.update(((1<<bit)-1,1<<bit,(1<<bit)+1))
    cases=list(itertools.product(sorted(edges),repeat=2));rng=random.Random(2026091664)
    cases += [(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(20000)]
    fc,nc=0,0
    for a,b in cases:
        inp=words(a)+words(b);assert integer(product.run(inp,output_count=8))==a*b
        r=integer(field.run(inp));assert (r-P if r>=P else r)==a*b%P
        first=(a*b&(B-1))+(a*b>>256)*C;second=(first&(B-1))+(first>>256)*C
        fc+=second>=B;nc+=r>=P
    stale=Program(ptx.replace('addc.cc.u32 z7, z7, 0;','addc.u32 z7, z7, 0;'))
    a,b=P-1,P-(1<<224);assert integer(stale.run(words(a)+words(b)))%P!=a*b%P
    broken=Program(raw.replace('madc.lo.cc.u32','mad.lo.cc.u32'))
    assert integer(broken.run(words(B-1)+words(B-1),output_count=8))!=(B-1)**2
    assert fc and nc and header.read_bytes()==data
    report={'status':'PASS','validation_level':'actual inline PTX CPU semantic model',
            'GPUMath_sha256':hashlib.sha256(data).hexdigest(),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),
            'semantic_model_sha256':hashlib.sha256((HERE.parent/'ptx_field_model.py').read_bytes()).hexdigest(),
            'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'full_product_and_reduction_cases':len(cases),'final_fold_carry_cases':fc,'precanonical_outputs_ge_p':nc,
            'mutations_rejected':['stale final carry flag','omitted narrow MAD carry-in'],
            'ptx_opcodes':dict(Counter(op for op,args in field.ops)),'gpu_executed':False,
            'limits':'No GPU correctness, runtime scheduling or timing. Host fallback and C++ canonicalization are unchanged.'}
    (HERE/'ptx-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
