#!/usr/bin/env python3
"""Source-bound PTX semantic experiment; neither PTX execution nor GPU timing."""
import hashlib,itertools,json,random,sys
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from ptx_field_model import Program,check_semantics,extract_ptx,function
P=(1<<256)-(1<<32)-977;B=1<<256;C=(1<<32)+977;H=1<<128
def words(x):return [x>>(64*i)&((1<<64)-1) for i in range(4)]
def integer(w):return sum(v<<(64*i) for i,v in enumerate(w))

def main():
    check_semantics()
    header=HERE/'candidate/GPUMath.h';data=header.read_bytes();src=data.decode()
    body=function(src,'__device__ __forceinline__ void _ModMultCore(')
    ptx=extract_ptx(body);assert ptx.strip()==(HERE/'field.ptx').read_text().strip()
    raw=(HERE/'product.ptx').read_text()
    # The full-product audit uses exactly the same arithmetic prefix, only
    # replacing the reduction/output tail with all eight product limbs.
    prefix=ptx[:ptx.index('.reg .u64 r0,r1,r2,r3')]
    assert raw.startswith(prefix)
    product=Program(raw);field=Program(ptx)
    edges=[0,1,2,H-1,H,H+1,P-65537,P-2,P-1,P,P+1,B-2,B-1]
    halves=[0,1,(1<<64)-1,1<<64,H-2,H-1]
    structured=[lo+(hi<<128) for lo,hi in itertools.product(halves,repeat=2)]
    cases=list(itertools.product(edges,repeat=2))+list(itertools.product(structured,repeat=2))
    rng=random.Random(2026091648);cases += [(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(20000)]
    signs=set();middle_high=0;final_carries=0;noncanonical=0
    for a,b in cases:
        inp=words(a)+words(b);got=integer(product.run(inp,output_count=8));assert got==a*b,(a,b,got,a*b)
        residue=integer(field.run(inp));canonical=residue-P if residue>=P else residue
        assert canonical==a*b%P,(a,b,residue,a*b%P)
        assert residue<B
        signs.add((int((a&(H-1))<(a>>128)),int((b&(H-1))<(b>>128))))
        middle=(a&(H-1))*(b>>128)+(a>>128)*(b&(H-1));middle_high+=middle>=B
        first=(a*b&(B-1))+(a*b>>256)*C;second=(first&(B-1))+(first>>256)*C
        final_carries+=second>=B;noncanonical+=residue>=P
    assert len(signs)==4 and middle_high and final_carries
    # Both historically dangerous carry/sign terms must matter to the oracle.
    mutant=Program(ptx.replace('addc.cc.u32 z7, z7, 0;','addc.u32 z7, z7, 0;'))
    a,b=P-1,P-(1<<224)
    assert integer(mutant.run(words(a)+words(b)))%P!=a*b%P
    wrong_sign=Program(ptx.replace('not.b32 neg, neg;','mov.u32 neg, 0;'))
    a,b=1,1;assert integer(wrong_sign.run(words(a)+words(b)))%P!=1
    assert header.read_bytes()==data
    result={'status':'PASS','validation_level':'actual generated PTX interpreted with independently modeled integer semantics','gpu_executed':False,'GPUMath_sha256':hashlib.sha256(data).hexdigest(),'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest(),'full_product_and_reduction_cases':len(cases),'difference_sign_combinations':sorted(signs),'cases_with_257bit_middle':middle_high,'final_fold_carry_cases':final_carries,'precanonical_outputs_ge_p':noncanonical,'mutations_rejected':['stale final carry flag','incorrect difference-product sign'],'ptx_opcodes':dict(Counter(op for op,args in field.ops)),'limits':'No GPU PTX execution, inline-asm constraint/alias runtime validation, scheduling or performance. Host fallback is unchanged; C++ canonicalization applied in oracle.'}
    (HERE/'ptx-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
