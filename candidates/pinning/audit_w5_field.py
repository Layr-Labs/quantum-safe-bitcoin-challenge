#!/usr/bin/env python3
"""Source-bound PTX integer semantics; no compiler, GPU or throughput evidence."""
from pathlib import Path
import random,json
from ptx_field_model import Program,function,extract_ptx,check_semantics
P=2**256-2**32-977; MASK=2**64-1
HERE=Path(__file__).resolve().parent

def run():
 check_semantics()
 sq=Program(extract_ptx(function((HERE/'W5Square.cuh').read_text(),'void w5_square_exact(')))
 mulptx=extract_ptx(function((HERE/'pinning.cu').read_text(),'void qsb_field_mul('))
 # Unused predicate declaration inherited from the promoted field function.
 mul=Program(mulptx.replace('.reg .pred take;',''))
 limbs=lambda x:[(x>>(64*i))&MASK for i in range(4)]
 val=lambda z:sum(t<<(64*i) for i,t in enumerate(z))%P
 edge={0,1,2,P-1,P,2**256-1,P-65537,P//2,P//2+1}
 for i in range(256):
  for d in (-1,0,1):
   if 0<=(1<<i)+d<2**256:edge.add((1<<i)+d)
 # Small quadratic residues have full-sized roots; include both roots.
 for i in range(1,256):
  x=pow(i,(P+1)//4,P)
  if x*x%P==i:edge.update((x,P-x))
 rng=random.Random(192016)
 vals=sorted(edge)+[rng.getrandbits(256) for _ in range(3000)]
 for x in vals:assert val(sq.run(limbs(x)))==x*x%P,x
 pairs=[(x,y) for x in list(sorted(edge))[:32] for y in list(sorted(edge))[-32:]]
 pairs += [(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(3000)]
 for x,y in pairs:assert val(mul.run(limbs(x)+limbs(y)))==x*y%P,(x,y)
 out=dict(status='PASS',square_cases=len(vals),multiply_cases=len(pairs),source_bound=True,compiled=False,gpu_measured=False)
 print(json.dumps(out));return out
if __name__=='__main__':run()
