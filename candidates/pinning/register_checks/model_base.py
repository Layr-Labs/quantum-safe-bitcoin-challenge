"""Eight-lane routed convolution and pre-carry folding. Python/source models only."""
from pathlib import Path
import random,json,hashlib
D=Path(__file__).resolve().parent
B=1<<32;U=B-1;T=1<<256;P=T-B-977;C=B+977
R=random.Random(0x9268)
stats=dict(cases=0,nonzero_first_high_word=0,second_carry=0,
           dropped_first_high_fail=0,dropped_final_carry_fail=0,max_column=0,max_first_high=0)

def val(x):return sum(w<<(32*j) for j,w in enumerate(x))

def normalize8(v):
    assert len(v)==8 and all(0<=x<2**46 for x in v)
    lo=[x&U for x in v];hi=[x>>32 for x in v]
    s=[(lo[j]+(hi[j-1] if j else 0))&U for j in range(8)]
    gen=sum((s[j]<lo[j])<<j for j in range(8));prop=sum((x==U)<<j for j,x in enumerate(s))
    carry=(prop+(gen<<1))^prop
    out=[(s[j]+((carry>>j)&1))&U for j in range(8)]
    top=hi[7]+((carry>>8)&1)
    assert val(out)+top*T==val(v)
    return out,top
