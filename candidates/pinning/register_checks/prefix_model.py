"""Interpret new prefix accumulation and complete carry reconstruction; Python only."""
from pathlib import Path
import random,json,hashlib
D=Path(__file__).resolve().parent
B=1<<32;U=B-1;W=1<<64;T=1<<256;P=T-B-977;C=B+977
R=random.Random(0x92696)
from model_base import normalize8,val
stats=dict(cases=0,max_total_carry=0,max_prefix_carry=0,max_upper_carry=0,
           max_raw_column=0,max_folded_column=0,max_first_high=0,
           first_high_word=0,second_carry=0,subtraction_borrows=0,
           omitted_upper_carry_failures=0,omitted_prefix_borrow_failures=0,
           omitted_last_correction_failures=0,negative_invariant_failures=0)

def finish3(words,cf):
    assert cf in [0,1]
    if cf:assert val(words)<2**65
    out=words[:];s0=(out[0]+977*cf)&U;c0=s0<out[0];out[0]=s0
    s1=(out[1]+cf+c0)&U;c1=s1<out[1];out[1]=s1
    assert out[2]+c1<B;out[2]+=c1
    assert val(out)==val(words)+cf*C
    return out

def product(a,b,negative=None):
    av=[a>>(32*j)&U for j in range(8)];bv=[b>>(32*j)&U for j in range(8)]
    L=[];LC=[];H=[];HC=[]
    for d in range(8):
        total=prefix=counts=0;exact=0;prefix_exact=None
        for i in range(8):
            term=av[i]*bv[(d-i)&7];exact+=term
            # Literal add.cc.u64/addc.u32 + predicated bfi.b32 model.
            z=total+term;total=z&(W-1);counts+=z>>64
            if i==d:
                prefix=total;counts=(counts&~0xff00)|((counts&255)<<8);prefix_exact=exact
            assert (counts&255)<=7
            assert total+(counts&255)*W==exact
            if prefix_exact is not None:assert prefix+((counts>>8)&255)*W==prefix_exact
        tc=counts&255;lc=counts>>8&255
        borrow=total<prefix;upper=(total-prefix)&(W-1)
        hc=tc-lc-(0 if negative=='borrow' else borrow)
        stats['subtraction_borrows']+=borrow
        if negative!='borrow':assert upper+hc*W==exact-prefix_exact
        stats['max_total_carry']=max(stats['max_total_carry'],tc)
        stats['max_prefix_carry']=max(stats['max_prefix_carry'],lc)
        stats['max_upper_carry']=max(stats['max_upper_carry'],hc)
        L.append(prefix);LC.append(lc);H.append(upper);HC.append(0 if negative=='upper' else hc)
    low=[(L[d]&U)+((L[d-1]>>32) if d else 0)+(LC[d-2] if d>=2 else 0) for d in range(8)]
    high=[(H[d]&U)+((H[d-1]>>32) if d else L[7]>>32)+(HC[d-2] if d>=2 else LC[6+d]) for d in range(8)]
    if negative is None:assert val(low)+T*val(high)==a*b
    stats['max_raw_column']=max(stats['max_raw_column'],max(low+high))
    assert max(low+high)<2*B+8 and high[7]<B
    folded=[low[d]+977*high[d]+(high[d-1] if d else 0) for d in range(8)]
    stats['max_folded_column']=max(stats['max_folded_column'],max(folded))
    assert max(folded)<1960*B
    words,top=normalize8(folded);tail=high[7]+top
    stats['max_first_high']=max(stats['max_first_high'],tail);assert tail<B+1960
    q0,q1=tail&U,tail>>32;stats['first_high_word']+=q1!=0
    second=words[:];second[0]+=977*q0;second[1]+=q0+977*q1;second[2]+=q1
    words,cf=normalize8(second);stats['second_carry']+=cf!=0
    if cf:assert val(words)<C*(B+1960)<2**65
    got=val(words) if negative=='last' else val(finish3(words,cf))
    if negative is None:assert got%P==a*b%P
    return got
