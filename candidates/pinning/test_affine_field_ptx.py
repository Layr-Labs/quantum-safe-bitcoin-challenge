#!/usr/bin/env python3
"""Interpret the actual carry-complete affine add/sub PTX against Python integers."""
import ast,random,re
from pathlib import Path
from test_parity_replay import function
P=(1<<256)-(1<<32)-977
MASK=(1<<64)-1

def program(name):
    src=function((Path(__file__).parent/'affine_block.cuh').read_text(),
                 '__device__ __forceinline__ void qsb_affine_'+name+'(')
    src=src[src.index('asm(')+4:src.index(': "=l"')]
    code=''.join(ast.literal_eval(m[0]) for m in re.finditer(r'"(?:[^"\\]|\\.)*"',src))
    code=re.sub(r'\.reg\s+[^;]+;','',code).replace('{','').replace('}','')
    return [(s.strip().split(None,1)[0],[a.strip() for a in s.strip().split(None,1)[1].split(',')]) for s in code.split(';') if s.strip()]

def run(code,a,b):
    r={f'%{i+4}':(a>>(64*i))&MASK for i in range(4)}
    r.update({f'%{i+8}':(b>>(64*i))&MASK for i in range(4)})
    carry=0
    def v(x):return r[x] if x in r else int(x,0)
    for op,args in code:
        dst=args[0]
        if op.startswith(('add.','addc.')):
            total=v(args[1])+v(args[2])+(carry if op.startswith('addc.') else 0)
            r[dst]=total&MASK
            if '.cc.' in op:carry=int(total>MASK)
        elif op.startswith(('sub.','subc.')):
            total=v(args[1])-v(args[2])-(carry if op.startswith('subc.') else 0)
            r[dst]=total&MASK
            if '.cc.' in op:carry=int(total<0)
        elif op=='and.b64':r[dst]=v(args[1])&v(args[2])
        elif op=='mov.u64':r[dst]=v(args[1])&MASK
        elif op=='setp.ge.s64':
            aa,bb=v(args[1]),v(args[2]);aa=aa-(1<<64) if aa>>63 else aa;bb=bb-(1<<64) if bb>>63 else bb
            r[dst]=aa>=bb
        elif op=='selp.u64':r[dst]=v(args[1]) if v(args[3]) else v(args[2])
        else:raise AssertionError(op)
    return sum(r[f'%{i}']<<(64*i) for i in range(4))

def main():
    values={0,1,2,P//2,P-1,P-2,(1<<32)+976,(1<<32)+977}
    for bit in (32,64,96,128,160,192,224,255):
        for delta in (-1,0,1):values.add(((1<<bit)+delta)%P)
    cases=[(a,b) for a in values for b in values]
    rng=random.Random(0x9162543);cases += [(rng.randrange(P),rng.randrange(P)) for _ in range(20000)]
    for name in ('add','sub'):
        code=program(name)
        for a,b in cases:assert run(code,a,b)==((a+b) if name=='add' else (a-b))%P,(name,a,b)
    print(f'{2*len(cases)} actual PTX add/sub comparisons; zero mismatches')
if __name__=='__main__':main()
