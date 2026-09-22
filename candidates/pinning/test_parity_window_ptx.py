#!/usr/bin/env python3
"""Interpret the exact generated PTX integer schedule against Python products."""
import ast
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
MASK64 = (1 << 64) - 1


def program():
    import subprocess
    source = subprocess.check_output(['clang','-E','-P','-x','c++',
        '-DQSB_C31=1','-DQSB_SHORT_CARRY=1','-DQSB_FIELD_SC=1','-DQSB_PARITY_SUM=1',
        '-DQSB_PARITY_WINDOW_NARROW=1',str(ROOT/'ParityWindow.cuh')],text=True)
    source = source[source.index('asm(')+4:source.index(': "=l"')]
    code=''.join(ast.literal_eval(m[0]) for m in re.finditer(r'"(?:[^"\\]|\\.)*"',source))
    code=re.sub(r"\.reg\s+[^;]+;", "",code).strip().removeprefix('{').removesuffix('}')
    result=[]
    for statement in code.split(';'):
        if not statement.strip():continue
        op,rest=statement.strip().split(None,1)
        result.append((op,[x.strip() for x in re.findall(r"\{[^}]*\}|[^,{}\s]+",rest)]))
    return result


def execute(code, a, b):
    regs = {f"%{i+2}": ((a if i < 4 else b) >> (64 * (i % 4))) & MASK64 for i in range(8)}
    carry = 0

    def value(arg):
        return regs[arg] if arg in regs else int(arg, 0)

    for op, args in code:
        if op == "mov.b64":
            dst, src = args
            if dst.startswith("{"):
                lo, hi = dst[1:-1].split(",")
                regs[lo.strip()] = value(src) & 0xffffffff
                regs[hi.strip()] = value(src) >> 32
            elif src.startswith("{"):
                lo, hi = src[1:-1].split(",")
                regs[dst] = value(lo.strip()) | (value(hi.strip()) << 32)
            else:
                regs[dst] = value(src) & MASK64
        elif op in ("mov.u32", "mov.u64"):
            regs[args[0]] = value(args[1]) & ((1 << int(op[-2:])) - 1)
        elif op == "mul.wide.u32":
            regs[args[0]] = (value(args[1]) & 0xffffffff) * (value(args[2]) & 0xffffffff)
        elif op.startswith(("add.", "addc.")):
            bits = int(op[-2:])
            total = value(args[1]) + value(args[2]) + (carry if op.startswith("addc.") else 0)
            regs[args[0]] = total & ((1 << bits) - 1)
            if ".cc." in op:
                carry = total >> bits
        elif op in ('and.b32','xor.b32'):
            aa,bb=value(args[1]),value(args[2])
            regs[args[0]]=(aa&bb if op.startswith('and') else aa^bb)&0xffffffff
        else:
            raise AssertionError(op)
    return [regs[f"%{i}"] for i in range(2)]


def oracle(a,b):
    aa=[(a>>(32*i))&0xffffffff for i in range(8)]
    bb=[(b>>(32*i))&0xffffffff for i in range(8)]
    def col(k):return sum(aa[i]*bb[k-i] for i in range(8) if 0<=k-i<8)
    mid=((col(6)>>32)+col(7)+((col(8)&1)<<32))&0x1ffffffff
    top=((col(13)>>32)+col(14))&MASK64
    return mid,top

def main():
    code=program();rng=random.Random(0xFA117)
    edges=[0,1,(1<<256)-1]+[1<<i for i in range(256)]+[((1<<256)-1)^(1<<i) for i in range(256)]
    pairs=[(a,b) for a in edges for b in [0,1,(1<<256)-1]]
    pairs += [(rng.getrandbits(256),rng.getrandbits(256)) for _ in range(20000)]
    p=(1<<256)-(1<<32)-977;accepted=0;rejected=0
    for a,b in pairs:
        mid,top=execute(code,a,b)
        assert (mid,top)==oracle(a,b)
        beta=rng.randrange(1,p);x7=mid&0xffffffff
        q=(top+977*(top>>32)+x7+(beta>>224))&MASK64
        if x7<0xfffffff9 and q&0xffffffff<0xfffff47f:
            got=((a&b)^(mid>>32)^beta^(q>>32))&1
            want=((a*b)%p+beta)%p&1
            assert got==want,(hex(a),hex(b),hex(beta),got,want)
            accepted+=1
        else:rejected+=1
    # Boundaries must defer before publishing an unverified parity.
    for x in [0xfffffff8,0xfffffff9,0xffffffff]:
        for q in [0xfffff47e,0xfffff47f,0xffffffff]:
            accept=x<0xfffffff9 and q<0xfffff47f
            assert accept==(x==0xfffffff8 and q==0xfffff47e)
    print({'ptx_pairs':len(pairs),'accepted_exact_parities':accepted,'replay_cases':rejected,'mismatches':0})

if __name__=='__main__':main()
