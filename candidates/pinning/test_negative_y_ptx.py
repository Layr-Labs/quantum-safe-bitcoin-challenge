#!/usr/bin/env python3
"""Audit generated multiply-add PTX, including its exact integer product seed.

The field reduction deliberately inherits C31/RP truncations from the leader.
Directed cases that trigger those specific tails are counted separately.
"""
import ast
import os
from pathlib import Path
import random
import re

ROOT=Path(__file__).resolve().parent
MASK64=(1<<64)-1
P=(1<<256)-(1<<32)-977

def program(raw=False):
 source=(ROOT/os.environ.get('QSB_MAC_HEADER','negative_y_mac.cuh')).read_text()
 source=source[source.index('asm(')+4:source.index(': "=l"')]
 code=''.join(ast.literal_eval(m[0]) for m in re.finditer(r'"(?:[^"\\]|\\.)*"',source))
 if raw:code=code[:code.index('.reg .u64 r0,r1,r2,r3,h0')]
 code=re.sub(r'/\*.*?\*/','',code,flags=re.S)
 code=re.sub(r'\.reg\s+[^;]+;','',code)
 code=re.sub(r'(?m)^[ \t]*[{}][ \t]*','',code)
 code=code.replace('; }',';')
 result=[]
 for statement in code.split(';'):
  if not statement.strip():continue
  op,rest=statement.strip().split(None,1)
  result.append((op,[x.strip() for x in re.findall(r'\{[^}]*\}|[^,{}\s]+',rest)]))
 return result

def execute(code, a, b, c):
    regs = {f"%{i+4}": ((a if i < 4 else b if i < 8 else c) >> (64 * (i % 4))) & MASK64 for i in range(12)}
    carry = 0
    truncated=False

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
        elif op.startswith(('mad.', 'madc.')):
            mul=(value(args[1]) & 0xffffffff)*(value(args[2]) & 0xffffffff)
            half=(mul>>32) if '.hi.' in op else (mul&0xffffffff)
            total=half+value(args[3])+(carry if op.startswith('madc.') else 0)
            regs[args[0]]=total&0xffffffff
            if '.cc.' in op:carry=total>>32
        elif op.startswith(("add.", "addc.")):
            bits = int(op[-2:])
            total = value(args[1]) + value(args[2]) + (carry if op.startswith("addc.") else 0)
            regs[args[0]] = total & ((1 << bits) - 1)
            if total>>bits and ((op=="addc.cc.u64" and args[0]=="g3") or (op=="addc.u32" and args[0]=="z8") or (op=="addc.cc.u32" and args[0]=="z2" and args[2]=="sfc")):
                truncated=True
            if ".cc." in op:
                carry = total >> bits
        elif op in ('and.b32','xor.b32'):
            aa,bb=value(args[1]),value(args[2])
            regs[args[0]]=(aa&bb if op.startswith('and') else aa^bb)&0xffffffff
        else:
            raise AssertionError(op)
    return regs,truncated


def main():
 raw=program(True);field=program(False);rng=random.Random(0x4E594D4143)
 edges=[0,1,P-1,P,(1<<256)-1]+[1<<i for i in range(256)]
 cases=[(a,b,c) for a in edges for b in (0,1,P-1,(1<<256)-1) for c in (0,1,P-1,(1<<256)-1)]
 cases += [(rng.getrandbits(256),rng.getrandbits(256),rng.getrandbits(256)) for _ in range(30000)]
 inherited=0
 for a,b,c in cases:
  rr,_=execute(raw,a,b,c)
  got=sum(rr[f'x{i}']<<(32*i) for i in range(16))
  assert got==a*b+c,('integer seed',hex(a),hex(b),hex(c))
  rr,truncated=execute(field,a,b,c)
  got=sum(rr[f'%{i}']<<(64*i) for i in range(4))%P
  want=(a*b+c)%P
  if got!=want:
   assert truncated,('unexpected field mismatch',hex(a),hex(b),hex(c))
   inherited+=1
 print({'triples':len(cases),'exact_512_bit_mismatches':0,'inherited_truncation_cases':inherited,'unexplained_field_mismatches':0})

if __name__=='__main__':main()
