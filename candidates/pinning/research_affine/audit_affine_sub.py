#!/usr/bin/env python3
"""Interpret the probe's actual canonical-subtraction PTX on the CPU.

This verifies integer semantics and output aliases, not CUDA register allocation.
"""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent
source = (ROOT/'persistent_affine_probe.cu').read_text()
body = source.split('struct AffineField {', 1)[1].split('static void neg(', 1)[0]
asm = re.search(r'asm volatile\((.*?)\n\s*:', body, re.S).group(1)
ptx = ''.join(ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"', asm))
P = (1<<256)-(1<<32)-977
MASK = (1<<64)-1

def parse(text):
    text = re.sub(r'\.reg[^;]*;', '', text).replace('{', '').replace('}', '')
    result = []
    for statement in text.split(';'):
        if not statement.strip():
            continue
        op, args = statement.strip().split(None, 1)
        assert op in {'sub.cc.u64', 'subc.cc.u64', 'subc.u64',
                      'add.cc.u64', 'addc.cc.u64', 'addc.u64', 'and.b64'}, op
        result.append((op, args.replace(' ', '').split(',')))
    return result

def execute(operations, a, b, alias=None):
    registers = {f'%{base+i}': (v>>(64*i))&MASK
                 for base, v in ((4,a), (8,b)) for i in range(4)}
    carry = 0
    def name(token):
        if alias is not None and token in ('%0', '%1', '%2', '%3'):
            return '%'+str(alias+int(token[1:]))
        return token
    def read(token):
        token = name(token)
        return registers[token] if token in registers else int(token, 0)
    for op, args in operations:
        dst, left, right = args
        x, y = read(left), read(right)
        if op == 'and.b64':
            value = x & y
        else:
            cy = carry if op.startswith(('addc.', 'subc.')) else 0
            value = x-y-cy if op.startswith('sub') else x+y+cy
            if '.cc.' in op:
                carry = int(value < 0) if op.startswith('sub') else int(value > MASK)
        registers[name(dst)] = value&MASK
    return sum(registers[name(f'%{i}') ]<<(64*i) for i in range(4))

edges = {0,1,2,976,977,978,(1<<32)+976,(1<<32)+977,P//2,P-1}
for bit in (1,31,32,33,63,64,65,95,96,127,128,159,160,191,192,223,224,254,255):
    for d in (-1,0,1):
        value = (1<<bit)+d
        edges.add(value)
        edges.add(P-value)
edges = sorted(v for v in edges if 0<=v<P)
rng = random.Random(0xAFF15AB)
ops = parse(ptx)
checked = aliases = negatives = 0
for a,b in itertools.chain(itertools.product(edges,repeat=2),
                           ((rng.randrange(P),rng.randrange(P)) for _ in range(20000))):
    want = (a-b)%P
    for alias in (None,4,8):
        out = execute(ops,a,b,alias)
        assert out == want and 0<=out<P, (a,b,alias,out,want)
        aliases += alias is not None
    assert execute(ops,0,a) == (-a)%P
    checked += 1
    negatives += 1

# Reject two mistakes which would otherwise resemble the correct expression.
wrong_constant = parse(ptx.replace('0xfffffffefffffc2f', '0xfffffffefffffc2e'))
assert execute(wrong_constant,0,1) != P-1
lost_borrow = parse(ptx.replace('subc.cc.u64 %1,%5,%9', 'sub.cc.u64 %1,%5,%9'))
assert execute(lost_borrow,1<<64,1) != ((1<<64)-1)

result = {
    'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
    'canonical_input_pairs': checked,
    'output_alias_checks': aliases,
    'negations': negatives,
    'negative_controls': 2,
    'mismatches': 0,
    'gpu_execution': False,
    'contract': 'Canonical inputs a,b<p; outputs (a-b) mod p canonically.',
}
(ROOT/'affine_sub_cpu_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
