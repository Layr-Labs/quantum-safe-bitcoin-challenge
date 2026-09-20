"""Interpret actual small field-add/sub PTX against Python integers.

Raw field-primitive diagnostics only; they do not establish a normal production
failure or authorize excluding any public submission from competitive checks.
Borrow semantics: NVIDIA PTX ISA, sections sub.cc and subc.
https://docs.nvidia.com/cuda/parallel-thread-execution/#extended-precision-arithmetic-instructions-subc
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
import sys

from audit_ptx_model import function, extract_ptx


def evaluate(text, a, b):
    regs = {'%' + str(4 + i): (v >> (64*j)) & ((1 << 64)-1)
            for i, (v, j) in enumerate((v, j) for v in (a, b) for j in range(4))}
    widths = {}
    flag = 0
    def read(x):
        return int(x, 0) if re.fullmatch(r'(?:0x[0-9a-fA-F]+|[0-9]+)', x) else regs[x]
    def signed(x, width):
        return x - (1 << width) if x >> (width - 1) else x
    for statement in text.strip()[1:-1].split(';'):
        statement = statement.strip()
        if not statement:
            continue
        if statement.startswith('.reg'):
            _, kind, names = statement.split(None, 2)
            for name in names.split(','):
                widths[name.strip()] = 1 if kind == '.pred' else int(kind[2:])
            continue
        if statement.startswith('@'):
            pred, statement = statement.split(None, 1)
            if not read(pred[1:]):
                continue
        op, arguments = statement.split(None, 1)
        args = [s.strip() for s in arguments.split(',')]
        dest, vals = args[0], [read(s) for s in args[1:]]
        width = 64 if dest.startswith('%') else widths[dest]
        if op in ('add.cc.u64', 'addc.cc.u64', 'addc.u64', 'add.u64'):
            value = sum(vals) + (flag if op.startswith('addc.') else 0)
            if '.cc.' in op:
                flag = value >> 64
        elif op in ('sub.cc.u64', 'subc.cc.u64', 'subc.u64', 'sub.u64'):
            value = vals[0] - vals[1] - (flag if op.startswith('subc.') else 0)
            if '.cc.' in op:
                flag = int(value < 0)
        elif op in ('mov.u64', 'mov.b64'):
            value = vals[0]
        elif op == 'and.b64':
            value = vals[0] & vals[1]
        elif op == 'neg.s64':
            value = -vals[0]
        elif op == 'setp.ge.s64':
            value = int(signed(vals[0], 64) >= signed(vals[1], 64))
        elif op == 'setp.eq.u64':
            value = int(vals[0] == vals[1])
        elif op == 'selp.u64':
            value = vals[0] if vals[2] else vals[1]
        else:
            raise ValueError(op)
        assert flag in (0, 1)
        regs[dest] = value & ((1 << width)-1)
    return sum(regs['%' + str(i)] << (64*i) for i in range(4))


def boundary_pairs(random_pairs=8192):
    b = 1 << 256
    k = (1 << 32) + 977
    p = b-k
    edge = [0, 1, 2, 977, k-1, k, k+1, (1 << 64)-1, 1 << 64,
            b//2, p-65537, p-2, p-1, p, p+1, p+2, b-2, b-1]
    pairs = list(itertools.product(edge, repeat=2))
    rng = random.Random(202609191221)
    pairs += [(rng.randrange(b), rng.randrange(b)) for _ in range(random_pairs)]
    # Densely cover the second carry/borrow region, not just random fields.
    near = [0, 1, 2, 976, 977, 978, k//2, k-2, k-1]
    pairs += [(x, p+y) for x, y in itertools.product(near, repeat=2)]
    pairs += [(p+x, p+y) for x, y in itertools.product(near, repeat=2)]
    return pairs


def audit(source, random_pairs=8192):
    p = (1 << 256) - (1 << 32) - 977
    pairs = boundary_pairs(random_pairs)
    text = source.read_text()
    result = {}
    for name in ('_ModAdd256', '_ModSub256', '_ModAddLazy'):
        body = function(text, '__device__ __forceinline__ void ' + name + '(')
        ptx = extract_ptx(body)
        mismatch = []
        for a, c in pairs:
            actual = evaluate(ptx, a, c)
            expected = (a-c if name == '_ModSub256' else a+c) % p
            if actual % p != expected:
                mismatch.append({'a':hex(a), 'b':hex(c), 'actual':hex(actual), 'expected_mod_p':hex(expected)})
        result[name] = {'passed':not mismatch, 'pairs':len(pairs),
                        'mismatches':len(mismatch), 'examples':mismatch[:6],
                        'ptx_sha256':hashlib.sha256(ptx.encode()).hexdigest()}
    return {'passed':all(r['passed'] for r in result.values()), 'functions':result,
            'header_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'scope':'CPU actual-PTX small-primitive diagnostic on arbitrary raw field representatives; no claim of normal production failure or throughput.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('header',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    assert not args.output.exists()
    result=audit(args.header)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'functions':{n:{k:v for k,v in r.items() if k!='examples'} for n,r in result['functions'].items()}},indent=2))
