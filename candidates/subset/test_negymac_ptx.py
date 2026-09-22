#!/usr/bin/env python3
"""Execute the actual inlined point PTX on the CPU; this is not a speed test.

The leader's speculative reduction tails are inherited. Random point tests must
agree modulo p; directed product tests check the exact 512-bit sum before folds.
"""
import ast
from pathlib import Path
import random
import re
import subprocess

HERE = Path(__file__).resolve().parent
P = 2**256 - 2**32 - 977
MASK = 2**64 - 1


def code(enabled, lean=1, raw=False):
    source = subprocess.check_output([
        'clang++', '-E', '-P', '-x', 'c++', '-D__CUDA_ARCH__=890',
        f'-DQSB_SUBSET_NEG_Y_MAC={enabled}', f'-DQSB_CHAIN_MUL_LEAN={lean}',
        str(HERE / 'hit_filter_field_sc.cuh')], text=True,
        stderr=subprocess.DEVNULL)
    source = source[source.index('void qsb_filter_point_add('):]
    source = source[source.index('asm(') + 4:source.index(': "+l"')]
    ptx = ''.join(ast.literal_eval(m[0]) for m in
                  re.finditer(r'"(?:[^"\\]|\\.)*"', source))
    if raw:
        ptx = ptx[ptx.index('.reg .u32 f2_a0'):]
        ptx = ptx[:ptx.index('.reg .u64 f2_r0')]
    ptx = re.sub(r'\.reg\s+[^;]+;', '', ptx)
    ptx = re.sub(r'(?m)^[ \t]*[{}][ \t]*', '', ptx).replace('; }', ';')
    result = []
    for stmt in ptx.split(';'):
        if stmt.strip():
            op, args = stmt.strip().split(None, 1)
            result.append((op, re.findall(r'\{[^}]*\}|[^,{}\s]+', args)))
    return result


def execute(program, initial):
    regs = initial.copy()
    carry = 0

    def val(s):
        return regs[s] if s in regs else int(s, 0)

    for op, args in program:
        dst = args[0]
        if op == 'mov.b64':
            src = args[1]
            if dst.startswith('{'):
                lo, hi = (x.strip() for x in dst[1:-1].split(','))
                regs[lo], regs[hi] = val(src) & 0xffffffff, val(src) >> 32
            elif src.startswith('{'):
                lo, hi = (x.strip() for x in src[1:-1].split(','))
                regs[dst] = val(lo) | val(hi) << 32
            else:
                regs[dst] = val(src) & MASK
        elif op in ('mov.u32', 'mov.u64'):
            regs[dst] = val(args[1]) & ((1 << int(op[-2:])) - 1)
        elif op == 'mul.wide.u32':
            regs[dst] = (val(args[1]) & 0xffffffff) * (val(args[2]) & 0xffffffff)
        elif op == 'mul.lo.u64':
            regs[dst] = val(args[1]) * val(args[2]) & MASK
        elif op.startswith(('add.', 'addc.', 'sub.', 'subc.')):
            bits = int(op[-2:])
            extra = carry if op.startswith(('addc.', 'subc.')) else 0
            subtract = op.startswith('sub')
            total = val(args[1]) + (-val(args[2]) - extra if subtract else val(args[2]) + extra)
            regs[dst] = total & ((1 << bits) - 1)
            if '.cc.' in op:
                carry = int(total < 0) if subtract else total >> bits
        elif op == 'and.b64':
            regs[dst] = val(args[1]) & val(args[2])
        elif op == 'neg.s64':
            regs[dst] = -val(args[1]) & MASK
        elif op == 'shr.s64':
            aa = val(args[1])
            if aa >> 63:
                aa -= 1 << 64
            regs[dst] = aa >> val(args[2]) & MASK
        elif op == 'shf.l.wrap.b32':
            combined = val(args[1]) | val(args[2]) << 32
            regs[dst] = (combined << (val(args[3]) & 31)) >> 32 & 0xffffffff
        elif op == 'setp.ne.u32':
            regs[dst] = int(val(args[1]) != val(args[2]))
        else:
            raise AssertionError(op)
    return regs


def put(regs, names, value):
    for i, name in enumerate(names):
        regs[name] = value >> (i * 64) & MASK


def get(regs, names):
    return sum(regs[name] << (64 * i) for i, name in enumerate(names))


def point(program, state, affine, anchor):
    regs = {'%16': 0}
    values = list(state) + [anchor] + list(affine)
    for start, value in zip((0, 4, 8, 12, 17, 21, 25), values):
        put(regs, [f'%{i}' for i in range(start, start + 4)], value)
    regs = execute(program, regs)
    return tuple(get(regs, [f'%{i}' for i in range(start, start + 4)]) % P
                 for start in (0, 4, 8, 12, 17))


def reference(state, affine, anchor, negative):
    x, y, zz, zzz = state
    ax, ay = affine
    u = ax * zz % P
    r = ((ay + anchor) * zzz + (y if negative else -y)) % P
    d = (u - x) % P
    pp, ppp = d*d % P, pow(d, 3, P)
    q = u * pp % P
    xx = (r*r + ppp - 2*q) % P
    yy = r * ((xx-q) if negative else (q-xx)) % P
    return xx, yy, zz*pp % P, zzz*ppp % P, ay


def add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    x, y = a
    xx, yy = b
    if x == xx and (y+yy) % P == 0:
        return None
    slope = ((3*x*x) * pow(2*y, -1, P) if a == b
             else (yy-y) * pow(xx-x, -1, P)) % P
    nx = (slope*slope-x-xx) % P
    return nx, (slope*(x-nx)-y) % P


def main():
    rng = random.Random(0x535542534554)
    whole = {(flag, lean): code(flag, lean) for flag in (0, 1) for lean in (0, 1)}
    edges = [0, 1, P-1, P, 2**256-1] + [1 << i for i in range(256)]
    triples = [(a, b, c) for a in edges for b in (0, 1, P-1, 2**256-1)
               for c in (0, 1, P-1, 2**256-1)]
    triples += [(rng.getrandbits(256), rng.getrandbits(256), rng.getrandbits(256))
                for _ in range(4096)]
    for lean in (0, 1):
        raw = code(1, lean, True)
        for a, b, c in triples:
            inputs = {}
            for prefix, v in [('S', a), ('ZZZ', b), ('YY', c)]:
                put(inputs, [f'{prefix}{i}' for i in range(4)], v)
            regs = execute(raw, inputs)
            got = sum(regs[f'f2_x{i}'] << (32*i) for i in range(16))
            assert got == a*b+c, ('raw MAC', lean, a, b, c)
    for key, program in whole.items():
        for _ in range(256):
            state = tuple(rng.randrange(P) for _ in range(4))
            affine = tuple(rng.randrange(P) for _ in range(2))
            anchor = rng.randrange(P)
            assert point(program, state, affine, anchor) == reference(state, affine, anchor, key[0]), key
    g = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
         0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
    points = [g]
    for _ in range(255):
        points.append(add(points[-1], g))
    for trial in range(128):
        selected = [rng.choice(points) for _ in range(15)]
        # Distinct initial X is required by the inherited incomplete seed.
        while selected[0][0] == selected[1][0]:
            selected[1] = rng.choice(points)
        expected = None
        for a in selected:
            expected = add(expected, a)
        a, b = selected[:2]
        d, r = (b[0]-a[0]) % P, (b[1]-a[1]) % P
        zz, zzz = d*d % P, pow(d, 3, P)
        q = a[0]*zz % P
        x = (r*r-zzz-2*q) % P
        seed_zz, seed_zzz = zz, zzz
        for negative in (0, 1):
            state = x, r*((x-q) if negative else (q-x)) % P, seed_zz, seed_zzz
            anchor = a[1]
            for affine in selected[2:]:
                result = point(whole[negative, 1], state, affine, anchor)
                state, anchor = result[:4], result[4]
            xx, yy, zz, zzz = state
            yy = ((-yy if negative else yy)-anchor*zzz) % P
            assert zz and zzz, ('incomplete-chain exception', trial)
            got = xx*pow(zz, -1, P) % P, yy*pow(zzz, -1, P) % P
            assert got == expected, ('curve sum', negative, trial)
    print({'raw_mac_cases_per_variant': len(triples), 'raw_variants': 2,
           'random_point_cases': 1024, 'curve_chains': 256,
           'point_addends_per_chain': 15, 'mismatches': 0})


if __name__ == '__main__':
    main()
