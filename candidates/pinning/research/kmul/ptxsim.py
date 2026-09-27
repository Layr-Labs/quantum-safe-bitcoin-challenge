#!/usr/bin/env python3
"""Minimal exact PTX interpreter for the Karatsuba product stage; checks x0..x15 == a*b.

Unknown opcodes fail closed. Carry flag semantics per PTX ISA (add.cc/addc/sub.cc/subc).
"""
import random, re, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from gen_kmul import product_stage

M32, M64 = (1 << 32) - 1, (1 << 64) - 1


def run(lines, a, b):
    R = {}
    cc = [0]
    params = {4 + i: (a >> (64 * i)) & M64 for i in range(4)}
    params.update({8 + i: (b >> (64 * i)) & M64 for i in range(4)})

    def val(tok, bits):
        tok = tok.strip()
        if tok.startswith('%'):
            return params[int(tok[1:])]
        if re.fullmatch(r'-?\d+', tok):
            return int(tok) & ((1 << bits) - 1)
        return R[tok]

    for ln in lines:
        ln = ln.strip().rstrip(';')
        if ln.startswith('.reg'):
            continue
        op, rest = ln.split(None, 1)
        if op == 'ld.const.u32':
            d = rest.split(',')[0].strip(); R[d] = 0; continue
        if op == 'mov.b64':
            d, s = [t.strip() for t in rest.split(',', 1)] if not rest.startswith('{') else (rest[:rest.index('}') + 1], rest[rest.index('}') + 2:].strip())
            if d.startswith('{'):
                lo, hi = [t.strip() for t in d[1:-1].split(',')]
                v = val(s, 64); R[lo] = v & M32; R[hi] = v >> 32
            else:
                lo, hi = [t.strip() for t in s[1:-1].split(',')]
                R[d] = R[lo] | (R[hi] << 32)
            continue
        args = [t.strip() for t in rest.split(',')]
        d = args[0]
        if op == 'add.u32':
            R[d] = (val(args[1], 32) + val(args[2], 32)) & M32; continue
        if op == 'mov.b32':
            R[d] = val(args[1], 32); continue
        if op == 'mul.wide.u32':
            R[d] = val(args[1], 32) * val(args[2], 32); continue
        if op == 'xor.b32':
            R[d] = val(args[1], 32) ^ val(args[2], 32); continue
        m = re.fullmatch(r'(add|addc|sub|subc)(\.cc)?\.u(32|64)', op)
        if not m:
            raise SystemExit('unknown op: ' + ln)
        kind, setcc, bits = m.group(1), m.group(2), int(m.group(3))
        mask = (1 << bits) - 1
        x, y = val(args[1], bits), val(args[2], bits)
        if kind == 'add':
            s = x + y
        elif kind == 'addc':
            s = x + y + cc[0]
        elif kind == 'sub':
            s = x - y
        else:
            s = x - y - cc[0]
        R[d] = s & mask
        if setcc:
            if kind in ('add', 'addc'):
                cc[0] = 1 if s >> bits else 0
            else:
                cc[0] = 1 if s < 0 else 0   # borrow
    out = 0
    for i in range(16):
        out |= R['x%d' % i] << (32 * i)
    return out


def main():
    lines = product_stage()
    rnd = random.Random(20260928)
    T = (1 << 128) - 1
    edge = [0, 1, T, T << 128, (1 << 256) - 1, 1 << 128, (1 << 128) + 1, (1 << 255),
            ((1 << 256) - 1) ^ (1 << 128), (1 << 256) - (1 << 32) - 977]
    cases = [(x, y) for x in edge for y in edge]
    for _ in range(20000):
        x = rnd.getrandbits(256); y = rnd.getrandbits(256)
        cases.append((x, y))
    # equal / near-equal halves (dA or dB zero, sign boundaries)
    for _ in range(5000):
        h = rnd.getrandbits(128); d = rnd.choice([0, 1, -1, 2, -2])
        lo = (h + d) & T
        x = (h << 128) | lo
        y = rnd.getrandbits(256)
        cases.append((x, y)); cases.append((y, x))
        h2 = rnd.getrandbits(128); lo2 = (h2 + rnd.choice([0, 1, -1])) & T
        cases.append((x, (h2 << 128) | lo2))
    # limb-extreme patterns
    for _ in range(5000):
        f = lambda: sum(rnd.choice([0, M32, rnd.getrandbits(32)]) << (32 * i) for i in range(8))
        cases.append((f(), f()))
    bad = 0
    for x, y in cases:
        if run(lines, x, y) != x * y:
            bad += 1
            if bad < 5:
                print('MISMATCH', hex(x), hex(y))
    n_mul = sum(1 for l in lines if l.startswith('mul.wide'))
    n_alu = sum(1 for l in lines if re.match(r'(add|addc|sub|subc)(\.cc)?\.u32|xor', l))
    print(f'cases={len(cases)} mismatches={bad} mul.wide={n_mul} u32-alu={n_alu}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
