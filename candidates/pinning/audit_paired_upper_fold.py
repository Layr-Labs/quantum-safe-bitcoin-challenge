"""Check the shipped multiplication PTX with an independent integer oracle.

This is a CPU semantic check. Native CUDA and benchmark evidence are separate.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from audit_ptx_model import Program, extract_ptx, function

B = 1 << 256
P = B - (1 << 32) - 977
BRANCH = 'setp.eq.u32 correction_skip,correction_carry,0;\n@correction_skip bra correction_done;\n'
CORRECTION = 'add.cc.u32 z0,z0,977;\naddc.cc.u32 z1,z1,1;\naddc.u32 z2,z2,0;\n'


def boundary_pairs():
    k = B-P
    edges = sorted({0, 1, 2, k-1, k, P-2, P-1, P, B-1, B-2,
                    *[min(B-1, (1 << bit)+delta) for bit in range(32, 256, 32)
                      for delta in (-1, 0, 1)]})
    pairs = list(itertools.product(edges, repeat=2))
    rng = random.Random(202609200214)
    pairs += [(rng.randrange(B), rng.randrange(B)) for _ in range(4096)]
    pairs += [(P-a, P-b) for a in range(65535, 65540) for b in range(65535, 65540)]
    pairs += [(P-(1 << bit)+i, P-(1 << bit)+j)
              for bit in (48, 64, 80, 96, 112)
              for i in range(-2, 3) for j in range(-2, 3)]
    assert len(pairs) == 5207
    return pairs


def make_model(ptx):
    if BRANCH not in ptx:
        straight = Program(ptx)
        return lambda words: (straight.run(words), None)
    assert ptx.count(BRANCH) == 2
    first, middle, rest = ptx.split(BRANCH)
    assert rest.startswith(CORRECTION + 'correction_done:')
    suffix = rest[len(CORRECTION + 'correction_done:'):]
    capture = 'mov.b64 %0,{correction_carry,0};\nmov.b64 %1,0; mov.b64 %2,0; mov.b64 %3,0;}'
    initial, final = Program(first+capture), Program(first+middle+capture)
    paths = [Program(first+suffix), Program(first+middle+suffix),
             Program(first+middle+CORRECTION+suffix)]

    def run(words):
        upper, *zeros = initial.run(words)
        assert upper in (0, 1) and zeros == [0, 0, 0]
        last = 0
        if upper:
            last, *zeros = final.run(words)
            assert last in (0, 1) and zeros == [0, 0, 0]
        kind = 0 if not upper else 1 if not last else 2
        return paths[kind].run(words), kind
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    pairs, results = boundary_pairs(), {}
    for filename, name in [('GPUMath.h', '_ModMultCore'), ('GPUMath.h', '_ModSqr'),
                           ('pinning.cu', 'qsb_field_mul')]:
        path = args.source/filename
        ptx = extract_ptx(function(path.read_text(), '__device__ __forceinline__ void '+name+'('))
        model, counts = make_model(ptx), [0, 0, 0]
        if name != '_ModSqr':
            assert ptx.count('mov.b64 odd_lc, {odd_cy, cy};') == 3
        for a, b in pairs:
            values = (a,) if name == '_ModSqr' else (a, b)
            words = [(x >> (64*i)) & ((1 << 64)-1) for x in values for i in range(4)]
            actual, kind = model(words)
            expected = a*a % P if name == '_ModSqr' else a*b % P
            assert sum(x << (64*i) for i, x in enumerate(actual)) % P == expected, (name, a, b)
            if kind is not None:
                counts[kind] += 1
        assert all(counts), (name, counts)
        results[name] = {'passed': True, 'pairs': len(pairs), 'branch_counts': counts,
                         'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
                         'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    proof = {'passed': True, 'functions': results, 'scope': __doc__}
    args.output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
