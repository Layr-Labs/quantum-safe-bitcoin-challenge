#!/usr/bin/env python3
"""Interpret actual generated PTX vs Python integers; no GPU execution."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ptx_field_model import Program, check_semantics, extract_ptx, function

P = (1 << 256) - (1 << 32) - 977
B = 1 << 256
C = (1 << 32) + 977


def words(x):
    return [x >> (64*i) & ((1 << 64) - 1) for i in range(4)]


def integer(w):
    return sum(x << (64*i) for i, x in enumerate(w))


def main():
    check_semantics()
    header = HERE / "candidate/GPUMath.h"
    data = header.read_bytes()
    body = function(data.decode(), "__device__ __forceinline__ void _ModMultCore(")
    ptx = extract_ptx(body)
    assert ptx.strip() == (HERE / "field.ptx").read_text().strip()
    raw = (HERE / "product.ptx").read_text()
    prefix = ptx[:ptx.index(".reg .u64 t;")]
    assert raw.startswith(prefix)
    product, field = Program(raw), Program(ptx)
    edges = {0, 1, 2, P-65537, P-2, P-1, P, P+1, B-2, B-1}
    for bit in range(29, 256, 29):
        edges.update(((1 << bit)-1, 1 << bit, (1 << bit)+1))
    for bit in range(32, 256, 32):
        edges.update(((1 << bit)-1, 1 << bit, (1 << bit)+1))
    cases = list(itertools.product(sorted(edges), repeat=2))
    rng = random.Random(2026091629)
    cases += [(rng.getrandbits(256), rng.getrandbits(256)) for _ in range(20000)]
    final_carry, noncanonical = 0, 0
    for a, b in cases:
        inp = words(a) + words(b)
        got = integer(product.run(inp, output_count=8))
        assert got == a*b, (a, b, got, a*b)
        residue = integer(field.run(inp))
        assert residue < B
        assert (residue-P if residue >= P else residue) == a*b % P, (a,b,residue)
        first = (a*b & (B-1)) + (a*b >> 256) * C
        second = (first & (B-1)) + (first >> 256) * C
        final_carry += second >= B
        noncanonical += residue >= P
    # Carry repair and three-limb repacking boundary must both be observable.
    mutant = Program(ptx.replace("addc.cc.u32 z7, z7, 0;", "addc.u32 z7, z7, 0;"))
    a, b = P-1, P-(1 << 224)
    assert integer(mutant.run(words(a)+words(b))) % P != a*b % P
    mutant = Program(raw.replace("shl.b32 tmp, d11, 31;", "shl.b32 tmp, d11, 30;"))
    a, b = (1 << 159), (1 << 160)
    assert integer(mutant.run(words(a)+words(b), output_count=8)) != a*b
    assert final_carry and noncanonical
    assert header.read_bytes() == data
    report = {"status": "PASS", "validation_level": "actual generated PTX CPU semantic model",
              "GPUMath_sha256": hashlib.sha256(data).hexdigest(),
              "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
              "semantic_model_sha256": hashlib.sha256((HERE.parent / "ptx_field_model.py").read_bytes()).hexdigest(),
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "full_product_and_reduction_cases": len(cases),
              "final_fold_carry_cases": final_carry, "precanonical_outputs_ge_p": noncanonical,
              "mutations_rejected": ["stale final carry flag", "wrong third-limb repack shift"],
              "ptx_opcodes": dict(Counter(op for op, args in field.ops)), "gpu_executed": False,
              "limits": "No GPU execution, inline-asm constraint/alias runtime validation or performance. Host fallback and C++ canonicalization unchanged."}
    (HERE / "ptx-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
