#!/usr/bin/env python3
"""Full-width oracle and mutation screen for secp256k1 product parity."""

import hashlib
import json
import random
from pathlib import Path

P = (1 << 256) - (1 << 32) - 977
C = (1 << 32) + 977
MASK = (1 << 256) - 1
HERE = Path(__file__).resolve().parent


def folded_features(a: int, b: int):
    n = a * b
    lo, hi = n & MASK, n >> 256
    t = lo + C * hi
    t0, t1 = t & MASK, t >> 256
    u = t0 + C * t1
    assert 0 <= u < 2 * P
    ge = int(u >= P)
    return (lo & 1, hi & 1, t1 & 1, ge), u


def predicted(a: int, b: int, mask: int = 15):
    features, _ = folded_features(a, b)
    return sum(features[i] for i in range(4) if mask & (1 << i)) & 1


def main():
    rng = random.Random(0x2565041521)
    edge = [0, 1, 2, C - 1, C, C + 1, P // 2, P - C, P - 2, P - 1]
    pairs = [(a, b) for a in edge for b in edge]
    pairs += [(rng.randrange(P), rng.randrange(P)) for _ in range(200_000)]
    failures = [0] * 16
    witnesses = [None] * 16
    max_u = 0
    for a, b in pairs:
        want = (a * b % P) & 1
        _, u = folded_features(a, b)
        max_u = max(max_u, u)
        for mask in range(16):
            if predicted(a, b, mask) != want:
                failures[mask] += 1
                if witnesses[mask] is None:
                    witnesses[mask] = {"a": hex(a), "b": hex(b)}
    assert failures[15] == 0
    assert all(failures[m] for m in range(15))
    result = {
        "status": "PASS",
        "cases": len(pairs),
        "unique_zero_failure_mask": 15,
        "features": ["lo_parity", "hi_parity", "second_fold_quotient_parity", "canonical_subtraction"],
        "mutation_failures": failures,
        "first_witnesses": witnesses,
        "max_folded_u_bits": max_u.bit_length(),
        "scope": "Boundary/random full-width oracle screen; Bend exhaustively synthesizes the analogous reduced-radix mask. Not a universal proof or CUDA performance result.",
        "model_sha256": hashlib.sha256((HERE / "MODEL.bend").read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "cases", "unique_zero_failure_mask", "max_folded_u_bits")}, indent=2))


if __name__ == "__main__":
    main()
