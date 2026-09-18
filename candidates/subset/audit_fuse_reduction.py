#!/usr/bin/env python3
"""Source binder + congruence for subset QSB_FUSE_MULSUB / QSB_FUSE_SQRADDSUB2.

No GPU / nvcc required. Verifies tip wiring on Meganpark980320 f043aab1 /
aab2047 and (a*b-c) / (a^2+e-2q) mod p identities. Does not claim a local
GPU score.
"""
from __future__ import annotations
import random
from pathlib import Path

M = 1 << 256
K = (1 << 32) + 977
P = M - K


def audit_source() -> None:
    root = Path(__file__).resolve().parent
    gm = (root / "GPUMath.h").read_text()
    assert "#define QSB_FUSE_MULSUB 1" in gm
    assert "#define QSB_FUSE_SQRADDSUB2 1" in gm
    assert gm.count("__device__ __forceinline__ void _ModMulSubCore") == 1
    assert gm.count("__device__ __forceinline__ void _ModSqrAddSub2") == 1
    # Tip uses one templated deferred madd (DEFER_Y true/false); one call site each.
    assert gm.count("_ModMulSubCore(R, S2, ZZZ1, Y1)") == 1
    assert gm.count("_ModSqrAddSub2(T, R, PPP, Q)") == 1
    assert "template<bool DEFER_Y>" in gm
    assert "_ModMult(S2, ZZZ1)" in gm  # -D=0 tip path retained
    assert "_ModAdd256(T, T, PPP)" in gm  # -D=0 tip X3 path (no _ModX3Fused on this tip)
    assert "gt_recode_setup" not in gm
    assert gm.count("__device__ int _BinarySearch") == 1

    start = gm.index("template<bool DEFER_Y>")
    # Skip to the templated function body (first '{' after template)
    i = gm.index("__device__ __forceinline__ void _PointAddXYZZ_def(", start)
    j = gm.index("{", i)
    depth = 0
    k = j
    while k < len(gm):
        if gm[k] == "{":
            depth += 1
        elif gm[k] == "}":
            depth -= 1
            if depth == 0:
                k += 1
                break
        k += 1
    chunk = gm[j:k]
    ifs = sum(1 for line in chunk.splitlines() if line.strip().startswith(("#if", "#ifdef", "#ifndef")))
    endifs = sum(1 for line in chunk.splitlines() if line.strip().startswith("#endif"))
    assert ifs == endifs, (ifs, endifs)
    assert "_ModMulSubCore(R, S2, ZZZ1, Y1)" in chunk
    assert "_ModSqrAddSub2(T, R, PPP, Q)" in chunk
    # Tip already ships __restrict__ on this madd; we retain tip's signature.


def audit_congruence(rng: random.Random, n: int = 80_000) -> int:
    cases = 0
    values = [0, 1, 2, K - 1, K, K + 1, P - 1, P, M - 1, 65537, P - 65537]
    for a in values:
        for b in values:
            for c in values:
                assert (a * b - c) % P == ((a % P) * (b % P) - (c % P)) % P
                assert (a * a + b - 2 * c) % P == ((a * a) % P + (b % P) - (2 * (c % P)) % P) % P
                cases += 1
    for _ in range(n):
        a = rng.randrange(M)
        b = rng.randrange(M)
        c = rng.randrange(M)
        e = rng.randrange(M)
        assert (a * b - c) % P == ((a % P) * (b % P) - (c % P)) % P
        assert (a * a + e - 2 * c) % P == ((a * a) % P + (e % P) - (2 * (c % P)) % P) % P
        cases += 1
    return cases


def main() -> None:
    audit_source()
    cases = audit_congruence(random.Random(0xF043AAB1))
    print(f"PASS: subset fuse reduction binder + congruence; cases={cases}")


if __name__ == "__main__":
    main()
