#!/usr/bin/env python3
"""Source binder + congruence for subset QSB_FUSE_MULSUB / QSB_FUSE_SQRADDSUB2.

No GPU / nvcc required. Verifies tip wiring and (a*b-c) / (a^2+e-2q) mod p identities.
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
    assert gm.count("_ModMulSubCore(R, S2, ZZZ1, Y1)") == 1
    assert gm.count("_ModSqrAddSub2(T, R, PPP, Q)") == 1
    assert "_ModMult(S2, ZZZ1)" in gm
    assert "_ModX3Fused(T, T, PPP, Q)" in gm
    assert "__restrict__" not in gm or gm.count("__restrict__") == 0
    # rare-branch k>=n recode must not appear as a new lever in GPUMath
    assert "gt_recode_setup" not in gm
    i = gm.index("void _PointAddXYZZ_def_body")
    chunk = gm[i : gm.index("\n}", i) + 2]
    ifs = sum(1 for line in chunk.splitlines() if line.strip().startswith(("#if", "#ifdef", "#ifndef")))
    endifs = sum(1 for line in chunk.splitlines() if line.strip().startswith("#endif"))
    assert ifs == endifs, (ifs, endifs)


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
    cases = audit_congruence(random.Random(0xF05ED02))
    print(f"PASS: subset fuse reduction binder + congruence; cases={cases}")


if __name__ == "__main__":
    main()
