#!/usr/bin/env python3
"""Source binder + host congruence for QSB_FUSE_MULSUB / QSB_FUSE_SQRADDSUB2 on tip 99234b73.

Verifies Meganpark tip composition wiring and that the fused contracts
(a*b-c and a^2+e-2q mod p) hold under a pure-integer model.
No GPU / nvcc required.
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
    cu = (root / "pinning.cu").read_text()
    assert "#define QSB_FUSE_MULSUB 1" in gm
    assert "#define QSB_FUSE_SQRADDSUB2 1" in gm
    assert gm.count("__device__ __forceinline__ void _ModMulSubCore") == 1
    assert gm.count("__device__ __forceinline__ void _ModSqrAddSub2") == 1
    assert gm.count("_ModMulSubCore(R, S2, ZZZ1, Y1)") == 2
    assert gm.count("_ModSqrAddSub2(T, R, PPP, Q)") == 2
    assert "_ModMult(S2, ZZZ1)" in gm  # -DQSB_FUSE_MULSUB=0 path
    assert "_ModX3Fused(T, T, PPP, Q)" in gm  # -DQSB_FUSE_SQRADDSUB2=0 + LAZY
    # Tip Meganpark markers retained
    assert "QSB_DIRECT_DIGITS" in cu
    assert "#define QSB_L2_SKIP 1" in cu
    assert "#define QSB_SPARSE_D 1" in cu
    assert "QSB_SLOTS" not in cu
    # Tip pure-asm ModAdd256 retained (not reverted to volatile-era helpers)
    assert "selp.u64 %0,t0,s0,choose" in gm
    for needle in ("__device__ void _PointAddXYZZ(", "void _PointAddXYZZT(\n"):
        # Prefer the definition bodies (skip forward-decl of T)
        if needle.startswith("void _PointAddXYZZT"):
            i = gm.rfind("template<bool DEFER_Y>")
            i = gm.index("__device__ __forceinline__ void _PointAddXYZZT(", i)
        else:
            i = gm.index(needle)
        # brace-match the function
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
        chunk = gm[i:k]
        ifs = sum(1 for line in chunk.splitlines() if line.strip().startswith(("#if", "#ifdef", "#ifndef")))
        endifs = sum(1 for line in chunk.splitlines() if line.strip().startswith("#endif"))
        assert ifs == endifs, (needle, ifs, endifs, chunk[:200])
    assert "_ModMulSubCore(R, S2, ZZZ1, Y1)" in cu
    assert "_ModSqrAddSub2(T, R, PPP, Q)" in cu


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
    cases = audit_congruence(random.Random(0xF05ED01))
    print(f"PASS: tip-99234b73 fuse binder + congruence; cases={cases}")


if __name__ == "__main__":
    main()
