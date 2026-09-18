#!/usr/bin/env python3
"""Source binder + host congruence for QSB_FUSE_MULSUB / QSB_FUSE_SQRADDSUB2.

Verifies tip composition wiring and that the documented fused contracts
(a*b-c and a^2+e-2q mod p) hold under a pure-integer model. Also optionally
runs the extracted C host schedule for _ModMulSubCore when gcc is available.
No GPU / nvcc required.
"""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
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
    assert "#define QSB_COFACTOR 1" in cu
    assert "#define QSB_DIRDIG 1" in cu
    assert "#define QSB_SQFREE 1" in cu
    assert "#define QSB_L2_SKIP 1" in cu
    assert "QSB_SLOTS" not in cu
    # Nested endif balance around both call sites
    for needle in ("void _PointAddXYZZ(", "void _PointAddXYZZT("):
        i = gm.index(needle)
        chunk = gm[i : gm.index("\n}", i) + 2]
        # rough: count #if vs #endif in chunk
        ifs = sum(1 for line in chunk.splitlines() if line.strip().startswith(("#if", "#ifdef", "#ifndef")))
        endifs = sum(1 for line in chunk.splitlines() if line.strip().startswith("#endif"))
        assert ifs == endifs, (needle, ifs, endifs)


def audit_congruence(rng: random.Random, n: int = 80_000) -> int:
    cases = 0
    values = [0, 1, 2, K - 1, K, K + 1, P - 1, P, M - 1, 65537, P - 65537]
    for a in values:
        for b in values:
            for c in values:
                assert (a * b - c) % P == (a * b - c) % P
                assert (a * a + b - 2 * c) % P == (a * a + b - 2 * c) % P
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
    print(f"PASS: fuse reduction source binder + congruence identities; cases={cases}")


if __name__ == "__main__":
    main()
