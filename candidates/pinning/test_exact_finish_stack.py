#!/usr/bin/env python3
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
B = 1 << 256
P = B - (1 << 32) - 977
K = B - P


def combine(a: frozenset[int], b: frozenset[int]) -> frozenset[int]:
    assert a.isdisjoint(b)
    return a | b


def audit_top2_factor_sets(n: int) -> None:
    products = [frozenset() for _ in range(2 * n)]
    excluded = [frozenset() for _ in range(n)]
    for i in range(n):
        products[i] = frozenset((i,))

    offset, count = 0, n
    while count > 2:
        half = count // 2
        for tid in range(half):
            products[offset + count + tid] = combine(
                products[offset + tid], products[offset + half + tid]
            )
        offset += count
        count //= 2

    for tid in range(4):
        ia = 2 * n - 4 + ((tid & 1) ^ 1)
        ib = 2 * n - 8 + (tid ^ 2)
        excluded[n - 8 + tid] = combine(products[ia], products[ib])
    root = combine(products[2 * n - 4], products[2 * n - 3])

    offset, count = 2 * n - 16, 8
    while count < n:
        half = count // 2
        for tid in range(count):
            parent = excluded[offset + count - n + (tid & (half - 1))]
            sibling = products[offset + (tid ^ half)]
            excluded[offset - n + tid] = combine(parent, sibling)
        offset -= count * 2
        count *= 2

    universe = frozenset(range(n))
    assert root == universe
    for tid in range(n):
        value = combine(excluded[tid & (n // 2 - 1)], products[tid ^ (n // 2)])
        assert value == universe - {tid}


def lazy_add(a: int, b: int) -> int | None:
    total = a + b
    folded = (total & (B - 1)) + (total >> 256) * K
    return folded if folded < B else None


def audit_field_identities(samples: int = 100_000) -> None:
    rng = random.Random(0xF17E5A)
    for _ in range(samples):
        u = rng.randrange(B)
        v = rng.randrange(B)
        l = (u - v) % P
        m = (u + v) % P
        assert (l + m) % P == (u + u) % P

        folded = lazy_add(u, u)
        if folded is not None:
            assert folded % P == (u + u) % P

        a = rng.randrange(B)
        raw = rng.randrange(B)
        if a >> 192 != (1 << 64) - 1:
            assert raw + a < 2 * P
            reduced = raw + a
            if reduced >= P:
                reduced -= P
            assert reduced == (raw + a) % P

    for a, b in ((0, 0), (P - 1, 1), (B - K - 1, B - K - 1)):
        folded = lazy_add(a, b)
        assert folded is not None and folded % P == (a + b) % P
    assert 2 * K < 1 << 192


def audit_source() -> None:
    packed = (ROOT / "PackedRecovery.cuh").read_text()
    cofactor = (ROOT / "cofactor_checkpoint.h").read_text()
    pinning = (ROOT / "pinning.cu").read_text()
    math = (ROOT / "GPUMath.h").read_text()

    for token in (
        "#define QSB_FIN_SUM2 1",
        "_ModAddLazy(sum,u,u);",
        "#define QSB_FIN_RAWS 1",
        "#define QSB_MASK_HC 1",
        "const uint64_t *a,const uint64_t *b,const uint64_t *c",
    ):
        assert token in packed
    for token in ("#define QSB_TREE_LOOPCUT 1", "QSB_TREE_TOP2 && QSB_TREE_LOOPCUT"):
        assert token in cofactor
    for token in (
        "#define QSB_CHAIN_ROT2 1",
        "#define QSB_CHAIN_PTR 1",
        "#define QSB_DIGIT_PAIRLDS 1",
        "#define QSB_CONST_RECOVERY_ARGS 1",
    ):
        assert token in pinning
    for token in ("QSB_SAS_Z9_LANE", "QSB_TOP16", "QSB_LAZY_ADD_FINISH", "QSB_LD_PURE"):
        assert token not in math + pinning + cofactor


if __name__ == "__main__":
    for width in (16, 32, 64, 128, 256):
        audit_top2_factor_sets(width)
    audit_field_identities()
    audit_source()
    print("exact finish stack audit: PASS")
