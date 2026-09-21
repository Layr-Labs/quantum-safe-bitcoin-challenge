#!/usr/bin/env python3
"""Host identities and source gates for the fused pinning point-add."""

from __future__ import annotations

import json
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
MASK64 = (1 << 64) - 1
MASK256 = (1 << 256) - 1
K = (1 << 32) + 977
P = (1 << 256) - K
C = (K - 1) // 2


def limbs(x: int) -> list[int]:
    return [(x >> (64 * i)) & MASK64 for i in range(4)]


def join(v: list[int]) -> int:
    return sum((x & MASK64) << (64 * i) for i, x in enumerate(v))


def lazy_off_words(a: int, b: int) -> tuple[int, bool]:
    """Model the fused PTX anchor add, including its documented short tail."""
    total = a + b
    t = limbs(total & MASK256)
    carry256 = total >> 256
    mk = (carry256 - 1) & MASK64
    c0 = (mk & 0xFFFFFFFEFFFFFC2F) + 1
    old0 = t[0]
    t[0] = (t[0] + c0) & MASK64
    cy = int(t[0] < old0)
    old1 = t[1]
    t[1] = (t[1] + mk + cy) & MASK64
    dropped = (carry256 == 0 and old0 < K - 1 and old1 == 0) or (
        carry256 == 1 and old0 == MASK64 and old1 == MASK64
    )
    return join(t), dropped


def audit(samples: int = 500_000) -> dict[str, int]:
    rng = random.Random(0x51A0FF)
    checked = 0
    dropped = 0
    vectors = [
        (0, 0),
        (1, P - 1),
        (P - 1, P - 1),
        (C, C),
        (C + 1, P - 1),
        (0, (1 << 128) - (K - 1)),
        (P - 1, (1 << 128) + 1),
    ]
    vectors.extend((rng.randrange(P), rng.randrange(P)) for _ in range(samples))
    for y0, y1 in vectors:
        a, b = y0 + C, y1 + C
        got, tail = lazy_off_words(a, b)
        want = (y0 + y1) % P
        if tail:
            dropped += 1
        else:
            assert got % P == want, (y0, y1, got, want)
        checked += 1
    return {"checked": checked, "documented_short_tail": dropped}


def source_audit() -> dict[str, bool]:
    cu = (HERE / "pinning.cu").read_text()
    lean = (HERE / "point_add_lean.cuh").read_text()
    return {
        "lean_default": "#define QSB_PIN_LEAN 1" in cu,
        "yoff_retained": "#define QSB_YOFF 1" in cu,
        "single_body_default": "#define QSB_CHAIN_ROT2 0" in cu,
        "lean_include": '#include "point_add_lean.cuh"' in cu,
        "lazyoff_fused": "0xFFFFFFFEFFFFFC2F" in lean
        and "addc.u64 S1,S1,f1_h" in lean,
        "lazyoff_fallback": "_ModAddLazyOff(S2" in lean,
        "anchor_inout": "mov.u64 %17,AY0" in lean,
        "finish_converts_yoff": "qsb_yoff_to_y(y0);" in cu,
        "exact_publication_gate": "#define QSB_HOST_GATE 1" in cu,
        "donor_kill_switch": "#if QSB_PIN_LEAN" in cu,
    }


def main() -> None:
    result = audit()
    src = source_audit()
    assert all(src.values()), src
    print(json.dumps({"test": "pin fused point-add + YOFF", **result, "source": src}))


if __name__ == "__main__":
    main()
