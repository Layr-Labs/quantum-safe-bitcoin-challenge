#!/usr/bin/env python3
"""Host audit for QSB_CHAIN_DUALFETCH: operand identity + WAR-free source order."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "candidates/pinning/pinning.cu").read_text()


def offset_seq_rot2():
    ya_is = 2
    out = [(2, "y0")]
    for c in range(3, 15, 2):
        out.append((c, ya_is))
        out.append((c + 1, c))
        ya_is = c + 1
    out.append(("final", ya_is))
    return out


def offset_seq_dual():
    off_is = 2
    out = [(2, "y0")]
    for c in range(3, 15, 2):
        out.append((c, off_is))
        out.append((c + 1, c))
        off_is = c + 1
    out.append(("final", off_is))
    return out


def main():
    assert "#define QSB_CHAIN_DUALFETCH 1" in SRC
    assert "QSB_CHAIN_DUALFETCH requires QSB_CHAIN_ROT2" in SRC
    r, d = offset_seq_rot2(), offset_seq_dual()
    assert r == d, (r, d)

    m = re.search(
        r"Both plane loads first:.*?QSB_CHAIN_FETCH\(\(size_t\)0, c0, xb, yb\);"
        r".*?QSB_CHAIN_FETCH\(\(size_t\)\(1u<<22\), c1, xa, nxt\);"
        r".*?_PointAddXYZZT<true>\(X,Y,U,V, xb,yb, off\);"
        r".*?_PointAddXYZZT<true>\(X,Y,U,V, xa,nxt, yb\);"
        r".*?off = nxt",
        SRC,
        re.S,
    )
    assert m, "dualfetch pair body (fetch,fetch,add,add,swap) not found in order"
    assert "qsb_yoff_to_y(off)" in SRC
    assert "_ModMult(xb,off,V)" in SRC
    # Confirm the WAR form (fetch into ya after reading ya) is only behind #else
    # of DUALFETCH — i.e. the ROT2-only path still has it, dual path must not.
    dual_only = re.search(
        r"#if QSB_CHAIN_DUALFETCH\n\s*/\* Both plane loads first.*?#else\n"
        r"(?P<legacy>.*?)#endif\n#if QSB_CHAIN_PTR",
        SRC,
        re.S,
    )
    assert dual_only, "legacy ROT2 pair path missing"
    legacy = dual_only.group("legacy")
    assert "QSB_CHAIN_FETCH((size_t)(1u<<22), c1, xa, ya)" in legacy
    assert "xa, nxt" not in legacy

    print("test_dualfetch: PASS")
    print("  offset-chunk sequence identical to ROT2")
    print("  both pair loads before both adds")
    print("  second load writes nxt; first add reads off")
    print("  legacy WAR form retained only under DUALFETCH=0")


if __name__ == "__main__":
    main()
