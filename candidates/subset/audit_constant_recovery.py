#!/usr/bin/env python3
"""Source-bound audit for the constant-memory recovery point on exact80."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TREE = ROOT / "tests/gpu_epochs/tree.cu"
PROBLEM = ROOT.parents[1] / "problems/subset.json"


def require_once(text: str, needle: str) -> None:
    count = text.count(needle)
    assert count == 1, f"expected one occurrence of {needle!r}, found {count}"


def limbs(value: str) -> list[int]:
    raw = int(value, 0).to_bytes(32, "little")
    return [int.from_bytes(raw[i:i + 8], "little") for i in range(0, 32, 8)]


def function(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    for end in range(brace, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end + 1]
    raise AssertionError(signature)


def main() -> int:
    tree = TREE.read_text(encoding="utf-8")
    problem = json.loads(PROBLEM.read_text(encoding="utf-8"))
    kernel = function(tree, "__global__ void __launch_bounds__(256, 2) kernel_digest(")

    require_once(tree, "__device__ __constant__ uint64_t QSB_U2R[8];")
    require_once(tree, "memcpy(h_u2r,dp.u2r_x,32);memcpy(h_u2r+4,dp.u2r_y,32);")
    require_once(tree, "cudaMemcpyToSymbol(QSB_U2R,h_u2r,sizeof(h_u2r))")
    require_once(kernel, "uint64_t u2rx[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]};")
    require_once(kernel, "uint64_t u2ry[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]};")
    assert "={d_u2rx[0]" not in kernel and "={d_u2ry[0]" not in kernel

    # Keep the ABI and allocation paths unchanged: this isolated rider only
    # substitutes the invariant coordinate loads in the recovery hot path.
    assert re.search(r"const uint64_t \* __restrict__ d_u2rx, const uint64_t \* __restrict__ d_u2ry", kernel)
    require_once(tree, "cudaMemcpy(d_u2rx,dp.u2r_x,32,cudaMemcpyHostToDevice);")
    require_once(tree, "cudaMemcpy(d_u2ry,dp.u2r_y,32,cudaMemcpyHostToDevice);")

    x_limbs, y_limbs = limbs(problem["u2r_x"]), limbs(problem["u2r_y"])
    packed = b"".join(v.to_bytes(8, "little") for v in x_limbs + y_limbs)
    expected = (int(problem["u2r_x"], 0).to_bytes(32, "little") +
                int(problem["u2r_y"], 0).to_bytes(32, "little"))
    assert packed == expected and len(packed) == 64
    print("constant recovery audit: symbol=64B, kernel x/y broadcasts=1/1")
    print("kernel ABI/global upload retained; problem limb packing: exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
