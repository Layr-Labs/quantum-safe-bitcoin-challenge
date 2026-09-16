#!/usr/bin/env python3
"""CPU/source audit for the exact PR77 mechanisms retained by this candidate."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "GPUMath.h": "2f99a04753920f29942cf15bb8153819306b259f35d2d518c3000a0d61fd9684",
    "tests/gpu_epochs/tree_inverse.cuh": "059e371b7ccedacd20c7c368a00c700e4c7f6be65a536e0b50fea58240303cd8",
    "tests/gpu_epochs/window_schedule_shared.cuh": "edce85dcd53d8827cd76962e19da4b578e1f946d31f519304a94694f011214d7",
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def second_key(window: tuple[int, int, int]) -> int:
    key = count = 0
    for index in range(12, -1, -1):
        if index not in window and count < 5:
            key = (key << 4) | index
            count += 1
    return key


def first_key(window: tuple[int, int, int]) -> int:
    key = count = 0
    for index in range(13):
        if index not in window and count < 6:
            key = (key << 4) | index
            count += 1
    return key


def selected_windows() -> list[tuple[int, int, int]]:
    values = []
    for a, b, c in itertools.combinations(range(13), 3):
        if a >= 1 and c <= 7 and not (a == 1 and b == 2):
            continue
        values.append((a, b, c))
    values.sort(key=lambda window: (second_key(window), first_key(window)))
    return values


def main() -> int:
    for relative, expected in EXPECTED.items():
        assert file_sha(ROOT / relative) == expected, relative

    tree = (ROOT / "tests/gpu_epochs/tree.cu").read_text()
    inverse = (ROOT / "tests/gpu_epochs/tree_inverse.cuh").read_text()
    window_source = (ROOT / "tests/gpu_epochs/window_schedule_shared.cuh").read_text()
    assert "const int easy_flag=0,single_hash_flag=1,calibrate_flag=0;" in tree
    assert "__device__ __constant__ uint64_t QSB_U2R[8];" in tree
    assert "if(a>=1 && c<=7 && !(a==1 && b==2))continue;" in tree
    assert "qsb_window_second_key(h_win3[j-1])" in tree
    assert "qsb_select_window_schedule" not in tree
    assert "__shared__ uint64_t tree[4][512];" in inverse
    assert "if(width>32)__syncthreads();else __syncwarp();" in inverse
    assert "if((width<<1)>32)__syncthreads();else __syncwarp();" in inverse
    assert "static uint32_t qsb_window_second_key" in window_source
    assert "static uint32_t qsb_window_first_key" in window_source

    problem = Path(sys.argv[1] if len(sys.argv) > 1 else "problems/subset.json")
    parsed = json.loads(problem.read_text())
    rows = [bytes.fromhex(value) for value in parsed["dummy_sigs"]]
    windows = selected_windows()
    assert len(windows) == 256 and len(set(windows)) == 256
    blocks = []
    for window in windows:
        kept = b"".join(rows[137 + index] for index in range(13) if index not in window)
        assert len(kept) == 100
        blocks.append((kept[:56], kept[56:]))
    first_classes = len({first for first, _ in blocks})
    second_classes = len({second for _, second in blocks})
    per_warp_second = [
        len({second for _, second in blocks[start:start + 32]})
        for start in range(0, 256, 32)
    ]
    assert first_classes == 54
    assert second_classes == 56
    assert per_warp_second == [17, 4, 15, 6, 7, 7, 4, 2]
    assert sum(per_warp_second) == 62
    print(json.dumps({
        "status": "PASS",
        "exact_pr77_files": len(EXPECTED),
        "selected_windows": len(windows),
        "first_classes": first_classes,
        "second_classes": second_classes,
        "per_warp_second_classes": per_warp_second,
        "per_warp_second_class_sum": sum(per_warp_second),
        "note": "Exact problem/source gives 62, not the PR77 note's stated 32.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
