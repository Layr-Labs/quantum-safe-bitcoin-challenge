#!/usr/bin/env python3
"""Static and combinatorial audit for the composed subset candidate."""
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
GPU = HERE / "tests" / "gpu_epochs"


def selected_windows():
    out = []
    for a, b, c in combinations(range(13), 3):
        if (a >= 6) or (a <= 5 and b >= 7) or (a == 0 and b == 1 and 8 <= c <= 10):
            out.append((a, b, c))
    return out


def main() -> None:
    tree = (GPU / "tree.cu").read_text()
    pair = (GPU / "pair_shared.cuh").read_text()
    groups = (GPU / "epoch_groups.cuh").read_text()
    tail = (GPU / "filter_tail_sc.cuh").read_text()
    schedule = (GPU / "window_schedule_shared.cuh").read_text()

    windows = selected_windows()
    assert len(windows) == 128 and len(set(windows)) == 128
    required = {
        "windows128": "#define QSB_SE_WINDOWS 128" in tree,
        "two_halves": "#define QSB_SE_HALVES  (QSB_SE_BLOCK / QSB_SE_WINDOWS)" in tree,
        "fast_producer": "#define QSB_EPOCH_FAST 1" in groups,
        "group_map": "d_epoch_group" in groups and "d_epoch_group" in tree,
        "schedule16": "#define QSB_FIRST_SLOTS (QSB_SE_WINDOWS==256?64:16)" in schedule,
        "shortcarry4": "#define QSB_SHORT_CARRY4 1" in tail,
        "negfold": "#define QSB_NEGFOLD_PARITY 1" in pair,
        "epoch_pair": "#define QSB_EPOCH_SHA_PAIR 1" in pair,
        "gate_pair": "#define QSB_GATE_PAIR 1" in pair,
    }
    assert all(required.values()), required
    assert pair.index("#define QSB_GATE_PAIR 1") < pair.index("#if QSB_EPOCH_SHA_PAIR && QSB_GATE_PAIR")
    print({"test": "subset composite source audit", "windows": len(windows), "checks": required})


if __name__ == "__main__":
    main()
