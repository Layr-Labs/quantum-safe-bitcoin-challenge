#!/usr/bin/env python3
"""Independent audit of the short-epoch window selector.

This models the host selection rule without importing or executing CUDA code.
It checks that the selected lanes are distinct valid candidates, reconstructs
their complete benchmark preimages, and reports the SHA schedule-class counts
before and after packing.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path


CUT = 137
WINDOW = 13
OMIT = 3
LANES = 256
PUSH_SIZE = 10


def load_problem(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def records(problem: dict) -> list[tuple[tuple[int, ...], bytes, bytes]]:
    rows = [bytes.fromhex(value) for value in problem["dummy_sigs"]]
    assert len(rows) == 150
    assert all(len(row) == PUSH_SIZE for row in rows)
    result = []
    for relative in itertools.combinations(range(WINDOW), OMIT):
        kept = b"".join(
            rows[CUT + index] for index in range(WINDOW) if index not in relative
        )
        assert len(kept) == 100
        result.append((relative, kept[:56], kept[56:100]))
    assert len(result) == 286
    return result


def select_packed(
    values: list[tuple[tuple[int, ...], bytes, bytes]],
) -> list[tuple[tuple[int, ...], bytes, bytes]]:
    first_frequency = Counter(first for _, first, _ in values)
    selected = sorted(values, key=lambda value: (-first_frequency[value[1]], value[0]))[
        :LANES
    ]
    return sorted(selected, key=lambda value: (value[2], value[1], value[0]))


def complete_preimage(problem: dict, relative: tuple[int, ...]) -> bytes:
    rows = [bytes.fromhex(value) for value in problem["dummy_sigs"]]
    early = set(range(6))
    skipped = early | {CUT + value for value in relative}
    return b"".join(
        [bytes.fromhex(problem["fixed_prefix"])]
        + [row for index, row in enumerate(rows) if index not in skipped]
        + [bytes.fromhex(problem["tail_section"]), bytes.fromhex(problem["tx_suffix"])]
    )


def main() -> int:
    problem_path = Path(sys.argv[1] if len(sys.argv) > 1 else "problems/subset.json")
    problem = load_problem(problem_path)
    values = records(problem)
    baseline = values[:LANES]
    packed = select_packed(values)

    combos = [value[0] for value in packed]
    assert len(combos) == LANES
    assert len(set(combos)) == LANES
    assert all(len(combo) == OMIT and tuple(sorted(combo)) == combo for combo in combos)
    assert all(0 <= index < WINDOW for combo in combos for index in combo)

    baseline_first = len({first for _, first, _ in baseline})
    baseline_second = len({second for _, _, second in baseline})
    packed_first = len({first for _, first, _ in packed})
    packed_second = len({second for _, _, second in packed})
    assert (baseline_first, baseline_second) == (84, 26)
    assert (packed_first, packed_second) == (54, 56)

    digests = set()
    expected_len = int(problem["total_preimage_len"])
    for combo in combos:
        preimage = complete_preimage(problem, combo)
        assert len(preimage) == expected_len
        first = hashlib.sha256(preimage).digest()
        digests.add(hashlib.sha256(first).digest())
    assert len(digests) == LANES

    second_runs = []
    for _, _, second in packed:
        if not second_runs or second_runs[-1] != second:
            second_runs.append(second)
    assert len(second_runs) == packed_second

    print(f"baseline classes: first={baseline_first} second={baseline_second}")
    print(f"packed classes:   first={packed_first} second={packed_second}")
    print(f"valid unique lanes: {len(combos)}; unique SHA256d outputs: {len(digests)}")
    print(f"contiguous second-schedule runs: {len(second_runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
