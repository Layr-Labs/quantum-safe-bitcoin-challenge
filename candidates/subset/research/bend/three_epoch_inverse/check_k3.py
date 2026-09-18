#!/usr/bin/env python3
"""Independent full-width oracle and scratch-ownership checks for K3."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

P = (1 << 256) - (1 << 32) - 977


def split3(a: int, b: int, c: int) -> tuple[int, int, int]:
    za, zb, zc = a == 0, b == 0, c == 0
    aa, bb, cc = (1 if za else a), (1 if zb else b), (1 if zc else c)
    ab = aa * bb % P
    inv_abc = pow(ab * cc % P, P - 2, P)
    inv_ab = inv_abc * cc % P
    return (
        0 if za else inv_ab * bb % P,
        0 if zb else inv_ab * aa % P,
        0 if zc else inv_abc * ab % P,
    )


def split3_mutation(a: int, b: int, c: int) -> tuple[int, int, int]:
    """Deliberate error: omit c from the combined product."""
    aa, bb, cc = (1 if a == 0 else a), (1 if b == 0 else b), (1 if c == 0 else c)
    ab = aa * bb % P
    inv_ab = pow(ab, P - 2, P)
    return (
        0 if a == 0 else inv_ab * bb % P,
        0 if b == 0 else inv_ab * aa % P,
        0 if c == 0 else inv_ab * ab % P,
    )


def expected(x: int) -> int:
    return 0 if x == 0 else pow(x, P - 2, P)


def scratch_word(slot: int, block: int, lane: int, word: int, blocks: int) -> int:
    assert 0 <= slot < 2
    assert 0 <= block < blocks
    assert 0 <= lane < 256
    assert 0 <= word < 8  # two 256-bit parked fields
    return (((slot * blocks + block) * 256 + lane) * 8) + word


def scratch_word_wrong_slot(_slot: int, block: int, lane: int, word: int, blocks: int) -> int:
    return ((block * 256 + lane) * 8) + word


def check_scratch(blocks: int = 33) -> dict[str, int]:
    seen: dict[int, tuple[int, int, int, int]] = {}
    for slot in range(2):
        for block in range(blocks):
            for lane in range(256):
                for word in range(8):
                    addr = scratch_word(slot, block, lane, word, blocks)
                    assert addr not in seen, (addr, seen[addr], (slot, block, lane, word))
                    seen[addr] = (slot, block, lane, word)
    expected_words = 2 * blocks * 256 * 8
    assert len(seen) == expected_words
    assert min(seen) == 0 and max(seen) == expected_words - 1

    wrong_seen: set[int] = set()
    wrong_collisions = 0
    for slot in range(2):
        for block in range(blocks):
            for lane in range(256):
                for word in range(8):
                    addr = scratch_word_wrong_slot(slot, block, lane, word, blocks)
                    wrong_collisions += addr in wrong_seen
                    wrong_seen.add(addr)
    assert wrong_collisions == blocks * 256 * 8
    return {"blocks": blocks, "words": expected_words, "wrong_slot_collisions": wrong_collisions}


def check_slot_protocol(launches: int = 257) -> dict[str, int]:
    live: dict[int, int] = {}
    drains = 0
    for launch in range(launches):
        slot = launch & 1
        if launch >= 2:
            prior = launch - 2
            assert live.pop(slot) == prior
            drains += 1
        assert slot not in live
        live[slot] = launch
    drains += len(live)  # mandatory final drain of both in-flight slots

    wrong_live: dict[int, int] = {}
    wrong_conflicts = 0
    for launch in range(launches):
        slot = launch & 1
        wrong_conflicts += slot in wrong_live
        wrong_live[slot] = launch
    assert wrong_conflicts == launches - 2
    return {"launches": launches, "drains": drains, "wrong_no_wait_conflicts": wrong_conflicts}


def main() -> None:
    rng = random.Random(0x513B3)
    vals = [0, 1, 2, 3, P - 1, P - 2, (1 << 255), (1 << 256) - (1 << 32) - 978]
    cases: list[tuple[int, int, int]] = []
    for a in vals:
        for b in vals:
            for c in vals:
                cases.append((a, b, c))
    for _ in range(20_000):
        cases.append((rng.randrange(P), rng.randrange(P), rng.randrange(P)))
    # Force every nonempty zero pattern through random nonzero partners.
    for mask in range(1, 8):
        row = [rng.randrange(1, P) for _ in range(3)]
        for i in range(3):
            if mask & (1 << i):
                row[i] = 0
        cases.append(tuple(row))

    mutation_rejections = 0
    first_mutation = None
    for case in cases:
        got = split3(*case)
        want = tuple(expected(x) for x in case)
        assert got == want, (case, got, want)
        if split3_mutation(*case) != want:
            mutation_rejections += 1
            if first_mutation is None:
                first_mutation = case
    assert mutation_rejections > 0 and first_mutation is not None

    source = Path(__file__).read_bytes()
    report = {
        "status": "PASS",
        "p": hex(P),
        "inverse_cases": len(cases),
        "zero_patterns": 7,
        "mutation_rejections": mutation_rejections,
        "first_mutation": [hex(x) for x in first_mutation],
        "scratch": check_scratch(),
        "slot_protocol": check_slot_protocol(),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "limits": [
            "Python big-integer oracle; not CUDA execution or a universal proof.",
            "Scratch check proves address injectivity, not host event ordering or GPU visibility.",
        ],
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
