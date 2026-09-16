#!/usr/bin/env python3
"""Bounded symbolic SHA work model; not a benchmark or production enumerator.

Model the submitted first-block state sharing and first 256 lexicographic
window choices. Each symbolic row byte is distinct. Count compression calls,
not instructions, latency, occupancy, or cache traffic. The sustainable-space
filter is an explicit planning assumption, not a change to the scoring rules.
"""
import argparse
import itertools
import json
import math


def geometry(window, omitted):
    early = 9 - omitted
    prefix_bytes = 42 + (150 - window - early) * 10
    prefix_blocks, remainder = divmod(prefix_bytes, 64)
    first, second = set(), set()
    samples = min(256, math.comb(window, omitted))
    for choice in itertools.islice(itertools.combinations(range(window), omitted), samples):
        skips = set(choice)
        message = tuple(
            [-1] * remainder
            + [100 + 10 * row + byte for row in range(window) if row not in skips
               for byte in range(10)]
            + list(range(10000, 10276))  # the 276-byte padded constant tail
        )
        first.add(message[:64])
        second.add(message[64:128])
        if len(message) % 64:
            raise ValueError("padded message is not block-aligned")
    remaining_blocks = len(message) // 64
    # First block shared by class; remaining first-hash blocks are per candidate.
    # Add one SHA256d outer block and two recovered-public-key hash blocks.
    # Rare early hit exit is ignored. Equal cost per compression is only a model.
    cost = (prefix_blocks + len(first)) / samples + remaining_blocks + 2
    return {
        "window": window, "window_omissions": omitted, "early_omissions": early,
        "sampled_choices": samples, "first_state_classes": len(first),
        "second_message_classes": len(second), "prefix_blocks": prefix_blocks,
        "remaining_blocks": remaining_blocks, "sha_calls_per_candidate": cost,
        "unique_candidate_space": math.comb(150 - window, early) * samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-per-second", type=float, default=500_000_000)
    parser.add_argument("--seconds", type=float, default=1200)
    args = parser.parse_args()
    current = geometry(13, 3)
    assert (current["first_state_classes"], current["second_message_classes"]) == (84, 26)
    required = math.ceil(args.candidates_per_second * args.seconds)
    rows = [geometry(w, t) for w in range(4, 33)
            for t in range(1, min(9, w - 1) + 1) if math.comb(w, t) >= 32]
    sustainable = [row for row in rows if row["unique_candidate_space"] >= required]
    sustainable.sort(key=lambda row: row["sha_calls_per_candidate"])
    print(json.dumps({
        "evidence": "symbolic operation counts only; no CUDA or timing",
        "scope": "window 4..32, omissions 1..9, first min(256,C(window,t)) choices",
        "required_space_assumption": required,
        "limitations": [
            "No block padding, inverse-group cost, schedule expansion, or hardware costs.",
            "Second-message equality permits schedule reuse, not state reuse.",
            "No hierarchical cross-epoch reuse or alternative choice ordering searched.",
            "Changing launch geometry still requires whole-warp inverse participation.",
        ],
        "submitted": current, "best_sustainable": sustainable[:5],
        "sha_only_equal_cost_speedup_ceiling": (
            current["sha_calls_per_candidate"] / sustainable[0]["sha_calls_per_candidate"]
            if sustainable else None),
    }, indent=2))


if __name__ == "__main__":
    main()
