"""Explore CPU-only subset families, without CUDA or any submission.

Costs count SHA-256 compression blocks, not time or measured throughput.
Fresh full preimages and hashlib validate the shared-prefix prototype.
"""
import hashlib
import itertools as it
import json
import math
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmark.json").is_file())
(ROOT / "research").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "harness"))
import problem


def gpu_pattern(w):
    a, b, c = (i - 137 for i in w)
    return a >= 6 or (a <= 5 and b >= 7) or (a == 0 and b == 1 and 8 <= c <= 10)


def families(prob, early_count):
    rows = [bytes.fromhex(x) for x in prob["dummy_sigs"]]
    rem = (42 + (137 - early_count) * 10) % 64
    groups = defaultdict(list)
    for w in it.combinations(range(137, 150), 9 - early_count):
        if early_count == 6 and gpu_pattern(w):
            continue
        tail = b"".join(rows[i] for i in range(137, 150) if i not in w)
        # Every prefix byte before this window is identical within an epoch.
        groups[tail[:64 - rem]].append(w)
    return groups


def choose(groups, min_per_epoch):
    # Whole groups only; preserve grouping; a multiple of four avoids SIMD waste
    # between epochs when desired, but the prototype accepts all sizes.
    picked = []
    for group in sorted(groups.values(), key=lambda g: (-len(g), g[0])):
        picked.append(group)
        if sum(map(len, picked)) >= min_per_epoch:
            break
    return picked


def prototype(prob, early, groups):
    rows = [bytes.fromhex(x) for x in prob["dummy_sigs"]]
    pfx = bytes.fromhex(prob["fixed_prefix"]) + b"".join(rows[i] for i in range(137) if i not in early)
    suffix = bytes.fromhex(prob["tail_section"]) + bytes.fromhex(prob["tx_suffix"])
    take = 64 - (len(pfx) % 64)
    prefix_state = hashlib.sha256(pfx)
    for group in groups:
        first = b"".join(rows[i] for i in range(137, 150) if i not in group[0])[:take]
        state = prefix_state.copy()
        state.update(first)
        for w in group:
            sk = early + w
            assert len(sk) == 9 and len(set(sk)) == 9 and tuple(sorted(sk)) == sk
            # GPU has exactly six omissions below 137, independently of epoch.
            assert len(early) != 6 or not gpu_pattern(w)
            tail = b"".join(rows[i] for i in range(137, 150) if i not in w)
            assert tail[:take] == first
            s = state.copy()
            s.update(tail[take:] + suffix)
            actual = hashlib.sha256(s.digest()).digest()
            full = problem.sub_preimage(prob, sk)
            expected = hashlib.sha256(hashlib.sha256(full).digest()).digest()
            assert actual == expected
            yield sk, actual


def main():
    prob = json.loads((ROOT / "problems/subset.json").read_text())
    rows = []
    baseline = families(prob, 6)
    assert len(baseline) == 77 and sum(map(len, baseline.values())) == 158
    basegroups = [g for g in baseline.values() if len(g) == 5]
    assert len(basegroups) == 20 and sum(map(len, basegroups)) == 100
    target_candidates = 1200 * 100_000_000  # capacity study, NOT a predicted rate
    chosen = {}
    for early in (6, 5, 4):
        groups = families(prob, early)
        ne = math.comb(137, early)
        minimum = math.ceil(target_candidates / ne)
        pick = basegroups if early == 6 else choose(groups, minimum)
        n = sum(map(len, pick))
        blocks = (42 + (150 - 9) * 10 + 218 + 44 + 9 + 63) // 64
        prefix_blocks = (42 + (137 - early) * 10) // 64
        tail_blocks = blocks - prefix_blocks
        row = dict(early=early, window=9-early, patterns=n, groups=len(pick),
                   available_patterns=sum(map(len, groups.values())),
                   group_histogram=dict(sorted(Counter(map(len, groups.values())).items())),
                   candidate_capacity=ne*n, capacity_meets_target=ne*n >= target_candidates,
                   suffix_and_second_hash_blocks=tail_blocks - 1 + 1 + 4*math.ceil(len(pick)/4)/n,
                   prefix_blocks_per_epoch=prefix_blocks)
        rows.append(row)
        chosen[early] = pick
    rng = random.Random(271828)
    checked = 0
    recovered = 0
    for seed in (0, 17, 83):
        # Fresh synthetic bytes at unchanged shape; keep valid EC problem fields.
        fresh = dict(prob)
        if seed:
            r = random.Random(seed)
            for key in ("fixed_prefix", "tail_section", "tx_suffix"):
                fresh[key] = r.randbytes(len(bytes.fromhex(prob[key]))).hex()
            fresh["dummy_sigs"] = [r.randbytes(10).hex() for _ in range(150)]
        for early_count, pick in chosen.items():
            for index in range(4):
                early = tuple(sorted(rng.sample(range(137), early_count)))
                outputs = list(prototype(fresh, early, pick))
                assert len({sk for sk, _ in outputs}) == len(outputs)
                checked += len(outputs)
                if index == 0:
                    # The prototype's actual z goes through the independent
                    # reference EC recovery, including both recovery parities.
                    for sk, digest in (outputs[0], outputs[-1]):
                        for recid in (0, 1):
                            actual = problem.recovered_hash(fresh, int.from_bytes(digest, "big"), recid)
                            expected = problem.candidate_hash(fresh, {"skip": sk}, recid)
                            assert actual == expected
                            recovered += 1
    result = dict(baseline_commit="582a99408761f904f7f92a5d64d9ca0dcc76924a",
                  full_preimage_hash_checks=checked, independent_recovery_checks=recovered,
                  capacity_target=target_candidates, families=rows,
                  limitation="Python prototype only; not production C++, CUDA, or a throughput measurement")
    (ROOT / "research/subset-family-results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
