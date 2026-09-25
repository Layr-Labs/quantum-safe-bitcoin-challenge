#!/usr/bin/env python3
"""Decode a ranked subset run artifact produced by the time-slot / telemetry / arm-comparison build.

usage: decode_time_slots.py run-subset.json [--slots-per-s 6.5] [--bin 30] [--ab]

* hits -> early skip set -> lex rank in C(137,6) -> slot = rank // 2^20
  (one slot per batch of 1,048,576 epochs = 134,217,728 candidates)
* slot / slots_per_s = launch time (s since main()) for time-driven slots
* prints the launch-rate / throughput profile, the startup delay and, with --ab, the per-arm
  batch counts, hit yield and throughput (arm of a batch = qsb_ab_arm_at(launch time));
* unpacks the telemetry or arm tokens packed into the self-reported numbers.
"""
import argparse, json, math, collections
from math import comb

CAP = 262144 * 4            # epochs per batch
CAND_PER_BATCH = CAP * 128  # candidates per batch
ARMS = ("GLV12", "GLV11", "GLV10")


def lex_rank(c, n):
    k = len(c); r = 0; prev = -1
    for i, x in enumerate(c):
        for v in range(prev + 1, x):
            r += comb(n - v - 1, k - i - 1)
        prev = x
    return r


def arm_at(t, phase_s=20.0):
    p = int(t // phase_s)
    return (p + p // 3) % 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--slots-per-s", type=float, default=6.5)
    ap.add_argument("--bin", type=float, default=30.0)
    ap.add_argument("--ab", action="store_true")
    ap.add_argument("--phase-s", type=float, default=20.0)
    ap.add_argument("--warmup-s", type=float, default=120.0)
    a = ap.parse_args()
    art = json.load(open(a.artifact))
    hits = art["hits"]
    slots = collections.Counter()
    for h in hits:
        s = sorted(h["skip"])
        slots[lex_rank(s[:6], 137) // CAP] += 1
    used = sorted(slots)
    print(f"hits={len(hits)} verified={art.get('verified_hits')} batches seen={len(used)} "
          f"first slot={used[0]} last slot={used[-1]} elapsed={art.get('elapsed_s')}")
    gaps = [b - a_ for a_, b in zip(used, used[1:])]
    print("slot gaps histogram:", sorted(collections.Counter(gaps).items())[:12])
    print(f"startup: first batch launched at ~{used[0] / a.slots_per_s:.2f} s after main()")
    t_of = {s: s / a.slots_per_s for s in used}
    nb = int(math.ceil((max(t_of.values()) + 1) / a.bin))
    per = collections.Counter(int(t // a.bin) for t in t_of.values())
    hits_per = collections.Counter()
    for s, c in slots.items():
        hits_per[int(t_of[s] // a.bin)] += c
    print(f"{'t0':>6} {'batches':>7} {'Mcand/s':>8} {'hits':>6} {'hit-implied M/s':>15}")
    for b in range(nb):
        n = per.get(b, 0)
        rate = n * CAND_PER_BATCH / a.bin / 1e6
        hr = hits_per.get(b, 0) * 2**23 / a.bin / 1e6
        print(f"{b * a.bin:6.0f} {n:7d} {rate:8.1f} {hits_per.get(b, 0):6d} {hr:15.1f}")
    tot = len(used) * CAND_PER_BATCH
    print(f"batches={len(used)} -> {tot / 1e9:.2f} B candidates launched; hit-implied "
          f"{art.get('hit_implied_candidates', 0) / 1e9:.2f} B")
    if a.ab:
        # per arm: batches, hits, yield; per phase: batches launched in the phase
        arm_b = collections.Counter(); arm_h = collections.Counter()
        phase_b = collections.defaultdict(int)
        for s in used:
            t = t_of[s]
            if t < a.warmup_s:
                continue
            arm = arm_at(t, a.phase_s)
            arm_b[arm] += 1; arm_h[arm] += slots[s]
            phase_b[int(t // a.phase_s)] += 1
        print("arm      batches   hits  yield/expected  launched M/s (per active second)")
        full = [p for p in phase_b if (p + 1) * a.phase_s < max(t_of.values()) - 1]
        for arm in range(3):
            ps = [p for p in full if (p + p // 3) % 3 == arm]
            nb_arm = sum(phase_b[p] for p in ps)
            secs = len(ps) * a.phase_s
            y = arm_h[arm] / (arm_b[arm] * CAND_PER_BATCH / 2**23) if arm_b[arm] else 0
            r = nb_arm * CAND_PER_BATCH / secs / 1e6 if secs else 0
            print(f"{ARMS[arm]:7s} {arm_b[arm]:8d} {arm_h[arm]:6d} {y:14.4f} {r:12.2f}  ({len(ps)} full phases)")
    cand = art.get("candidates")
    thr = (art.get("self_reported") or {}).get("throughput_Mps")
    if cand and cand >= 10**18:
        c = str(cand // 10**6)
        mark = int(c[0])
        if a.ab:
            e0 = int(c[1:5]) / 10.0
            e1 = e0 * (1 + (int(c[5:9]) - 5000) / 1e4)
            e2 = e0 * (1 + (int(c[9:13]) - 5000) / 1e4)
            src = {1: 'energy counter', 2: 'energy counter', 5: 'power readings', 6: 'power readings'}.get(mark, 'UNAVAILABLE')
            print(f"arms (process, NVML energy): carrier={'on' if mark in (1, 3, 5) else 'off'} "
                  f"energy={src}  nJ/candidate: "
                  f"GLV12 {e0:.1f}  GLV11 {e1:.2f}  GLV10 {e2:.2f}")
        else:
            print(f"telemetry: carrier={'on' if mark in (1, 3) else 'off'} nvml={'ok' if mark in (1, 2) else 'FAILED'} "
                  f"power_limit={int(c[1:4])} W  late: power={int(c[4:7])} W sm={int(c[7:11])} MHz temp={int(c[11:13])} C")
    if thr is not None and thr < 1000:
        r = f"{round(thr, 6):010.6f}"
        if a.ab:
            r0 = int(r[0:3]); r1 = 1 + (int(r[4:7]) - 500) / 1e3; r2 = 1 + (int(r[7:10]) - 500) / 1e3
            print(f"arms (process clock): GLV12 {r0} M/s  GLV11 x{r1:.3f}  GLV10 x{r2:.3f}")
        else:
            print(f"telemetry: late throttle tenths swpwr={r[0]} swthermal={r[1]} hw={r[2]}  "
                  f"early: sm={int(r[4:7]) * 10} MHz power={int(r[7:10])} W")


if __name__ == "__main__":
    main()
