#!/usr/bin/env python3
"""Decode a ranked subset run artifact produced by the time-slot / telemetry build.

usage: decode_run.py run-subset.json [--slots-per-s 6.5] [--bin 30]

* hits -> early skip set -> lex rank in C(137,6) -> slot = rank // 2^20
  (one slot per batch of 1,048,576 epochs = 134,217,728 candidates)
* slot / slots_per_s = launch time (s since main()) for time-driven slots
* prints the launch-rate / throughput profile, startup delay and the
  telemetry fields packed into the self-reported numbers.
"""
import argparse, json, math, collections
from math import comb

CAP = 262144 * 4            # epochs per batch
CAND_PER_BATCH = CAP * 128  # candidates per batch


def lex_rank(c, n):
    k = len(c); r = 0; prev = -1
    for i, x in enumerate(c):
        for v in range(prev + 1, x):
            r += comb(n - v - 1, k - i - 1)
        prev = x
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--slots-per-s", type=float, default=6.5)
    ap.add_argument("--bin", type=float, default=30.0)
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
    # launch-time profile (time-driven slots: t = slot / rate)
    t_of = [s / a.slots_per_s for s in used]
    nb = int(math.ceil((t_of[-1] + 1) / a.bin))
    per = collections.Counter(int(t // a.bin) for t in t_of)
    hits_per = collections.Counter()
    for s, c in slots.items():
        hits_per[int((s / a.slots_per_s) // a.bin)] += c
    print(f"{'t0':>6} {'batches':>7} {'Mcand/s':>8} {'hits':>6} {'hit-implied M/s':>15}")
    for b in range(nb):
        n = per.get(b, 0)
        span = a.bin
        rate = n * CAND_PER_BATCH / span / 1e6
        hr = hits_per.get(b, 0) * 2**23 / span / 1e6
        print(f"{b * a.bin:6.0f} {n:7d} {rate:8.1f} {hits_per.get(b, 0):6d} {hr:15.1f}")
    tot = len(used) * CAND_PER_BATCH
    print(f"batches={len(used)} -> {tot / 1e9:.2f} B candidates launched; hit-implied "
          f"{art.get('hit_implied_candidates', 0) / 1e9:.2f} B")
    # telemetry fields
    cand = art.get("candidates")
    thr = (art.get("self_reported") or {}).get("throughput_Mps")
    if cand and cand >= 10**18:
        c = str(cand // 10**6)
        mark = int(c[0])
        print(f"telemetry: carrier={'on' if mark in (1, 3) else 'off'} nvml={'ok' if mark in (1, 2) else 'FAILED'} "
              f"power_limit={int(c[1:4])} W  late: power={int(c[4:7])} W sm={int(c[7:11])} MHz temp={int(c[11:13])} C")
    if thr is not None and thr < 1000:
        r = f"{round(thr, 6):010.6f}"
        print(f"telemetry: late throttle tenths swpwr={r[0]} swthermal={r[1]} hw={r[2]}  "
              f"early: sm={int(r[4:7]) * 10} MHz power={int(r[7:10])} W")


if __name__ == "__main__":
    main()
