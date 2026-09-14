#!/usr/bin/env python3
"""
CPU reference grinder — a slow, correctness-only grinder used to (a) make the
benchmark runnable/testable without a GPU and (b) demonstrate the hit-record
I/O contract. It is NOT a competitive implementation and is NOT scored against
GPU kernels; it exists so the verifier + scoring can be exercised anywhere.

Emits a run artifact JSON: {bench, zeros_n, mode, candidates, elapsed_s,
throughput_Mps, hits:[...]} where each hit is a candidate the verifier can
independently re-derive.

Usage:
  python3 harness/cpu_grind.py --bench pinning --zeros 12 --mode fixed_hits --hits 20 --out run.json
  python3 harness/cpu_grind.py --bench subset  --zeros 10 --mode fixed_time --seconds 30 --out run.json
"""
from __future__ import annotations
import argparse, itertools, json, time
from pathlib import Path

import crypto as C
import problem as PB

ROOT = Path(__file__).resolve().parent.parent


def candidates_pinning():
    """Enumerate (sequence, locktime); mirrors the pinning kernel's sweep."""
    seq = 0
    while True:
        for lt in range(0, 1 << 32):
            yield {"sequence": seq, "locktime": lt}
        seq += 1


def candidates_subset(n, t):
    """Enumerate t-of-n skip combinations; mirrors the subset kernel."""
    for combo in itertools.combinations(range(n), t):
        yield {"skip": list(combo)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["pinning", "subset"], required=True)
    ap.add_argument("--zeros", type=int, required=True, help="N leading zero bits (difficulty)")
    ap.add_argument("--mode", choices=["fixed_time", "fixed_hits"], default="fixed_hits")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--hits", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "run.json"))
    ap.add_argument("--report-every", type=float, default=5.0)
    args = ap.parse_args()

    prob = PB.load_problem(args.bench)
    N = args.zeros
    if args.bench == "pinning":
        gen = candidates_pinning()
    else:
        gen = candidates_subset(prob["n"], prob["t"])

    neg_r_inv = int(prob["neg_r_inv"], 16)
    u2R, neg_2u2R = PB.precomputed_points(prob)

    hits = []
    candidates = 0
    t0 = time.time()
    t_report = t0
    for cand in gen:
        candidates += 1
        z = int.from_bytes(C.sha256d(PB.candidate_preimage(prob, cand)), "big")
        for recid in (0, 1):
            h = PB.recovered_hash_fast(prob, z, recid, neg_r_inv, u2R, neg_2u2R)
            if C.is_hit(h, N):
                rec = dict(cand); rec["bench"] = args.bench; rec["recid"] = recid
                hits.append(rec)
                break
        now = time.time()
        if now - t_report >= args.report_every:
            rate = candidates / (now - t0) / 1e6
            print(f"  {rate:.4f}M/s  candidates={candidates:,}  hits={len(hits)}", flush=True)
            t_report = now
        if args.mode == "fixed_hits" and len(hits) >= args.hits:
            break
        if args.mode == "fixed_time" and (now - t0) >= args.seconds:
            break

    elapsed = time.time() - t0
    artifact = {
        "bench": args.bench,
        "zeros_n": N,
        "mode": args.mode,
        "grinder": "cpu_reference",
        "candidates": candidates,
        "elapsed_s": round(elapsed, 4),
        "throughput_Mps": round(candidates / elapsed / 1e6, 6),
        "verified_hits": None,          # filled by verify.py
        "hits": hits,
        "problem_seed": prob.get("seed"),
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2))
    print(f"\n  {args.bench}: {candidates:,} candidates in {elapsed:.1f}s "
          f"= {artifact['throughput_Mps']:.4f}M/s, {len(hits)} hits → {args.out}")


if __name__ == "__main__":
    main()
