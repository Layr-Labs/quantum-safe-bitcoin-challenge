#!/usr/bin/env python3
"""
Emit the ranked score file for ONE bench.

Pinning and subset-selection are scored SEPARATELY and can be submitted
independently: their per-candidate algorithms differ (a 75-byte varying tail vs
a ~1.7 KB section rebuilt per candidate), so a combined number would hide where
an implementation wins or loses. Each is its own track in benchmark.json with
its own scorePath, and a submission may enter either or both.

Writes score-<bench>.json:  {"score": <candidates/s>, "metrics": {...}}
Refuses to write a score unless the run is valid (every hit verified).

Usage:
  python3 harness/score.py --out score-pinning.json benchmark-results/run-pinning.json
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

UNIT = "verified candidates per second"


def build_score(artifact: dict) -> dict:
    score = artifact.get("score") or {}
    if not score.get("valid"):
        raise ValueError("is not a valid run — no score written")

    thr_Mps = float(score["primary_throughput_Mps"])
    if not math.isfinite(thr_Mps) or thr_Mps <= 0:
        raise ValueError("has non-positive throughput — no score written")

    return {
        # Published in base units so the ranked number needs no conversion.
        "score": thr_Mps * 1e6,
        "metrics": {
            "bench": artifact["bench"],
            "unit": UNIT,
            "direction": "higher is better",
            "throughput_Mps": thr_Mps,
            "hits_per_s": score.get("secondary_hits_per_s"),
            "leading_zero_bits": artifact.get("zeros_n"),
            "mode": artifact.get("mode"),
            "candidates": artifact.get("hit_implied_candidates"),
            "candidates_self_reported": artifact.get("candidates"),
            "elapsed_s": artifact.get("elapsed_s"),
            "verified_hits": artifact.get("verified_hits"),
            "hit_relative_variance": artifact.get("hit_relative_variance"),
            "problem_seed": artifact.get("problem_seed"),
            "gpu": artifact.get("gpu"),
            "verified": True,
        },
    }


def write_score(artifact, out_path):
    result = build_score(artifact)
    Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
    print(f"score.py: {artifact['bench']} score = {result['score']:.1f} {UNIT}  →  {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("artifact")
    args = ap.parse_args()

    a = json.loads(Path(args.artifact).read_text())
    try:
        write_score(a, args.out)
    except ValueError as e:
        sys.exit(f"score.py: {args.artifact} {e}")


if __name__ == "__main__":
    main()
