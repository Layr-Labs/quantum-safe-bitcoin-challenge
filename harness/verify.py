#!/usr/bin/env python3
"""
Standalone verifier (anti-cheat core of the benchmark).

Given a run artifact + the synthetic problem, INDEPENDENTLY re-derives every
claimed hit from scratch — rebuild preimage → SHA-256d → z → ECDSA recover →
SHA-256(compress(Q)) → h — and asserts h really satisfies the gate
(leading_zero_bits(h) ≥ N). Any bad hit rejects the run, so a
kernel cannot report throughput without doing the real EC work.

Also emits a statistical diagnostic tying the claimed candidate count to the
number of verified hits (you can't claim 10⁹ candidates but show 3 hits at a
difficulty where 10⁹ should yield ~thousands, nor vice-versa). The band never
rejects a run: the ranked count is derived from verified hits, not from
`candidates`. `--strict` is accepted but does not change the exit code.

Usage:  python3 harness/verify.py --artifact run.json [--strict]
Exit 0 iff every hit verifies (and there is ≥1 hit).
"""
from __future__ import annotations
import argparse, json, math, multiprocessing, os, sys
from pathlib import Path

import crypto as C
import problem as PB


UNSET = object()   # argparse default: flag not given, so fall back to artifact/config

_WORKER_PROB = None
_WORKER_N = None


def verify_workers() -> int:
    """How many processes to use for per-hit re-derivation.

    Returns int(os.environ["QSB_VERIFY_WORKERS"]) when that variable is set
    and ≥ 1. Otherwise ``os.cpu_count() or 1``. The env var exists for tests
    and diagnostics only — the judge uses the host CPU count at runtime.
    Never hard-coded; not read from the runner profile.
    """
    raw = os.environ.get("QSB_VERIFY_WORKERS")
    if raw is not None:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = 0
        if n >= 1:
            return n
    return os.cpu_count() or 1


def _init_worker(prob, N):
    global _WORKER_PROB, _WORKER_N
    _WORKER_PROB = prob
    _WORKER_N = N


def _check_hit(item):
    """Re-derive one hit. Returns (index, failure_message_or_None)."""
    index, hit = item
    recid = hit["recid"]
    try:
        h = PB.candidate_hash(_WORKER_PROB, hit, recid)
    except Exception as e:
        return (index, f"re-derivation error: {e}")
    if h is None:
        return (index, "recovery failed (r not on curve)")
    lz = C.leading_zero_bits(h)
    if lz < _WORKER_N:
        return (index, f"only {lz} leading zero bits (need {_WORKER_N})")
    return (index, None)


def opt_float(v: str):
    """A float, or None for 'none'/'null' (gate off)."""
    return None if v.lower() in ("none", "null") else float(v)


def hit_relative_variance(N: int, K: int) -> float:
    """Relative std-dev of the hit-rate estimate from K verified hits at
    difficulty N:  Var = sqrt((1-p)/K),  p = 2^-N.  Infinite for K == 0."""
    if K <= 0:
        return math.inf
    p = 2.0 ** -N
    return math.sqrt((1.0 - p) / K)


def hit_implied_candidates(verified: int, N: int) -> float:
    """Invert expected hits ≈ candidates · 2 · 2⁻ᴺ.

    The /2 is the two recids tried per candidate. Returns the candidate
    count implied by `verified` hits at difficulty N.
    """
    return verified * (2 ** N) / 2


def _require_nonbool_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s must be an int" % name)
    return value


def canonical_key(bench, hit, prob) -> tuple:
    """Identity of a candidate after the same canonicalization hashing uses.

    Pinning masks sequence/locktime to 32 bits; subset sorts the skip set.
    recid is part of the key because both recids are tried per candidate.
    Raises ValueError(reason) on malformed input.
    """
    recid = hit.get("recid")
    _require_nonbool_int("recid", recid)
    if recid not in (0, 1):
        raise ValueError("recid must be 0 or 1")
    if bench == "pinning":
        sequence = _require_nonbool_int("sequence", hit.get("sequence"))
        locktime = _require_nonbool_int("locktime", hit.get("locktime"))
        if sequence < 0:
            raise ValueError("sequence must be >= 0")
        if locktime < 0:
            raise ValueError("locktime must be >= 0")
        return (sequence & 0xFFFFFFFF, locktime & 0xFFFFFFFF, recid)
    if bench == "subset":
        skip = hit.get("skip")
        if not isinstance(skip, list):
            raise ValueError("skip must be a list")
        t, n = prob["t"], prob["n"]
        if len(skip) != t:
            raise ValueError("skip must have %d entries" % t)
        seen = set()
        for i in skip:
            _require_nonbool_int("skip entry", i)
            if not (0 <= i < n):
                raise ValueError("skip index %s out of range" % i)
            if i in seen:
                raise ValueError("skip has repeats")
            seen.add(i)
        return (tuple(sorted(skip)), recid)
    raise ValueError("unknown bench %s" % bench)


def verify_artifact(artifact: dict, max_rel_var=None, prob=None, bench=None, N=None):
    eff_bench = bench if bench is not None else artifact["bench"]
    if prob is None:
        prob = PB.load_problem(eff_bench)
    eff_N = N if N is not None else artifact["zeros_n"]
    hits = artifact.get("hits", [])
    failures = []
    seen = set()

    if (bench is not None and artifact.get("bench") != bench) or \
       (N is not None and artifact.get("zeros_n") != N):
        failures.append((-1, f"artifact claims bench={artifact.get('bench')} "
                             f"zeros_n={artifact.get('zeros_n')} but the harness "
                             f"ran bench={bench} at N={N}"))

    pending = []
    for i, hit in enumerate(hits):
        try:
            key = canonical_key(eff_bench, hit, prob)
        except ValueError as e:
            failures.append((i, f"malformed candidate: {e}"))
            continue
        if key in seen:
            failures.append((i, "duplicate candidate"))
            continue
        seen.add(key)
        pending.append((i, hit))

    workers = verify_workers()
    if workers <= 1 or len(pending) < 2:
        _init_worker(prob, eff_N)
        results = [_check_hit(item) for item in pending]
    else:
        # Workers must be memory copies of the already-loaded harness.
        # spawn/forkserver re-import judge modules from disk after the
        # candidate ran, which both violates the trust boundary and (if a
        # module was tampered) makes every worker die on import and the
        # Pool respawn forever. fork is available on Linux and macOS; this
        # code has no threads, so the macOS fork caveat does not apply.
        chunksize = max(1, len(pending) // (workers * 4))
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=workers, initializer=_init_worker,
                      initargs=(prob, eff_N)) as pool:
            results = list(pool.imap_unordered(_check_hit, pending,
                                               chunksize=chunksize))
    for index, msg in results:
        if msg is not None:
            failures.append((index, msg))
    failures.sort(key=lambda f: f[0])

    verified = len(hits) - len([f for f in failures if f[0] >= 0])

    # Bind the run to its instance: a run that names a different seed than the
    # problem on disk was ground against something else, and its hits verifying
    # here would be a coincidence worth rejecting rather than crediting.
    claimed_seed, actual_seed = artifact.get("problem_seed"), prob.get("seed")
    if claimed_seed is not None and actual_seed is not None and claimed_seed != actual_seed:
        failures.append((-1, f"run claims problem_seed={claimed_seed} but the loaded "
                             f"instance has seed={actual_seed}"))

    # statistical consistency: expected hits ≈ candidates · 2 · 2^-N  (2 recids per candidate)
    # Diagnostic only — never a reason to reject. The ranked count is hit-derived.
    cand = artifact.get("candidates")
    warn = None
    if cand:
        expected = cand * 2 * (2 ** -eff_N)
        if expected > 0:
            # crude Poisson band; flag only gross inconsistency
            lo = max(0, expected - 6 * math.sqrt(expected) - 5)
            hi = expected + 6 * math.sqrt(expected) + 5
            if not (lo <= verified <= hi):
                warn = (f"verified hits ({verified}) outside expected Poisson band "
                        f"[{lo:.0f},{hi:.0f}] for {cand:,} candidates at N={eff_N} "
                        f"(expected≈{expected:.1f}) — candidate count may be misreported")

    # hit-count sufficiency: the sample must be large enough that the hit-rate
    # estimate's relative variance sqrt((1-p)/K) is below the tunable bound.
    rel_var = hit_relative_variance(eff_N, verified)
    artifact["hit_relative_variance"] = None if math.isinf(rel_var) else round(rel_var, 6)
    if max_rel_var is not None and rel_var > max_rel_var:
        k_needed = math.ceil((1.0 - 2.0 ** -eff_N) / (max_rel_var ** 2))
        failures.append((-1, f"too few verified hits: K={verified} gives relative variance "
                             f"{rel_var:.4f} > max {max_rel_var} (need K ≥ {k_needed})"))
    return verified, failures, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--strict", action="store_true",
                    help="accepted for compatibility; the Poisson band is "
                         "diagnostic-only and does not affect the exit code")
    ap.add_argument("--max-rel-var", type=opt_float, default=UNSET,
                    help="reject unless sqrt((1-2^-N)/K) <= this (K = verified hits); "
                         "'none' disables the gate; "
                         "default: artifact['max_relative_variance'] or harness/config.json")
    args = ap.parse_args()
    artifact = json.loads(Path(args.artifact).read_text())

    max_rel_var = args.max_rel_var
    if max_rel_var is UNSET:                     # not given: artifact, then config
        max_rel_var = artifact.get("max_relative_variance")
        if max_rel_var is None:
            cfg = Path(__file__).resolve().parent / "config.json"
            if cfg.exists():
                max_rel_var = json.loads(cfg.read_text()).get("max_relative_variance")

    verified, failures, warn = verify_artifact(artifact, max_rel_var)
    n = len(artifact.get("hits", []))
    if artifact.get("hit_relative_variance") is not None:
        print(f"hit relative variance sqrt((1-p)/K) = {artifact['hit_relative_variance']}"
              + (f"  (max {max_rel_var})" if max_rel_var is not None else ""))
    print(f"bench={artifact['bench']} N={artifact['zeros_n']} "
          f"hits={n} verified={verified} failed={len(failures)}")
    for i, why in failures[:20]:
        print(f"  ✗ hit[{i}]: {why}" if i >= 0 else f"  ✗ {why}")
    if warn:
        print(f"  ⚠ {warn}")

    # write verified count back into the artifact
    artifact["verified_hits"] = verified
    Path(args.artifact).write_text(json.dumps(artifact, indent=2))

    ok = (len(failures) == 0 and n > 0)
    print("RESULT:", "PASS ✅" if ok else "REJECT ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
