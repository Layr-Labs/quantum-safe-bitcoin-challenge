#!/usr/bin/env python3
"""
Benchmark orchestrator: run a grinder → verify every hit → compute the score.

A grinder is ANY command that writes a run-artifact JSON to the path given by
--out (fields: bench, zeros_n, mode, candidates, elapsed_s, throughput_Mps,
hits[]). The reference CPU grinder (grinder="cpu") already conforms; a GPU
kernel plugs in via a shell command (grinder="cmd:<...>", or harness/gpu_wrap.py).

The harness owns the clock: it times the grinder process itself and recomputes
throughput from its own wall clock, so a grinder's self-reported timing cannot
inflate the score (it is kept under "self_reported" for reference only).

Score:
  primary   = candidate throughput (M/s)   — low-variance over a 15-20 min run
  secondary = verified hits / second       — anti-cheat rate; N & target_hits tunable
A run is only scored if EVERY submitted hit independently re-derives (verify.py).

Usage:
  python3 harness/run_benchmark.py                        # uses harness/config.json
  python3 harness/run_benchmark.py --bench subset --N 24 --mode fixed_time --seconds 900
  python3 harness/run_benchmark.py --grinder "cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu"
  python3 harness/run_benchmark.py --grinder bridge:/opt/starkware-challenge/bench-exec.sh
  python3 harness/run_benchmark.py --seed random          # fresh instance (ranked)
  python3 harness/run_benchmark.py --score-out score-pinning.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, secrets, shlex, shutil, signal, subprocess, sys, time
from pathlib import Path

import problem as PB
import score
import verify as V

ROOT = Path(__file__).resolve().parent.parent

UNSET = object()   # argparse default: flag not given, so keep the configured value


def opt_float(v: str):
    """A float, or None for 'none'/'null' (gate off)."""
    return None if v.lower() in ("none", "null") else float(v)


def snapshot_tree(paths) -> dict:
    """relpath -> sha256 for files and walked directories. Missing paths skipped."""
    out = {}
    for p in paths:
        p = Path(p)
        files = [p] if p.is_file() else (
            sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else []
        )
        for f in files:
            try:
                key = str(f.resolve().relative_to(ROOT))
            except ValueError:
                key = f.name
            out[key] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def snapshot_diff(before, after):
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def build_cpu_cmd(cfg, out):
    c = ["python3", str(ROOT / "harness" / "cpu_grind.py"),
         "--bench", cfg["bench"], "--zeros", str(cfg["leading_zero_bits"]),
         "--mode", cfg["mode"], "--out", out]
    if cfg["mode"] == "fixed_time":
        c += ["--seconds", str(cfg["max_seconds"])]
    else:
        c += ["--hits", str(cfg["target_hits"])]
    return c


def run_bridge(cfg, bench_exec, probdir, out_path):
    """Run the kernel through the root-owned sandbox bridge (yukon-runners).

    The bridge validates its own arguments (empty OUTDIR outside the checkout,
    problem dir inside it, bounds) and publishes root-owned run-<bench>.json and
    metrics.json into OUTDIR. Returns (artifact, metrics, rc).
    """
    try:
        problem_rel = str(probdir.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        sys.exit(f"bridge: --problem-dir {probdir} must be inside the checkout so the "
                 "read-only sandbox can see it")
    outdir = Path(os.environ.get("QSB_BRIDGE_OUTDIR") or (ROOT.parent / f"qsb-bridge-out-{cfg['bench']}"))
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = ["sudo", "-n", bench_exec, "qsb-bench-v1", "run", cfg["bench"], str(ROOT), str(outdir),
           str(int(cfg["max_seconds"])), str(cfg["leading_zero_bits"]), problem_rel]
    print(f"▶ grinder: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode

    try:
        metrics = json.loads((outdir / "metrics.json").read_text())
        assert metrics["schema"] == "yukon.gpu-bench-metrics.v1" and metrics["bench"] == cfg["bench"]
        float(metrics["wall_s"])
    except (OSError, ValueError, KeyError, AssertionError) as e:
        sys.exit(f"bridge: missing or invalid {outdir / 'metrics.json'}: {e}")

    published = outdir / f"run-{cfg['bench']}.json"
    if not published.is_file():
        if rc != 0:
            return {}, metrics, rc
        sys.exit(f"bridge: missing published artifact {published}")
    text = published.read_text()
    Path(out_path).write_text(text)
    return json.loads(text), metrics, rc


def _bridge_summary_line(metrics):
    return (f"bridge: exit={metrics.get('command_exit_status')} "
            f"wall={metrics.get('wall_s')}s evidence={metrics.get('evidence_dir')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "harness" / "config.json"))
    ap.add_argument("--bench", choices=["pinning", "subset"])
    ap.add_argument("--N", type=int, help="leading_zero_bits override")
    ap.add_argument("--mode", choices=["fixed_time", "fixed_hits"])
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--hits", type=int)
    ap.add_argument("--grinder", help="'cpu', 'cmd:<shell command that writes artifact to --out>', "
                                     "or 'bridge:<absolute path to bench-exec>'")
    ap.add_argument("--out", default=str(ROOT / "run.json"))
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--seed", default=None,
                    help="regenerate the problem instance before the run: an int, or "
                         "'random' for a fresh unpredictable instance (ranked runs). "
                         "Omit to grind the instance already on disk.")
    ap.add_argument("--problem-dir", default=None,
                    help="where the regenerated instance goes (default $QSB_PROBLEM_DIR "
                         "or problems/)")
    ap.add_argument("--max-rel-var", type=opt_float, default=UNSET,
                    help="max_relative_variance override: reject unless sqrt((1-2^-N)/K) <= this; "
                         "'none' disables the sufficiency gate (smoke tests only)")
    ap.add_argument("--score-out", default=None,
                    help="write the ranked score JSON here (in-process) when the run is valid")
    ap.add_argument("--score-copy", default=None,
                    help="if --score-out is written, also copy it into this directory")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.bench:   cfg["bench"] = args.bench
    if args.N is not None: cfg["leading_zero_bits"] = args.N
    if args.mode:    cfg["mode"] = args.mode
    if args.seconds is not None: cfg["max_seconds"] = args.seconds
    if args.hits is not None:    cfg["target_hits"] = args.hits
    if args.grinder: cfg["grinder"] = args.grinder
    if args.max_rel_var is not UNSET: cfg["max_relative_variance"] = args.max_rel_var

    use_bridge = str(cfg["grinder"]).startswith("bridge:")
    if use_bridge and cfg["mode"] != "fixed_time":
        print("bridge: mode must be fixed_time (the protocol has no fixed_hits)")
        sys.exit(1)

    # ---- problem instance ----
    # The committed instance is generated from a public seed, so hits found
    # against it can be precomputed offline and replayed as a fake fast run.
    # A ranked run therefore grinds a FRESH instance the submission could not
    # have seen: --seed random picks it after the submission is fixed.
    probdir = Path(args.problem_dir) if args.problem_dir else \
        Path(os.environ.get("QSB_PROBLEM_DIR") or (ROOT / "problems"))
    seed = None
    if args.seed is not None:
        seed = secrets.randbelow(1 << 31) if args.seed == "random" else int(args.seed)
        subprocess.run([sys.executable, str(ROOT / "harness" / "gen_problem.py"),
                        "--seed", str(seed), "--out-dir", str(probdir)], check=True)
    # Point both the grinder subprocess and the in-process verifier at it.
    os.environ["QSB_PROBLEM_DIR"] = str(probdir)
    # Load once before the candidate runs; verification must not re-read disk.
    prob = PB.load_problem(cfg["bench"])

    # ---- run the grinder ----
    bench_exec = None
    if cfg["grinder"] == "cpu":
        cmd = build_cpu_cmd(cfg, args.out)
    elif cfg["grinder"].startswith("cmd:"):
        # the vendor command is responsible for honoring bench/N/mode and writing --out.
        # config.json holds one grinder for both benches, so "{bench}" lets it name the
        # per-track kernel (candidates/{bench}/{bench}.cu) without a second config key.
        base = cfg["grinder"][4:].replace("{bench}", cfg["bench"])
        cmd = shlex.split(base) + ["--bench", cfg["bench"], "--zeros", str(cfg["leading_zero_bits"]),
                                   "--mode", cfg["mode"], "--out", args.out,
                                   "--seconds", str(cfg["max_seconds"]), "--hits", str(cfg["target_hits"])]
    elif use_bridge:
        bench_exec = cfg["grinder"][len("bridge:"):]
        cmd = None
    else:
        print(f"unknown grinder: {cfg['grinder']}"); sys.exit(2)

    watch = [probdir]
    watch.extend((ROOT / "harness").glob("*.py"))
    watch.extend([
        ROOT / "harness" / "config.json",
        ROOT / "benchmark.sh",
        ROOT / "setup.sh",
        ROOT / "benchmark.json",
        ROOT / "harness" / "gpu_wrap.py",
    ])
    before = snapshot_tree(watch)

    bridge_metrics = None
    if use_bridge:
        _copied, bridge_metrics, rc = run_bridge(cfg, bench_exec, probdir, args.out)
        harness_elapsed = float(bridge_metrics["wall_s"])
        if rc != 0:
            print(_bridge_summary_line(bridge_metrics))
            print(f"grinder exited {rc}"); sys.exit(rc)
    else:
        print(f"▶ grinder: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        proc = subprocess.Popen(cmd, start_new_session=True)
        rc = proc.wait()
        harness_elapsed = time.time() - t0
        # Defense in depth: reap the grinder's session. A process that calls setsid
        # itself escapes this; the outer sandbox's PID namespace is the real guarantee.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if rc != 0:
            print(f"grinder exited {rc}"); sys.exit(rc)

    after = snapshot_tree(watch)
    tampered = snapshot_diff(before, after)

    # ---- the harness owns the clock ----
    # A grinder's own elapsed_s/throughput_Mps are advisory only. Scoring them
    # directly would let a submission inflate its rate by under-reporting its
    # runtime, or by reporting a peak rate instead of a mean. Score from the
    # harness's wall clock, which brackets the whole grinder process — so
    # anything the submission does (including compiling) is timed. Build in
    # setup.sh, not inside the grinder.
    # In bridge: mode the clock is root's wall_s (GPU admission is not scored).
    artifact = json.loads(Path(args.out).read_text())
    if bridge_metrics is not None:
        artifact["bridge_metrics"] = bridge_metrics
    artifact["self_reported"] = {"elapsed_s": artifact.get("elapsed_s"),
                                 "throughput_Mps": artifact.get("throughput_Mps")}
    claimed = artifact.get("elapsed_s")
    artifact["elapsed_s"] = round(harness_elapsed, 4)

    # A grinder cannot have run longer than the harness watched it run; if it
    # claims otherwise, its whole accounting is untrustworthy. (Claiming LESS is
    # normal — startup and teardown sit inside the harness interval — and is
    # already handled by scoring the harness clock.)
    clock_failure = None
    if claimed is not None and claimed > harness_elapsed + 1.0:
        clock_failure = (f"grinder claimed {claimed:.1f}s but the harness measured only "
                         f"{harness_elapsed:.1f}s of wall time")
    max_rel_var = cfg.get("max_relative_variance")
    artifact["max_relative_variance"] = max_rel_var
    N = cfg["leading_zero_bits"]
    verified, failures, warn = V.verify_artifact(
        artifact, max_rel_var, prob=prob, bench=cfg["bench"], N=N)
    if tampered:
        failures.append((-1, "judge or problem files modified during the run: "
                             + ",".join(tampered)))
    artifact["verified_hits"] = verified
    implied = V.hit_implied_candidates(verified, N)
    artifact["hit_implied_candidates"] = int(implied)
    artifact["throughput_Mps"] = round(implied / harness_elapsed / 1e6, 6)
    artifact["gpu"] = cfg.get("gpu")
    # The seed is the harness's to record, not the grinder's to claim.
    artifact["problem_seed"] = prob.get("seed")
    ok = (len(failures) == 0 and clock_failure is None
          and len(artifact.get("hits", [])) > 0)

    # ---- score ----
    thr = artifact["throughput_Mps"]
    hps = verified / artifact["elapsed_s"] if artifact.get("elapsed_s") else 0.0
    artifact["score"] = {"primary_throughput_Mps": thr, "secondary_hits_per_s": round(hps, 6),
                         "valid": ok}
    Path(args.out).write_text(json.dumps(artifact, indent=2))

    if args.score_out and ok:
        try:
            score.write_score(artifact, args.score_out)
            if args.score_copy:
                dest_dir = Path(args.score_copy)
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.score_out, dest_dir / Path(args.score_out).name)
        except ValueError:
            sys.exit(1)

    print("\n" + "=" * 56)
    print(f"  BENCH   : {artifact['bench']}   GPU: {cfg.get('gpu')}")
    print(f"  N (zeros): {artifact['zeros_n']}   mode: {artifact['mode']}")
    print(f"  candidates (self-reported): {artifact['candidates']:,}  in {artifact['elapsed_s']:.1f}s (harness clock)")
    print(f"  candidates (from N verified hits): {artifact['hit_implied_candidates']:,}")
    if artifact["self_reported"]["elapsed_s"] is not None:
        print(f"  grinder self-reported: {artifact['self_reported']['elapsed_s']:.1f}s, "
              f"{artifact['self_reported']['throughput_Mps']} M/s (not scored)")
    print(f"  verified hits: {verified} / {len(artifact.get('hits', []))}")
    if artifact.get("hit_relative_variance") is not None:
        print(f"  hit rel. variance: {artifact['hit_relative_variance']}"
              + (f"  (max {max_rel_var})" if max_rel_var is not None else ""))
    if failures:
        for i, why in failures[:10]:
            print(f"    ✗ hit[{i}]: {why}" if i >= 0 else f"    ✗ {why}")
    if clock_failure:
        print(f"    ✗ {clock_failure}")
    if warn:
        print(f"    ⚠ {warn}")
    print("-" * 56)
    print(f"  SCORE (throughput) : {thr:.4f} M/s        ← primary (from verified hits)")
    print(f"  SCORE (hit-rate)   : {hps:.6f} hits/s")
    print(f"  RESULT: {'PASS ✅ (scored)' if ok else 'REJECT ❌ (not scored)'}")
    if bridge_metrics is not None:
        print(f"  {_bridge_summary_line(bridge_metrics)}")
    print("=" * 56)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
