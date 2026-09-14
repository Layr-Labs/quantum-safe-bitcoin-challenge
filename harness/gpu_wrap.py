#!/usr/bin/env python3
"""
GPU→artifact bridge for the candidate kernels.

Compiles a track .cu with -DQSB_ZEROS_N=<N>, runs it under a timeout on a
single GPU, parses throughput + candidate count from stdout, collects hits from
the kernel's results/*.txt, and writes the run-artifact JSON the verifier scores.

BEST-EFFORT / UNTESTED WITHOUT A GPU. On your first GPU run at a low N (many
hits), confirm the harness's independent verify agrees with the kernel
(kernel↔verifier agreement) — especially the subset STORAGE index convention.

Called by run_benchmark.py as:  cmd:python3 harness/gpu_wrap.py --src <file>
(run_benchmark appends --bench --zeros --mode --out --seconds --hits).
Ranked/bridge runs pass --no-build so a read-only checkout never invokes nvcc
(the kernel must already be prebuilt by setup.sh). cwd is the parent of --out,
not this file's directory; hits are collected from <out.parent>/results/.

Each track is a self-contained directory (candidates/pinning/,
candidates/subset/) with its own kernel and GPUMath.h/GPUHash.h. This bridge
sits in harness/ and is outside both editable surfaces.
"""
from __future__ import annotations
import argparse, glob, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ROOT = HARNESS.parent
MPS = re.compile(r"([0-9.]+)M/s")
SEARCHED = re.compile(r"\((\d+)M/")          # progress "(12M/34567M)"
DONE = re.compile(r"Done:\s*(\d+)M")


def compile_kernel(src: Path, zeros: int, no_build: bool = False) -> Path:
    """Build src for this N, reusing an existing binary when it is already
    current. The harness times this whole process, so a ranked run should have
    compiled during setup.sh and hit the cache here."""
    # The binary and its build stamp sit beside the source, inside that track's
    # editable directory, so the two tracks never overwrite each other's build.
    binp = src.parent / src.stem
    stamp = src.parent / f".{src.stem}.build"
    want = f"QSB_ZEROS_N={zeros} {src.stat().st_mtime_ns}"
    if binp.exists() and stamp.exists() and stamp.read_text() == want:
        print(f"  compile: reusing {binp} (already built for N={zeros})", flush=True)
        return binp
    if no_build:
        print(
            f"kernel for N={zeros} is not prebuilt; run ./setup.sh {src.parent.name}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(3)
    # Quoted includes resolve against the .cu's own directory. Each track
    # ships GPUMath.h and GPUHash.h beside the kernel, so no extra include path.
    cmd = ["nvcc", "-O3", f"-DQSB_ZEROS_N={zeros}",
           "-o", str(binp), str(src), "-lcrypto", "-lm"]
    print("  compile:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    stamp.write_text(want)
    return binp


def subset_seq_lt(tx_suffix: bytes) -> tuple[int, int]:
    """Values already encoded in tx_suffix that the subset kernel would patch."""
    seq = int.from_bytes(tx_suffix[0:4], "little")
    lt = int.from_bytes(tx_suffix[len(tx_suffix) - 8 : len(tx_suffix) - 4], "little")
    return seq, lt


def kernel_argv(bench: str, binp: Path, probdir: Path, seconds: float) -> list[str]:
    if bench == "pinning":
        run = [str(binp), str(probdir / "pinning.bin"), "0", "1", "0"]
    else:
        # Kernel patches tx_suffix[0:4] with sequence and tx_suffix[len-8:len-4]
        # with locktime. The verifier hashes those bytes verbatim, so pass the
        # values already there — the patch is then a no-op.
        suffix = bytes.fromhex(
            json.loads((probdir / "subset.json").read_text())["tx_suffix"]
        )
        seq, lt = subset_seq_lt(suffix)
        run = [str(binp), str(probdir / "subset.bin"), "0", str(seq), str(lt), "1", "0"]
    run.append("single_hash")
    return ["stdbuf", "-oL", "timeout", str(int(seconds))] + run


def candidate_count(searched_m: list[int], thr_mps: float, elapsed_s: float, returncode: int) -> int:
    """The candidate total for the artifact, from the kernel's stdout.

    The kernels print their cumulative count only on progress lines (subset: about every 60 s),
    so when `timeout` ends the run (rc 124) the last printed count can lag the truth by a whole
    interval. Seen on the 4090: a 120 s subset run whose last sample was at 60 s reported half
    its candidates and failed the Poisson band. On a timeout kill, the rate-based extrapolation
    over the full elapsed time is the better estimate; keep the reported figure only if larger.
    """
    reported = max(searched_m) * 1_000_000 if searched_m else 0
    extrapolated = int(thr_mps * 1e6 * elapsed_s)
    if returncode == 124:
        return max(reported, extrapolated)
    return reported or extrapolated


def collect_hits(bench: str, results_dir: Path):
    hits = []
    if bench == "subset":
        for f in glob.glob(str(results_dir / "digest_hit_*.txt")):
            cur_skip = None
            cur_recid = None
            lines = Path(f).read_text().splitlines()
            for line_number, line in enumerate(lines):
                mc = re.search(r"(?:combo|indices)=([0-9,]+)", line)
                mr = re.search(r"recid=(\d)", line)
                if mc:
                    if cur_skip is not None:
                        if cur_recid is None:
                            raise ValueError(f"incomplete subset hit in {f}")
                        hits.append({"bench": "subset",
                                     "skip": cur_skip, "recid": cur_recid})
                    raw_skip = mc.group(1).split(",")
                    if any(not x for x in raw_skip):
                        if line_number == len(lines) - 1:
                            cur_skip = None
                            break
                        raise ValueError(f"incomplete subset hit in {f}")
                    cur_skip = [int(x) for x in raw_skip]
                    cur_recid = int(mr.group(1)) if mr else None
                elif mr is not None:
                    cur_recid = int(mr.group(1))
            # timeout can stop the writer midway through its final record.
            if cur_skip is not None and cur_recid is not None:
                hits.append({"bench": "subset",
                             "skip": cur_skip, "recid": cur_recid})
    else:  # pinning: accumulate a record per 'sequence=' line
        for f in glob.glob(str(results_dir / "pinning_hit_*.txt")):
            cur = {}
            for line in Path(f).read_text().splitlines():
                for key, cast in (("sequence", int), ("locktime", int), ("recid", int)):
                    m = re.search(rf"{key}=(-?\d+)", line)
                    if m:
                        if key == "sequence" and cur:
                            if not {"sequence", "locktime", "recid"} <= cur.keys():
                                raise ValueError(f"incomplete pinning hit in {f}")
                            hits.append(cur); cur = {}
                        cur[key] = cast(m.group(1))
            # timeout can stop the writer midway through its final record.
            if {"sequence", "locktime", "recid"} <= cur.keys():
                hits.append({"bench": "pinning", "sequence": cur["sequence"],
                             "locktime": cur["locktime"], "recid": cur["recid"]})
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="candidates/pinning/pinning.cu or candidates/subset/subset.cu")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--zeros", type=int, required=True)
    ap.add_argument("--mode", default="fixed_time")
    ap.add_argument("--seconds", type=float, default=1200)
    ap.add_argument("--hits", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-build", action="store_true",
                    help="do not invoke nvcc; exit 3 if the kernel is not prebuilt")
    args = ap.parse_args()

    work = Path(args.out).resolve().parent
    results_dir = work / "results"
    shutil.rmtree(results_dir, ignore_errors=True)
    results_dir.mkdir(exist_ok=True)
    src = Path(args.src).resolve()
    binp = compile_kernel(src, args.zeros, no_build=args.no_build)

    # A ranked run generates a fresh problem instance and points QSB_PROBLEM_DIR
    # at it; grind the instance for THIS run, not the committed example.
    probdir = Path(os.environ.get("QSB_PROBLEM_DIR") or (ROOT / "problems"))
    run = kernel_argv(args.bench, binp, probdir, args.seconds)
    print("  run:", " ".join(run), flush=True)

    t0 = time.time()
    proc = subprocess.run(run, cwd=str(work), capture_output=True, text=True)
    elapsed = time.time() - t0
    out = proc.stdout + proc.stderr

    rates = [float(x) for x in MPS.findall(out)]
    thr = max(rates) if rates else 0.0
    searched = [int(x) for x in SEARCHED.findall(out)] + [int(x) for x in DONE.findall(out)]
    candidates = candidate_count(searched, thr, elapsed, proc.returncode)
    hits = collect_hits(args.bench, results_dir)

    artifact = {
        "bench": args.bench, "zeros_n": args.zeros, "mode": args.mode,
        "grinder": src.relative_to(ROOT).as_posix() if src.is_relative_to(ROOT) else str(src),
        "candidates": candidates, "elapsed_s": round(elapsed, 3),
        "throughput_Mps": round(thr, 6), "verified_hits": None, "hits": hits,
        "problem_seed": json.loads((probdir / f"{args.bench}.json").read_text()).get("seed"),
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2))
    print(f"  → {args.out}: {thr:.1f}M/s, {candidates:,} candidates, {len(hits)} hits")


if __name__ == "__main__":
    main()
