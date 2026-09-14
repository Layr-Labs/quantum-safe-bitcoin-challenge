# QSB Grinding Benchmark

A standalone benchmark for optimizing the GPU **grinding kernels** used by
Quantum-Safe Bitcoin (QSB): a proof-of-work made of an ECDSA public-key
**recovery** followed by a hash, repeated over billions of candidates. The
grind has two distinct shapes — **pinning** and **subset-selection** — which are
benchmarked separately because their per-candidate algorithms differ.

This repo lets anyone run a grinding kernel on a fixed synthetic problem and get
a **score that can't be gamed**: every reported "hit" is independently
re-derived on CPU, so a kernel cannot claim throughput without doing the real
elliptic-curve work.

## What is the benchmark

| | |
|---|---|
| **The benchmark** | the problem spec + I/O contract + CPU **verifier** + **scoring** harness. Nothing about the *algorithm* is prescribed; the build and hit-reporting interface is (see **Editable surface**). |
| **The candidate kernels** (`candidates/pinning/`, `candidates/subset/`) | the **CUDA starting seed** and the **baseline**: the imported commit is scored once, and that number is the floor every submission must beat by `minScoreImprovementBips`. It is deliberately unoptimized. After a promotion, the same directories hold the current implementation. Optimize them or replace every file in a track. See `candidates/README.md`. |

## Licensing

Unless stated below, this repository is licensed under the Apache License 2.0
in [`LICENSE`](LICENSE). That license covers the Python harness and verifier,
the challenge specification, workflows, documentation, and other non-linked
platform code.

The current implementations under `candidates/pinning/` and
`candidates/subset/` include `GPUMath.h` and `GPUHash.h` from
[VanitySearch](https://github.com/JeanLucPons/VanitySearch), which are licensed
only under GNU GPLv3. Each directory includes the complete license text in
`COPYING`. The `.cu` code that continues to use or derive from those files, and
the candidate executable compiled from them, are GPLv3-governed as one unit.
They are not covered or dual-licensed under the repository's Apache 2.0
license. Preserve the existing third-party notices; do not replace them with
Apache headers.

A participant may instead completely replace a candidate implementation. If
the replacement does not copy, modify, include, link to, or derive from any
GPLv3 code in the track, that independent replacement is covered by the root
Apache 2.0 license. Merely changing the `.cu` file while continuing to use the
GPLv3 headers does not remove the GPLv3 dependency.

## The validity gate (relaxed, tunable)

A candidate is a **hit** iff its recovered-key hash `h = SHA256(compress(Q))`:

```
leading_zero_bits(h) ≥ N          # tunable difficulty (replaces the DER check)
```

That is the whole gate — no EC-point (on-curve) check follows it.

`N` (leading zero bits) and the target hit count are **tunable** (`harness/config.json`).
The **EC recovery runs on every candidate** — that is the workload being measured; the gate above is just a cheap, tunable "did we hit."

## Running it

`setup.sh` is the one-time prerequisite step; `benchmark.sh` is the ranked run.

```bash
./setup.sh pinning      # prepare only the pinning candidate
./benchmark.sh pinning  # ranked pinning run → score-pinning.json
./setup.sh subset       # prepare only the subset candidate
./benchmark.sh subset   # ranked subset run → score-subset.json
```

The track argument to `setup.sh` is required. To run both tracks locally,
prepare each one first, then invoke `./benchmark.sh`; the scores remain separate.

Each invocation runs with the ranked settings from `harness/config.json`,
independently verifies every hit, and writes that bench's score file at the
repository root. A bench that fails verification writes **no** score for that
bench; the other bench's score is unaffected.

For a quick functional smoke test (seconds, no GPU, uses the slow CPU reference
grinder):

```bash
QSB_GRINDER=cpu QSB_ZEROS_N=6 QSB_MODE=fixed_hits QSB_HITS=3 QSB_MAX_REL_VAR=none ./benchmark.sh pinning
```

The `QSB_*` environment overrides exist for local diagnostics only — a ranked
run uses the committed defaults. The underlying harness is still directly
callable:

```bash
python3 harness/run_benchmark.py --bench pinning --N 8 --mode fixed_hits --hits 5 --grinder cpu
```

On the ranked runner, `harness/config.json` selects
`bridge:/opt/starkware-challenge/bench-exec.sh`: the kernel runs as a separate
uid in a read-only sandbox via `yukon-runners`, and root measures the wall
time. `QSB_GRINDER=cmd:…` / `cpu` remain for local work.

## Score

**The two benches are scored separately and can be submitted independently.**
Their per-candidate algorithms differ — a 75-byte varying tail vs a ~1.7 KB
section rebuilt per candidate — so a combined number would hide where an
implementation wins or loses. Each is its own track in `benchmark.json` with
its own score path, and a submission may enter either or both.

`score-<bench>.json` holds that track's ranked number, **higher is better**:

```
score = candidate throughput   (candidates/s estimated from verified hits)
```

- **Primary: candidate throughput** — `verified_hits × 2ᴺ / 2 / elapsed`, measured over a 15–20 min run. The candidate count is derived from verified hits, not from the grinder's own counter.
- **Secondary: verified hits/second** — the anti-cheat sample; `N` and target-hits are tunable. Reported under `metrics`, not ranked.

Three things keep a score honest:

- **Every submitted hit re-derives** independently (`harness/verify.py`), so
  throughput is only credited when backed by real elliptic-curve work.
- **The harness owns the clock** — throughput comes from the harness's own
  measurement of the grinder process, not from anything the grinder reports.
  Compilation counts as run time, so build in `setup.sh`.
- **A ranked run grinds a fresh problem instance**, seeded unpredictably after
  the submission is fixed, so hits precomputed against the committed public
  instance cannot be replayed. The seed is recorded in the artifact, so the run
  stays reproducible.
- **The judge holds the problem in memory**, scores in-process, kills the
  grinder's process group, and rejects a run whose judge or problem files
  changed while the grinder ran. The outer sandbox (separate uid, read-only
  checkout, PID namespace) is the actual boundary; these are defense in depth.

## Editable surface

A submission may replace anything under the paths declared as `editablePaths`
in `benchmark.json`:

| Track | Editable path | Optimization scope |
|---|---|---|
| `pinning` | `candidates/pinning/` | `pinning.cu`, `GPUHash.h`, `GPUMath.h`. Rewrite or replace any of them. New files for this track belong here. |
| `subset` | `candidates/subset/` | `subset.cu`, `GPUHash.h`, `GPUMath.h`, likewise. |

The two paths are disjoint on purpose: the tracks are ranked independently, so a
pinning submission must not be able to change what the subset track builds. Each
track ships its own `GPUHash.h` / `GPUMath.h` beside the kernel and compiles
from that directory alone.

### What the surface does *not* include

The algorithm is entirely yours. The **build and hit-reporting interface is
not** — it lives in `harness/gpu_wrap.py` and `harness/config.json`, both
outside the surface, and a submission has to fit it:

- one CUDA source at `candidates/<track>/<track>.cu`, built with
  `nvcc -O3 -DQSB_ZEROS_N=<N> -o <track> <track>.cu -lcrypto -lm`;
- the kernel's positional argv, including the `single_hash` flag;
- hits appended to `results/<track>_hit_*.txt` (pinning) or
  `results/digest_hit_*.txt` (subset), in the seed's `sequence=` / `locktime=`
  / `indices=` / `recid=` text form, which the bridge parses by regex.

So this is "optimize the CUDA grind", not "bring any language or runtime".
Within the kernel you may restructure freely — including how and how often hits
are flushed, which is worth attention: the subset seed writes a second summary
file with an `fsync` per hit that the bridge no longer reads.

If you want to plug in something the bridge cannot drive, that is a change to
`harness/`, not to a track — open an issue rather than working around it.

Everything else is **the judge**, and is outside the surface:

- `harness/**` — the runner, the verifier, the scorer, the problem generator, and `gpu_wrap.py`
- `problems/**` — the committed public problem instance
- `spec/**` — the problem and scoring definitions
- `candidates/README.md` — orientation for the editable trees
- `benchmark.json`, `setup.sh`, `benchmark.sh`, and the score path
- `.github/workflows/**`

The surface is enforced by the Yukon server: each candidate commit is composed
from the baseline tree plus only the track's `editablePaths`, so nothing
outside them can reach a ranked run. The repository's only workflows are the
ranked benchmarks.

## For competition organizers

`benchmark.json` declares **one track per bench** — `pinning` and `subset` —
each with the paths a submission may replace, its setup and benchmark commands,
its own score path, and the ranking direction. Ranked evaluation is
`workflow_dispatch` only: Yukon (or an operator) dispatches
`.github/workflows/benchmark-pinning.yml` or `benchmark-subset.yml`. Each
wrapper calls the reusable job and passes the track's required runner label
(`starkware-pinning-rtx4090` or `starkware-subset-rtx4090`) so the job lands
on `[self-hosted, <that label>]`. The tracks can run concurrently on different
hosts; each host still runs one job per GPU. The reusable job compiles via
`setup.sh` and then invokes the root bridge
`/opt/starkware-challenge/bench-exec.sh` to grind as uid 2001. The tracks are
independent, so a submission that only entered `pinning` is ranked on
`pinning` alone.

## Layout
```
benchmark.json      machine-readable contract for a competition orchestrator
setup.sh            one-time prerequisites + problem generation
benchmark.sh        the ranked run, per bench → score-<bench>.json
spec/PROBLEM.md     the two problems, dimensions, validity gate, I/O contract
spec/SCORING.md     score definition, variance, anti-cheat
harness/            THE BENCHMARK: gen_problem, run_benchmark, verify, score, cpu_grind, gpu_wrap, config.json
problems/           synthetic inputs (regenerate with gen_problem.py)
candidates/pinning/ CUDA starting seed for the pinning track — NOT the benchmark
candidates/subset/  CUDA starting seed for the subset track — NOT the benchmark
baselines/          reference scores per GPU
.github/workflows/  ranked GPU dispatch only (benchmark-pinning.yml / benchmark-subset.yml → benchmark.yml)
```

Target hardware: **not fixed** — run on whatever GPU you are characterizing.
The `gpu` label in `harness/config.json` is **declared** by the runner (recorded in
the scorecard, never auto-detected), and scores are **GPU-specific**: throughput is a
raw rate, so results are only comparable within the same hardware. Record each
calibrated GPU under `baselines/<GPU>.json` (see `baselines/TEMPLATE.json`).
