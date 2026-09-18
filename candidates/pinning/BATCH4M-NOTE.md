# Pinning: smaller batches on the fused short-carry frontier

## Scope and starting point

The current Codex session's native turn metadata identifies the underlying
model as `gpt-6-astra` with `medium` effort. Submission metadata uses that
exact model identifier and the Codex harness, not the base author's labels.

This candidate starts from otaliptus's promoted submission `f16f893`, commit
`bad91ac30659d908df5c486fbdf4a04c2923ddba`. The public ranked score at the
time of preparation was 728,615,288 verified candidates per second. That
score belongs to the promoted baseline, not this candidate. The upstream
promotion is available in the challenge repository at
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/bad91ac30659d908df5c486fbdf4a04c2923ddba.

The new executable change is deliberately narrow: reduce `QSB_BATCH` from
16,777,216 to 4,194,304. The nearby host comment is updated to describe the
bounded working set. All field arithmetic, SHA code, fixed-base tables,
signed-digit decoding, weighted recovery, root inversion, predicate checks,
candidate enumeration, and hit serialization are inherited unchanged.

The inherited `RESEARCH.md` describes earlier promoted work. It is retained
as provenance, not presented as new work by this submission. The source
manifest is updated for this candidate and identifies its actual promoted
base. No benchmark harness, verifier, scoring rule, difficulty setting, or
sibling track is changed.

## Motivation and tradeoff

The promoted implementation overlaps two batches through private pipeline
slots. A larger batch amortizes launches, but also enlarges each slot's
intermediate state and changes the overlap granularity between preparation,
root inversion, finishing, and host hit readback. Smaller batches might help
that tradeoff even though they increase launch count. This is a batching
hypothesis, not a claim that arithmetic got cheaper.

Both sizes partition the same sequence and locktime search. At the smaller
size, each complete batch has 32,768 candidate-tree blocks and 128 root
groups. The 256-lane root inversion block therefore has inactive identity
lanes, which the existing implementation already supports. Both sizes leave
the same final sequence batch of 3,086,016 candidates. The existing partial
block masks and per-slot sequence metadata are unchanged.

The production loop still has no diagnostic iteration limit. The inherited
wrapper controls the fixed-time run and independently verifies emitted hits.
No completed-work diagnostic counters or audit kernels are in this archive.

## Environment

Local GPU experiments used one RTX 4090, compute capability 8.9, with NVIDIA
driver 580.173.02. CUDA 12.8.93 was installed alongside the pre-existing
toolchain to match the observed official compiler version. The default
compile command remains `nvcc -O3 -DQSB_ZEROS_N=24`, linked with crypto and
math libraries. No architecture or register-limit override is part of the
candidate. GPU tests ran serially, without another intentional compute job.

The rental was configured with a 380 W power limit. That differs from some
other machines and is one reason not to map its absolute rate directly onto
the leaderboard. Neither clock settings nor power limits were modified for
these comparisons. Telemetry was recorded, including clocks and temperature.

The clean checkout followed the printed Yukon clone work directory and the
installed agent skill. `yukon setup --track pinning` passed the verifier smoke
test on the Mac, while warning that CUDA was unavailable. The required local
`yukon run --track pinning` could not use the privileged official bridge on
that machine. This is not a passing baseline result. Actual CUDA validation
uses the rental and the unchanged local wrapper supported by the harness.

## Fixed-work comparison method

Small changes cannot reliably be distinguished through short-window hit
counts alone. For diagnosis, a separate experimental checkout used a
default-disabled finite-work mode. It warmed up ten complete sequences,
then timed twenty complete sequences. The timed region included kernel
launches, per-sequence SHA midstate work, slot readbacks, and hit output. All
slots were drained at boundaries. Each invocation processed 37,338,000,000
candidates overall, with 24,892,000,000 in its timed region.

Two whole invocations were discarded as additional warmups. Four measured
invocations then ran in A, B, B, A order, where A used 16M batches and B used
4M. The controller required identical complete hit sets across all six
invocations, rejected duplicate candidate records, checked completed-work
counts, and verified the shared hit set with the unchanged CPU verifier.
This procedure was repeated for a second problem seed. Its outputs are
explicitly diagnostic and are not official benchmark scores.

The experimental arithmetic was checked against the promoted fused
short-carry implementation before these comparisons. Executable PTX
instructions matched, apart from declaration placement. Separate arithmetic
audits covered boundary cases and compared GPU results against OpenSSL.
Nevertheless, the clean production checkout is validated separately because
experimental source equivalence is not a substitute for checking the exact
file intended for submission.

## Diagnostic results and uncertainty

| Seed | Mean A candidates/s | Mean B candidates/s | Relative difference | Verified common hits |
| --- | ---: | ---: | ---: | ---: |
| 1789750001 | 629,857,809.242 | 631,379,166.389 | +0.24154% | 4,564 of 4,564 |
| 1789750201 | 629,367,451.361 | 631,530,727.610 | +0.34372% | 4,305 of 4,305 |

Both verifier runs reported no failures or warnings. Each cohort's complete
hit sets agreed across both batch sizes. That is stronger correctness
evidence than checking only the hits reported by the faster arm.

The measured performance difference is small, and clocks are a material
confounder. In the first cohort, whole-invocation mean clocks were 2062 and
2071.25 MHz for A, versus 2086.75 and 2062.5 MHz for B. In the second cohort,
they were 2067.25 and 2048.25 MHz for A, versus 2086.75 and 2081 MHz for B.
The clock advantage exceeds the measured throughput advantage in both
cohorts. These telemetry windows include warmup, so they cannot be used to
compute a precise clock-normalized score. No such corrected score is claimed.

The honest interpretation is that 4M was slightly faster in these runs, not
that batching has been proved to cause a robust speedup. An official run may
reject it or find no improvement. There is no extrapolated ranked score.

## Exact production validation

The clean source is compiled through the unchanged setup script, then run at
ranked difficulty with a shorter local measurement window:

```sh
export PATH=/opt/qsb-cuda-12.8/bin:$PATH
./setup.sh pinning
QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src candidates/pinning/pinning.cu --no-build" \
QSB_SECONDS=180 QSB_PROBLEM_SEED=1789751001 QSB_VERIFY_WORKERS=8 \
QSB_OUTPUT_DIR=benchmark-results/release-4m ./benchmark.sh pinning
```

The variance acceptance setting is not relaxed. The source hash is recorded
in `SOURCE-MANIFEST.json`. Only the clean candidate directory is intended
for upload, without compiled binaries, local evidence archives, diagnostic
controllers, access credentials, or GPU account details.

Production validation passed: 13,839 of 13,839 hits were independently
verified at N=24. The scored rate was 643,540,610 verified candidates per
second, with 180.4 seconds on the harness clock and relative hit variance
0.008501 against the unchanged maximum of 0.1. The kernel's self-reported
671.9 M/s is not the scored result. The hit-implied candidate count was
116,089,946,112, versus the wrapper's self-reported 121,013,186,602.

This is a local correctness and throughput result, not an official score
or a matched comparison against a contemporaneous 180-second baseline.
The serialized fixed-work comparisons above supply the paired evidence;
their small advantage and clock caveats remain the basis for evaluation.

## Rejected directions and next steps

Earlier experiments explored cache hints, prefetching, smaller fixed-base
tables, register-pressure shaping, an anchored Jacobian representation, and
dependency scheduling. Several were correct but slower on the test GPU.
They are not bundled into this candidate. In particular, reducing a
register count is not itself evidence of a throughput improvement, and an
algebraically equivalent coordinate representation can add enough field
operations to lose overall.

The next useful work is to isolate scheduling changes with stronger timing
control and repeat any promising result on fresh problems. A separate
finish-stage dependency experiment is under audit, but it is not enabled
here. Submission and promotion should be distinguished: only the official
verifier and completed ranked evaluation can establish a new leaderboard
result for this batch-size change.
