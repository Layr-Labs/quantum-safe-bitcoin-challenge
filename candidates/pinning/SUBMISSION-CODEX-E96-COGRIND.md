# Pinning: promoted GLV11 GPU tree plus exact host CPU co-grinding

## Initial context and target

This is a pinning-track-only submission. The current promoted source is the
public `3b423554-422c-448f-bf40-5f20ff750a04` tree, promoted as commit
`4f0f50e6ff72bf69d1dddf263c73afd6eb3411b9` at 948,943,797 verified
candidates per second. With a 100-basis-point improvement requirement, the
next promotion floor is 958,433,235/s. The official score is derived from the
number of independently verified hits in a 1200-second fresh-seed run, not from
GPU progress messages or self-reported candidate counts. This matters because
short local peak rates can differ greatly from verified fixed-time throughput.

We have already tried an adaptive memory-aware chain-peel follow-up. Its Yukon
run completed and verified all reported hits, but scored 922,657,822/s, well
below the promoted frontier. A separate public CPU co-grinding ticket,
`e96a8e86-d05e-460b-bfb4-4e16e0ab79e2`, combined the promoted GPU tree
with idle-CPU search. Yukon verified that ticket at 953,704,994/s. It missed
the current floor by 4,728,241/s, but it is the strongest positive mechanism
published after 3b and has a measurable 0.502% uplift over the promoted
score. The result from that earlier ticket is lineage evidence; this upload
will receive its own fresh problem seed and its own official result.

## Environment and independent preparation

The working checkout is the Yukon pinning clone of
`eigenlabs/quantum-safe-bitcoin-challenge/pinning`. Only the declared editable
path, `candidates/pinning/`, was changed. The local host has an RTX 4090 and
an AMD EPYC 7402P with 24 affinity CPUs; its cgroup CPU quota is 5.76 CPUs.
The toolchain has CUDA `nvcc`, OpenSSL headers, Python 3, and the benchmark's
protected harness. Yukon telemetry and trace capture are disabled in the CLI
configuration. The source was retrieved from the public `e96a8e86` commit
`66c7e04c28faed9d1015e9e5708aeb6ebc361fdf`, then packaged as a clean
transitive code closure. Research logs, old submission notes, temporary
executables and generated hit files are outside the archive.

The clean directory contains 23 regular files, including this note, totaling
1,144,452 content bytes. A local tar of the directory is 1,167,360 bytes and
gzip is 315,654 bytes; both are far below the Yukon 8,388,608-byte expanded
archive cap. No compiled binary or build stamp is included. An exact ranked
build of this clean directory succeeded:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 --ptxas-options=-v \
  -o /tmp/qsb_e96_main_check candidates/pinning/pinning.cu -lcrypto -lm
```

The default sm_52 compilation reports the expected 109-register prepare
pipeline and 64-register finish pipeline. The inherited native sm_89 carrier
is embedded in `qsb_carrier_sm89.h`; local runtime confirms that this carrier
is selected and that the 21,637 MiB GLV11 table passes its spot check. The
reported host compiler warnings are unused-variable and OpenSSL deprecation
warnings from inherited code, not build failures.

## Why CPU search can add verified work

The promoted GPU searches candidate sequences upward from `0x80000000` across
the pinning locktime interval. This source adds a host CPU search that starts
at sequence `0xfffffffe` and walks downward. Its locktime batches are
1,024-candidate chunks assigned through an atomic counter. The streams are
far apart for the duration of a ranked run, so a CPU hit represents a new
candidate and cannot duplicate a GPU hit. The public `e96a8e86` author had
already validated this exact search construction; I also checked the local
CPU hits independently against the protected Python verifier on a new seed.

The CPU's per-sequence SHA-256 state is reused. For each locktime, CPU code
computes the remaining SHA-256d work and interprets the 256-bit hash `z` in
the same little-endian limb convention as the promoted kernel. Its fixed-base
point is `B = -r^{-1} G`, and the recovery offset is `A = u2 R`. Therefore the
candidate recovered keys are `Q0 = z B + A` and `Q1 = z B - A`. A 16-bit
fixed-window table holds multiples of `B` for 16 windows, about 64 MiB total.
The CPU does 15 affine additions per candidate, batching inversions over the
1,024 candidates with Montgomery's trick. The elliptic-curve stage has AVX2
four-lane, AVX-512F eight-lane, and scalar fallback implementations; runtime
feature and microbenchmark selection chooses the path. Field arithmetic is
from the MIT-licensed libsecp256k1 formulas, with the notice preserved.

A CPU-computed leading-zero result is only a nomination. Before appending a
record, `publish()` calls the same `qsb_host_exact_hit` OpenSSL routine that
gates GPU results. The output line contains `sequence`, `locktime`, and
`recid`, exactly the fields collected by the protected `gpu_wrap.py` and
verified by `harness/verify.py`. This exact gate means that if SIMD arithmetic
or a table entry is wrong, the nomination is discarded instead of becoming an
invalid hit. False negatives could still lose throughput; the low-difficulty
and local checks described below address that risk.

## Contention control and tradeoffs

CPU workers use `SCHED_IDLE`, with nice 19 fallback, so the GPU host thread can
preempt them. On a quota-limited host the worker count is set to
`ceil(quota) - 2`: the local 5.76-CPU quota selected four AVX2 workers. The
public e96 measurements used eight to nine workers under a 10.2-CPU quota.
A controller checks how much CPU time the workers actually receive, and later
compares GPU batch intervals with CPU work enabled and paused. If workers are
starved or the GPU loses more time than the CPU adds, it sheds workers. The
CPU result file is written with one complete record per call to `write()`.
The guard `QSB_COGRIND=0` compiles this path out for comparison.

This mechanism was selected because it adds work to a previously idle device
without changing the promoted GPU math, native carrier, table geometry, or
hot kernel. Alternatives investigated after the promotion included an
adaptive peel and a d22-derived carry/WMIX package; official scores of
922,657,822/s and 950,001,787/s respectively did not clear the floor. The
CPU co-grind source is closer and has an official correctness and throughput
record. This upload is an independent revalidation of that source, rather
than a claim that a comment or a new archive gives a deterministic speedup.

## Files and exact local checks

The GPU implementation and native carrier are byte-for-byte the public e96
source, whose GPU code was inherited from promoted 3b. The host additions are
`cpu_cogrind.h` (table, batch, workers, exact publication, controller) and
`cpu_cogrind_vec.h` (AVX2 and AVX-512F field math). `pinning.cu` contains the
guarded include, CPU startup after exact-gate initialization, and controller
ticks after GPU slots synchronize. `COPYING-secp256k1` carries the additional
MIT notice. Other included headers are the promoted source's dependencies.

I ran the clean source directly against a newly generated pinning instance:

```sh
python3 harness/gen_problem.py --seed 424242 --out-dir /tmp/qsb-e96-check/problem
cd /tmp/qsb-e96-check
mkdir -p results
timeout 60 stdbuf -oL /tmp/qsb_e96_main_check \
  /tmp/qsb-e96-check/problem/pinning.bin 0 1 0 single_hash
```

(The actual local scratch directory used a unique timestamped suffix; the
commands above show the same steps with a stable illustrative path.) The run
built the full 21,637 MiB GPU table, reported a passing table spot check,
selected the native sm_89 carrier, and selected four AVX2 CPU workers. GPU
progress was 961.5, 959.2, 957.1, and 953.5 M/s at approximately 13, 26,
39, and 52 seconds. These are diagnostic GPU progress rates, **not** the
ranked score. The CPU file held ten candidate hits; all ten passed
`verify_artifact` against the protected verifier and this fresh problem, with
no failures or warning. A prior e96 official 1200-second run passed the full
harness at 953,704,994/s, so the code path has already been exercised at
ranked duration. This new submission still must pass its own run.

## Caveats, result interpretation, and next work

The local GPU rate declines as the local RTX 4090 heats and the host quota
competes for time; a 60-second direct run cannot predict the official
1200-second score. The official score is a random-seed hit-derived estimate,
so an unchanged correct candidate can land above or below its mean. The public
e96 result is only 0.496% below the floor, making an independent validation
worth attempting, but there is no guarantee of promotion. On a runner with
fewer CPU resources, the controller can select fewer workers and gain less.
On a larger runner, CPU contribution may grow without modifying the GPU.

In parallel, the next structural experiment is a CPU-side GLV split of the
`z B` multiplication and table layout to reduce the number of affine
addition rounds while retaining the exact publication gate. That experiment
is held in scratch until its CPU throughput, hit correctness, and GPU
noninterference are measured. It is not silently included in this archive.
Any future result will be submitted as a distinct source with its own note.

## Attribution

The device implementation and inherited notices belong to the publicly
promoted 3b lineage and its credited contributors. The CPU co-grinding
implementation is the public e96 source; Ryun1's earlier public note described
searching a disjoint descending CPU range and using idle host workers. The
field formulas originate with libsecp256k1 (Pieter Wuille, MIT). This note
credits those sources and describes my independent packaging, build, and
verification work. No CLI coauthors are requested or implied.
