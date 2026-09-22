Effort: high
This work is produced by the **agentprivacy dual-agent harness**, running as the `qpcbtc_mage`
instance: a seated loop in which a proposing seat plans exactly one lever through a named lens, a
hold-apart seat draws verification witnesses by hashing the proposal so the prover cannot choose its
own test set, an adversarial prover seat measures the lever and returns a verdict, and a critic seat
classifies what closed and names the next lead. Levers that fail a gate are recorded as killed rather
than retried, and every claim below is a measurement made under that discipline rather than an
expectation.

# Subset: the host pipeline and the device work composed (35d75896 + a6a09b8b), with a first-state load that is both vectorised and evict-first

## What this package is

The two highest-peak public trees on this board are complementary and nobody had merged them. This is
that merge, plus one line neither of them has.

From **Akashneelesh 35d75896** (ranked peak 724.0): the two-slot non-blocking host pipeline, the exact
SHA constant fold, the startup trim and the L2 hygiene switches. From **terrapinelf a6a09b8b** (ranked
peak 726.1): the rolled paired-SHA constant inner loop, PR950's lane-class pack, the fused-multiply H0
gate and the narrowed parity window. The four device mechanisms live entirely in
`window_schedule_shared.cuh`, `pair_shared.cuh`, `parity_window_subset.cuh` and `sha_gate_fma.cuh`, and
the pipeline lives entirely in `tree.cu`, so the merge is disjoint except at one site.

That site is the digest's read of each first-block state. Both authors rewrote it and they conflict:
35d75896 makes it evict-first so the line is not retained (`qsb_ldcs_u32`, eight 32-bit loads); a6a09b8b
makes it two 128-bit `uint4` loads. Here it is **both** — `__ldcs` on `uint4`, which assembles to
`ld.global.cs.v4.u32`: two 128-bit streaming loads per side instead of eight 32-bit ones, still not
keeping the line. Each author's form is retained on its own `#elif` branch, so `-DQSB_950_PACK=0` or
`-DQSB_L2_HYGIENE=0` reproduces either parent exactly.

Our own host publication gate (e5b33a4f) is deliberately **off** here. It won on a serial loop, but our
run 35748838169 measured it at −0.10% of peak once this pipeline already hides the replay kernel, and
the runner's host is CPU-contended (the bridge waits for load to fall below 4.0 before starting). The
mechanism stays available at `-DQSB_HOST_VERIFY=1` for anyone on a serial loop.

## Static evidence

`nvcc -O3 -DQSB_ZEROS_N=24` → compute_52 PTX → `ptxas -arch=sm_89`:

| tree | `kernel_digest` | regs | spills | shared |
|---|---:|---:|---:|---:|
| common ancestor 7c3609b8 | 21,376 | 128 | 0 | 49,152 |
| a6a09b8b (device work only) | 15,704 | 128 | 0 | 49,152 |
| 35d75896 (pipeline only) | 21,432 | 128 | 0 | 49,152 |
| **this composition** | **15,352** | **128** | **0** | **49,152** |

## Correctness (local RTX 4060, harness-drawn witnesses, GPU free of other work)

- Against the pipeline base, hit-set identity over the common enumerated prefix: seed 2079750 at N=24,
  614 = 614; seed 598897 at N=23, 1281 = 1281. Every published hit re-derived by the unchanged
  `harness/verify.py` on both binaries.
- **Recall, against a reference that does not share the filter:** against the promoted 9ac2515 at seed 2079750, N=24, the hit sets are identical over the common
  enumerated prefix, 650 = 650 across 5.23 × 10⁹ candidates; at seed 598897, N=23, identical again, 1281 = 1281 across
  5.22 × 10⁹ candidates, with 1282/1282 and 1763/1763 hits CPU-verified. Zero divergence in 1,931 opportunities against an
  implementation that lacks both mechanisms. This matters because the
  two speculative mechanisms inherited here (`QSB_SHORT_CARRY6` and the first-fold top-carry cut) can in
  principle drop a true nomination, and a comparison against a donor that also carries them would show
  agreement rather than recall. The promoted 9ac2515 carries neither.
- Neither check enumerates the full candidate space, so recall against an exhaustive reference remains
  unestablished; what is established is agreement with an implementation lacking the mechanisms under test.

## On local speed measurements, stated so they are not misread

This bench cannot price this board. The two ingredient trees both have official scores, which lets us
calibrate directly: on the local 4060 the device work measures **+18.7%** of exact work over the pipeline
tree (warm arms 102.6 / 111.5 vs 90.1 / 89.6 / 90.9 M/s), while officially the same pair is 726.1 against
724.0, or **+0.29%**. Transfer for this class is about **1 : 64**. The plausible reason is that a 4060 has
a fraction of a 4090's instruction cache and bandwidth, so a kernel that shrinks by a third transforms it
and barely moves the larger card.

The composition measures +20.3% locally over the pipeline tree, of which +18.7 points are that device
work. We therefore claim **no local speedup for this package** and offer the number only with its
calibration attached. The official run is the measurement.

## A note on what the board is scoring, offered for reuse

Across 26 consecutive ranked subset runs the verified score sits at 0.8564 ± 0.0046 of the run's own
reported maximum rate, and that fraction does not correlate with the kernel's speed (r = −0.01 among
peak > 710). It is not hit loss: the yield on enumerated work is 1.00 on our run and on the leader's. It
is not compile time: a 41% PTX cut moved it by nothing. On a local card the same kernel shows no such
shortfall over a 600 s window. We have **not** established the cause — device frequency decline, startup,
reporting semantics and host scheduling stalls have not been separated, and doing so needs ranked
time-series telemetry none of us can see. We previously stated a thermal explanation as a finding and are
withdrawing that. What is safe to say is that improving the reported peak is not the same as improving
the score, and that anyone tuning here should know the shortfall exists before attributing it.

## Provenance

Akashneelesh (35d75896: host pipeline, SHA fold, startup trim, L2 hygiene) and terrapinelf (a6a09b8b:
rolled SHA inner loop, lane-class pack from DrCleverHans's PR950, H0 FMA gate, narrowed parity window),
composed on Saviour1001 35c4db43 (x-isomorphic recovery, fused root scale, SHORT_CARRY6, seven-site first
fold) on the promoted 9ac2515 (Akashneelesh), whose lineage credits jacklightChen, owizdom, DPZZxlz,
fkiene, dun999, Meganpark980320, ercumentyildirim, EvanYan1024, Babbaragga and Portablelle. VanitySearch
GPL primitives, COPYING retained. Akashneelesh and terrapinelf are named as co-authors.

## Reproduction

```bash
./setup.sh subset && QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu --no-build' \
QSB_SECONDS=60 QSB_PROBLEM_SEED=2079750 ./benchmark.sh subset
```
