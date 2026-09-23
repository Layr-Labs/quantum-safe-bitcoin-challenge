# Pinning: lean scalar primitives and an exact high-word coefficient screen

Effort: xhigh. This composition and its new coefficient work were developed with
GPT 6 Astra through Codex. The exact model and effort were checked in the active
session metadata. The authoring machine has no usable NVIDIA GPU, so this note
contains source, arithmetic, and compilation evidence, not a claimed local score.

## Frontier and prior result

The clean starting checkout is the promoted source
`b59484345df5208f5caffc82c25a4a3b50cbe523`. Its live score remains 826,926,066
verified candidates/s; the one-percent promotion floor is 835,195,327 after
integer rounding. The new archive is confined to `candidates/pinning/`.

Our previous [PR #1224](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1224),
submission `20a669fe-4ba8-4f89-b0ac-6cb8800cb34d`, was independently verified but
rejected at 814,044,005 candidates/s, approximately 1.56 percent below the
frontier. Its reduced static instruction count did not establish a throughput
win. The previous even/odd modulo-2^129 product experiment is not in this archive.
The old checkout is preserved separately.

The subsequent public PR review found these relevant official results:

| PR | Implementation summary | Verified candidates/s |
| --- | --- | ---: |
| #1205 | Dense-first GLV and host/seed composition | 829,282,307 |
| #1223 | Smaller dense cache reserve and prefetch | 793,354,720 |
| #1229 | Six refinements including lean scalar primitives | 824,638,381 |
| #1237 | #1205 composition plus multiply SFC2 shortcut | 829,084,805 |
| #1243 | Reported repeat of the #1237 runtime | 791,023,360 |
| #1249 | Further reported repeat of that runtime | 827,827,523 |

The small improvements in #1205, #1237, and #1249 are recorded publicly but did
not meet the one-percent promotion threshold. #1237 and #1249 improve the
frontier by about 0.261 and 0.109 percent respectively. The wide spread of
reported repeated runs limits causal conclusions; it does not justify assuming
that any particular source change is responsible for all of the difference.
Self-reported search rates are not used as the ranked score in this note.

PRs #1251, #1252, #1253, and #1254 were still open during the final review. The
latter two are described as repeat measurements; #1252 isolates dense-first
placement. Their pending results are not claimed as evidence for this archive.
The benchmark reports that research Discussions are disabled.

## Selected composition and provenance

The runtime source is deliberately assembled from individually identified files:

- `pinning.cu` and `negative_y_mac.cuh` are byte-identical to
  [PR #1205](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1205),
  commit `6c3409ff19f90651f2c4baca9c55f8d20eb1e8b2`. This retains dense-first
  table placement, exact host publication, saved slot-report publication after
  refill, independent sequence overlap, and register-resident Q seed codes.
- `GPUMath.h` is byte-identical to
  [PR #1237](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1237),
  commit `0ec5edf1a017e2a69c9c58f317120dedac659859`. Relative to #1205, this
  selects `QSB_MUL_SFC2_DROP` at the two multiply second-fold sites. Square
  paths retain their distinct carry handling. The shortcut remains guarded by
  `QSB_C31` and `QSB_HOST_GATE`. This inherited field optimization is speculative
  in rare carry cases and can lose nominations; the exact host publication
  gate remains essential. It is not presented as universally exact arithmetic.
- `GLVScalar.cuh` starts from
  [PR #1229](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1229),
  commit `d7d4278249b44ad21ac6214d878d4e2a55a57f6f`, and adds the two exact
  changes described below. The imported `QSB_GLV_LEAN` path uses explicit
  `mul.wide.u32` and `mad.wide.u32` operations and carry-flag overflow accounting
  in the coefficient and residual helpers. The rest of #1229's SHA, paired-code,
  loop, offset, and zero-component modifications are not imported.

This excludes the donor archive's extra research directories and prebuilt cubin.
All other runtime files remain from the promoted checkout. The implementation
continues to use the required nvcc translation unit, argument format, and hit
record format. No harness, verifier, workflow, problem generator, or sibling
track changes are required.

Coauthorship credits Saviour1001 for the dense composition; dun999, i34-9, and
DrCleverHans for its unpromoted host and seed donor chain; jrcarlos2000 for the
SFC2 mechanism; ItlaStudent for the reviewed union composition; and kaankolcu
for the imported lean scalar implementation. The promoted source and its upstream
contributors retain their existing notices. Existing GPLv3 and secp256k1 MIT
license materials are preserved. The new coefficient screen and round helper
are the additional work of this submission, not attributed to the donors.

## New change 1: bounded high-word diagonal

`QSB_GLV_HIGH10_HI=1` changes how the existing high15 coefficient screen forms
its first carry. Each GLV coefficient is the exact nearest-integer result
`floor((k*g + 2^383) / 2^384)` for one of the two existing reciprocal constants.
The older high15 screen retains complete product diagonals 10 through 14 in
base B=2^32 and invokes the full coefficient fallback near a rounding boundary.

The new screen keeps only the upper 32 bits of each of the five products in
diagonal 10, then processes diagonals 11 through 14 as before. Its initial carry
is the sum of `mulhi(a3,b7)` through `mulhi(a7,b3)`. The first two high words fit
in 32 bits because `b7+b6 < B` for both production constants. The remaining sum
uses 64 bits. This removes the dependence on the discarded low halves and their
carry accounting without changing the exact final coefficient contract.

The rounding guard must be widened. All omitted terms are nonnegative. For
constant words b_j, an input-independent bound on the omitted integer is:

```
Emax = sum((B-1)*b_j*B^(i+j), for 0<=i,j<8 and i+j<10)
       + 5*(B-1)*B^10
```

For the actual two constants, this bound is strictly below `9*B^11` and
`8*B^11` respectively. The new thresholds in word 11 are therefore
`0x7ffffff7` for g1 and `0x7ffffff8` for g2. When the approximate word is below
its threshold, the omitted nonnegative amount cannot cross the half boundary.
When it is already at or above `0x80000000`, it is already rounded upward. A
possible wrap through word 11's end increments the next word and removes that
round-up, leaving the same coefficient. It cannot reach the following half
boundary under the stated error bound. Values in the intervening guard interval
use the existing exact full-product fallback.

The screen is exact for the existing constants and every 256-bit input. It does
not discard candidates or depend on the exact host gate for coefficient
correctness. The wider guard can increase fallback frequency; that cost is part
of the performance experiment. `QSB_GLV_HIGH10_HI=0` restores the old first
carry and guard. With `QSB_GLV_LEAN=0`, the original C product path is selected
and the original guard remains in force.

## New change 2: direct rounding carry

`QSB_GLV_ROUND_CC=1` uses a two-instruction PTX carry chain for the fast path's
128-bit rounding increment: `add.cc.u64` on the low word followed by
`addc.u64` on the high word. This is exactly the former low-word addition plus
unsigned-overflow comparison. It removes comparison/select work in the generated
kernel. Setting the switch to zero restores the former expression. The source
also retains a portable expression for CPU auditing.

## Arithmetic validation

`test_glv_coeff.py` extracts the actual production coefficient function and its
rounding helper for compilation on CPU. CUDA multiplication/carry primitives
are modeled by equivalent unsigned integer operations, and the rare full-product
fallback uses a complete 8-by-8 word multiplication. Python arbitrary-precision
rounding supplies the independent expected result. This executes the production
screen's control flow; it does not execute PTX or NVIDIA scheduling on a GPU.

Both the old and new screens passed **601,636 coefficient cases**, split evenly
between the two production constants. The fixtures include 200,000 random inputs,
bit boundaries, sparse/all-high word patterns, inputs constructed immediately
around true rounding boundaries, and offsets that enter and leave both guards
and cross word 11's wrap point. The test separately computes the omitted-term
bounds and verifies the first-two-high-word sum bound.

Fallback counts in these deliberately adversarial fixtures were 95,976/93,060
for g1/g2 with the old screen and 110,925/107,982 with the new screen. These are
fixture coverage counts, not estimates of fallback rates for random benchmark
scalars. They show that the test exercises both fast and fallback behavior and
the additional guarded cases. Every returned coefficient matched the independent
big-integer oracle.

The inherited host-publication test passed its 64 midstate samples and recovery
comparison. All five priority-pipeline tests and all three slot-readback tests
passed. Their dependency/reuse/error checks do not constitute a complete GPU
execution test of the host search loop.

## Build and resource evidence

`yukon setup --track pinning` passed, including the CPU verifier smoke test.
The required `yukon run --track pinning` was attempted on the clean checkout;
it could not produce a ranked score because the configured organizer bridge
and an NVIDIA GPU are absent on this machine. The previous checkout was retained
and this experiment uses a new clean benchmark clone.

CUDA 12.8.93 compilation is available through a local GCC 13/compatible-header
wrapper. The default-target organizer-style executable compiled and linked
successfully with N=24, O3, and OpenSSL/math. Resource comparisons used the
compute_52-to-sm_89 compilation route, preserving the ranked source preprocessing
architecture rather than enabling native-only source alternatives.

| Static preparation-kernel screen | SASS instructions | Registers | Shared bytes | Stack / spill bytes |
| --- | ---: | ---: | ---: | --- |
| Union composition, scalar lean and both new switches off | 6,688 | 106 | 12,288 | 0 / 0 |
| Union plus imported scalar lean, new changes off | 6,616 | 106 | 12,288 | 0 / 0 |
| Submitted scalar lean plus both new changes | 6,608 | 106 | 12,288 | 0 / 0 |

The first high-word implementation had the same total static count as the lean
control. Bounding its first sum to 32 bits and using the direct rounding carry
produced the final eight-instruction reduction. This count is a compiler screen,
not a throughput measurement; the donor lean change contributes the other 72
instructions in the table. All counts refer to the same preparation kernel.
No percentage speedup is inferred by dividing static instruction counts.

## Reproduction and evaluation limits

From the benchmark work directory:

```
yukon setup --track pinning
python3 -B candidates/pinning/test_glv_coeff.py
python3 -B candidates/pinning/test_host_gate.py
python3 -B candidates/pinning/test_priority_pipeline.py
python3 -B candidates/pinning/test_slot_readback.py
nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/pinning candidates/pinning/pinning.cu -lcrypto -lm
yukon run --track pinning
```

For the resource comparison add `-gencode arch=compute_52,code=sm_89 -Xptxas=-v`.
Disable the two new switches to isolate their contribution; also disable
`QSB_GLV_LEAN` to recover the union scalar implementation. The inherited SFC2
switch separately controls the imported multiply shortcut. There are no runtime
benchmark-selection rules or hidden timing overrides in the candidate.

The trusted fresh-problem run must determine validity, verified yield, elapsed
time, and promotion. This composition has no locally executed GPU hit set.
Compiler scheduling, driver JIT, memory traffic, and hardware/run variation can
outweigh this scalar reduction. The old refusal remains negative evidence and
none of the donor improvements are assumed additive. A follow-up should compare
completed equal-work A/B/B/A runs on one GPU, with sorted verified-hit equality
and the coefficient switch toggles, before attributing a small score difference.
The new source inventory records exact file hashes; no claimed-score flag is
needed because the benchmark reports that claimed scores are recorded only.
