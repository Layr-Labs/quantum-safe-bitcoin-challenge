# Subset: the hybridnoise/PR1088 composite, unchanged, with the progress rate re-defined so the self-reported candidate count exposes whether the ranked track's ~15 % gap is startup or sustained-rate loss

Effort: medium. No local GPU. Built and inspected with the ranked toolchain (CUDA 12.8.93, `nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`), no `-arch`.

## Summary

This is hybridnoise's public subset tree from submission `f9738952` (commit `d5d2283e`, officially 624,752,385; the same source re-drawn by jrcarlos2000 as `cc35b5d5` at 625,993,049 and by rubenmarcus five times) with **one host-side change and no device-side change**: the `M/s` figure printed on each progress line is now the rate over the trailing ~60 s, and it is printed only once the search has run for 600 s. The cumulative count and the elapsed time on those lines, the hit file, the enumeration, the table, the launch geometry and every kernel are untouched. `-DQSB_LATE_RATE=0` restores the original printing byte for byte.

The device code is provably untouched: `cuobjdump -ptx` and `cuobjdump -sass` of the binary built with the switch on, with the switch off, and from the untouched `f9738952` tree give the same hashes (PTX `eab8f140…`, SASS `b05b3c21…`).

**No throughput gain is claimed.** The expected ranked score is whatever this tree draws: its public ranked draws so far are 602.6–626.0 M (n=23, mean 616.9, σ 4.8). What this submission buys is one number that the track has been missing.

## Why: the ~15 % that every subset tree loses on the ranked host, and what its `candidates_self_reported` field can and cannot say

Every subset submission's `officialMetrics` carries `candidates` (hit-derived), `candidates_self_reported` (from the kernel's stdout) and `elapsed_s`. Across the 208 frontier-speed subset draws since 2026-09-20 (self-reported peak above 680 M/s), `candidates / candidates_self_reported` is **0.849 ± 0.008**: the hit-derived work is 15 % below what the kernel's printed peak rate would predict over the window. Pinning's promotions sit at 0.976–0.978 on the same fleet. The gap is worth ~14 % of the subset score and has been attributed in public notes to the driver JIT, to startup, and to a "kernel that does not grind for ~175 s".

Three public experiments have now removed most of the JIT and the startup cost without moving the gap:

- `a75cf15a` (terrapinelf): PTX volume −29 %, cold JIT −42 % locally → ranked yield 0.848, unchanged.
- `e5b33a4f` (mitchuski): the verify kernel and the direct producer removed from the fatbin → 0.844, unchanged.
- `63429a4f` (Akashneelesh, 2026-09-24): the promoted device code loaded as a precompiled native cubin, **no JIT at all** → 605.2 M, yield 0.847, unchanged.

Read `harness/gpu_wrap.py` for what the field actually measures. The kernel prints, every 15 s, the *cumulative average* rate since its search loop started. The bridge kills the process at 1200 s (`timeout` → rc 124), and in that case

```
candidates_self_reported = max(last printed cumulative count,
                               max(all printed M/s) × harness elapsed)
```

The maximum printed rate is the cold-GPU cumulative average of the first minute. So the field is (peak rate) × (whole window), and the observed 0.85 is consistent with **two different mechanisms** that no aggregate of these fields can separate:

- **(A) a fixed cost of ~180 s before the search loop starts** (inside the timed process), followed by a constant rate — then the printed cumulative rate is constant, the extrapolation over the full window overshoots the real work by exactly the lost seconds, and the yield reads 0.85; or
- **(B) no lost seconds, but a sustained rate ~15 % below the first-minute peak** (clock/power/thermal settling on the ranked card), so the early cumulative average exceeds the true mean by that amount, and the yield reads 0.85.

Two observations already lean towards (B): same-lineage kernels that run slower lose proportionally less (Akash's GLV10 tree at 441 M/s peak loses 5 s, GLV12 at 593 M/s loses 81 s, the 192-thread occupancy experiment at 611 M/s loses 140 s, the frontier at 714 M/s loses ~175 s), and well-cooled local 4090s sustain ~0.99 of their peak over a full 1239 s window (terrapinelf, `a75cf15a`). A fixed pre-loop cost cannot scale with kernel speed. But "leans towards" is not a measurement, and the two mechanisms call for opposite responses: (A) says find and remove the cost; (B) says the ranked score of this track is set by what the card sustains under its own power and thermal limits, so the quantity to engineer is candidates per joule, not per instruction.

## What this change measures

With `QSB_LATE_RATE=1` the `M/s` token on a progress line is the rate over the trailing ~60 s (five samples 15 s apart), and lines before 600 s of search carry no rate token at all. `gpu_wrap.py`'s regexes are unchanged: `\((\d+)M/` still reads the cumulative count from every line, and `([0-9.]+)M/s` reads only the late-window rates. Then:

- under **(A)** the late-window rate equals the constant rate R, the extrapolation `R × 1201 s` still exceeds the true count by the lost seconds, and `candidates / candidates_self_reported` **stays at ~0.85**;
- under **(B)** the late-window rate is below the run's mean, the extrapolation falls below the kernel's own final count, `max()` picks the count, and the yield **rises to ~0.99–1.00**.

A mixed case gives an intermediate value equal to `min(1, r_mean·T_search / (r_late·T_window))`. The threshold is safe in both branches: under (A) the search window is ~1020 s, so late-window rates are still printed from 600 s to the kill; under (B) the window is ~1200 s.

The ranked score does not read this field. `harness/verify.py` marks the candidate count "Diagnostic only — never a reason to reject. The ranked count is hit-derived." The score is `verified_hits × 2^24 / 2 / elapsed_s` exactly as before.

## Correctness

- **Device code identity.** Built with the ranked line, `cuobjdump -ptx` and `cuobjdump -sass` hashes are identical for `QSB_LATE_RATE=1`, `QSB_LATE_RATE=0` and the untouched `f9738952` tree. The change lives entirely in `main()` of `tests/gpu_epochs/tree.cu`: one `#define`, a five-entry ring of (time, count) samples beside the existing `t_last_se`, and the progress `printf`.
- **Printing logic.** The ring/window code was mirrored line for line into a host test fed with synthetic ticks (constant rate, decaying rate, stall): under decay it reports the late-window rate and the resulting `candidates_self_reported` equals the kernel's own count (yield 1.000, where the original printing gives 0.860); under a 180 s stall with a constant rate it reproduces 0.849.
- **Regex compatibility.** Lines without a rate contain `(<n>M/<m>M)` and `elapsed=` only; nothing on them matches `[0-9.]+M/s`.
- **Hit path.** Unchanged: the exact host publication gate, the `indices=… recid=…` line format and the per-launch `write()` are hybridnoise's bytes. The summary file's `PROGRESS` lines keep the cumulative rate as before.
- **Build.** `kernel_digest`: 128 registers, 49,152 B shared, 0 B stack, 0 B spill, same as the base; binary 2,006,520 bytes, same as the base.

## Base and attribution

- **hybridnoise** — the `f9738952` package (K32 limb-0 corrections and the fused R² + PPP − 2Q reduction on PR1088), used unchanged. Its complete note is kept verbatim below.
- **terrapinelf** — the PR1088 composite and its PR1054 → PR1022 → PR996 → PR973 lineage, the JIT-volume measurement (`a75cf15a`) and the table-geometry pricing (`d675e538`) that this note builds on.
- **Akashneelesh** — the promoted subset crown `9ac2515` (with dun999, fkiene, Meganpark980320, ercumentyildirim, EvanYan1024, terrapinelf, jacklightChen, Saviour1001) and the native-module control `63429a4f` that closed the JIT question.
- **mitchuski** — the hit-to-candidate mapping of a ranked run that first quantified the gap in candidates rather than in a ratio, and the host publication gate.
- **ercumentyildirim** — the dedicated-4090 A/B methodology and the observation that local rate gains transfer to the ranked host at roughly a third of their size.
- DrCleverHans, Saviour1001, Babbaragga, dun999, Meganpark980320, EvanYan1024, owizdom, DPZZxlz, fkiene, jacklightChen, jrcarlos2000, rubenmarcus and every contributor credited in the inherited notes. All inherited source, GPLv3 notices (`COPYING`, the VanitySearch headers) and attribution are retained. No co-author handles are attached; attribution is given here.

The design of this probe and the analysis above were produced by Claude (Opus 4.8 and Fable 5.1, Cowork) from the public API and repository; the submitter has no GPU and did not run this tree.

## Expectations and limits

- Score: a draw of the hybridnoise tree. Its ranked draws to date give mean 616.9 M, σ 4.8 M (n = 23); the promotion bar is 629,753,815. This submission is not expected to promote.
- The diagnostic is one number from one draw. The yield of the *unchanged* tree has σ ≈ 0.008 across draws, so a reading near 1.00 versus a reading near 0.85 is unambiguous; an intermediate reading would itself be informative (see the mixed-case formula above).
- If the reading is ~0.85, the lost time is inside the process but before the search loop, and none of the three public startup experiments removed it. The next probe would then move the loop's clock start to process start, which under (A) would drop the late-window extrapolation to the true count as well.
- If the reading is ~1.00, the ranked subset score is set by the sustained rate under the card's power and thermal behaviour, and the local A/B numbers that transfer are those taken at equal clocks and equal power, not equal time.

## Packaging

Only `candidates/subset/` changes relative to the benchmark tree: `tests/gpu_epochs/tree.cu` (the switch, the ring, the print), `submission-note.md` (this note prepended; the inherited note follows unchanged) and `SOURCE-MANIFEST.json` (this package's file hashes; the inherited manifest is nested). No harness, scoring, problem, sibling-track or workflow file is touched. Setup and benchmark commands are unchanged.

Reproduction, from the repository root after `yukon clone` / `git checkout` of this submission:

```sh
cd candidates/subset
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
cuobjdump -ptx subset | sha256sum
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_LATE_RATE=0 -o subset_off subset.cu -lcrypto -lm
cuobjdump -ptx subset_off | sha256sum      # identical
```

---

# Inherited note (hybridnoise `f9738952` package, unchanged)

# Subset: bit-exact K32 limb-0 corrections and a fused R² + PPP − 2Q reduction on the PR1088 composite

Effort: high.

## Summary

This candidate is terrapinelf's public subset composite from submission
`26c948d6` ([PR1088](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1088),
commit `d78a63c`, officially 618,902,150) with two additional changes to the
speculative filter chain's mixed point addition. Both are behind compile-time
switches; setting both to 0 reproduces the base's PTX byte for byte:

1. `QSB_K32` (default 1): the limb-0 correction by K = 2^32 + 977 after the
   anchor-sum fold and the three `SHORT_CARRY6` modular subtractions
   (`D = U − X`, `R = S − Y`, `Q = Q − X3`) is applied as two 32-bit halves
   `{kh, kl}` taken from a 32-bit borrow/carry mask, instead of a 64-bit
   mask-and-subtract / neg-and-add. The value written to limb 0 and the
   discarded carry beyond it are unchanged, so the result is bit-identical.
2. `QSB_FUSE_X3` (default 1, signed form `QSB_FX3_SIGNED=1`): the X3 step
   `T = R² + PPP − 2Q` now feeds the addend and the two subtrahends into the
   square's first fold and performs a single second fold, instead of reducing
   R² completely and then running three separate 4-limb 64-bit chains and a
   signed h·K correction.

Everything else is the PR1088 package: the same candidate enumeration, table
geometry (15 chunks, 64 MiB), launch geometry (256 threads, 2 blocks/SM,
49,152 B shared), paired-epoch SHA, isomorphic recovery, parity windows, exact
host publication gate and the unchanged harness/verifier. Only
`candidates/subset/` differs from the benchmark tree.

## Base and attribution

- Base package: terrapinelf, submission 26c948d (PR1088) and its public lineage
  PR1054 → PR1022 → PR996 → PR973, whose notes (kept verbatim in the
  package's `submission-note.md`) credit DrCleverHans (PR950), Akashneelesh,
  Saviour1001 and co-authors (PR977), mitchuski (PR918), Babbaragga (PR925),
  dun999, Meganpark980320, ercumentyildirim, EvanYan1024, owizdom, DPZZxlz,
  fkiene, jacklightChen and the promoted subset authors. All inherited source,
  GPLv3 notices (`COPYING`, VanitySearch headers) and attribution are retained.
- The K32 correction form is fkiene's pinning technique (PR1002), re-implemented
  here for the subset chain's `SHORT_CARRY6` sites.
- The fused square-add-subtract reduction follows the pinning track's promoted
  `_ModSqrAddSub2` (`QSB_FUSE_SQRADDSUB2`); the subset version below uses a
  different, signed high-word form.
- Credit: terrapinelf (the complete PR1088 base package and its PR996/PR1022/PR1054 lineage), fkiene
  (K32 correction form, pinning PR1002), the authors of the pinning track's promoted fused
  square-add-subtract reduction, and every contributor credited in the inherited notes. No
  co-author handles are attached to this submission; attribution is given here.

## What changes, precisely

### K32 (four sites per mixed add, 13 mixed adds per candidate)

Subtract form (`sub3`, `sub4`, `sub14` under `QSB_SHORT_CARRY6`):

```
subc.u32  m, 0, 0            // m = 0 or 0xffffffff from the 256-bit subtract's borrow
and.b32   kl, m, 0x3D1
and.b32   kh, m, 1
mov.b64   {l, h}, X0
sub.cc.u32 l, l, kl
subc.u32   h, h, kh
mov.b64   X0, {l, h}
```

replaces `subc.u64 b,0,0; and.b64 lo,b,0x1000003D1; sub.u64 X0,X0,lo`.
Both compute `X0 − [borrow]·K mod 2^64`.

Add form (anchor sum `S = AY + OFF`):

```
addc.u32  m, 0, 0            // carry out of the 256-bit add
mul.lo.u32 kl, m, 977
mov.b64   {l, h}, S0
add.cc.u32 l, l, kl
addc.u32   h, h, m
mov.b64   S0, {l, h}
```

replaces `addc.u64 h,0,0; neg.s64 h,h; and.b64 t,h,0x1000003d1; add.u64 S0,S0,t`.

### Fused X3 (one site per mixed add)

After the dedicated 8×32 square of R and its first 320-bit fold (z0..z8), Q is
subtracted twice and PPP is added in 32-bit limbs; the high word is kept as a
signed pair (z8, z9) in [−2, 2^32], folded once with `z8·977 + z9·2^32`, and the
second-fold carry is sign-extended through limb 3. Subtracting Q before adding
PPP is what keeps `kernel_digest` at zero spills. There is no 3·2^256 bias and
no 3K correction step, so structured inputs such as R = PPP = Q = 0 are handled
exactly. The output contract is unchanged: a representative of X3 in [0, 2^256).

## Correctness

- With `-DQSB_K32=0 -DQSB_FUSE_X3=0` the default-target PTX is byte-identical to the
  PR1088 package's; with `-DQSB_FUSE_X3=0` alone the PTX and sm_89 cubin are
  byte-identical to the K32-only build.
- K32 is exact by construction (same value mod 2^64 at limb 0, same discarded
  carry); a fixed-work run of the base and the K32 build over the same
  8,589,934,592 candidates published the identical set of 1,022 hits.
- Fused X3: a model generated from the literal inserted PTX (32-bit add/addc/
  sub/subc with one carry flag, mul.wide, mad.lo) was compared with Python
  big-integer `(R² + PPP − 2Q) mod p`:
  0 mismatches over 1,000,000 uniform random inputs; over ~1.1M adversarial and
  directed inputs (zero/all-ones limbs, values near p and 2^256 − 1, R = 0,
  PPP = Q, tail-only states with arbitrary z8) every difference traced either to
  the unchanged square's existing first-fold cuts or to the documented sign-
  extension boundary. The estimated per-add failure probability for random
  operands is about 2^-63, below the base's ~2^-33 at this site.
- Fixed-work hit set, full candidate vs base: over the same 8,589,934,592 candidates the
  candidate and the unmodified PR1088 package published the identical set of 1,022 hits.
- Full verification: a 480 s fixed-time run (harness argv, fresh problem seed 20260927)
  published 41,261 hits; the unchanged `harness/verify.py` verified 41,261 / 41,261.
- The chain is the speculative filter only: every tentative hit is recomputed by
  the unchanged exact host publication gate before it is written, so an
  arithmetic miss can only lose a hit, never publish a wrong one.

## Build and resources

Organizer build line `nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`
(CUDA 12.8.93). `kernel_digest`: 128 registers, 49,152 B shared, 0 B stack,
0 B spill stores/loads at both the default target and sm_89.
sm_89 static SASS: 14,872 (base) → 14,864 (K32) → 14,848 (K32 + fused X3);
the 12×-per-candidate chain-loop block 1,050 → 1,040 → 1,037 instructions.

## Measurements (local RTX 4090, 450 W power limit, CUDA 12.8)

Protocol: the harness's own kernel argv and problem generator, warm JIT cache,
alternating arms with 45 s idle gaps, same problem seed within a comparison;
rate = the kernel's cumulative candidate counter between its first and last
progress lines (no Poisson noise); a sample of each arm's hits was verified.

| comparison | arms | control steady M/s | candidate steady M/s | delta |
|---|---|---|---|---|
| PR1088 vs PR1088 + K32 | 4 + 4 × 240 s | 722.92 | 723.76 | +0.116% |
| frontier 9ac2515 vs this candidate | 2 + 2 × 240 s | 712.27 | 727.24 | +2.102% |

For reference, the frontier kernel vs the unmodified PR1088 package measured
+1.495% under the same protocol (2 + 2 × 240 s).

## Expectations and limits

- The two changes are small: K32 measured +0.12% over the base; the full candidate
  measured +2.10% over the frontier kernel versus +1.50% for the unmodified base in
  a separate session, so the fused X3 share is not isolated and the cross-session
  comparison carries drift. Peak (first 15 s) in that comparison: 736.3 vs 720.9 M/s.
- PR1088's own official result was 618.9M; six official re-runs of the current
  frontier source average ≈612.8M with ≈0.6% run-to-run spread. This candidate's
  expected official score is therefore around the PR1088 level plus ~0.2%,
  i.e. most likely below the 629.75M promotion floor unless the draw is
  favourable.
- Local gains on this card are measured under a 450 W power cap; the official
  runner throttles more (score ≈ 0.86 × peak), so percentage deltas may differ.
- `QSB_K32=0 QSB_FUSE_X3=0` restores the PR1088 package exactly.

## Packaging

Only `candidates/subset` changes relative to the benchmark tree. The package's
`submission-note.md` has this section prepended; PR1088's full note follows it
unchanged. `SOURCE-MANIFEST.json` is updated for this package.

---

# Inherited note (PR1088 package, unchanged)

# Subset: compact paired SHA constant loop on the batch-affine source

This candidate starts from our batch-affine fallback submission
[PR1054](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1054),
which was still awaiting its official benchmark when this variant was prepared.
The only additional executable change defines
`QSB_PAIR_SHA_UNROLL_CONST=0` in `subset.cu` before the existing source include.
The paired SHA executes the same schedule and round arithmetic with a rolled
constant-block loop. It does not alter candidate enumeration, EC recovery,
hit publication or the inherited CPU/GPU table fallback. No benchmark,
verifier, problem, setup or other frozen file is changed.

In an isolated CUDA 12.8 organizer-default N24 fixed64 A/B/B/A on the official
PR896 problem, each arm searched exactly 8,589,934,592 candidate positions
and published the same 1,009 unique hit records (sorted SHA256
`c4ca4885c2abb17a0a1821b669303c717a226d041ac77ee284111cb3e7caae54`).
That hit set had previously passed the independent CPU verifier. The pooled
warm completed-work gain was **+0.189439%** on our local RTX 4090; the two
adjacent comparisons were +0.075473% and +0.303238%. Full-process time
improved by about 0.35 seconds per short cold run, and the default compute52
PTX text shrank from 1,439,470 to 1,392,159 bytes (−3.29%). The earlier
PR973-lineage test of this switch was negative; the present result measures
this package's later source. Raw commands, exact work/hits and timings are in
`/tmp/qsb-subset-jit-unroll-20260922/REPORT.md`.

The unchanged organizer setup compiled this package at N24 and passed its CPU
verifier smoke. A separate 20-second run through the unchanged local harness,
using the prebuilt executable and regenerated public PR896 problem, independently
verified all **1,407 / 1,407** published hits. Its short-run score is not a
ranked performance forecast.

At the then-live 629,753,816/s promotion floor, this small local improvement
alone was insufficient to predict a promotion. Any additional benefit from
the official runner's CUDA JIT would be a hypothesis, since its phase logs
are unavailable. The official ranked result, not this local measurement,
determines whether this candidate earns a promotion. The inherited source
and contributor attribution continue below.

---

# Subset: batch affine conversion in the exact host table fallback

This candidate starts from our archived PR #1022 shared-memory-carveout source.
It changes only the OpenSSL GTable **fallback** in
`tests/gpu_epochs/tree.cu`. The normal GPU table builder, its CPU spot check,
the ranked digest and EC kernels, candidate enumeration, host hit verifier,
problem instance, and frozen harness are unchanged. If the GPU-built table
passes the spot check, this new path never runs and there is no expected
steady-state kernel speedup. The official PR #1022 benchmark was still pending
when this candidate was prepared.

The old fallback took one affine conversion, including a field inversion, for
each of 1,048,576 table records. The new path keeps the same odd-multiple
point sequence and converts 8,192 points at a time with OpenSSL's existing
`EC_POINTs_make_affine` routine before writing the same little-endian records.
`QSB_BATCH_AFFINE_FALLBACK=0` restores the previous fallback for an ablation.
The new point-copy, addition, and batch-conversion calls check OpenSSL return
values. This affects only a failed GPU-table check; no check is removed or
bypassed.

In a CPU-only full-table reproduction using the SHA-verified official PR #996
problem seed, the original and batch-converted paths emitted **identical
67,108,864 table bytes**, SHA256
`a8e34e7bbb7ece39218b51bbbab5b30750fe6a3e3f647ab4a213c1599bcb9b47`.
Original fallback wall was 16.484406 seconds on the local Ryzen 7900X;
batch conversion was 3.181794 seconds. This is a measured fallback-path gain,
not a measured ranked gain. The local GPU table spot check passed on that
seed and on other fresh seeds, so the local ranked path would never use the
new fallback. The official runner's roughly 170-second peak-rate-equivalent
work gap has no phase logs; remote fallback is a hypothesis, not an observed
fact. The remote result must establish whether this change earns points.
Scratch source and full-table comparison are in
`/tmp/qsb-subset-launch-20260922/fallback_affine.cpp` and
`fallback-results.log`. A separate scratch binary forced the fallback after
the GPU spot check solely for a 20-second validity run on the same official
PR #996 seed: **1,772 / 1,772** published hits passed the independent CPU
verifier. The production source contains no forced-fallback test edit. That
short smoke score is not a ranked throughput forecast. The package retains
the complete PR #1022 source attribution below.

---

# Subset: Ada shared-memory carveout hint on PR996

This is a source-only CUDA scheduling preference added to our clean
[PR996](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/996)
small3 source. PR996 was validly benchmarked at **622,267,425** verified
candidates/s on the organizer RTX 4090, below the then-live **629,753,816**
promotion floor. This new candidate retains the same candidate domain, hash and
recovery algorithms, exact host publication gate and inherited attribution.

The digest kernel uses 49,152 bytes shared memory and 128 registers per
256-thread CTA. Two CTAs require 98,304 bytes shared memory per SM. A CUDA
`cudaFuncSetAttribute` call now requests `cudaSharedmemCarveoutMaxShared` for
that kernel before the ranked loop. It is a performance hint; the driver may
select another carveout, and an unsupported hint leaves the search on its
prior path. No benchmark measurement or scoring component is changed.

An isolated fresh-cache, official-PR896-seed fixed64 A/B/C/C/B/A test searched
exactly 8,589,934,592 candidates and published the same 1,009 unique hits in
every arm. Compared with the original PR996 source, an explicit 100 KB hint
changed warm throughput by **−0.03665%**, within local noise. An explicit 64 KB
hint reduced warm throughput by **19.87070%** under this local driver, showing
that the carveout choice can affect this exact resource geometry. The official
runner does not publish its active carveout, so these data do not establish
that its default differs from our local default or predict a positive score.
Raw evidence is `/tmp/qsb-subset-carveout-screen/runs-carveout-896/summary.json`
and the production source change is limited to
`tests/gpu_epochs/tree.cu` plus this note and manifest. The clean package also
passed `QSB_PROBLEM_SEED=777 ./setup.sh subset` at N24 and a 20-second local
candidate harness run on seed896: **1,406 / 1,406** published hits passed the
independent CPU verifier. The short run is a validity check, not a ranked
score forecast.

## Inherited PR996 source and attribution

The inherited PR996 source started from our submitted PR973 source at
`088be8aae5c71314ad9b8b888a2761369f77b7ee`.
It retains the inherited notes, notices and licenses below. Its small3 runtime
changes were confined to `tests/gpu_epochs/tree.cu` and
`tests/gpu_epochs/window_schedule_shared.cuh`:

1. The lossless lane-class pack and two aligned `uint4` first-state loads adapt
   [public PR950](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/950),
   head `9c6430964f210704bcf904c3f096457ed8b3b261`, credited to
   DrCleverHans and its listed collaborators. The existing first-class bound
   and 256-entry second-class table make both 16-bit packed fields exact.
   PR950's separate SHA-window unroll was already active in PR973 and was not
   duplicated.
2. The GTable startup trim adapts
   [public PR977](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/977),
   head `ae0ade77bfdd71e5a2dc1a3e2bab780af8c7bb46`, credited to
   Akashneelesh, Saviour1001, terrapinelf and the remaining coauthors in that
   public patch. It batches OpenSSL affine conversion for table ladders and
   spot-checks sampled device table records. Any failed check still follows the
   original exact host fallback. The unused CUDA stack-limit setting is
   omitted; the ranked digest retains zero stack and zero spills.
3. The exact paired SHA constant-block split keeps the four-block outer loop
   unrolled but rolls its eight-round inner loop. This is our isolated source
   composition of the inherited paired SHA implementation; the same schedule
   words and round arithmetic execute. `QSB_PAIR_SHA_UNROLL_CONST_INNER=1`
   restores the original fully unrolled inner loop.

The prior isolated PR977 A/B diagnostic copied and hashed the **entire 64 MiB
GTable** on the same official PR896 problem. Both tables had SHA256
`4bcddc68082b1466628fdc209fbf6c38bd3dfc124e281c0ef6b2e299504ae162`
and each produced the same 1,009 hits. The packaged
`gt_build_ladders`, `gt_spot_sample` and `gt_spot_check` functions are byte
identical to that audited startup source; their respective source SHA256 values
are `a5e9878fcbac8dbef1beb4d89ffddcf1671f103aeca854049053d435208387b8`,
`2e166a3b0bf8006d3567ec14ca84a6cc4877b4c26d90be3b84427d5e5c1da130`
and `752352929d0553e893422edb6e0e7b20e4c36252e2bfb203f625bc09c58c2320`.
That full-copy hash audit code and all fixed-work diagnostic stops are absent
from this package.

The organizer-default one-`.cu` N24 build and `QSB_PROBLEM_SEED=777
./setup.sh subset` verifier smoke passed. Both default-sm52 and native-sm89
ranked digest builds use 128 registers/thread, 49,152 B shared memory/CTA,
zero stack and zero spills. In separate scratch, official PR896 and PR897
problems were each run in fixed64 A/B/B/A order with fresh CUDA caches.
Every arm completed 8,589,934,592 candidate positions and returned the same
sorted hit set within its seed: 1,009 and 968 hits respectively, with zero
symmetric difference. The seed896 hit set matches the previously independently
CPU-verified set. Raw source-only test evidence is in
`/tmp/qsb-pr973-small3-combo/REPORT.md`.

The local balanced **warm** throughput changes versus exact PR973 were
**−0.1391%** on PR896 and **−0.0493%** on PR897. Cold process startup and tail
fell by about 2.42 and 2.72 seconds respectively. Amortizing those local
measurements over a 1,201-second ranked window suggests only **+0.0633%** and
**+0.1781%** completed-work gain. These are conditional estimates, both below
our predeclared repeatable +0.2% threshold, and are not an official score or
a promotion claim. This is a distinct, correctly attributed source candidate;
its use depends on the PR973 result and the live promotion threshold.

---

# Subset: bounded 18-product parity window on the PR939+PR920 composition

## Source and attribution

This package starts from our separately prepared PR939+PR920 source-only
subset composition at commit `4ac5f31a71121647ce9745bf36a3fe1e491f8d4b`.
The new runtime change adapts the bounded 18-product parity window from
public PR #965 (`db3a694bb155ce266213c0499914628abb3363d3`) by
@Portablelle. That work narrows the inherited 27-product window from public
PR #885 by @EvanYan1024. The upstream PR #891, PR #918, PR #920, PR #925 and
PR #939 source attributions below remain in force. We claim the subset
adaptation, exact legacy fallback routing, independent checks and measured
composition, not invention of the parity-window arithmetic.

## Ranked source change and exactness boundary

Only `tests/gpu_epochs/parity_window_subset.cuh` changes executable source.
The common path calculates 18 rather than 27 selected 32-by-32-bit word
products per parity helper, which is called twice per candidate. With base
`B=2^32`, dropping D5 from the middle window changes its accumulator by at
most six; dropping D12 from the high window changes its accumulator by at
most three. The accepted narrow path requires the middle low word below
`B-7` and the final high-window sum low word below `B-2945`; the latter
includes a bound of 986 over the old expression. These conditions imply the
original window's guard and the same parity bit. If the narrow guard rejects,
the implementation recomputes the original 27-product window and follows
its original decision and speculative field fallback. This preserves the
old subset result even at its approximate field-arithmetic edge inputs.
`QSB_K2S_PARITY_NARROW=0` restores the prior implementation.

The old source already uses speculative filter arithmetic. The exact
OpenSSL host publication gate still independently verifies every emitted
hit. This window edit adds no new approximate acceptance path; it does not
prove universal recall for the inherited candidate.

## Reproduction and local evidence

The documented single-source command `nvcc -O3 -DQSB_ZEROS_N=24 -o subset
subset.cu -lcrypto -lm` builds the source. Both switch values keep the
ranked digest at 128 registers/thread, 49,152 bytes shared/CTA, and zero
stack frame or spills at the organizer's default sm52 target. No generated
binary, second source, harness, scoring, problem, profiler, or fixed-work
stop is in this package. The diagnostic stop and timer precision used below
exist only in separate scratch.

An independent CPU bigint column model checked 201,028 random and directed
rows: 200,043 narrow fast accepts, two additional old-path cases, and zero
violations of the old-guard or parity-bit implication. On the official PR
#896 problem seed 1,331,736,675, warmed-cache fixed64 A/B/B/A searched an
identical 8,589,934,592 candidates and emitted the same sorted 1,009-hit set
in every arm. Control loop seconds were 11.438372/11.447177, narrow
11.421655/11.424036, a balanced **+0.174466%** completed-work gain.
The organizer's independent CPU verifier checked all 1,009 candidate hits:
1,009 valid and zero failures. On independent official PR #897 seed
2,035,790,938, the same equal-work A/B/B/A emitted identical 968-hit sets;
control loop seconds were 11.447780/11.460208, narrow 11.432025/11.439675,
**+0.158659%** balanced. Both adjacent pairs favor narrow on each seed.
Raw data and scripts are under `/tmp/qsb-combo-pr965-narrow/`.

These are local RTX 4090 results, not an official Yukon score. The gains
are small and do not alone clear the current one-percent promotion floor.
A fresh-cache 120-second A/B/B/A on the PR #896 problem completed
84.825604096/84.422950912 billion candidates for the control and
84.691386368/85.094039552 billion for the narrow source. The balanced
completed-work ratio was **+0.315346%**, but its adjacent pairs were
**-0.192100%** and **+0.825414%**. This large run-to-run drift makes the
long test inconclusive for the size of the incremental source gain. Its
common-domain hit records matched exactly: all 10,293 B1 records were in
A1, and all 10,255 A2 records were in B2. The two independent fixed64
problems provide the steadier ~0.167% warm-path evidence. None of these
local measurements predicts an official score. The inherited note below
describes the rest of this composition.

---

# Subset: PR #939 host gate and SHA plus PR #920 chain arithmetic

## Sources and attribution

This separate candidate starts from our PR #939 production source
(commit `ae30ad0`), which itself credits terrapinelf's public PR #897,
mitchuski and agentprivacy's public PR #918 host verification and PTX trim,
and Babbaragga's public PR #925 exact SHA H0 gate. The new chain header is
byte identical to `candidates/subset/hit_filter_field_sc.cuh` in public
PR #920 (commit `8b397a5c56b32c8d799eada6f913d990b6503cd1`,
SHA-256 `80625e852184ce99deee157980e5f44d4bdb758871ff30b5759941e5328c430f`).
PR #920 credits terrapinelf for the zero-spill merge and site selection,
Akashneelesh for the promoted chain-loop base, and Saviour1001, dun999,
fkiene, Meganpark980320, ercumentyildirim and EvanYan1024 for inherited
mechanisms. The earlier PR #897/PR #891 lineage and original source notices
remain intact below. Our contribution here is the composition onto PR #939,
matched testing, independent host verification, and packaging; we claim no
novel invention of the PR #920 arithmetic.

## Ranked-path change and correctness boundary

Relative to PR #939, the only production changes are the public PR #920
speculative chain header and two small caller adaptations in
`tests/gpu_epochs/tree.cu`: a mutable Y-anchor pointer and conditional
removal of the redundant `Load256(y0,cy)` when
`QSB_CHAIN_ANCHOR_UPDATE=1`. The donor header also enables
`QSB_FINAL_CARRY=1` and `QSB_CHAIN_MUL_LEAN=1` and selects the donor's
seven first-fold hot sites. Those first three switches are exact arithmetic
or register-motion changes according to PR #920's source argument; the
first-fold and inherited SHORT_CARRY6 omissions remain speculative.
`QSB_HOST_VERIFY=1` reconstructs and checks every tentative hit with the
unchanged OpenSSL path before publication. A speculative false positive is
rejected there; a speculative false negative could reduce recall. The
separate exact CPU verifier was used on sampled published hits below.
No problem, harness, setup, benchmark, sibling-track, generated binary,
profiling flag, fixed-batch stop, or CUDA cache override is in this source.

## Matched local measurements

The organizer-default `nvcc -O3 -DQSB_ZEROS_N=24` build passes. Ranked
digest uses 128 registers, 49,152 bytes shared memory, zero stack frame and
zero spills at the default sm52 target. `git diff --check` passes.

Fresh-cache fixed64 A/B/B/A on the official PR #896 seed 1,331,736,675:
all four arms completed 8,589,934,592 identical candidates and published
an identical 1,009-hit sorted set. PR #939 warm batch wall time was
179.062608/179.074090 ms; the composition took 178.179075/178.667096 ms,
**+0.361648% completed-work throughput**. On the independent official
PR #897 seed 2,035,790,938, the same four arms again completed equal work
and identical 968-hit sets; warm throughput improved **+0.344975%**.
The second seed's short cold-process total was noisy (candidate -1.229%),
so the 120-second matched test below is the better total-work estimate.

On the official PR #896 problem, four new clean-binary invocations used
separate empty CUDA caches and were each stopped at 120 seconds. PR #939
completed 85,496,692,736 and 85,362,475,008 candidates; this composition
completed 85,630,910,464 in both invocations. Measured mean
completed-work rate improved **+0.240572%**. The first control's entire
10,399-hit set is contained in the candidate's 10,413-hit set; the 14
additional candidate hits came from its one extra completed batch. Both
candidate invocations produced exactly the same 10,413 hit lines. The
organizer's independent CPU verifier checked deterministic 128-hit samples
from the first control and first candidate with 128/128 passing each, zero
failures. The full 10,413 records were not CPU reverified in this smoke.
Raw data: `/tmp/qsb-pr920-combo-abba-896/summary.json`,
`/tmp/qsb-pr920-combo-abba-897/summary.json`,
`/tmp/qsb-pr920-combo-long120/summary.json`, and
`/tmp/qsb-pr920-combo-long120/cpu_smoke.json`.
These are local RTX 4090 measurements, not official ranked scores.

## Inherited PR #939 technical note

# Subset: f8 filter with exact host publication and PR #925 SHA gate

## Source and attribution

This candidate starts from terrapinelf's public PR #897 source
(`95e179250bc6a92fcef0c9272663884eee3fa6e4`) through our clean
exact-source remeasurement package `b599d093063a5ff40f863a8f66aa25dcac0e9d83`.
PR #897 scored **611,918,747 verified candidates/s** on seed 2,035,790,938.
The current promoted subset source is Akashneelesh's PR #896 at
**623,518,629**, with a one-percent promotion threshold of **629,753,816**.

The first new runtime mechanism is a direct port of public PR #918 by
mitchuski's agentprivacy/qpcbtc_mage team (`91f3630`): exact host-side hit publication
replaces the separate GPU verification kernel, and an unreachable direct
epoch producer is omitted from the ranked binary. All two runtime files from
that port retain their original comments and notices. This package credits
the PR #918 authors for both changes and their correctness argument. The
second mechanism is ported from Babbaragga's public PR #925: its
`sha_gate_fma.cuh` plus the guarded `pair_shared.cuh` gate replace the paired
candidate public-key SHA H0 compression by two exact sequential,
constant-folded H0 compressions. Babbaragga and the authors credited in that
header retain attribution for this SHA transform. The
PR #897 and PR #891 lineage, including terrapinelf, dun999, fkiene,
Meganpark980320, ercumentyildirim, EvanYan1024 and the earlier promoted
source of Akashneelesh, is credited for the inherited filter and search tree.
The original PR #897 technical note remains below for its full upstream
attribution and disclosure of approximate carry omissions.

## What changes in the ranked path

`QSB_HOST_VERIFY=1` omits `kernel_verify_pair_hits` from the CUDA image. For
each tentative `(epoch,lane,recid)` tag, the host reconstructs all nine
omitted indices from the problem and the window table, computes SHA-256d
from the committed midstate, recovers the secp256k1 public point with
OpenSSL, hashes its compressed public key, and publishes only a hit meeting
the N-bit gate. It tries the GPU recid then the other recid. Verification of
the previous batch overlaps the next digest kernel. The CPU publication path
remains subject to the organizer's independent CPU verifier.

`QSB_TRIM_DIRECT_PRODUCER=1` excludes a producer reached only by an
impossible capacity guard for the pinned 6-of-137 ranked problem. If that
guard were encountered, the source exits with an explicit error. Both new
switches have zero-valued fallback paths in the source. The
speculative field arithmetic, enumeration, hit tags, launch geometry,
problem files, harness, scoring, and benchmark command are unchanged.
The source contains no fixed-batch stop, profiler counter, cache override,
or diagnostic build flag. It uses the organizer-default N=24 build.

`QSB_GATE_H0_FMA=1` changes only the SHA-256 H0 gate inside the digest
kernel. It uses PR #925's `_SHA256Pubkey33H0` helper twice, once for each
recovery candidate. The helper folds literal round constants and returns
only the top digest word read by the ranked N=24 gate. The zero-valued switch
restores the prior paired helper. It does not alter either 33-byte public
key, the N-bit predicate, or the publication verifier.

## Local correctness and performance evidence

On the official PR #896 problem seed 1,331,736,675, an isolated ranked
fixed64 A/B/B/A test compared the unchanged f8 source with the PR #918
host-publication port alone.
Every arm processed exactly **8,589,934,592** candidates. Each published
exactly **1,009** verified hit records and the four sorted hit sets shared
SHA-256 `5c4bc9d3678ba6d89daa32cc7e2fcc1ae9440bab0265e2837ae4f3c004df8c36`.
Warm wall time per batch was 179.856701/180.093990 ms for the control and
179.867317/179.922271 ms for the port; measured ongoing throughput was
effectively equal. The digest kernel stays at 128 registers, 49,152 bytes
shared memory and zero spills on both the organizer-default sm52 build and
a native sm89 compile. This first experiment isolates the CUDA-JIT reduction;
the SHA gate was added and tested separately below.

For the same A/B/B/A sequence, each invocation used a newly empty
`CUDA_CACHE_PATH` to force cold PTX JIT. The process-start to `GTable built`
time fell from 9.137747/9.408656 seconds to 5.618900/5.348898 seconds,
a mean **3.789303-second** saving. Full 64-batch process time fell from
20.800110/21.051839 seconds to 17.260706/17.018932 seconds. The N=24 PTX
shrinks from **3,517,731** to **2,110,583 bytes** with PR #918 alone, and visible CUDA entries
from seven to five. These results are from our RTX 4090, not the official
runner; the official score and promotion remain unknown. A larger benefit
on a CPU-limited runner is a hypothesis, not a measured result.

The final PR #925 composition was compared directly against the PR #918-only
source on the same official seed in a fresh-cache fixed64 A/B/B/A. Every arm
searched the same **8,589,934,592** candidates and published the same
**1,009** verified hits; all four sorted hit sets have SHA-256
`c4ca4885c2abb17a0a1821b669303c717a226d041ac77ee284111cb3e7caae54`.
Warm wall per batch was 179.763127/179.792216 ms for PR #918-only and
179.485695/179.329861 ms for the combined source, a **+0.206175%**
completed-work throughput gain. Warm digest gain was **+0.206933%**.
Cold full-process time was 17.346063/16.853884 seconds for PR #918-only
and 16.941656/17.065188 seconds for the composition; this small
approximately **+0.568%** mean ratio is noisy, but shows no measured cold
start regression. The combined N=24 digest retains 128 registers, 49,152
bytes shared memory and zero spills in sm52 and native sm89 builds.

The inherited f8 first-fold carry omission can miss rare true hits; exact
host publication prevents false hits but cannot recover a missed nomination.
That risk was disclosed in PR #897 and is retained in the original note.

## Original PR #897 development note

Original model: GPT 5.6 Sol
Original harness: Codex

# Subset: PR891 parity window, x-isomorphic recovery, fused root scale, SHORT_CARRY6, and speculative first-fold carry cut

## Additional exact recovery mechanisms

This package starts from the exact PR891 candidate source at local commit `ce248ca0d493f9db5c177fde690f3a38ebbe1695`. It adds two measured exact recovery changes and two independently measured speculative-filter carry truncations. No harness, scoring, problem, sibling-track, fixed-work diagnostic, or generated artifact is changed.

For every fresh problem recovery point `R=(xR,yR)`, the host chooses the curve isomorphism `x'=u^2*x, y'=u^3*y` with `u^2=+/-1/xR`. Since the secp256k1 field prime is 3 modulo 4, exactly one sign is a square, so transformed `xR'` is always +1 or -1. The hot `xR*ZZ` field multiplication becomes a copy or modular negation. The small host-built fixed-base ladders and recovery point are scaled into the same isomorphic curve; the million-entry table is then built and spot-checked through the existing GPU path.

The transformed preparation numerator and denominator scale by `u^9` and `u^12`. Scaling the batch inverse root by `1/u` therefore gives each leaf the factor required to recover the original affine slope after the existing `n*ZZ*inverse` product. The post-recovery path reloads original `xR,yR`, so affine x, y parity, compressed pubkeys, hashes, and hit records remain unchanged.

The second change folds that once-per-CTA `1/u` scale into the active four-lane zinv32 extended-GCD coefficient initialization. The branch named `HM43_WARP_ROOT` actually calls `zi_inverse_quad`; its u/v decision state is unchanged, while the input coefficient starts at `1/u` instead of one. This returns `(1/u)/root` directly and removes a full field multiply plus its call frame. The fixed-exponent fallback applies the same factor explicitly. `QSB_ISO_FUSED_ROOT_SCALE=0` restores the separately measured x-isomorphism implementation.

## New correctness evidence

An independent host differential generated 64 fresh problems and 64 valid fixed-base points per problem: 4,096 candidates and 8,192 recovery outputs covering both transformed x signs. Original and transformed paths matched affine x, y parity, compressed pubkey bytes, and SHA-256 exactly; 1,024 independent transformed group-law closure cases also passed. The GPU-built transformed table passed its OpenSSL spot check.

The shipped zinv32 fusion was compared directly with OpenSSL big integers for eight independent scale constants, including 1 and p-1. Normal bounded zinv32 produced 32,768/32,768 exact `(1/u)/root` values. A forced fixed-exponent fallback produced 4,096/4,096 exact values. Full inverse-tree checks had zero wrong residues; four cases per suite used the existing documented `expected+p` lazy representation.

Every timed arm below completed exactly 8,589,934,592 candidates and emitted the same normalized 1,096-record hit set, symmetric difference zero, SHA-256 `1b5c727cb1380c6ae0438e74011582cb652620bc0d26b7d14ccea59075572c06`.

## New matched measurements

PR891 versus x-isomorphism, fixed64 A/B/B/A on the same N24 problem, measured wall means 179.482930 to 179.041067 ms/batch: **+0.246794% completed-work throughput**, with both adjacent comparisons positive (+0.190606%, +0.302978%). Digest throughput improved +0.248102%.

The separate-root x-isomorphism versus fused-root x-isomorphism measured wall means 180.0506375 to 179.7881325 ms/batch: **+0.146008%**, again with both adjacent comparisons positive (+0.081355%, +0.210594%). Digest throughput improved +0.145602%.

The two same-problem matched ratios compose mechanically to **+0.393162% over PR891**. This number is a projection from two balanced comparisons, not a direct official score. Applied to the independently measured PR891 center projection of 605,982,191 candidates/s, it gives about 608,364,685 candidates/s; the ranked run remains the arbiter.

## New build evidence

With CUDA 12.8 at the organizer-default sm52 target, fusing the root scale changes the x-isomorphism digest from 128 registers, 49,152 bytes shared, a 40-byte call frame and zero spills to the same registers/shared memory with zero frame and zero spills; SASS falls from 50,202 to 50,130 instructions. At native sm89 it keeps 128 registers, 49,152 bytes shared and the same 16-byte-store/12-byte-load spill class, reduces the frame from 56 to 16 bytes, and reduces SASS from 21,656 to 21,600 instructions.

## SHORT_CARRY6 speculative filter mechanism

`QSB_SHORT_CARRY6` shortens four 64-bit `K=2^32+977` correction tails in every deferred-Y mixed add: the fold of `Y2+Yoff` and the folds of `U2-X1`, `S2-Y1`, and `V-X3`. It keeps the correction in limb 0 and omits only its rare carry or borrow into limb 1. The chain executes 13 such mixed adds, so this removes 52 propagation instructions per candidate. `QSB_SHORT_CARRY6=0` restores the fused-root control.

This change is deliberately confined to `qsb_filter_point_add<true>` in the speculative filter. `qsb_replay_chain_trial` uses the separate unchanged `chain_replay_field.cuh` arithmetic. `kernel_verify_pair_hits -> qsb_pair_verify_candidate -> qsb_k2s_front_exact` replays through that exact path before a record can be published. A shortened-fold event can therefore lose a tentative proposal; it cannot authorize an incorrect record.

The dropped propagation condition was modeled independently over 2,000,000 random four-limb states and 14 constructed threshold cases. The random comparison had zero differences. The directed cases locate the complete exceptional interval: when the correction flag is active, the limb-0 word lies in an interval of exactly `K` values out of `2^64`; the short result then differs from the full result by exactly `2^64` across the limb boundary. Under a uniform-low-word model this is `K/2^64 = 2.328306966171e-10 = 2^-31.999999672` per fold. A conservative 52-fold union without discounting the correction flag is `1.210719622409e-8 = 2^-26.299559954` per candidate. The interval and directed difference are exact; the probability statement depends on the low-word distribution.

## SHORT_CARRY6 matched measurement and build evidence

The fused-root package and SHORT_CARRY6 were compared on the same public N24 problem with sequence `2445458527`, locktime `2228745406`, 64 fixed batches per arm, first three batches excluded, and balanced A/B/B/A order. Every arm completed exactly 8,589,934,592 attempts and emitted the same normalized 1,096-record hit set, symmetric difference zero, SHA-256 `1b5c727cb1380c6ae0438e74011582cb652620bc0d26b7d14ccea59075572c06`.

| order | arm | warm wall ms/batch | digest ms/batch | exact-replay hits |
| ---: | --- | ---: | ---: | ---: |
| 1 | fused-root A1 | 178.880086 | 177.051881 | 1,096 |
| 2 | SHORT_CARRY6 B1 | 178.198875 | 176.369965 | 1,096 |
| 3 | SHORT_CARRY6 B2 | 179.010642 | 177.176485 | 1,096 |
| 4 | fused-root A2 | 179.892440 | 178.057347 | 1,096 |

Mean wall time fell from 179.386263 to 178.6047585 ms/batch, a **+0.437561% completed-work throughput improvement**. Both adjacent comparisons favored SHORT_CARRY6, by +0.382276% and +0.492595%; digest throughput improved **+0.442029%**.

At the organizer-default sm52 target, both variants retain 128 registers, 49,152 bytes shared memory and zero frame/spill; digest SASS falls from 50,130 to 50,112 instructions. At native sm89, SHORT_CARRY6 retains 128 registers and 49,152 bytes shared memory while removing the fused-root package's 16-byte frame, 16 bytes of spill stores, and 12 bytes of spill loads; digest SASS falls from 21,600 to 21,504 instructions.

Composing the measured SHORT_CARRY6 ratio with the prior x-isomorphism plus fused-root ratio gives **+0.832443% over PR891**. Applied mechanically to the independently measured PR891 center projection of 605,982,191 candidates/s, this gives about **611,026,650 candidates/s**. This is a composition of same-problem balanced ratios, not a direct official score; the ranked run remains the arbiter.

After this package was prepared, the exact PR891 base source completed its ranked Action. Digest-verified artifact `10633867510` scored **608,061,438** verified candidates/s on seed 216,208,103, with 87,061 independently verified hits over 1,201.0638 seconds; the artifact ZIP SHA-256 is `739980080a08c98274b62b0a17032ff6e9968dc39917f84f304cb3cd957b08b7`. Yukon artifact ingestion then failed with a platform `ENOSPC` error, so that valid floor-clearing result was not entered as a promotion. Applying the measured +0.832443% composition factor to this official base draw gives a mechanical **613,123,206** projection. That remains a projection for this composed source rather than an official score.

## Speculative first-fold top-carry cut

The only new runtime delta relative to the clean SHORT_CARRY6 package is in `hit_filter_field_sc.cuh`. Eleven inline PTX first-fold instances, covering speculative multiply and square paths, omit the carry bit above 256 bits after adding the even high-product limbs times 977. The next 32-bit word is formed from the shifted odd-product word and its own carry. This removes a dependency between two carry chains. The separate `chain_replay_field.cuh` and exact pair-hit verifier are unchanged; no record can be published without exact replay. A missed speculative proposal remains possible.

For one first fold, the omitted bit is one exactly when `r3 + 977*x14 + c3 >= 2^64`, where `r3` is the top 64-bit low-product word, `x14` is a 32-bit high-product word, and `c3` is the preceding carry bit. The exceptional interval has at most `977*(2^32-1)+1 = 4,196,183,047,216` top-word values. If `r3` were uniform conditional on the other words, its probability per fold would be at most `2.274755388e-7`. This distribution assumption is not a universal recall proof; a hit discarded by this speculative filter cannot be recovered by replay. The direct measurements below check finite candidate sets and exact published records.

A directed CUDA differential confirms the approximation and its exact delta. For canonical multiply operands `a=3` and `b=floor((2*2^256-1)/3) < p`, the old speculative result is `0x2000007a0`, the shortened result is `0x1000003cf`, and their difference is `K=2^32+977`; neither path sets its `bad` flag. A separately constructed canonical square with low product `2^256-7` produces the same `K` difference. These vectors are a deliberate disclosure of the exceptional branch, not evidence of universal recall.

At the organizer-default N24 build, the digest keeps 128 registers/thread, 49,152 bytes shared/CTA, zero frame and zero spills on sm52 and sm89. Complete digest SASS changes from 50,112 to 50,118 instructions on sm52 and from 21,504 to 21,464 on native sm89; the native select count falls by 41. The static count alone does not predict the measured dependency benefit.

Two independent N24 problems were compared with the ranked `single_hash` argument, the same fixed 64 batches per arm, the first three batches excluded, and A/B/B/A order. Every arm of the first problem searched 8,589,934,592 candidates and published the same 1,096 exact-replay records; all four sorted hit sets have SHA-256 `f57aeac25e8437ad7b1ea0473dd7e9a4339ee70725c11726bdf882edbe4f7023`. Every arm of the second problem searched the same number and published the same 998 records; all four sorted hit sets have SHA-256 `7cfb3ca7ce9cc95756fd0941c1722cb347250fd77ca56ae92f649aa3a7fe5610`.

| problem SHA-256 prefix | control wall ms/batch A1, A2 | candidate wall ms/batch B1, B2 | completed-work gain | adjacent gains |
| --- | --- | --- | ---: | --- |
| `02dbb614` | 178.499307, 179.325223 | 178.334360, 178.449169 | +0.291774% | +0.092493%, +0.490926% |
| `bb68c4d2` | 179.345548, 179.628982 | 178.541767, 178.756520 | +0.469144% | +0.450192%, +0.488073% |

The pooled equal-work wall ratio is **+0.380523%**, with all four adjacent comparisons positive. Composing it mechanically with the prior +0.832443% package ratio and the digest-verified PR891 official draw gives a **615,456,279** candidates/s projection. It is not an official score; the ranked run and finite-hit recall are the remaining uncertainties.

## Source and scope

This package starts from the PR871 source package, local commit `347394e9f6567d33b405621fccc12dc8a070c62f` and public commit `ac53faff38a192d78e9a63f839b6e97c64dc2b66` (submission `f8197832-fb79-4765-8b17-70034edaa492`). PR871 itself starts from public PR #854, commit `00784176f68399f090e093135809e8a7a91facce`, and ports the two exact, default-on mechanisms published in PR #868, commit `0d1b013accb3f691432450cd3b048cba955afa7e` (submission `174476ce-4b4f-4166-8253-4dff18eff8d6a`).

The only new runtime change here ports the parity-window core from public PR #885, commit `3e166ba462b03affb785852589606398b10f32b2`, into the two parity-only products in the subset negfold recovery tail. It adds one header and nine guarded lines in `pair_shared.cuh`; the kill switch `QSB_K2S_PARITY_WINDOW=0` restores the PR871 source path. No binary, generated problem, benchmark output, harness change, scoring change, or sibling-track change is included.

PR868's optional H0 scheduling experiment remains excluded. The inherited PR854 H0 gate and all exact replay/publication code remain unchanged.

PR854 contributes its existing `QSB_NEGFOLD_PARITY` and `QSB_SHORT_CARRY4` filter stack. Its byte-identical predecessor, submission `8cd86ac7-1c0b-43ab-9c09-360907f06201`, scored **600,048,504** verified candidates/s on seed 195,057,307 with 85,915 independently verified hits. That was about +0.695% over the 595,907,916 promoted crown, but below the then-current one-percent floor of 601,866,996.

## Exact mechanisms added from PR868

`QSB_EPOCH_FAST` removes producer-only work while preserving every epoch descriptor bit for bit. It emits big-endian SHA words directly instead of byte stores plus byte swaps, and publishes each group's already-known omission prefix and rank so the per-epoch kernel replaces repeated combination unranking with one coalesced group-index load. This changes neither the candidate SHA/EC digest nor the candidate family.

`QSB_SE_WINDOWS=128` selects 128 valid omission triples from the same `C(13,3)` pool. Those triples use eight distinct first-block schedules instead of the 54 used by the 256-window layout. The digest CTA remains 256 threads, 49,152 bytes of shared memory, and two resident CTAs per SM; warps 0..3 process one epoch pair and warps 4..7 process a second epoch pair. Each thread still represents exactly one `(epoch, window)` candidate, and the hit tag is decoded as `epoch * 128 + lane`.

The ranked family contains exactly `C(137,6) * 128 = 1,051,964,508,672` distinct candidates. A 1,200-second run at 600M/s needs about 720 billion candidates, and even 604M/s needs about 725 billion, so the reduced family retains substantial headroom without wraparound.

## Exact parity-window mechanism added from PR885

The negfold recovery needs only the low parity bit of each of two products, `p1*m1 + yR` and `p2*m2 + yR`; their full 256-bit residues are discarded. The parity-window helper evaluates the 27 cross-products that determine that bit after secp256k1 reduction, instead of running two complete 8-by-8-limb field multiplications and reductions. Its bounded carry test identifies the rare rows where the partial window is insufficient and sends those rows through PR871's unchanged `qsb_fmul` plus `qsb_fadd` path. In a 16,777,216-row CUDA differential, 16,777,207 rows used the fast path, nine used the fallback, and the result disagreed with the inherited path zero times.

The helper's contract is the one exercised by this recovery tail: arbitrary full-width product operands, a canonical nonzero `yR` addend, and a requested sign bit. It is filter-only. Every tentative hit is still recomputed by the unchanged exact recovery kernel before publication.

## PR871 base local matched measurement

The PR868 composition was compared with PR854 on an RTX 4090 at the organizer-default `nvcc -O3 -DQSB_ZEROS_N=24` target, official subset seed 526487517, ranked arguments, fixed 64 batches per arm, and balanced A/B/B/A order. Each arm completed exactly 8,589,934,592 candidates. A is PR854 and B is this composite.

| order | arm | warm wall ms/batch | epoch ms | first-state ms | digest ms | exact-replay hits |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PR854 A1 | 184.237451 | 1.061899 | 1.791768 | 181.256810 | 1,014 |
| 2 | composite B1 | 183.158792 | 1.150580 | 0.539826 | 181.341217 | 1,048 |
| 3 | composite B2 | 183.213873 | 1.150842 | 0.539791 | 181.397150 | 1,048 |
| 4 | PR854 A2 | 184.193130 | 1.065747 | 1.790725 | 181.208169 | 1,014 |

Mean wall time was 184.215291 ms for PR854 and 183.186333 ms for the composite, a **+0.56170% completed-work throughput improvement**. Both adjacent comparisons favored the composite, by +0.58892% and +0.53449%. The expected trade is visible directly: first-state construction falls by about 1.25 ms/batch, while the larger epoch producer rises by about 0.09 ms/batch and digest time remains essentially flat.

The fixed-work timing code and its identical chain-side diagnostic overlay existed only in isolated test copies and are absent here. Public PR868 separately reported +0.703% for `QSB_EPOCH_FAST + QSB_SE_WINDOWS=128` on its e876-derived source, with two-seed verification; that is donor evidence rather than a result reproduced by this package.

## Parity-window matched measurement

The new parity-window port was compared directly against PR871 with identical diagnostic source, official subset seed 526487517, organizer-default N24, and fixed 64 batches in A/B/B/A order. Every arm completed exactly 8,589,934,592 candidates and published the same byte-identical 1,096-hit file.

| order | arm | warm wall ms/batch | digest ms/batch | exact-replay hits |
| ---: | --- | ---: | ---: | ---: |
| 1 | PR871 A1 | 180.463236 | 178.564401 | 1,096 |
| 2 | parity window B1 | 179.584988 | 177.684009 | 1,096 |
| 3 | parity window B2 | 179.617404 | 177.718394 | 1,096 |
| 4 | PR871 A2 | 180.906521 | 179.008001 | 1,096 |

Mean wall time fell from 180.684879 to 179.601196 ms/batch, a **+0.60338% completed-work throughput improvement**. Both adjacent comparisons favored the parity window, by +0.48904% and +0.71770%; digest throughput improved by +0.61058%. Fixed-work timing and raised hit-cap diagnostics are absent from this source-only package.

## Correctness checks

The 128-window layout searches a different valid subset of the full window family, so its complete fixed64 hit set is not expected to equal PR854's 256-window hit set. Correctness was checked in four ways:

- The parity-window CUDA differential tested 16,777,216 random and directed rows against the inherited full-product path: zero mismatches, 16,777,207 fast rows, and nine exact-fallback rows.
- All four PR871/parity-window fixed64 arms produced the byte-identical 1,096-record hit file, raw SHA-256 `8dda10dcf41d84faca98fb0da3de20763709a3c15cb4f002da8ae3cc3bba7c4f`; the normalized set SHA-256 is `1b5c727cb1380c6ae0438e74011582cb652620bc0d26b7d14ccea59075572c06`.
- The unchanged PR871 base previously passed independent CPU verification of **1,048 / 1,048** fixed-work hits, zero failures, when its 128-window mechanism was introduced. The parity-window ABBA additionally traverses the unchanged exact replay for every one of its 1,096 published records.
- All 128 selected windows are a subset of PR854's 256 windows. Restricting PR854 to those windows and restricting the composite to PR854's first 33,554,432 epochs produced **541 versus 541** normalized hits with symmetric set difference **zero**.

The mapping also has a direct accounting proof: a 256-thread block covers four consecutive epochs as two warp-aligned 128-lane epoch pairs; full launches therefore cover the same 134,217,728 candidate count as PR854 without overlap or gaps.

## Build and risk

The clean organizer-default sm52/N24 build retains the ranked digest resource class: 128 registers per thread, 49,152 bytes shared memory per CTA, one barrier, zero stack, and zero spills. The two fast epoch producer kernels use 48 registers and zero spills. The matched control and candidate also had the same resource class. The complete digest SASS contains 49,304 instructions in control and 49,926 in the candidate because the cold full-product fallback remains in the binary; the guarded hot path avoids the two full products. `QSB_PROBLEM_SEED=777 ./setup.sh subset` and its CPU verifier smoke pass on the clean package.

The two PR868 mechanisms and the guarded PR885 parity-window calculation ported here are exact under their stated input contracts. The inherited PR854 `QSB_SHORT_CARRY4` speculative filter remains approximate and can theoretically lose a rare true proposal; exact replay prevents false records from being published but cannot restore a missed proposal. The prior 85,915-hit official run, the local CPU verification above, and the common-domain equality are finite evidence rather than a universal recall proof.

## Attribution

The promoted e876 base was submitted by **jacklightChen**. Its H0-only gate credits **Saviour1001**, and its paired preparation lineage credits **owizdom**, **DPZZxlz**, and **fkiene**. The negfold-parity research was published by **fkiene**. `QSB_SHORT_CARRY4` traces to **Meganpark980320** PR #654. **dun999** assembled and measured the exact PR854 negfold + carry4 runtime. **ercumentyildirim** authored and published the PR868 fast epoch producer and 128-window/two-pair CTA mechanism. **EvanYan1024** published the PR885 parity-window mechanism; its public validation commit also credits **terrapinelf** and **DrCleverHans**. All inherited source and license notices remain intact.

This subset port, vector differential, balanced local comparison, and packaging were performed with **GPT 5.6 Sol** using **Codex**. Co-authorship should retain dun999, fkiene, Meganpark980320, and ercumentyildirim for PR871 and add EvanYan1024 for the parity-window mechanism.

The speculative first-fold carry cut, two-problem comparison, and final package update were performed with **GPT 5.6 Sol** using **Codex**.
