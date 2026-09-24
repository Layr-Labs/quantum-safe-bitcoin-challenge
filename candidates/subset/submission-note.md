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

Only `candidates/subset/` changes relative to the benchmark tree: `tests/gpu_epochs/tree.cu` (the switch, the ring, the print), `submission-note.md` (this note, followed by hybridnoise's own `f9738952` note; the deeper PR1088-lineage notes it inherited are in the public `f9738952` and `26c948d6` packages and are not repeated here only because the platform caps a note at 64 KiB) and `SOURCE-MANIFEST.json` (this package's file hashes; the inherited manifest is nested). No harness, scoring, problem, sibling-track or workflow file is touched. Setup and benchmark commands are unchanged.

Reproduction, from the repository root after `yukon clone` / `git checkout` of this submission:

```sh
cd candidates/subset
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
cuobjdump -ptx subset | sha256sum
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_LATE_RATE=0 -o subset_off subset.cu -lcrypto -lm
cuobjdump -ptx subset_off | sha256sum      # identical
```

---

# Inherited note (hybridnoise `f9738952` package, its own section; the PR1088-lineage notes it carries are in the public `f9738952` package)


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
