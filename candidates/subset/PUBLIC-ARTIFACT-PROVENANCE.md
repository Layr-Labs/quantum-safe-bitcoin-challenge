# Historical public source notes

The following donor notes are preserved for attribution and reproducibility. Their model labels, measurements, dates, and status statements describe their authors' original work, not this resubmission. Current independent checks and limitations are in submission-note.md.

## PR1211 / Saviour1001

# Subset successor from the PR1137 source

Model: GPT 6 Sol, xhigh reasoning. Harness: Codex.

This subset candidate starts from the public
[PR1137](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1137)
source (`d5d2283e47fdb0108ff36b3e6acc5d16c115d8ad`), whose official score was
624,752,385 candidates/s. The current leader is 623,518,629 candidates/s, so
the 1% promotion floor is 629,753,816 candidates/s. The PR1137 source fell
short of that floor. The original benchmark source base for this worktree is
`b59484345df5208f5caffc82c25a4a3b50cbe523`.

## Changes

- `QSB_DROP_Z2_EARLY=1` removes the early carry propagation into limb 2 at
  nine lean field-square fold sites. The later carry propagation remains. This
  belongs to the speculative filter and can very rarely lose a hit; an exact
  host gate rechecks every tentative hit before publication. Setting the switch
  to 0 restores the inherited fold code.
- `QSB_RECODE_BASE_A=1` uses the base point A directly rather than A/2 in the
  fixed-base table. The scalar recoding, GPU table construction, host spot
  check, and CPU fallback were changed together. Setting the switch to 0
  restores the inherited A/2 path.
- `QSB_SHA_FOLD=1` uses the packaged `_SHA256TransformDigest32Q` for the second
  SHA-256 transform in the paired-epoch path. The routine already exists in
  the inherited package; this change connects it to the active call site.
  Setting the switch to 0 restores `_SHA256Transform`.

All other executable changes in this directory are inherited from PR1137,
including its exact host verification gate, field chain, and SHA routine.
Source licenses and contributor notices are retained. The PR1137 source
inherits the PR1088 composite credited to terrapinelf and its earlier
contributors. The active SHA idea follows the public pinning SHA work; the
base-A and field carry changes were prepared here.

## Verification and limits

- The organizer-style CUDA 12.6 build completed on arm64, including native
  `sm_89`. `kernel_digest` uses 128 registers and 49,152 B shared memory per
  CTA, with zero stack frame and zero spills in that static build.
- The base-A scalar recoding passed 20,008 random and boundary identity/bounds
  checks. The default preprocessed PTX point-update path matched a reference
  over 1,000 random inputs.
- `tests/sha_digest32_host.cpp` compiles the actual optimized SHA digest
  function as host C++ and compares 10,000 random 32-byte messages with
  OpenSSL SHA-256; all eight output words matched in every case. This tests
  the algebra and call contract, not CUDA scheduling or target-GPU throughput.
- A native sm_89 static ablation at CUDA 12.6 found 14,896 `kernel_digest`
  SASS instructions and 4 B spill stores/loads with all three new switches
  disabled. Enabling only the carry cut gave 14,880 instructions and zero
  spills; only base-A gave 14,840 and 4 B spills; only SHA fold gave 14,888
  and 4 B spills. All three enabled gave 14,808 instructions and zero spills.
  These are compile artifacts, not an end-to-end throughput measurement.
- No NVIDIA GPU is available in this environment. No candidate-vs-base
  throughput or full hit-set measurement exists for this combination. Static
  resource counts do not establish a ranked speedup or correctness on the
  official CUDA 12.8 runner. The three changes may interact, especially in
  register scheduling and the SHA path. This submission is the target-GPU
  measurement for the strongest locally verified subset candidate available
  in this handoff; no score claim is made from the static measurements.

The package contains only `candidates/subset/` changes relative to the
promoted benchmark source. `SOURCE-MANIFEST.json` records every source file and
hash, excluding this note and the manifest itself to avoid self-reference.

## Why this package is a separate candidate

The promoted subset source and PR1137 differ substantially. PR1137 already
bundles the optimized paired-epoch and field paths, which makes it the
relevant starting point for these three cuts. The source tree here was copied
from that public submitted package before the new switches were applied. The
package itself stays within the editable subset directory; the benchmark
schema, harness, problem generation, scoring, and verification files remain at
the promoted source commit. `SOURCE-MANIFEST.json` identifies the precise
implementation commit and records byte hashes, so the candidate can be
reconstructed without relying on a worktree name or an untracked build output.

The direct base-A recode removes the need to multiply each scalar by two modulo
the group order before digit extraction. Its table uses A in both the GPU build
and the host fallback, and the host sample checker expects the same point. A
one-sided change at any of these sites would yield a valid-looking table with
incorrect scalar products. The scalar identity/bounds property test covers
zero, boundary, order-adjacent, and random inputs. It does not substitute for
an end-to-end hit-set comparison on an NVIDIA GPU.

The paired SHA transform is called only from `qsb_pair_second_sha_z` when the
paired-epoch path is enabled. The inherited `sha_gate_fma.cuh` implementation
contains the second-block schedule and digest extraction. The switch is local
to that call; the first SHA transform, candidate enumeration, and host
publication check follow the inherited PR1137 package. The file's CRLF line
endings were normalized for a clean source diff; the code tokens are the same
as the public donor file.

The field cut removes one `addc.u32` at each selected fold. PTX carry flags are
sensitive to instruction order, so the verification model used the candidate's
preprocessed PTX rather than only a handwritten algebraic rewrite. There were
no differences in the 1,000 tested point updates. That sample cannot bound
extremely rare carries. The exact host gate can reject false tentative hits,
while any lost tentative hit would lower verified throughput; that risk is part
of the reason this source has not been uploaded.

## Promotion gate

A rank-worthy upload needs a target-GPU comparison against the same source
with these switches disabled, with identical problem seeds and verified hit
sets, plus a comparison against the current frontier. The official score uses
verified hits over elapsed time. A favorable SASS count or a donor's earlier
rate is not an official throughput measurement for this combination. A future
submission should refresh the leader and 1% floor immediately before upload,
then replace this note's status and measurements with the observed data.


## PR1137 / hybridnoise

Model: Claude Opus 5.5
Harness: Claude Code

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
