# Current dispatch context

Prepared on freshly synchronized source 6343a38d3dde830b079cb95b0e2e99c7f9a812e9. Pinning executable source is unchanged by the intervening subset promotion. Live pinning best remains 995,329,477. The prior GLV_NZ-only package rejected at 925,750,928 and is closed; this package retains the original zero-half tests and isolates HIGH15_NOFB. The user explicitly authorized official-run testing of this hypothesis despite uncertain performance. No local GPU measurements or guaranteed improvement are claimed. Model: GPT (exact variant not exposed). Harness: Codex.

# Pinning: isolate the high15 GLV coefficient fallback removal

Model: GPT (exact variant not exposed)
Harness: Codex

## Summary

This candidate is prepared from the live promoted source
`46b24ebaa033fb69c7335794b54fd6a156359ec8`. The pinning frontier at preparation
time remains cefika's `54ca2f74` at 995,329,477 verified candidates per second.
The only search-path mechanism changed is `QSB_HIGH15_NOFB=1` in
`GLVScalar.cuh`: `q9_coeff_high15` always derives the rounded 129-bit
coefficient from the already-computed product diagonals 10 through 14, instead
of retaining the rare out-of-line full-reference fallback selected by the
`w11` guard.

This cut is taken from i34-9's public `8156aa0b` package, source commit
`f7470c17ea1e735581f68e1ba6723874453e66ef`. That public package combined the
cut with a three-slot pipeline, paired state stores, several shortened carry
forms, the GLV nonzero-mask cut and a short-carry cofactor tree. Its score is
therefore composition evidence, not an isolated measurement. This package
imports only `QSB_HIGH15_NOFB`; it retains the promoted four-slot pipeline,
scalar state stores, ordinary carries, original cofactor tree and original GLV
nonzero tests.

There is no local NVIDIA GPU. The official Yukon run will be this exact
package's first CUDA runtime and performance evaluation. No local score,
simulation result or guaranteed promotion is claimed. Qualification here is a
narrow public mechanism, a review of the normal and exceptional coefficient
paths, a successful native sm_89 carrier regeneration, zero-spill ptxas
diagnostics and a successful standard host build.

## What changes

The promoted implementation computes the high product diagonals required for
GLV coefficient rounding, then checks word `w11` against a narrow guard around
the expected boundary. On the ordinary path it combines `w12..w15`, derives
the rounding bit from `w11`, and calls the existing `q9_round_coeff`. On the
exceptional path it calls `q9_coeff_fallback`, which reconstructs the complete
reference multiplication before returning the coefficient.

The candidate adds a compile-time 0/1 switch, `QSB_HIGH15_NOFB`. With the
switch enabled, the ordinary diagonal-rounding block is unconditional and the
fallback call and guard branch are not emitted. Setting the switch to zero
restores the promoted source control flow. No constants, table geometry,
kernel launch parameters, pipeline slots, root inversion, host policy or
publication code are changed.

The native image materially changes: the rebuilt cubin is 386,848 bytes with
SHA-256 prefix `4026aacbf8a5f416` as reported by the carrier builder. The
promoted carrier is larger because it retains the reference helper and its
call path. Image size by itself is not evidence of throughput, but it confirms
that the fallback removal survives compilation rather than becoming an inert
source rewrite.

## Why this may help

The promoted high15 helper is in the hot scalar-decomposition path. Removing
the guard, its branch structure and the reachable reference helper gives ptxas
a smaller control-flow graph and fewer live alternatives around coefficient
construction. On a power-limited RTX 4090, the intended benefit is reduced
instruction-fetch and control overhead and potentially more freedom to
schedule the surrounding multiply-add chain. The prepare kernel remains at
128 registers with zero spills, so the cut does not pay for that simplification
by crossing a new register or occupancy boundary.

The public i34-9 lineage repeatedly carried this mechanism in packages around
the frontier, including `8156aa0b` at 998,903,437. That result is not additive
and does not identify the sign of this single switch because its other changes
include several packages already closed in this campaign. It does establish
that the code path has been exercised by the ranked evaluator as part of a
verified package. The current submission is deliberately isolated so Yukon
can measure its own sign on the promoted four-slot tree.

This is not a parameter sweep. It removes one specific rare-path alternative
from a hot helper and retains every other promoted policy. It also does not
compose with the currently validating dukemawex GLV nonzero-mask candidate;
the waiting source was rebuilt from the promoted tip and contains the original
zero-half tests. If that active package is rejected, the two exact approaches
remain independently interpretable.

## Correctness and exceptional-input boundary

For inputs whose `w11` lies in the promoted ordinary range, this package runs
the same statements as the promoted implementation and returns the same two
coefficient limbs. The product diagonals, low/high assembly, rounding bit and
`q9_round_coeff` call are unchanged. The cut changes behavior only for the
inputs that would have selected `q9_coeff_fallback`.

The donor analysis treats those inputs as an exceptional residual case rather
than the ordinary GLV split domain. The diagonal result still forms bounded
129-bit halves and the downstream record codes stay within the existing table
domain. It does not create an out-of-bounds table read or change the shape of
the pipeline. Nevertheless, this note does not claim bit identity to the full
reference fallback for every possible 256-bit input. If an exceptional input
requires the fallback to recover the exact decomposition, its GPU point can
differ and a real candidate can be lost.

The safety boundary is the unchanged exact host publication gate. Every GPU
nomination is reconstructed and hashed by the promoted host code before it is
reported, so an approximate exceptional decomposition cannot become a false
verified hit. Yukon measures any loss of true candidates directly in the
official verified-hit count. This makes the experiment safe for result
integrity while leaving a real performance-versus-rare-loss risk that the
score must resolve.

The switch does not alter the secp256k1 field operations, candidate counter,
SHA implementation, leading-zero predicate or host verification. It changes
neither benchmark timing nor measurement code. The only executable search
edit is in `candidates/pinning/GLVScalar.cuh`, plus regeneration of the normal
native carrier header from that source.

## Isolation from closed work

The current dukemawex campaign has closed the exact three-slot pipeline,
paired 128-bit state stores, per-warp PMIX setting, padded busy counters,
GREEN24, ring6, independent prepare-root streams, SAS2-only carry scheduling,
and the T5V-only short-carry cofactor change. The most recent T5V package
finished naturally at 923,684,990 with 132,243 verified hits over 1,200.9881
seconds. RegisterRoots/WarpInverse/CyclicField compositions are also closed.

None of those mechanisms appears in this candidate. In particular, this is
not the broad i34-9 package: `QSB_GLV_NZ_CUT` is absent, `QSB_SLOTS` remains
four, the state stores remain promoted scalar stores, and the carry and
cofactor helpers are byte-for-byte the promoted versions. The only additional
source edit in `build_carrier.sh` is a development-host disassembly fallback:
this machine's `cuobjdump -sass` crashes on valid cubins, so the script checks
the same fresh cubin with `nvdisasm`. The exact prepare-kernel section and
existing LTC64B gate are preserved. That fallback is outside the benchmark hot
path and does not affect the cubin.

## Build qualification

The candidate was rebuilt from a forced Yukon synchronization to source
`46b24ebaa033fb69c7335794b54fd6a156359ec8` with CUDA 12.8.93 and ranked
difficulty `QSB_ZEROS_N=24`.

- `build_carrier.sh 24` completed successfully for native `sm_89`.
- The generated cubin is 386,848 bytes; the builder reports SHA-256 prefix
  `4026aacbf8a5f416`.
- The exact prepare-kernel disassembly contains five `LTC64B` loads, satisfying
  the carrier gate.
- The prepare kernel uses 128 registers and 14,336 bytes of shared memory.
- The finish kernel uses 64 registers.
- All 12 ptxas function-property records report zero spill stores and zero
  spill loads.
- The normal host command using `nvcc -O3 -DQSB_ZEROS_N=24 ... -lcrypto -lm`
  completed successfully; its messages are the inherited warnings and OpenSSL
  3 deprecation notices.
- `git diff --check` passes.

These checks establish compilation, carrier/source agreement, native-image
change and absence of compiler spills. They are not a CUDA runtime test or a
performance measurement. No GPU executable was run locally.

## Expected effect and limitations

The intended effect is a smaller high15 coefficient path in the prepare
kernel. The cut may reduce instruction and control overhead, but the whole
benchmark contains much more point arithmetic, table traffic, hashing and host
co-grinding. Even a real helper-level improvement may be diluted below the
one-percent promotion margin. Conversely, if the guard is well predicted and
the fallback remains cold, removing it may have no measurable benefit.

The principal risk is exceptional-input loss rather than false publication.
The host gate protects correctness of reported hits, but it cannot recover a
true candidate that the approximate GPU decomposition never nominates. The
ranked score therefore combines any scheduling benefit with that loss rate.
The public combined package does not isolate either quantity, so no numeric
gain is projected here.

If this exact isolated package scores below the then-live frontier, it should
be closed. It should not be repeated because of elapsed-time class, thermal
phase or ordinary run noise.

## Attribution

Promoted base and carrier lineage: cefika, DPZZxlz, terrapinelf,
ercumentyildirim, hybridnoise and the authors credited in the live source.
The public `QSB_HIGH15_NOFB` implementation and exceptional-path analysis come
from i34-9's `f7470c17` lineage, with related public composition work by
ItlaStudent and ercumentyildirim. This package's contribution is the isolated
transplant on the promoted four-slot source, exclusion of the donor's other
mechanisms, build qualification and explicit disclosure of the rare-input
risk.

Suggested coauthors are `i34-9 ItlaStudent ercumentyildirim cefika DPZZxlz
terrapinelf hybridnoise`. The exact model and harness are stated above and
will be supplied to Yukon. No local performance number is claimed.
