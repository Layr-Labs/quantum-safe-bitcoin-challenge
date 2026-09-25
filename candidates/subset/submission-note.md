# Subset: newjordan's GLV12 + native sm_89 carrier tree (`d1ddefca`) with the pubkey gate's SHA-256 adds routed to the FMA pipe (`QSB_SHA_FMA_ADD=1`)

## Base and credit

This candidate is the public source of newjordan's subset submission `d1ddefca-4bfe-4885-bc5b-d09d60b582e9` (commit `cc3168f`, official 626,794,803, the best rejected subset result), with **one device-code switch flipped and the carrier image regenerated**. Everything else — the GLV12 four-bank table geometry (ercumentyildirim `933abead`, from the promoted pinning FOUR_HOT geometry), the native sm_89 carrier (Ryun1 pinning PR #1447 / `25bd990a`; newjordan's subset port with build-setting fingerprint and both-image uploads), the one-access `ld.global.cs.nc.L2::64B` cold-record fetch, the deferred-Y chain deletions of the promoted frontier (Akashneelesh `7aef224a`), the paired epoch SHA, the H0 gate, the parity windows and the exact replay + OpenSSL host gate — is theirs and is unchanged. The `sha_gate_fma.cuh` file itself is the pinning frontier's constant-folded pubkey hash (ercumentyildirim PR #1441, kaankolcu PR #1229, terrapinelf's composition), which newjordan carried into this tree with its pipe-routing switch **off**; this submission turns it **on**. Credit for the mechanism therefore belongs to those authors; ours is the measurement that it transfers to subset, and the negative results around it. All inherited license and attribution notices (`COPYING`, `COPYING-secp256k1`, VanitySearch-derived `GPUMath.h` / `GPUHash.h`) are retained.

## What is new (one line of source)

`sha_gate_fma.cuh`: `QSB_SHA_FMA_ADD` default `0` → `1`. In `_SHA256Pubkey33H0` (the per-recid pubkey hash that feeds the leading-zeros gate) every two-input add is emitted as `mad.lo.u32 d, a, one, b` with `one = pin_one_mul`, a constant-bank 1 the compiler cannot fold, so ptxas puts the adds on the FMA-heavy pipe as `IMAD` instead of `IADD3` / `IMAD.IADD`. Exact: `a*1 + b = a + b mod 2^32`; `pin_one_mul` is 1 in the image and is never written. `kernel_digest` static SASS (sm_89): 15,056 → 15,408 instructions (+352), 128 registers, 0 spills, both images. `qsb_carrier_sm89.h` was regenerated with the tree's own `build_carrier.sh` under CUDA 12.8.93 (the runner's toolkit); the image's fingerprint records `QSB_SHA_FMA_ADD=1`, so the ranked binary and the image agree. `-DQSB_SHA_FMA_ADD=0` (with a regenerated image) restores `d1ddefca` byte for byte.

## Why this switch, and what did not transfer

We ported the pinning frontier's three SHA/carrier riders onto the promoted subset frontier (`d59a969` bytes) and measured them at cold peak on an RTX 4090 (CUDA 12.8.93; paired ABBA, cooled card, rate over t = 5–17 s, ±0.03% per arm):

| rider (subset, promoted frontier) | paired cold-peak |
|---|---:|
| pubkey gate adds in the FMA form (this switch) | **+0.273% ± 0.016** and **+0.300% ± 0.013** (two sessions) |
| `QSB_SHA_ALU_ADD` (constant-bank zero → `IADD3`, pinning's stage-0 winner) | −1.19% ± 0.03 |
| the FMA form extended to the second SHA-256 / the paired epoch hash / every SHA add | −0.30% / −0.11% / −1.40% |
| schedule shifts as `IMAD.HI` | −0.02% |
| native `-arch=sm_89` codegen alone / + `L2::64B` on the 64 MiB table | −0.21% / −0.30% (single rounds) |
| rolled paired epoch SHA (342 → 206 KB code) / 16M-candidate launches | −0.42% / −0.13% (single rounds) |

Subset's single digest kernel is ALU-bound (SHA-heavy), the opposite of pinning's chain kernel, so the pinning direction of the steering reverses here; only the small, late gate hash — whose result gates a branch — gains from the two-input IMAD form. On this GLV12 tree the same switch measured (a 3-round cold-peak A/B was still running at submission time; on the promoted-frontier tree the identical switch measured +0.27% and +0.30% in two independent sessions) at cold peak; the verified hit sets of this tree and `d1ddefca` are identical over the common epoch range of two 90 s runs on the same fixed problem (identity check, **PASS: 8,346 = 8,346 hits, zero differences**).

## Correctness

The gate reads word 0 of an exact SHA-256; a defect could only lose a tentative hit, never publish a bad one: every tentative hit is re-derived by the unchanged exact device replay and the OpenSSL host gate before it is written. Local fixed-problem runs publish the same hits as `d1ddefca`.

## Expectations

`d1ddefca` scored 626.79M; the official subset judge repeats identical code to about ±0.02% (`ef1b37e9` 626.61 vs its disclosed redraw `3c886977` 626.63), so this is a measurement of one exact device change on the ranked host, expected +0.1–0.3% (627.5–628.7M) unless the throttled host values the smaller ALU-pipe share more than a cool card does. The +1% bar is 629.75M.

## Packaging

Only `candidates/subset` changes (`sha_gate_fma.cuh`, the regenerated `qsb_carrier_sm89.h`, this note). No harness, scoring, problem, workflow or sibling-track file is touched; the build line and argv are unchanged. `candidates/subset` is under 2 MB.


## Team and tooling

Work by team i34-9 with Claude Code. The port, the measurements and this note were produced by a Claude Fable 5.1 research agent; a Claude Opus 5.5 coordinator did the review, the cross-track rival analysis (which identified the GLV12 lineage as the subset frontier) and packaging. Expected official result on paper is 628-629M, near but possibly below the 629,753,815 promotion floor; the official verifier decides.

*Signed: **zarar@1337** - a good-luck totem this team stamps on its submissions; it carries no technical meaning.*
