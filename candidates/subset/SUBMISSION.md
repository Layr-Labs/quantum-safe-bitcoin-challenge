# Subset submission — speculative stack on the `e876032` crown

Model: deepseek-v4.1-flash · Harness: opencode · Base: `e876032f79e6f4f3af2732bbba39403e29f0e227` (crown 595,907,916).

No local nvcc/GPU: no measured speedup is claimed; the official remote run
determines build, verified throughput and score.

## Changes (each behind its own kill switch, default `1`)

`hit_filter_field_sc.cuh`, `QSB_SEED_SHORT`: the three speculative subtractions
inside `qsb_filter_point_seed` (`P = X2-X1`, `R = Y2-Y1`, `Q-X3`) move from the
canonical `_ModSub256` to `qsb_seed_sub`, the same truncated-borrow fold
(`K = 2^32+977`, borrow kept through limb 1) the inlined point-add asm and
`qsb_fsub` already use.

`pair_shared.cuh`, `QSB_NEGFOLD_PARITY` and `QSB_EPOCH_SHA_PAIR`: attributable
ports of the public frontier submissions `63b0fcc9` / `252f6acb`. Negfold forms
the two `qsb_k2s_post3` y-parities from negated differences (two fewer 256-bit
subtractions per pair); the epoch pair hash routes both epochs' SHA-256d digest
compressions through the existing round-interleaved `qsb_sha256_init_transform_pair`
(two serial 64-round chains become one interleaved pass).

`parity_window_subset.cuh`, `QSB_K2S_PARITY_WINDOW`: port of public PR885
(EvanYan1024, `3e166ba4`). The two full field multiplies that exist only to read
a y parity are replaced by the exact parity-window core, with a guarded
`qsb_fmul`+`qsb_fadd` fallback.

Every tentative hit is still re-derived exactly by `kernel_verify_pair_hits` ->
`qsb_k2s_front_exact` -> `qsb_k2s_post` -> `qsb_k2s_gate` before publication, so a
wrong speculative value can only lose a tentative hit, never publish one. Each
`-DQSB_*=0` restores the promoted body.

## Evidence

- `SOURCE-MANIFEST.json` — hashes and byte count for this tree.
- `work_yukon_subset/host_audit_seed_short.py` — exact model of both seed
  reductions (10^7 random pairs + 36 boundary cases, 0 divergences).
- `work_yukon_subset/win3_opt.py` — shows the current 256-window `WIN3` selection
  already attains the minimum possible 54 first-block classes.
- `parity_window_subset.cuh` is byte-identical to public `252f6acb`; the new
  header passes `g++ -fsyntax-only` with CUDA stubs.
- The full submission note (provenance, accounting, attribution) is appended to
  `submission-note.md`.

## Reproduction

```
git checkout e876032f79e6f4f3af2732bbba39403e29f0e227
# apply this diff to candidates/subset/
yukon setup --track subset && yukon run --track subset
```
