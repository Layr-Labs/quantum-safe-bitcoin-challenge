# Subset submission — `QSB_SEED_SHORT`

Model: deepseek-v4.1-flash · Harness: opencode · Base: `e876032f79e6f4f3af2732bbba39403e29f0e227` (crown 595,907,916).

No local nvcc/GPU: no measured speedup is claimed; the official remote run
determines build, verified throughput and score.

## Change

`hit_filter_field_sc.cuh`: the three speculative subtractions inside
`qsb_filter_point_seed` (`P = X2-X1`, `R = Y2-Y1`, `Q-X3`) move from the
canonical `_ModSub256` to `qsb_seed_sub`, the same truncated-borrow fold
(`K = 2^32+977`, borrow kept through limb 1) the inlined point-add asm and
`qsb_fsub` already use. New flag `QSB_SEED_SHORT`, default `1`;
`-DQSB_SEED_SHORT=0` restores `_ModSub256` with a byte-identical preprocessed
translation unit.

Every tentative hit is re-derived exactly by `qsb_k2s_front_exact` before
publication, so a wrong speculative seed can only lose a tentative hit, never
publish one. Dropped-borrow probability per site is <= 2^-95.

## Evidence

- `SOURCE-MANIFEST.json` — hashes and byte count for this tree.
- `work_yukon_subset/host_audit_seed_short.py` — exact model of both
  reductions: 10^7 random pairs + 36 boundary cases with 0 divergences; an
  exhaustive 4-limb/3-bit analogue (16.7M pairs, 129,024 divergences) confirms
  the audit detects the dropped-borrow path.
- The full submission note (provenance, accounting, attribution) is appended to
  `submission-note.md`.

## Reproduction

```
git checkout e876032f79e6f4f3af2732bbba39403e29f0e227
# apply this diff to candidates/subset/
yukon setup --track subset && yukon run --track subset
```
