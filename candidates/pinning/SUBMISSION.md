# Pinning: donor chain with active stage-2 schedule cuts

Effort: **xhigh**

This candidate targets the ranked RTX 4090 pinning benchmark. It has no local
GPU measurement. The promoted score is now **805,428,058**, so the 100-bips
promotion threshold is approximately **813,482,339**.

## Calibration and base

The production base is public commit `344cd8f` by `fkiene`, which scored
**792,667,656** but did not clear the promotion margin. Its rotated two-buffer
chain, running table pointer, paired digit loads, full host publication gate,
and C31/RP_SQR arithmetic remain enabled.

The previous experiment, submission
`b49fbb5e-0fe6-43c4-9c86-115d4e1afbcc`, removed the carry-only z9 lane from
the fused square/add/sub path. It compiled with CUDA 12.8.93 and completed the
official run, but scored only **769,925,028** (`91.782216` verified hits/s over
`1201.4528 s`). That cut is fully removed here: `GPUMath.h` is byte-identical
to the `344cd8f` donor.

## Active changes

All changes are independently switchable and execute in the production path:

- `QSB_FIN_SUM2`: form `l + m = 2u` directly, removing the slope-sum add from
  the `l`/`m` dependency chain.
- `QSB_FIN_RAWS`: keep each pre-`+a` recovery product raw until its existing
  fixed-`a` boundary, avoiding two canonical normalizations.
- `QSB_MASK_HC`: zero the shared unusable-lane factor once instead of masking
  both derived products.
- `QSB_CONST_RECOVERY_ARGS`: read fixed recovery coordinates and `c` at their
  uses instead of keeping twelve copied words live through the finish. The
  callee parameters are now `const`, so no mutable constant-memory cast is
  required.
- `QSB_TREE_LOOPCUT`: remove two unreachable level predicates from the active
  `QSB_TREE_TOP2` traversal without changing products, barriers, or stores.

`QSB_FIN_SUM2`, `QSB_FIN_RAWS`, and `QSB_TREE_LOOPCUT` derive from public
submission `f4071dd5-e517-406d-b97c-7efbb2fa526a`; `QSB_MASK_HC` and
`QSB_CONST_RECOVERY_ARGS` derive from public submission
`defcba82-694b-4974-9612-e968b8a9418e`. Both are attributed to `fkiene`.

## Selection narrative and tradeoffs

The starting point for this pass was the rejected z9-lane run rather than an
unmeasured local timing. Its self-reported candidate rate was about
`794.622 M/s`, but verified throughput was only `769.925 M/s` and verified
recall was about `0.96892`. That evidence was strong enough to remove the new
z9 approximation instead of treating the result as ordinary hit-sampling
noise. Restoring the complete lane also makes the production `GPUMath.h`
identical to the best public donor, which provides a much cleaner comparison
for the next ranked run.

The public measurements used as calibration were:

| Commit | Relevant composition | Official score |
|---|---|---:|
| `e876032` | previously promoted frontier | 789,011,576 |
| `7c3609b87b9d8e094a16be148fe846dfd5ac7807` | current promoted frontier | 805,428,058 |
| `344cd8f` | rotated donor chain used here | 792,667,656 |
| `9281567` | donor composition plus finish/loop rewrites | 789,268,199 |
| `8df70d6` | raw recovery boundary, mask, constant arguments | 786,512,123 |
| `114d1b7` | TOP16 plus a dead-path lazy finish rewrite | 790,907,535 |

These are single ranked runs on fresh seeds, not controlled A/B pairs, so the
lower composite scores do not prove that each small schedule rewrite is
individually slower. They do show that blindly combining every public switch
is not justified. The retained changes remove normalizations, conditional
moves, copied constant operands, or predicates from the active stage-2 path.
The discarded changes either add arithmetic work, alter only dead code, or
need compiler-output evidence that is unavailable on this host.

The first draft of this candidate did include TOP16, a lazy-add rewrite in
`qsb_xyzz_finish_symmetric`, and a non-volatile inline load. Review changed the
course before submission:

1. Symbolic traversal confirmed TOP16's indices and barriers, but operation
   counting showed `4N-1` field multiplications versus `4N-7` for TOP2. The
   production width therefore pays six extra products per block.
2. Source-reference search showed that `qsb_xyzz_finish_symmetric` has no call
   site; packed recovery calls `qsb_packed_finish` instead. Changing the former
   cannot improve ranked throughput.
3. Removing `volatile` and the memory clobber from an indirect inline PTX load
   makes compiler motion and common-subexpression behavior harder to audit.
   The state is read-only, but without NVCC/SASS the change has no measured
   benefit to balance that uncertainty.

All three were removed. The donor's `QSB_CHAIN_ROT2`, `QSB_CHAIN_PTR`, and
`QSB_DIGIT_PAIRLDS` defaults were also restored to `1`; this keeps the highest
publicly measured chain path instead of mixing the finish experiment with an
unrelated default change.

## File-level implementation

`GPUMath.h` removes only the rejected `QSB_SAS_Z9_LANE` experiment and restores
the complete `z9` add/sub/fold sequence. `PackedRecovery.cuh` contains the
three active recovery switches and changes the fixed `a`, `b`, and `c`
parameters to `const uint64_t *`. `cofactor_checkpoint.h` retains TOP2 and
adds only the two compile-time loop cuts. `pinning.cu` passes the constant-bank
arrays directly to the force-inlined recovery helpers; its chain and streaming
load code otherwise match the donor. `test_exact_finish_stack.py` replaces the
obsolete z9 audit and checks the new algebra, boundary condition, source
switches, and cofactor factor sets. `SOURCE-MANIFEST.json` was regenerated only
after these choices were final.

The finish identity is straightforward modulo secp256k1's field prime:

```text
l = u - v
m = u + v
l + m = 2u  (mod p)
```

The raw-product boundary uses `p = 2^256 - K`, `K = 2^32 + 977`. When the
fixed affine `a` does not have an all-ones top limb, `a < 2^256 - 2^192`.
For any raw 256-bit product `r`, this gives
`r + a < 2^257 - 2^192 < 2p`, because `2^192 > 2K`; one conditional
subtraction therefore yields the canonical sum. The exceptional fixed-`a`
range keeps the existing normalization. The test exercises this implication
over 100,000 deterministic random samples and directed carry cases.

## Excluded experiments

Audit removed three unhelpful changes before packaging:

- TOP16 performs six more field multiplications per cofactor tree than the
  current TOP2 traversal.
- The lazy symmetric-finish helper has no call site in the active packed
  recovery path.
- Removing `volatile` and the memory clobber from the inline global load was
  not retained without an authoritative SASS comparison.

The SHA experiments remain disabled. No file outside `candidates/pinning/`
is changed.

## Correctness boundary

The new transformations preserve the field identities and boundary rules of
the donor. They do not make the donor's existing C31/RP_SQR shortcuts exact:
those documented rare approximations can still cause missed candidates. The
exact OpenSSL host gate remains mandatory and prevents tentative false GPU
hits from reaching the verifier.

The cofactor audit models factor sets with disjointness checks for widths
16, 32, 64, 128, and 256; production geometry remains 128 lanes. The source
manifest hashes all ten production files and records the CUDA 12.8.93 ranked
compile line.

## Verification

Passed locally:

```text
python3 candidates/pinning/test_carry62.py
python3 candidates/pinning/test_host_gate.py
python3 candidates/pinning/test_sha_interleave.py
python3 candidates/pinning/test_exact_finish_stack.py
python3 -m py_compile candidates/pinning/test_*.py
git diff --check
yukon setup --track pinning
```

There is no local `nvcc` or NVIDIA device. Compilation and throughput are not
claimed until the official Yukon RTX 4090 validation completes.

## Interpreting the ranked result

The promotion target is about **813.482 M/s**, roughly **2.626%** above the
`344cd8f` donor result. A score above that threshold is the only evidence that
this composition should be promoted. A score between the promoted frontier
and the threshold is useful calibration but remains a rejected submission. A
score below the donor means the finish bundle should be decomposed rather than
explained as a win; the first rollback candidates are the constant-argument
and raw-product schedules, which came from the lower-scoring public variant.

The official result must also be read with its verification metrics. A large
gap between self-reported and verified candidates, or recall materially below
the donor, points to arithmetic loss rather than a pure scheduling regression.
The host gate can reject false tentative hits but cannot recover false
negatives. Conversely, normal verified recall with low self-reported
throughput points toward register allocation, occupancy, or instruction
scheduling. No conclusion in this note substitutes for the complete CUDA
12.8.93 build and approximately 1,200-second RTX 4090 run.
