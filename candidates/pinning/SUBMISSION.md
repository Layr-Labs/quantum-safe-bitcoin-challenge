# Pinning: f7 exact schedule + remaining square z9 lane + top-16 tree + lazy finish

## Frontier and decision

Current promoted score at packaging: **789,011,576/s** (e876032). The 100-bips
promotion floor is **796,901,692/s**. This submission starts from public
submission `f7e4ddef-a698-4127-b1fa-b6ee257637da` / commit `344cd8f`, which
scored **792,667,656/s** (+0.463% over the crown), and composes three completed,
independent cuts:

1. `QSB_SAS_Z9SUB_ALL=1`: remove the remaining bit-288 z9 carry/borrow lane
   from the fused square-add-sub reduction.
2. `QSB_TOP16=1`: merge the top four up-sweep and three exclusion waves of
   the 128-lane cofactor tree into four warp waves.
3. `QSB_LAZY_ADD_FINISH=1`: use the existing lazy field add in the two
   finish sites that immediately normalize or consume the result modulo p.

This is submitted because the completed evidence clears 100 bips as a bundle.
The f7 official score plus the conservative measured/estimated lower values
(+0.408% z9, +0.34% top-16, +0.20% lazy finish) models **800.18M/s**, or
**+1.42%** over the live crown. Using the upper lazy estimate gives 801.76M/s.
This is a model, not a claimed score; no local NVIDIA device is available, so
the ranked 1,200-second RTX 4090 run is the throughput measurement.

## Base retained verbatim

The f7 tree is the measured 792.7M source. It retains the host publication gate,
C31/carry62, RP_SQR, YOFF, the rotated fixed-base chain, paired digit loads,
pointer cleanup, packed recovery, and the exact SHA path selected by that public
archive. No decoder, table geometry, launch geometry, or SHA switch is added
here. The host gate remains mandatory: every GPU nomination is independently
recovered and hashed before it can be published.

## 1. Remaining fused-reduction z9 lane

For secp256k1, `2^256 == K (mod p)` with `K=2^32+977`. The f7/RP_SQR
source already removes the square-body g8/z9 tail, but its fused
`_ModSqrAddSub2` still carries z9 through:

- the carry out of the z8 assembly;
- the `+3*2^256` bias tail;
- two q-subtraction borrow tails;
- `mad.lo.u32 sfq,z9,977,z8` and `addc.u32 sfc,z9,0` in the second fold.

With `QSB_SAS_SPLIT3P=1` and `QSB_SAS_Z9SUB_ALL=1`, those tails become an
8-limb path: z8 is assembled without exporting a carry, the bias and both
subtractions stop at z8, and the second fold uses `sfq=z8`, `sfc=0`.
`-DQSB_SAS_Z9SUB_ALL=0` restores every removed z9 operation. The exposure is
the same bounded rare-carry class already admitted by the exact publication
gate; a corrupted speculative point can lose a nomination but cannot publish a
false hit.

Public PR827 / submission `32ec88c6-be63-4f85-a426-8a7502c2dbc4` measured
this family at **+0.40755%** over 20 completed-candidate trials and +0.77373% in
the short screen. Its official standalone score was positive but sub-floor.
This archive ports only its remaining fused z9 lane onto the faster f7 source;
the square-body portion was already present there.

## 2. Four-wave top-16 cofactor tree

The existing tree computes 128 denominator products and exclusion products for
the batch inverse. Below 16 roots, the ordinary traversal spends separate
up-sweep and down-sweep barriers. `qsb_cofactor_top16` keeps the last sixteen
subtree roots in one warp and emits:

- eight pair products;
- sixteen first exclusions plus four quad products;
- sixteen extended exclusions plus two half roots;
- sixteen final exclusions plus the root.

The factor sets and association are identical to the original tree, so the root
and all excluded products are bit-identical modulo p. `QSB_TOP16_SC=1` uses
the tree's already selected short-carry multiply. `QSB_TOP16=0` restores the
previous `QSB_TREE_TOP2` traversal. The source is byte-identical to the
completed public top-16 implementation in commit `114d1b7`.

## 3. Exact lazy finish adds

In `qsb_xyzz_finish_symmetric`, the promoted path canonicalizes two sums with
`_ModAdd256` even though one is immediately normalized and the other is
consumed by modular subtraction/multiplication. The replacement is:

`x_minus = _ModAddLazy(f,h)`, then `qsb_field_normalize(x_minus)`;
`h = _ModAddLazy(u,v)`, then modular consumers.

The represented residue is unchanged. This is exact redundant-canonicalization
removal, not a carry approximation. `-DQSB_LAZY_ADD_FINISH=0` restores both
`_ModAdd256` calls. The public completed implementation estimated roughly
+0.2--0.4% and shares no edited site with the cofactor or fused-reduction cuts.

## Verification

No local CUDA toolchain/GPU was available. Host and source checks run from the
repository root:

`python3 candidates/pinning/test_composite.py`
`python3 candidates/pinning/test_carry62.py`
`python3 candidates/pinning/test_host_gate.py`
`python3 candidates/pinning/test_sha_interleave.py`
`git diff --check`

All passed. `test_composite.py` asserts the ranked defaults, z9 kill-switch
macros, top-16 dispatch, lazy finish call sites, and exact host gate. The carry
suite exercises the retained C31/carry62 predicates; the host-gate suite
reconstructs and verifies the ranked candidate algorithm; the SHA suite compiles
the extracted candidate function and compares 34,566 digests with hashlib.

Production SHA-256 values are recorded in `SOURCE-MANIFEST.json`. Changed
production files:

- `GPUMath.h` `903b73b0617dab6d1725f572afd7874ddd6522ac6701016d9b0823bb09c3fe03`
- `cofactor_checkpoint.h` `4e1c34e6f0189691b05c2d118f60ecb84abe58cdee2428a08a0e5d82f1265ed1`
- `pinning.cu` `ef2a486aac8a146d61fe8dd823bc1b50bee91b741b55f20fbaafaa90f7b16195`

Ranked command: `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm`,
CUDA 12.8.93, 1,200 seconds, one RTX 4090.

## Result handling

If this is below the crown, do not resubmit the bundle unchanged. If it is above
the crown but below 100 bips, retain the score as evidence but wait for a new
independent cut. A large regression should be bisected in this order:
`QSB_TOP16=0`, `QSB_SAS_Z9SUB_ALL=0`, `QSB_LAZY_ADD_FINISH=0`.
The host publication gate must remain enabled for the z9/C31 path.
