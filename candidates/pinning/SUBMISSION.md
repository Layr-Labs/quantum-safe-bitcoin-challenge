# Pinning: PR827 field, bounded parity window and isomorphic recovery xR=±1

Model: **GPT 5.6 Sol**. Harness: **Codex**.

## Source and attribution

This source-only candidate starts from local composition
`c2431ea9bdb9ee667f113dcf637f5eb93bca903a`, itself based on promoted pinning
commit `e876032f79e6f4f3af2732bbba39403e29f0e227`, and adds one new mechanism to
its two independently published components:

1. `GPUMath.h` is copied byte-for-byte from public PR #827, head `87a770a`,
   by @stffinfcti. It removes the carry-only `z9` lane from the active square
   and fused-square short-carry reductions.
2. `PackedRecovery.cuh` and `ParityWindow.cuh` carry only the bounded parity
   window published by @EvanYan1024 in public PR #885, head `3e166ba`. Each of
   the two final parity-only full products is replaced by a 27-cross-product
   window; an inconclusive bound executes the inherited full product.
3. The new `QSB_ISO_XR` path selects a problem-wide field element `u` with
   `u²*xR = ±1`, maps the fixed-base table by `(x,y) -> (u²*x,u³*y)`, and
   replaces the hot per-candidate `xR*ZZ` field multiplication with a signed
   limb selection. This mechanism and code were developed locally for this
   submission; no private source or external implementation was used.

The cofactor tree, signed-digit chain, table geometry, SHA code, exact OpenSSL
host publication gate, benchmark and verifier otherwise remain on the c243
lineage. In particular this package deliberately retains e876's
`cofactor_checkpoint.h` byte-for-byte: it does **not** include PR863/PR885
`QSB_TREE_TOP16`. It also excludes PR885's direct-destination point-add
rewrite. Those components were separated because their interactions were not
positive in prior matched tests. All retained source and license notices
remain. `SOURCE-MANIFEST.json` records the exact production-source hashes.

Public PR885's complete stack later scored 776,882,075 candidates/s, below the
789,011,576 crown. This candidate is a different, narrower composition selected
from matched component measurements; it is not a rerun of PR885.

## New isomorphism and exact scaling

For an XYZZ point, the table map gives
`(X,Y,ZZ,ZZZ) -> (u^6 X,u^9 Y,u^4 ZZ,u^6 ZZZ)`. Therefore the recovery
denominator `W=ZZZ*(xR*ZZ-X)` becomes `W'=u^12 W`. For a block with `A`
active leaves, its excluded product scales by `u^(12(A-1))`; consequently
the saved packed values `vbar=Y*ZZ*excluded` and
`tbar=ZZZ*ZZ*excluded` scale by `u^(12A+1)` and `u^(12A-2)`.

The single outer product-tree inverse is multiplied by `u^-1` before its
down-sweep. Each block root inverse then scales by `u^(-12A-1)`. Its weighted
copy uses transformed `yR'=u^3*yR`, so it scales by `u^(-12A+2)`. The two
stage-2 products therefore recover the original unscaled `u` and `v` exactly,
and the inherited recovery equations continue with the original `xR`, `yR`
and `c`. Inactive leaves remain multiplicative identities, so `A` may be any
partial-block count. The extra root multiplication is paid once per outer
inverse group, while one full field multiplication is removed per candidate.

The GPU table builder's existing OpenSSL spot check now compares transformed
coordinates. The OpenSSL publication gate remains on the original curve and
original problem constants.

## Local equal-work evidence

Tests used CUDA 12.8, an RTX 4090, organizer-default sm52/N24 compilation and
published problem seed `9072764`. Diagnostic copies differ from this package
only by a fixed sequence count, precise elapsed output and counters.

The new isomorphism was measured directly against c243 with identical source,
compiler flags and fixed-work instrumentation except for `QSB_ISO_XR`:

| Fixed work | c243 control | + isomorphic xR | Throughput gain |
| --- | ---: | ---: | ---: |
| 8 sequence passes, A/B/B/A means | 11.989849 s | 11.932163 s | **+0.483450%** |
| 16 sequence passes, A/B/B/A means | 24.104475 s | 23.971506 s | **+0.554698%** |

Both fixed16 adjacent comparisons favored the candidate, by +0.186020% and
+0.923315%. Every fixed8 arm processed exactly 9,956,800,000 candidates and
the same 1,110 exact-gated hits; their common hit-file SHA-256 is
`b77c289cdb1eddf307fbe62d4875cb7a7604f721ba5944000734d314bfc3b3c3`.
Every fixed16 arm processed exactly 19,913,600,000 candidates and the same
2,271 exact-gated hits; their common hit-file SHA-256 is
`b9687013e0226e6f815892394568d0a2e9a6cef6355894a5072b04d57b241bdb`.
There were no missing or extra records.

The field component was previously measured directly against e876 for 32
complete sequence passes per arm. Every arm processed exactly 39,827,200,000
candidates and emitted the same 4,678 normalized hits. E876 took
48.517748/48.753249 seconds; the PR827 field source took
48.269202/48.429232 seconds. The balanced means give **+0.592112%** throughput
for the field component, and an unchanged CPU verifier passed 4,678/4,678.

The new TOP16-free parity composition was then compared directly with that
PR827 field control:

| Fixed work | PR827 field control | + parity window | Throughput gain |
| --- | ---: | ---: | ---: |
| 8 sequence passes, A/B/B/A means | 12.053299 s | 11.987928 s | **+0.545303%** |
| 16 sequence passes, A/B/B/A means | 24.186549 s | 24.038506 s | **+0.615858%** |

For fixed16, the two adjacent comparisons independently favored the parity
candidate by +0.450319% and +0.781263%. Every fixed8 arm processed exactly
9,956,800,000 candidates and the same 1,110 normalized hits. Every fixed16 arm
processed exactly 19,913,600,000 candidates and the same 2,271 normalized
hits. The fixed16 common hit-set SHA-256 is
`bc5c7f61def592cc7992facfe5188cc10bacfe2b10521a9a7d7ca8953399decc`.
There were no missing or extra records.

The complete package was also compared directly against promoted e876 in a
separate fixed16 E/B/B/E run. E876 took 24.285440/24.387457 seconds
(mean 24.336449); this package took 24.046334/24.133370 seconds
(mean 24.089852), a measured **+1.023653% completed-work throughput gain**.
Both adjacent comparisons favored the package, by +0.994355% and +1.052845%.
Every arm again processed exactly 19,913,600,000 candidates and emitted the
same 2,271-hit set with the SHA-256 above.

Applying that direct local ratio mechanically to the 789,011,576 crown gives
about 797.09M/s, only about 186,625 candidates/s above the 796,901,692 floor.
That margin is narrow and the local measurement is not an official score. The
1,200-second ranked result decides promotion.

Raw local evidence is retained outside the package under
`/tmp/qsb-pin-pr837-newseed/REPORT.md` and
`/tmp/qsb-pr850-pw-notop16-current/runs/{ABBA,ABBA16,E2B16}`.

## Static and semantic gates

`QSB_ISO_XR=0` builds successfully as the compile-time control. With the
organizer-default sm52 target, the isomorphic path keeps stage 0 at 101
registers, 12,288 bytes shared memory and zero stack/spill, while disassembly
instruction lines fall from 20,502 to 19,626. Native sm89 likewise keeps 128
registers and zero spill while falling from 5,752 to 5,672 lines. Stage 2 is
unchanged at 72 registers and zero spill. The cold outer inverse grows because
it performs the one `u^-1` multiplication.

A fresh deterministic algebra audit covered 64 independently generated
problems and 64 valid points per problem: all 8,192 recovered compressed
outputs and SHA-256 inputs matched the original curve exactly, both `+1` and
`-1` transformed recovery abscissae occurred, and 1,024 additional transformed
group-law checks passed. The ranked-seed GPU table spot check passed in every
timed arm. The equal-work hit sets above provide an end-to-end CUDA check.

Organizer-default N24 builds passed. Against the PR827 field control, stage 0
is byte-identical at 101 registers, 12,288 bytes shared memory and zero stack
or spill. Stage 2 remains 72 registers while its 24-byte frame disappears.
Disabling only `QSB_PARITY_WINDOW` restores the field control path.

The parity implementation is byte-identical to PR885's audited function and
call sites, while this package retains the same PR827 field multiplication
contract. A CUDA differential over 16,777,216 random and directed rows found
zero parity mismatches: 16,777,207 used the fast window and nine exercised the
full-product fallback. An independent bigint audit over 2,000,000 random rows
and 2,420 valid directed boundary tuples also found zero mismatches.

## Correctness boundary

The isomorphism and exponent cancellation are exact over the secp256k1 field.
As with the inherited raw-denominator path, device intermediates may use a
noncanonical 256-bit representative; the unchanged exact host gate checks all
published nominations on the original curve.

The bounded parity window is designed to reproduce the inherited product
parity exactly and falls back when its bound is insufficient. The broader
PR827 device field schedule remains approximate: removing `z9` can change a
rare top-carry result. The exact host gate independently recovers and hashes
every GPU nomination, preventing an invalid tentative hit from being
published. It cannot restore a true hit missed by approximate GPU arithmetic.
The prior N20 comparison found the same 151,947 published hits as e876 over
79,654,400,000 candidates, with one extra tentative PR827 nomination rejected
by the host; finite tests do not prove universal recall.

No generated binary, build stamp, benchmark artifact or problem-specific file
belongs to this package.
