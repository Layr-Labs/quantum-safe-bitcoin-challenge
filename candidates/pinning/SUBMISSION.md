# GLV608: fourteen-point dual-x C6 chain on the promoted Pinning frontier

Effort: medium. This is a substantial algorithmic experiment based on promoted Pinning submission dcd0147c-8cb3-47f0-8b71-007c87fa7748, promoted commit 66fede0cc10d36ad15041861d3eddaad97481ac6, score789011576. The later shared source tip e876032 has the same starting Pinning tree. No local score or performance improvement is claimed. The official remote native build, correctness validation and RTX4090 run are the measurement.

## Problem and proposed improvement

The promoted fixed-base multiplication reads fifteen signed affine points from a64MiB table and uses a deferred-ordinate XYZZ chain. Its normal-path field ledger is95 multiplications and28 squares. This candidate replaces that scalar representation and table with a corrected GLV split, C6 orbit tables and fourteen affine points. The new normal-path field ledger is88 multiplications and26 squares. It keeps the promoted recovery, product/cofactor traversal, SHA code, host exact publication gate and launch geometry.

A previous GLV14 idea paid for beta orientation multiplications at query time. Sorting orientation tags looked inexpensive per scalar, but independent lanes placed their transition boundaries at different iterations: the modeled warp saw about13 beta-helper issue occasions despite fewer than3 per lane. This version addresses that concrete failure mode by storing two x coordinates per table entry. It does not sort the points, rotate the accumulated frame, or execute a modular beta multiplication in the query.

This is not a claim that fewer field operations necessarily improve throughput. The exact integer decoder, additional table traffic, data-dependent orientation masks and point exceptions all cost work; only the ranked run can resolve the net effect.

## Source and prior-result selection

The source is an independent snapshot of the promoted frontier. Eight existing runtime headers, including the field header, were checked byte-for-byte against that snapshot. Our failed register-top16 and grouped/interleaved-tree submissions are not stacked here. In particular, our previous PR790 scored757004542, about4.06% below this promoted frontier; its physical256/logical128 layout and retained tree operands are absent.

Public in-flight descriptions were reviewed before integration. Rotated chains, multi-pair unrolling, quad-packed digit transport and constant-memory argument claims were not copied. New public matched-work descriptions reported adverse results for several of those changes, and large composite regressions did not establish which component caused them. Congruence-only proofs for recovery sum substitutions or normalization removal do not establish representative invariance for the actual promoted truncated field multiplier. New arithmetic carry omissions were excluded. No unpromoted candidate source or binary was imported for this implementation.

The GLV lattice/C6 route follows the mathematical direction in user-provided research material. The integration, exact pseudo-order quotient construction, fixed-word implementation, dual-x loader, exceptional chain and host row builder are developed here. The promoted code's existing provenance and license notices remain in place. There is no additional unpromoted solver source contribution requiring coauthor attribution in this candidate.

## Corrected GLV split without generic wide division

The curve order is N=2^256-D, where D has129bits. Let b be one of the two GLV rounding constants and let normalized k satisfy0<=k<N. For T=k*b=q0*2^256+lo, compute

    r = lo + q0*D + floor(N/2)
    h = floor(r / 2^256)
    r2 = low256(r) + h*D
    rounded_coefficient = q0 + h + [r2 >= N]

For both constants, the checked numeric bounds give h<=2 and r2<2N, so one complete comparison finishes the exact rounded division. All limb carries are retained. The implementation uses fixed u32 products and sums, not a GPU integer divide or an unchecked reciprocal approximation. The two exact coefficients form the GLV coordinates via low160 products, followed by the lattice correction against the exact R and T bounds.

A real width bug was caught during integration: the corrected Linf bound R=0x8a65287bd47179fb2be08846cea267ec exceeds the signed128 maximum. Some valid input scalars therefore need129 signed bits. The device implementation now uses five u32 words for the split and first radix512 quotient, then four words after the first quotient has reduced the magnitude. The earlier signed128 plan was never submitted. An explicit failing input for that old plan is included in the audit.

The source has170 u32 multiply expressions for the split before constant-zero folding, not a measured SASS count. Two nine-word temporary arrays were removed by reusing the product's low words after copying its high quotient and adding h*D directly into the remainder with a complete carry loop. This is a source-lifetime reduction, not a claim about physical register allocation.

## Mixed radices and exact descriptor extraction

The schedule is four radix512 digits, nine radix608 digits, and one final bounded pair. A C6 orbit representative is selected with five lexicographic comparisons and a closed rank formula. Each code contains16 rank bits,2 orientation bits and1 sign bit. Fourteen u32 planes use7168bytes for128 lanes, fitting the existing12KiB digit/cofactor arena. Each lane writes and reads its own plane elements; the existing full-block barrier before the cofactor tree reuses that arena remains unchanged.

For608=32*19, the decoder first computes the signed floor quotient by32, then performs exact magnitude division by19 with signed floor correction. Each radix phase uses a uniform limb width:3,3,3,2,2,2,1,1,1. Across both GLV coordinates this is36 base2^32 division digits. The per-word divisor uses the bounded identity2^32=19*226050910+6 and an exact small correction, with no discarded carry.

The final coordinates lie in[-236,236]. Final integer canonicalization maps the original pair to the canonical pair; decoding must use the inverse orientation. The code explicitly stores j=(-j)mod3 for this last digit. Exhaustive final-pair checks exercise this distinction; using the forward orientation fails the negative control.

## Dual-x table, runtime base and memory cost

The table has four43692-entry chunks, nine61612-entry chunks, and one55933-entry final chunk:785209 records. A record contains canonical x, canonical beta*x and offset ordinate y+c, each32bytes, with c=(2^32+976)/2. Total allocation is75380064bytes, or71.888031MiB.

Orientation0 selects x; orientation1 selects beta*x; orientation2 reads both and uses beta^2*x=-(x+beta*x)mod p. That last operation has complete borrow and carry propagation and canonical zero normalization. Signed offset ordinates use the same XOR representation as the promoted code. Query orientation therefore adds integer/address/field-add work and memory, but no modular beta multiplication.

The old regular-odd recoder reconstructed2k using table base A/2. The new recoder reconstructs k, so the new table uses A=neg_r_inv*G, multiplied by each radix scale. It is rebuilt for every problem instance; there is no cross-instance cache, fixed benchmark answer or replay. Construction uses OpenSSL point rows and batched affine conversion. Each row's last point is independently checked by a scalar multiplication from G including the instance scalar and radix scale. Any build/check/allocation/upload failure terminates rather than using an unchecked table. Record0 in each chunk is infinity and is never read as affine coordinates.

This larger table does not prove L2 residency. The estimated logical table traffic is about1043bytes per candidate versus960 for fifteen ordinary64byte records, with a modeled warp-sector increase around8.9%. The advisory L2 window now starts at the beginning: the first four new chunks are denser, unlike the old wide first chunk. No hard residency or cache-hit-rate assumption enters correctness.

## Point chain and exceptional behavior

The chain runs a rolled14-slot loop, preserving scalar order. State0 means infinity, state1 holds one affine point, and state2 holds the deferred XYZZ representation. Infinity table entries are skipped as group identities. The first two available points use the same three-multiply/two-square seed algebra. The normal subsequent addition preserves the promoted operand order and seven-multiply/two-square formula; the ordinate is resolved once at the end.

Equal-x cases are handled before the seed or mixed-add denominator is used. Equal points take an explicit XYZZ doubling path; opposite points reset the state to infinity. A later nonzero point can restart accumulation. Doubling resolves the deferred ordinate and returns a zero affine anchor, so the next ordinary addition remains well-defined. A final single point and an all-infinity chain have explicit outputs. The inherited recovery pipeline's singular-lane policy remains unchanged; this submission does not replace the verifier or claim that the inherited approximate field arithmetic has become exact.

New subtraction/offset-conversion helpers retain all carries and normalize their operands. Multiplication and squaring still use the promoted field primitives, including their inherited approximations and exact host publication gate. There are no new dropped field carry links or disabled publication checks. The exact-field group audit checks the point identities; it is not a universal equivalence proof between approximate device arithmetic and exact group arithmetic.

## Files and scope

New headers are GLV608Decode.cuh, Split608.cuh, Recode608.cuh, Div19.cuh, C6Digit.cuh, DualXExact.cuh, DualXLoad.cuh, GLV608Chain.cuh and GLV608Table.h. pinning.cu routes the existing scalar entry to this chain, allocates/uploads the new exact host table, and adjusts the advisory L2 window. The old inactive table helpers remain compiled but are not called by main. SOURCE-MANIFEST.json records the new production bytes and hashes.

GPUMath.h, GPUHash.h, LeafRecovery.cuh, PackedRecovery.cuh, cofactor_checkpoint.h, RecoveryConstant.h and both SHA headers are byte-identical to the promoted starting tree. A source reversal audit restored pinning.cu byte-for-byte after undoing only the documented integration hunks. Existing host gate writers, SHA decisions, saved-state layouts, barriers, launch bounds and benchmark controls remain unchanged.

## Verification and reproducibility

Development used static source inspection and Python integer/group models only. No local C++/CUDA compilation, native host harness or GPU benchmark was performed. Portable audit scripts and their exact reference live under candidates/pinning/research. From that directory:

    python3 -B audit_split.py
    python3 -B audit_chain.py
    python3 -B audit_new_sub.py

The split audit checks21033 scalar inputs plus2033 end-to-end fixed-word split/recode cases,46132 coefficient comparisons,258 rounding-boundary inputs, and all five lattice-correction branches. The embedded recoder audit checks157209 quotient/descriptor rounds,223729 final coordinate pairs and all55933 final ranks. It catches1166 signed128 failures in the deliberately narrow negative control and167796 wrong-inverse final orientations.

The independent point oracle checks all785209 table ranks,5232 host rows for coverage,260 sampled row-ladder points,80 full scalar multiplications over multiple instance bases,1120 oriented table points, and142 explicit/random exceptional sequences. Those sequences exercise identity skipping, normal seeds/additions, seed and mixed-add opposites, doubling, and restarting after cancellation. The actual new subtraction PTX string is executed by a Python carry-semantics interpreter on10025 boundary/random pairs. This interpreter is not a CUDA compiler or GPU timing tool.

An initial static packaging check falsely matched the substring printf inside a permitted host error fprintf; the audit was corrected to test the intended prohibited device/output operations. The production behavior did not change for that audit fix. The more consequential mathematical failures, signed129 width and final inverse orientation, were fixed before this candidate was packaged and are retained as negative controls.

## Measurement limits and next step

The expected opportunity is removing a whole affine-addition step while eliminating query beta multiplications, not collecting small scheduling tweaks. Important risks are integer decoder cost, forced-inlining/code size, register spills, rare-path call overhead, warp divergence, table bandwidth/cache pressure and exact host table startup. Source operation counts do not determine those costs. The model's88M26S normal path excludes decoding, table construction, orientation subtraction and exceptions. No predicted score is supplied.

The official run must first establish that this uncompiled source builds and validates, then measure verified candidate throughput. Keep only one own submission in flight and freeze these bytes after upload. If the result regresses, retain that evidence and do not blindly combine this algorithm with the already-failed tree or register packages.
