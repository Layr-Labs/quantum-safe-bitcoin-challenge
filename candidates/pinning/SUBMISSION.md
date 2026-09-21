# Certified GLV608 decoder and nonexceptional common chain

Effort: medium. This candidate integrates two exact integer cost reductions and a proved common-chain specialization into our GLV608 route on the promoted Pinning frontier. The baseline is promoted submission dcd0147c-8cb3-47f0-8b71-007c87fa7748, commit 66fede0cc10d36ad15041861d3eddaad97481ac6, score 789,011,576. The current shared source tip e876032 has the same Pinning starting bytes. Development used GPT 6 Astra through Codex. No local CUDA/C++ compilation, native host harness or GPU benchmark was performed; the official remote build and validation supply native evidence.

## Motivation and the failed previous experiment

Our previous PR826, a9bde073-1efb-4a7e-a003-bdfad09f52b0, completed with verified=true and score 520,573,216: 34.0221% below the promoted baseline. It verified 74,535 hits in the reported 1,201.07 seconds. That large regression is real evidence against its overall implementation. It does not identify the individual cause. Public metrics do not give register allocation, spills, cache misses or instruction profiles; the diagnostic artifact was not accessible anonymously. No such explanation is asserted here.

PR826 reduced the nominal normal-chain field ledger from 95 multiplications and 28 squares to 88 multiplications and 26 squares, but introduced a costly exact scalar decoder, a generic exception state machine and a larger table. This submission makes concrete cumulative corrections to the first two costs. It is not a byte-equivalent remeasurement, an inert macro change or a claimed recovery of the entire 34% loss. The table cost remains and could still dominate. Remote validation is necessary to decide whether the corrected route is competitive.

## Certified reciprocal head instead of a complete high product

For a normalized scalar 0 <= k < N and either positive GLV rounding coefficient b, precompute g=floor(b*2^384/N). Let H be the scaled high-product lower approximation obtained from the ten complete 32-bit products on diagonals i+j>=11 and the five high halves on diagonal i+j=10. The omitted product tail is strictly less than 12 units at scale 2^352. The downward reciprocal error contributes strictly less than one further unit. Therefore

    H <= k*b*2^32/N < H+13.

If the rounded integers at both interval ends agree, that value is the exact nearest-integer coefficient. Otherwise the complete original qsb_round_glv function is mandatory. The device guard tests whether low32(H+2^31) is at least 0xfffffff3. No ambiguous interval is accepted speculatively. Boundary tests contain 512 cases where ignoring this guard would give a wrong coefficient; they are retained as negative controls. The old full coefficient function and its complete carries are still present in Split608.cuh.

Each fast coefficient uses 15 source 32-bit product expressions instead of 57. These are source-level arithmetic counts, not SASS instruction counts and not a speed prediction. The two constants are recomputed from the curve order in the independent audit. Accumulator bounds keep each partial diagonal within unsigned 64-bit range. The input is normalized once as before; all lattice-correction branches and signed five-word output handling remain unchanged.

## Three-product exact residual identity

For the GLV basis, A2=A1+B1 and B2=A1. Instead of separately constructing four residual products, compute

    t = A1*(c1+c2)
    x = k-t-B1*c2
    y = A2*c1-t.

This is an integer identity before reduction. Its low-160-bit implementation has the same signed output semantics as the previous four-product construction. The sum c1+c2 needs 129 bits; its fifth word is explicitly retained. The former p[5] scratch is reused and y temporarily holds t. The old residual needs 56 source product expressions; the replacement needs 43. Together with the two certified reciprocal heads, the fast split budget changes from 170 to 73 source product expressions. No output range, rounding rule, field carry or correction branch is weakened to obtain this reduction.

## Proved common-chain specialization

The digit schedule remains four radix-512 digits, nine radix-608 digits and a final radius-236 digit: fourteen points. The recoder now accumulates a boolean certificate while writing descriptors. It returns true only when all fourteen ranks are nonzero. There is no second shared-memory scan. If any rank is zero, the original complete generic state machine runs as a cold noinline function. Its body, doubling helper, infinity reset, restart, seed and mixed-add exceptions are source-identical to PR826 apart from returning a value object instead of writing caller output arguments. All four coordinate arrays are copied back on that path.

For nonzero ranks a group-theoretic bound excludes exceptional additions. The scalar eigenvalue obeys lambda^2+lambda+1=0 modulo N. Any relation a+lambda*b=0 modulo N implies a^2-a*b+b^2=0 modulo N. The quadratic form is positive for a nonzero integer pair. When both coordinates have absolute value at most M and 3*M^2<N, such a nonzero relation is impossible.

Exhaustive enumeration of all 631,808 residue pairs establishes digit coordinate bounds floor(2*m/3): 341 for radix 512 and 405 for radix 608, and verifies rank zero exactly means the zero digit. Write W_i for the current radix weight and L_i for the coordinate bound on all prior digits. At every regular step L_i<W_i and 3*(L_i+D_i*W_i)^2<N. A nonzero current digit has at least one coordinate of absolute value >=W_i, so neither the prefix plus nor minus that digit can vanish over the integers. The quadratic bound then excludes infinity, equality and opposition in the exact group.

At the final step W=780210979034070658749485424425566208 and L=520569104627345332444055349088398165. The corrected split coordinate bound is R=183958706508226550111917910514569734124. The final sum is the split pair, and the final difference is twice the prefix minus that pair. Both 3*R^2<N and 3*(R+2*L)^2<N hold; the latter ratio to N is about 0.886717118. Together with L<W and a nonzero final digit this closes the final exceptional cases. The proof is independent of the nonzero instance base.

The common path therefore uses the promoted three-multiply seed and a rolled twelve-add deferred XYZZ loop without per-add state or zero-denominator branches. Its operand order and original field helpers are retained. The final ordinate conversion uses the full-carry subtraction helper. This is an exact-group domain certificate, not a proof that inherited approximate field arithmetic is exact. No valid exact-group exception is deliberately discarded. Approximation effects from the existing promoted field routines still require the unchanged exact publication gate and official validation.

## Retained table and pipeline costs

The exact instance-specific table contains 785,209 records of 96 bytes: canonical x, canonical beta*x and offset y+c. Four chunks contain 43,692 records, nine contain 61,612 and the final chunk contains 55,933. Allocation is 75,380,064 bytes, about 71.888 MiB. Orientation two still reads both x coordinates and forms the full-carry negative sum. Estimated logical traffic remains about 8.9% above the old fifteen-point table. This candidate does not establish L2 residency or remove that risk.

The table is rebuilt for each instance using A=neg_r_inv*G, exact OpenSSL point rows and batched affine conversion. Each row endpoint receives an independent scalar multiplication check. No problem answer, table cache across instances or generated native binary is packaged. GPUMath.h, GPUHash.h, LeafRecovery.cuh, PackedRecovery.cuh, cofactor_checkpoint.h, RecoveryConstant.h and both SHA headers remain byte-identical to the promoted baseline. The exact host publication gate, launch dimensions, recovery tree, saved-state layout, SHA decisions and collective barriers are unchanged. No new arithmetic carry omission is included.

## Public candidate screening

All current in-flight public descriptions were reviewed. 5494489 describes a comment-only retry; 781e335 describes a frontier remeasurement with an unspecified micro mechanism. Neither establishes an increment for this route. ce1682b describes a dual-fetch ordinate-buffer hazard fix, but adds register pressure on an unrelated rotated chain without a demonstrated gain here. b1d48f4 describes address and seed-register microchanges; those are not imported from its losing composite. 32ec88c describes the PR827 z9 carry omission, explicitly with directed wrong residues. It is excluded because this integration does not introduce further approximate arithmetic. The completed 94100d4 composite also does not isolate a beneficial contribution. No unpromoted source or binary from another solver was imported. The promoted frontier retains its existing attribution; the GLV/C6 direction derives from user-provided mathematical research and our own implementation. There is no new substantial unpromoted coauthor contribution.

## Audits and reproducibility

Portable models are under candidates/pinning/research. Run them with Python only:

    python3 -B interval_model.py
    python3 -B residual_model.py
    python3 -B nonzero_proof.py
    python3 -B audit_split.py
    python3 -B audit_recode.py
    python3 -B audit_chain.py
    python3 -B audit_new_sub.py

The reciprocal audit passed 225,156 coefficient checks including 1,028 exercised fallback calls and 11,289 full corrected splits. The residual audit passed 100,049 fixed-word cases; 7,336 needed the retained fifth sum word. The nonzero proof exhausts 631,808 residue pairs and checks 32,000 sampled complete scalar sequences, with two zero-digit samples correctly requiring fallback. Those samples are not a universal frequency estimate. The existing exact chain audit covers all table ranks, 5,232 host rows, 80 full scalar-instance combinations and 142 exceptional sequences. The added common-chain model compares full fourteen-point results against an independent Jacobian EC oracle, including zero-digit fallback inputs.

Static source reversal confirms unchanged correction branches, descriptor operations and loader arithmetic. The cold chain and all arithmetic helpers match the previous validated source. Restoring only documented integration hunks reconstructs promoted pinning.cu exactly. New full-carry subtraction PTX was interpreted with explicit carry semantics on 10,025 boundary/random pairs. Include closure, changed-header bracket structure, shared digit layout and preserved barriers were checked. Previous submitted source hashes remain frozen. These checks are necessary preflight evidence, not a native compiler result or throughput measurement.

The official remote result remains the decision point. A verified but slow outcome would reject this corrected combination as a performance route; it would not justify another unchanged retry. A useful next analysis would distinguish decoder/control overhead from the unchanged table and orientation costs using actual runner diagnostics when accessible.
