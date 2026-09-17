# HM43 bounded-range and cap-plan review

This is an independent mathematical/source review of the preserved unbounded6cb9b006 prototype (`hm43_warp_inverse.cuh` SHA256940f4c6ad4272a71d38533123a809c8f1affc7a9e0beca959db48380f24cd1fc) and the proposed16-batch capped variant. It does not qualify an unbounded implementation, prove an eight/sixteen-batch convergence theorem, or measure performance. No candidate, stage or VM changes were made.

The proposed production contract is canonical nonzero input x∈[1,p−1], p=2^256−C, C=2^32+977. Initial U=p,V=x,R=0,S=1, with U≡Rx and V≡Sx modulo p. U is odd. Tests of zero/noncanonical roots are useful extensions but are not silently included in that tree-input contract.

## Matrix coefficient bound

Let t be the number of trailing-zero bits consumed in a batch. Immediately after stripping zeros, both matrix rows have L1 norm at most2^t. Initially both norms are1. Stripping z bits doubles the U row z times and leaves the V row unchanged; swapping preserves the bound. Subtraction can make the V-row norm at most2^(t+1). Since U and the then-stripped V are odd, their difference is even; unless the batch has already ended, the next trailing-zero count is at least1. That next strip absorbs the temporary factor2: both row norms are again at most2^(t+z).

The last strip reaches t=62 and exits before a subtraction. Thus both **terminal** row norms are at most2^62. A subtraction occurs only at t≤61, so its transient coefficients also have magnitude at most2^62. All signed coefficient arithmetic and coefficient negations fit int64; no coefficient can reach INT64_MIN. The initial strip may be zero, but the initial rows already satisfy the bound. Treating the transient post-subtraction row as bounded by2^t would be an incorrect proof step.

Low-word simulation is sufficient for these parity/divisibility statements: after t discarded bits, its low64−t bits agree with the corresponding exact current integer. The62-bit sentinel never requests more relevant low bits than are available. Approximate high-word comparisons can choose a different swap than exact comparison; this does not affect the L1 bound, low-bit divisibility, or gcd preservation. It **does** prevent importing an exact-comparison binary-GCD iteration bound without a separate proof.

## Integer ranges and congruences

For a terminal row(a,b), its U/V numerator has magnitude at most2^62·max(U,V). It is divisible by2^62; applying the row's sign correction and dividing therefore gives nonnegative U′,V′≤max(U,V)≤p. Inductively this remains true for every completed batch. Predivision magnitudes are at most2^62p<2^318, safely inside signed320-bit range. Reading the sign from word4 (bit319) is valid. Swaps, sign changes, subtraction and dividing an even value while the other is odd preserve gcd; U remains odd and nonzero. The high-word search therefore never sees both U,V zero for the production contract.

Let M_t=max(|R_t|,|S_t|). Initially M_0=1. Apply the same sign-adjusted matrix row as U or V. Its coefficient numerator T satisfies |T|≤2^62M_t. Define m=(T·MM64) mod2^62. The actual constant obeys C·MM64≡1 mod2^62, hence T+mp≡0 mod2^62. With0≤m≤2^62−1,

|R_(t+1)|,|S_(t+1)| ≤ M_t+(1−2^−62)p < M_t+p.

Thus M_t≤1+t·p. For at most16 completed batches, M_t≤1+16p<2^260 (the proposed<2^261 bound is conservative). Before the sixteenth division, every coefficient sum plus correction is bounded in magnitude by

2^62(1+15p)+(2^62−1)p <2^62(1+16p)<2^322.

This is comfortably within signed384 bits. The proposed<2^324 bound is also safe. Individual row products and their sum fit the same six-word arithmetic. The updated quotient fits signed320 bits, so retaining five words and reconstructing word5 by sign extension is exact. Sparse m·p construction has m<2^62 and fits below2^318. Guard words prevent carry leakage between the four eight-lane groups; discarding bits above the six live words implements exact signed arithmetic under these bounds.

The correction adds only a multiple of p. Because2^62 is invertible modulo odd p, exact division preserves U′≡R′x,V′≡S′x. If a batch terminates with V=0, preserved gcd gives U=1 for canonical nonzero x. Then R is an inverse; the existing finite additions/subtractions of p canonicalize it. The accumulator bound makes these normalization loops finite (each direction needs at most17 iterations under the conservative16-batch bound). This is correctness **conditional on termination before the cap**, not a convergence theorem.

## Cap and fallback obligations

A16-batch cap is sufficient to close the new fixed-width arithmetic argument without proving that every input converges within16 batches. It is a conservative engineering limit, not a claim that16 is necessary or optimal. The inspected implementation and finite validation do not supply an unconditional eight-batch or sixteen-batch theorem for this approximate-high matrix algorithm. A standard exact binary-GCD step bound cannot simply be transferred. An unbounded variant needs a global accumulator/convergence argument that is not established here.

The proposed fallback is sound if all of the following are enforced and source-tested:

1. Count complete62-bit matrix batches uniformly. Execute at most16. On successful V=0 completion, finalize the inverse; otherwise return false uniformly before starting any seventeenth-batch shuffle/ballot. All32 lanes, including six-word guard lanes, take the same return. A forced cap of0 or1 is a useful test configuration.
2. The helper may change private state but must not publish a partial root into shared tree[1]. Keep the original canonical tree root intact until success or completed scalar fallback. All other EC warps may wait at the existing root-publication barrier; they must not read or overwrite the root meanwhile.
3. On false, only ecid0 reloads tree[0..3][1], explicitly sets root[4]=0, runs the unchanged `_ModInv(root)`, and publishes the resulting four words. Do not call `_ModInv` on the partial cooperative state. The first EC warp must reconverge before the existing full-warp/subgroup synchronization.
4. Keep the existing five EC subgroup joins and192-owner participation. No early return from the surrounding inverse-tree helper and no leader-only execution of HM43 full-mask collectives. The fallback scalar helper contains no new warp collective. Arithmetic products, identity padding and descent are unchanged.

This restores the baseline inverse on cap without adding a new total-termination assumption beyond the existing `_ModInv` path. The cap branch and root reload require their own actual-source positive/negative tests; tests of the preserved unbounded6cb source do not qualify the proposed source automatically. No GPU timing, race-freedom or native resource claim follows from this proof review. Ready3ac remains unchanged.
