# Thirteen-window exceptional domain

The proposed order has **no seed or intermediate equal/opposite exception** for any uint256 scalar, assuming a valid nonzero order-n runtime table base and correct table/field primitives. The final window8 guard is necessary and sufficient for the point-add exceptional domain. This is an integer-domain result; the parent separately checks actual point helpers, recovery, table construction and CUDA compilation.

The reviewed closure is `thirteen_candidate`, fingerprint `5d17e5fad6cb9d632d7b8b4977751da574242488d428cfedfb455accaa21ca5b`. `exception_domain.py` verifies all16 closure-file hashes against `thirteen-prepared.json`, and records exact helper hashes in `exception_domain.json`. No candidate was changed. This generalizes the earlier `asymmetric_windows/cold_seed/check_exceptions.py` argument to four cold windows and nine hot windows.

## Geometry and recoding

Widths are `[18,18,18,17,17,17,17,17,17,25,25,25,25]`; shifts are `[0,18,36,54,71,88,105,122,139,156,181,206,231]`. Order is `[9,10,11,12,0,1,2,3,4,5,6,7,8]`. The actual source seeds9+10, uses its ten-add loop for11,12,0..7, then guards8. Only window12 uses the positive top-digit rule.

Let n be the secp256k1 group order, U=2^256, B=156, H=139, and δ=U−n. Exactly δ=432420386565659656852420866394968145599; its bit length is **129**, not128. The necessary inequalities are δ<2^B and δ<2^B−2^H, both with large margins.

The actual `gt_recode_setup` reduces a uint256 k once modulo n, doubles with its high carry, reduces again, then makes the representative odd. Thus it produces 1≤M≤n, M odd, sign σ∈{−1,+1}, and σM≡2k (mod n). M=n occurs exactly at k≡0, with σ=−1. Every odd M<n is reachable with both signs. This handles raw hashes above n, parity and doubling overflow.

Write t_c=d_c·2^s_c without the global sign. For a nonfinal digit,

`d_c = (((M >> s_c) & (2^(w_c+1)−1)) | 1) − 2^w_c`.

The top digit is `(M>>231)|1`. Each digit is odd, nonzero, and has |d_c|≤2^w_c−1. Repeated peeling leaves `(M>>S)|1` after a prefix ending at bit S; hence the low prefix is `L(S)=M−((M>>S)|1)·2^S`, with |L(S)|≤2^S−1. The entire sum is M. Direct indexing reads at `s_c+1`, maps to magnitude `2·idx+1`, and XORs the local sign with the global sign; it is equivalent to this peeling identity. Top `last=true` is essential at12, including its zero bit256.

## Four cold windows

For a partial cold prefix plus or minus its next term, every term is a multiple of2^B. Its quotient is odd: the lowest cold coefficient d9 is odd, and all later terms are divisible by2^25. Therefore the integer cannot be zero. For m cold terms, including the seed m=2 and subsequent m=3,4, its absolute value is at most

`(2^25−1)·2^B·(1+2^25+...+2^(25(m−1))) = 2^(B+25m)−2^B`.

For m≤4 this is ≤U−2^B<n. Consequently neither the sum nor difference is zero modulo n. Both seed points and every intermediate cold accumulator are nonzero; the seed cannot be equal or opposite. The same holds for adding cold11 and12. This proof permits either global sign.

The completed cold sum telescopes to `C=((M>>B)|1)·2^B`, so `2^B≤C≤U−2^B`. Positivity is needed only after all four cold terms have been accumulated; individual cold prefixes can be negative.

## Hot windows0 through7

Before adding hot c, the accumulator without its global sign is A=C+L(s_c). For either potential exceptional relation, consider A±t_c. The triangle bound gives

`|L(s_c)|+|t_c| ≤ (2^s_c−1)+(2^w_c−1)·2^s_c = 2^(s_c+w_c)−1`.

For c=0 the preceding L is exactly zero and the same bound holds. If S=s_c+w_c, both A+t_c and A−t_c lie between `2^B−2^S+1` and `U−2^B+2^S−1`. Through c7, S≤H=139. The lower bound is positive and the upper bound is below n because δ<2^B−2^H. Thus neither equal nor opposite points can occur. The JSON records each concrete bound. Negating every term only negates these integers and leaves the nonexception result unchanged.

## Final window8: precise exceptions

At the final addition, S=B. Both A+t8 and A−t8 lie in `[1,U−1]`, which is below2n. Thus the only exceptional multiple is n; after global sign it is σn. No extra zero or ±2n case was omitted.

Opposite points require A+t8=M=n. These are exactly raw uint256 scalars k=0 and k=n. The actual setup produces σ=−1 and the final sum is −n. This path must return infinity, which the existing final guard represents with zero coordinates/scales and the separately checked recovery handles.

Equal points require `M−2d8·2^H=n`. Necessarily d8<0. Since δ<2^H, `n>>H=2^117−1`. Substituting `M=n+2d8·2^H` into the digit equation gives `d8=2^17−1+2d8`, with no wrap in its18-bit extraction for all allowed negative odd d8. Therefore uniquely:

`d8=−131071`, and `M_equal=n−(2^18−2)·2^139`.

This M is odd and in(0,n). The source-matched recoder confirms it. The checker additionally enumerates all65,536 possible negative odd final digits and finds this one representative. Define D=(2^17−1)·2^139. Its two scalar residues are k=D (σ=−1) and k=n−D (σ=+1); neither admits an additional uint256 alias k+n. Thus the complete exceptional scalar set has four inputs: `{0,n,D,n−D}`. Equal and opposite cannot coincide because the nonzero table point has odd prime order; doubling it is valid and nonzero. Removing the final guard exposes both equal witnesses to the incomplete addition.

Exact scalar hex values, signed accumulator/last-term values, and ±n identities are in the JSON.

## Checks and scope

Run `python3 -B candidates/subset/research/geometry_frontier/exception_domain.py`. It passed22,853 scalar cases,297,089 direct-versus-peeling digit comparisons,137,118 cold sum/difference tests,365,648 hot sum/difference tests, both equal signs and both opposite raw scalars. Inputs include deterministic random uint256 values, every single-bit boundary, n and n/2 neighborhoods, limb/window edges, digit extrema, and constructed exceptions. Wrong global sign, one-bit-wrong direct extraction, and a wrong top-last flag are detected by negative controls. The entire negative final-digit domain is separately enumerated.

The tested Python models explicitly retain the source's uint64 wrapping and128-bit borrow semantics. They are hash-bound translations, **not execution of extracted C++**. Full-domain correctness here follows the inequalities and recoder identities, not the finite sample count. There was no compilation, CUDA/GPU execution, or candidate modification.

Storage is exactly48MiB hot plus4GiB cold: `(3·2^17+6·2^16)·64` and `4·2^24·64` bytes. One7M2S intermediate addition is removed relative to the14-window geometry. Logical point requests change from12 hot+2 cold to9 hot+4 cold, so total64-byte points fall from14 to13 while cold requests double. These counts do not establish a throughput gain; they identify the intended tradeoff.
