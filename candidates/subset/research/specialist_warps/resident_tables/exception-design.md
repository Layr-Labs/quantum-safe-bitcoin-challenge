# Resident-table orders: complete exceptional-domain argument

**Both proposed all-small orders are safe for the existing incomplete seed/intermediate helpers, provided the final guard is retained.** This covers every raw uint256 scalar and every valid nonzero order-n secp256k1 runtime base, conditional on correct tables and field arithmetic. It is a design proof, not qualification of an implemented new geometry.

| Geometry | Table bytes | Seed | Incomplete intermediate windows | Guarded final | Only top `last=true` |
|---|---:|---|---|---:|---:|
|[18]+[17]*14|64MiB|13+14|0..11,12 additions|12|14|
|[16]*16|32MiB|14+15|0..12,13 additions|13|15|

Thus the exact orders are `[13,14,0,1,2,3,4,5,6,7,8,9,10,11,12]` and `[14,15,0,1,2,3,4,5,6,7,8,9,10,11,12,13]`. Reversing the two seed inputs is also mathematically safe, but the listed order minimizes changes to the earlier ascending-within-seed convention. The deferred-Y seed anchor must remain the actual first seed point's Y. The top-digit flag follows the window index, not its position in execution order: the second seed input gets `last=true`, while the guarded final window does not.

## Source/model binding and notation

Reference source is the qualified ring experiment `32411a6a248c7ab53beba0f025450d95f31a0f2fc3da98a549456a47f93162d7`. `check_exception_design.py` binds all16 files and confirms byte equality of its actual `gt_recode_setup`, direct-bit/direct-digit helpers and `qsb_asym_last_add` with the previously proved thirteen-window source. It reuses the prior Python limb translations with independent bigint checks and new geometry parameters. No geometry source is generated or edited.

Let n be the prime secp256k1 order, U=2^256, and δ=U−n=432420386565659656852420866394968145599. δ has129bits. For any raw hash scalar k in[0,U), actual setup reduces once mod n, doubles with its overflow carry, reduces again and chooses an odd representative M and global sign σ. It gives:

`1 ≤ M ≤ n`, M odd, σ in{−1,+1}, and `σM ≡ 2k (mod n)`.

M=n occurs exactly for raw k=0,n, with σ=−1. Every odd M<n is reachable with both signs.

For window c of width w_c at shift s_c, omit the global sign and define

`d_c = ((((M >> s_c) & (2^(w_c+1)−1)) | 1) − 2^w_c)`

for each nontop digit. The top digit is `(M>>s_top)|1`. All digits are odd and nonzero, |d_c|≤2^w_c−1. Direct digit extraction at position s_c+1 is equivalent to the peeling recoder, including limb shifts and wrap behavior. Signed table terms are σ*d_c*2^s_c times the runtime base.

For a low prefix ending at bit S, telescoping gives

`L(S) = M − ((M>>S)|1)*2^S`, with `|L(S)|≤2^S−1`.

Because the runtime base is nonzero in a prime-order group, equality/opposition of two accumulated points is equivalent to their scalar difference/sum being0 mod n. This converts the curve-domain question into the following exact integer bounds.

## Highest-two seed and all intermediate additions

Let B be the shift of the lower seed window and H the shift of the guarded final window:

- 15 windows: B=222, H=205, final width17.
- 16 windows: B=224, H=208, final width16.

For the two seed terms, either their sum or difference is a multiple of2^B with odd quotient: the lower term's coefficient is odd, and the upper term has at least16 additional factors of2. Therefore neither integer is zero. Their absolute values are at mostU−2^B<n because δ<2^B. This excludes equal/opposite seed points and an infinite seed result. It also remains true if the seed order is swapped or all signs are negated.

The completed seed sum telescopes to

`C=((M>>B)|1)*2^B`, so `2^B≤C≤U−2^B`.

Before an ascending low-window addition c, A=C+L(s_c). For either equal/opposite relation, bound A±d_c*2^s_c. With S=s_c+w_c,

`|L(s_c)|+|d_c*2^s_c| ≤ 2^S−1`.

Hence both integers lie in `[2^B−2^S+1, U−2^B+2^S−1]`. Through every incomplete intermediate addition, S≤H. Both geometries satisfy the exact checked inequality

`δ < 2^B−2^H`.

The lower endpoint is positive and the upper endpoint is strictly below n. Neither sum nor difference can vanish mod n; all intermediate accumulators remain finite and nonzero. Global sign only negates these values. JSON records each window's concrete lower/upper bound and margin below n. The argument covers all M, not merely tested scalars.

## Guarded final window: exact four-input exception set

At the final window, S=B. Both A+Q and A−Q lie in[1,U−1], which is below2n. Thus n is the only possible exceptional multiple (or σn after restoring sign).

Opposition requires A+Q=M=n, giving exactly raw k=0,n. These must take the existing infinity path.

Equality requires `M−2*d*2^H=n`. Since M≤n, d must be negative. Set `M=n+2*d*2^H`. Because δ<2^H, `n>>H=2^(256−H)−1`. For every allowed negative odd d, the low(w+1)-bit substitution has no wrap and the actual digit equation simplifies to

`d = (2^w−1)+2d`, hence uniquely `d=−(2^w−1)`.

Define `D=(2^w−1)*2^H`. The unique odd representative is `M_equal=n−2D`, and the two scalar residues are k=D with σ=−1 and k=n−D with σ=+1. Both are greater than δ, so neither has another raw uint256 alias obtained by adding n. Therefore the complete raw exceptional set is exactly

`{0, n, D, n−D}`.

For15windows, D=(131071)*2^205. For16windows, D=(65535)*2^208. Both equal and opposite witnesses appear in JSON with their recoded M, signs, accumulator and final term. Equality uses a valid nonzero doubling point because n is odd; the guard's separately qualified doubling/infinity paths are the relevant behavior. Removing the guard exposes all four constructed inputs. The model enumerates every negative odd final digit (65,536 or32,768 cases) and independently finds only the derived equal representative.

## Executed checks and limits

Run from the benchmark root:

```sh
python3 -B candidates/subset/research/specialist_warps/resident_tables/check_exception_design.py
```

The15-window model passed22,252 scalar cases and333,780 direct-versus-peeling digit comparisons; the16-window model passed22,284 scalars and356,544 digit comparisons. Inputs include all single-bit boundaries, n/n/2/doubling-overflow neighborhoods, limb/window/digit boundaries, the constructed guard witnesses and20,000 deterministic random uint256 values per geometry. Wrong sign, one-bit-wrong direct extraction and wrong top-last flag are detected in each geometry. Exhaustive final-digit enumeration and the integer bounds, rather than sampling, establish the full-domain conclusion.

This model is not execution of extracted C++ with either new geometry, nor an actual curve/helper test of a new candidate. When a port exists, execute its actual direct digits, signed loads, new point order and final guard with these four witnesses and the existing equal/opposite/ordinary constructed curve cases. Retain independent table-builder and field validation. New small-table geometry also requires a real startup-policy check; old cold-window/48MiB boundary assertions must not survive by accident. Public scan owns builder/cache design details; this proof does not qualify their implementation or performance.

Both geometries are acceptable from the exceptional-domain standpoint. This does not rank64MiB versus32MiB throughput, prove cache residency, or remove the additional point additions associated with more windows. Natural ascending promoted orders are outside this specific ordering proof; do not borrow this result for a different schedule.
