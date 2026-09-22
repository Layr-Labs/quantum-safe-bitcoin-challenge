# An exact certificate for raw sum doubling in the promoted C31 finish

This is a new guarded identity for the actual promoted add/sub functions. It is
not the previously rejected unconditional `l+m=2u` substitution. It neither
restores nor removes any field carry. On a failed certificate, the prototype
replays the original complete finish.

Let `B=2^256`, `L=2^64`, `T=2^32`, `K=T+977` and `p=B-K`. Inputs `u,v` are
arbitrary raw integers in `[0,B)`. Define `S(u,v)` as promoted `_ModSub256`, and
`A(u,v)` as promoted `_ModAddLazy`, with `QSB_C31 && QSB_SHORT_CARRY` active.
Both functions first perform a full 256-bit add/sub, then apply the selected
correction only to the low 64-bit word. Their actual definitions were checked
in promoted `GPUMath.h`, at the active functions starting around lines 467/503.

The old sum is `A(S(u,v),A(u,v))`. The proposed precomputed sum is `d=A(u,u)`.

## Sufficient full-width bounds

Write:

```
x = (u-v) mod L
y = (u+v) mod L
q = 2u mod L
```

It is sufficient that

```
x >= K
y < L-K
K <= q < L-2K
K < d < p
```

The first bound ensures that subtracting K in the low word of `S` cannot lose a
borrow, whether or not the full 256-bit subtraction borrowed. The second ensures
that adding K in the low word of `A(u,v)` cannot lose a carry, whether or not the
full 256-bit addition carried. These two operations are therefore congruent to
`u-v` and `u+v` modulo p on the certified inputs.

Let the corresponding global borrow/carry be `b,c` in `{0,1}`. After those two
corrections, the low word of their sum is

```
(q + (c-b)*K) mod L.
```

The q interval makes that expression an ordinary integer in `[0,L-K)`, so the
final low-word addition of either zero or K cannot overflow. Thus the old sum
is also congruent to `2u` modulo p, without relying on a discarded low-word
carry. The same q bound makes `d=A(u,u)` congruent to `2u` without a discarded
low-word carry.

Both results are in `[0,B)`. Since `K<d<p`, there is only one representative of
the residue d in that range: `d-p<0` and `d+p>B`. Consequently the two raw
256-bit outputs are **identical**, not merely congruent.

## A conservative certificate using only upper 32-bit pieces

Let `uh=(u>>32) mod T`, `vh=(v>>32) mod T`, and use wrapping u32 arithmetic for

```
dh = uh-vh
sh = uh+vh
qh = uh<<1
top = d>>224
```

The implementation accepts exactly when

```
dh >= 3
sh < T-3
2 <= qh < T-5
1 <= top < T-1
```

The last two interval tests are written as wrapping subtract-and-compare in
`GuardedSum2u.cuh`, giving four comparisons in total.

- The upper half of x is `dh-e`, where e is the low-half subtraction borrow in
  `{0,1}`. Since `dh>=3`, no wrapping ambiguity remains and `x>=2T>K`.
- The upper half of y is `sh+f`, for low-half carry f in `{0,1}`. The sh bound
  prevents wrapping and gives `y<=L-2T-1<L-K`.
- The upper half of q is `qh+g`, where g is the most significant bit of the low
  32 bits of u. Therefore `q>=2T>K` and `q<=L-4T-1<L-2K`.
- The top-word test gives `d>=2^224>K` and `d<=B-2^224-1<p`.

The conservative certificate therefore implies every sufficient bound above.
It may reject inputs where equality nevertheless holds. Those inputs take the
original full replay; rejection is not an approximation or a missing-hit policy.
This reasoning applies to arbitrary raw input values, even if an earlier
approximate field multiply produced them. It requires only the selected add/sub
contract and does not assume that earlier multiplications are exact.

## Source integration and verification limits

The guarded adjacent-lane prototype forms d, evaluates the certificate, then
computes only the lane's own slope. It marks the whole candidate for original
full replay if the sum certificate or either parity probe fails. Both endpoints
must be resolved before either can nominate. The unchanged replay kernel still
uses the original `S/A/A` sum.

The Python checks include 524,288 exhaustive reduced low-word/high-state pairs
(153,600 accepted) and 80,772 full-width random/directed pairs (50,480 accepted).
The reduced analogue uses T=16, K=17 and the same conservative interval pattern.
Every accepted pair satisfies the full bounds and yields the identical raw sum.
Those acceptance fractions are dominated by deliberate boundary inputs and must
not be treated as production exception rates.

The unguarded substitution already fails at `u=0,v=1`: the original raw sum is p,
while doubling gives zero. The certificate rejects it. A further complete
adjacent-lane/raw-hash transport test covers 10,692 candidates, including 3,086
replays and 81 matching simulated nominations, and exercises forced parity
failures, unusable rows, tails and sum-certificate rejection.

No native compilation, physical register allocation or device execution was
performed. The certificate and fewer live slope arrays may shorten dependencies
or reduce allocation, but add predicates. Complementary lane branches still
issue both add/sub paths at warp level. This is a correctness result and a
performance hypothesis, not a measured speedup.
