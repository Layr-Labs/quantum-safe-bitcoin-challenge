# Pinning: signed digit decoding and weighted cofactor recovery

This candidate removes work from the fixed-base scalar decoder, point-chain
scheduling and public cofactor recovery pipeline. The search still visits the
same sequence and locktime domain, derives both recovery keys, applies the same
SHA predicates and emits the same first-hit choice for each candidate. The
production candidate has no diagnostic iteration limit. The organizer's problem
generator, verifier, fixed-time wrapper and score calculation are unchanged.

The comparison baseline is ercumentyildirim's public submission
`ce0aff4e-be9b-4fae-a8a3-b0ed7acabb8e`, source
`3b81be52f08e3f412f9823f9dce46c12f2d1fad3`. It was the fastest artifact in our
local public-source cohort, then became the official frontier at
**713,225,734 verified candidates/s**, promoted as
`33753cc7bc2ebb7054b667805ce216dc5fe7fecd`. The submitted tree is based on that
promotion; its baseline production include closure matches the exact source
used in the comparison. Only `candidates/pinning` changes.

## Implementation

### Decode the signed odd representative directly

Let `n` be the secp256k1 group order and `D = 2*(k mod n)-n`. `D` is odd and lies
in `[-n,n)`. The existing table represents multiples of half of the fixed base,
so `D*(A/2) = k*A` in the group. The new decoder keeps D's raw 256-bit residue
and its mathematical sign directly. It avoids converting D to an absolute
value and then applying the global sign to all fifteen decoded digits.

For an odd integer x, define the regular digit and remaining scalar by
`d_w(x) = (x mod 2^(w+1))-2^w` and
`T_w(x) = 2*floor(x/2^(w+1))+1`. Both functions commute with negation.
Consequently the direct signed extraction gives exactly the same signed table
indices as the inherited absolute-value extraction followed by a global sign.
The last window uses sixteen stored bits and the explicit sign of D; its index
is f for positive D and `65535-f` for negative D. The table layout, lookup
records, window widths and addition order are unchanged.

The raw `k >= n` case retains an exact conditional reduction. Computing
`2*k-n` uses `C = 2^256-n`, which is a 129-bit constant: the third 64-bit limb
is one and must not be omitted. The fifteen immutable index/sign words are
stored in per-thread shared-memory planes. This lets scalar limbs die before
the point-chain loop. The shared arena is reused by the cofactor tree only
after every lane reaches a full block barrier.

### Keep one deferred point-add body and shorten register lifetimes

All thirteen remaining mixed additions use one rolled deferred-Y
specialization. The final ordinate is resolved once after the loop, instead
of retaining a separately compiled resolving-add body. The field-operation
count and algebra are preserved. The slope numerator is formed before the
X-difference so the old ordinate is no longer live during the later products.

The independent even and odd rows of the existing multiplication and squaring
PTX are interleaved. Complete linear carry chains are expressed as single pure
inline-PTX expressions with explicit operands. They do not have memory side
effects, so the previous per-instruction volatile memory clobbers are
unnecessary. These changes preserve the inherited hot field primitives' bit
contract; they do not claim to repair every pre-existing noncanonical raw-input
case in those hot primitives.

### Weight one root instead of every recovery leaf

For a fixed recovery point `R=(a,b)` and a prepared point
`P=(X/U,Y/V)`, with `U=Z^2` and `V=Z^3`, use the denominator
`D_i=V_i*(a*U_i-X_i)`. The public 128-leaf cofactor traversal produces the
product excluding each leaf. Define `h_i=U_i*C_i`, then save only
`vbar_i=Y_i*h_i` and `tbar_i=V_i*h_i`, four 128-bit state planes per leaf.
Inactive and unusable leaves retain their masks.

At a tree root, publish both `I=1/T` and `J=b/T`. Each finish lane obtains
`u=tbar*J` and `v=vbar*I`; this hoists a multiplication by the fixed b from
every leaf to one operation per root. With canonical u and v, set
`l=u-v`, `m=u+v`, `S=l+m`, and the host-derived constant `c=3*a^2/(2*b)`.
The square-free recovery identities are
`x_plus=a+S*(l-c)` and `x_minus=a+S*(m-c)`. Their ordinate parities come from
`l*(a-x_plus)-b` and `b-m*(a-x_minus)`.

The exact recovery multiplier retains the last reduction carry for every raw
256-bit input. Write `B=2^256`, `K=2^32+977`, and `p=B-K`. After the second
pseudo-Mersenne fold the value is below `B+K^2`. If it carries, the remaining
low residue plus K is below `K^2+K < 2^65`, so only three 32-bit limbs need
that last correction. No rare carry is dropped. Tree-internal representatives
may remain raw; canonicalization remains at the inversion and zero-test
boundaries and wherever the recovery algebra requires it.

### Compute only the ordinate parity that the hash needs

For `-p < r-b < p`, the canonical difference has parity
`((r_low xor b_low) xor (r<b)) & 1`, because p is odd. An exact raw product r
lies in `[0,B)`. If the fixed `b[3] != 0`, then `b >= 2^192 > K`, which proves
that interval bound without first normalizing r. For smaller b, the original
normalization is retained. The reverse difference uses the same argument.
This removes two redundant normalizations while preserving the small-b path.
At the denominator boundary, X remains canonical; the raw `a*U-X` result is
consumed by the exact full-width multiplier and normalized before the zero test.

## Correctness evidence

The final composed source passed native CUDA checks before performance
qualification. In particular:

- The actual old and new C++ decoders were extracted and run under UBSan on
  **93,576 raw scalars** around limb, group-order and sign boundaries, plus
  generated full-width values. All **1,403,640 signed table codes** agree.
  Independent integer reconstruction checks the scalar modulo n. The native
  CUDA decoder produces exactly the same 1,403,640 codes.
- Native scalar multiplication matches OpenSSL on **16,520 cases for each of
  two fixed bases**, for 33,040 point comparisons in the final composition.
- The recovery-boundary component passed **108,437 full-width input pairs**
  and **433,748 comparisons**, including small-b normalization fallback,
  raw values at p and B-1, carry and borrow edges, and aliases. Original and
  loose-representative recovery fixtures passed 42,448 cases per finish arm.
  The existing singular-mask behavior is preserved; these fixtures are not a
  claim of a new complete exceptional-point addition formula.
- Four native pipeline modes cover ordinary single/double hashing and easy
  single/double hashing. Each uses 2,352 candidates with non-power-of-two tail
  batches; the respective 291, 553, 276 and 533 emitted records match the
  independent control. First-hit priority remains unchanged.
- All eighteen invocations in the final short cohort produce the same 858
  complete-prefix hit tuples. They match an independently OpenSSL-verified
  reference. Short timing is not used as an official score.
- The long matched comparison uses a fresh problem seed, **1713034501**.
  Every one of its four complete 80-sequence prefixes yields the same
  **11,970 unique hits**. Independent OpenSSL verification passes all 11,970,
  deriving recovery from r, s and the preimage instead of trusting the problem's
  precomputed point shortcuts.

The unchanged production fixed-time wrapper then ran on a different fresh
problem, seed **1605866842**, for **1,200.119 seconds** (outer process wall
time 1,201.492 seconds). Every one of its **103,447 reported hits** passed
independent OpenSSL recovery and hash verification, with no failures or
duplicates. The production binary SHA-256 was
`d0090dbd589ef0837ac41ad789aff1084ad62e0f742da4b1ab103cae00907753`.
The source and binary hashes were checked before and after this run. This is
local correctness evidence for the submitted source, not an official score.

## Matched performance evidence

One RTX 4090, CUDA 12.8.93 and the organizer's unchanged compilation flags are
used for both arms. The GPU is shared through an exclusive bounded lease.
No clocks, power limit or driver settings are changed. The full comparison
starts with a complete long baseline warm-up and then uses ABBA order. Each
observation contains ten complete warm-up sequences and seventy measured
sequences, **87,122,000,000 completed candidates** in the timed region.

| Order | Source | Measured seconds | Local wall rate, M candidates/s |
|---|---|---:|---:|
| A1 | ce0aff4e frontier | 121.829278 | 715.115458 |
| B1 | submitted candidate | 120.165246 | 725.018282 |
| B2 | submitted candidate | 120.275803 | 724.351849 |
| A2 | ce0aff4e frontier | 122.003528 | 714.094106 |

The two paired gains are **+1.38479%** and **+1.43647%**. These are local
complete-work wall measurements, not hit-derived ranked scores and not a
cross-division of local throughput by the official record. The candidate is
submitted because it repeatedly exceeds the fastest currently public artifact
in this cohort. Ranked hit sampling and future competitors can affect promotion.

The contemporary short cohort also included the previous 260879f4 frontier,
ca8ed548, f4c21084, 0a26e48f, ad772199 and 9f06e0e5, alongside earlier relevant
pending artifacts. None exceeded ce0aff4e in the matched local comparisons.
A final refresh on 2026-09-18 at 06:12:34 UTC confirmed that the frontier
remained ce0aff4e at 713,225,734/s. Every still-pending public artifact was
covered by the fresh comparison or exact token-identity deduplication. The
newest sources were built with the same compiler and measured in mirrored
order after four complete control warm-ups:

| Source | Mean local wall rate, M candidates/s |
|---|---:|
| submitted candidate | 728.191416 |
| ce0aff4e frontier | 716.777155 |
| 1c8e12c4 | 702.871449 |
| 0c6d3205 | 718.257090 |
| 1fd5ccd1 | 700.437554 |

The fastest new competitor was 0c6d3205. The candidate led it by
**1.41857%** and **1.34763%** in the two directions; the mean-rate
lead was **1.38312%**. Control spread within
this short cohort was 0.13854%. All fourteen invocations
reproduced the same 858 independently verified hit tuples. The two-slot source
was drained before both timer boundaries, so queued work is not mistaken for
completed work. Submission 287a16b9 was deduplicated with 1c8e12c4 after
verifying that its only executable-source difference was an ordinary comment;
quoted literals and token boundaries remained part of the identity check.

The production source has its ordinary unbounded search loop terminated by
the official fixed-time wrapper. Finite-work diagnostic sources, their stop
conditions, prebuilt binaries, local result files and private infrastructure
configuration are excluded from the submission. The code payload consists of
the eight production source/license files, plus this research note and a
public source manifest. It is below the 8 MiB editable-path limit.

## Provenance

This is a composition and further optimization of public work, not a claim
that the full grinder was independently invented here. The modern comparison
base is **ercumentyildirim's ce0aff4e**. **tekkac's 31e98e47** provides the
public cofactor checkpoint and square-free denominator architecture.
**odinfree's e00f5566** supplies the earlier square-free two-recovery identity.
The regular direct-digit lineage includes **dun999's f535811**, and the exact
cold raw-scalar reduction follows **scarletbright's e7a648c7**.
Earlier 128-leaf and compact-state work by **0xCramJam** and **xlib**, the
VanitySearch-derived GPL field primitives, sparse hashing, and prior point and
table authors remain credited in the source and retained license.

Our work here combines the shared predecoded digits, direct signed recoder,
rolled deferred ordinate resolution, independent field-row scheduling,
weighted root recovery and proven raw-parity boundary. The selected production
source retains the baseline one-slot host execution path. GPL notices and the
full COPYING file are included with the production source.

Production source hashes are recorded in `SOURCE-MANIFEST.json`. The main
`pinning.cu` SHA-256 is
`2459223ae4692b1850b279bd3dc492275a5aa337149b5e167739d396907c6a98`.
The eight source/license files total 247374 bytes before documentation.

---

# Addendum: static-only round on 208bbcb6 (03e399c) — no GPU

Claude Opus 5 / Claude Code, effort max. No card was available; every number here is a
compile-time artifact from the ranked compiler (`cuda-nvcc-12-8=12.8.61-1`), produced with
the method owizdom established in `aff38dd0`:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
cuobjdump -ptx pinning | awk '/^\.version/{f=1} f' > c.ptx
ptxas -arch=sm_89 -O3 -v c.ptx -o c.cubin
nvdisasm -c c.cubin > c.sass
```

The instrument reproduces `547a64cf`'s published table exactly (chain body 1136,
prepare 5944, finish 3784, 120 registers, no spills), which is what makes the rest
reproducible by anyone else without a card. When counting SASS, match `/*[0-9a-f]{4,}*/`
and not `{4}`: these kernels pass 0xffff and a four-digit regex truncates `prepare`
to exactly 4096 instructions.

## Calibration

`547a64cf` removed ~992 dynamic instructions per candidate (-4.0% of ~24,760) and
measured +3.40% / +3.92% on a real 4090. Elasticity is ~0.85-0.98: on this kernel
throughput tracks dynamic instruction count roughly 1:1. Cross-check: 766.67e6
candidates/s x ~23,770 instructions ~= 1.82e13 instr/s against a 4090's
128 SM x 64 int32/clk x ~2.5 GHz ~= 2.05e13, i.e. ~85-90% of peak integer issue.

Dynamic budget per candidate (SHA rows ablation-measured by stubbing each transform):

| component | instructions |
|---|---:|
| chain loop, 13 x 1136 | 14,768 |
| tail SHA-256 | 1,304 |
| SHA-256d second | 1,312 |
| pubkey SHA-256 x 2 | 2,656 |
| cofactor tree up/down (warp-weighted) | ~700 |
| rest of prepare | ~1,900 |
| rest of finish | ~1,128 |

## What this rules out

- **PTX JIT**: `ptxas -arch=sm_89 -O3` over the shipped PTX takes 838 ms total for all
  six kernels, ~0.07% of a 1200 s window. The lever owizdom opened is now spent.
- **GLV / shrinking the 64 MiB table**: dominated by plain window narrowing (16-bit
  windows give the same 32 MiB and the same 16 lookups with no beta multiply and no
  lattice decomposition), and counterproductive either way: +1 chain step is +1136
  instructions (+4.8%), which the calibrated model says costs more than the L2 win.
- **Karatsuba** in `_ModMultCore`: -16 partial products but ~+40 IADD3, net +26.
  ptxas already fuses the even/odd merge into the `0x3d1` fold chain, which is why
  `_ModMult` lands at 119 SASS and not ~135.
- **More occupancy**: 5 blocks/SM is feasible (96 registers, zero spills) but costs
  +50 instructions in the chain body; at ~85-90% of peak issue the extra warps have
  nothing to hide.
- Primitive floors measured: `_ModMult` 119, `_ModSqr` 101, `_ModSqrAddSub2` 139,
  `_ModSub256` 19, `_ModAdd256` 31; SHA-256 at ~14 instructions/round.

## What changed here

No arithmetic. (1) `_PointAddXYZZT`'s statements are reissued in a different valid
topological order of the same dependence graph — the emitted PTX instruction multiset
for `kernel_pinning_pipeline<true,0>` is identical opcode-for-opcode (9,104 ops both
ways) — taking the rolled body 1136 -> 1134 and prepare registers 120 -> 114.
(2) The chain loop is unrolled by two (`QSB_CHAIN_UNROLL`, default 2) so the
loop-carried ordinate-anchor copy becomes register alternation: the fused pair is
2247 SASS where two rolled steps are 2272. Dynamic chain instructions
14,768 -> 14,613, i.e. -155 per candidate or -0.65%. Registers 126 stay inside the
128 that `__launch_bounds__(128,4)` allows, so occupancy is unchanged, and ptxas
reports no spill stores or loads. `-DQSB_CHAIN_UNROLL=1` restores the rolled loop.

Unmeasured risk: unrolling takes the hot loop from 18.2 KB to 36.0 KB of SASS. Both
already exceed a per-partition L0 and both sit inside the per-SM L1I, and the body runs
strictly sequentially, but instruction fetch is the one thing a static instrument cannot
bound. `-DQSB_CHAIN_UNROLL=1` isolates it in a single build.
