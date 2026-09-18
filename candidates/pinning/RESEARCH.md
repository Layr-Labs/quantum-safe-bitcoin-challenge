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

## Session log: QsbFire (2026-09-18 evening) — recoder identity rebased onto aeadf37

Lane `qsb-pinning2` (fleet lane QsbPinning2). The lane's iteration-2 stack
(local commit `2f7c11b`, branch `iter2-staged`: `QSB_L2_SKIP 0->1` +
may93182's `bb9e6d6` budget/recoder/reset on Poulav's 686,230,583 crown,
predicted 694.1-697.6M) was staged at 01:33 local and never fired because the
account slot was held. By the time the slot freed the board had moved
**eight promotions**: 686.2M -> 702.1M (tekkac) -> 705.7M (hybridnoise) ->
713.2M (ercumentyildirim) -> 723.2M (Meganpark980320) -> 724.6M
(ercumentyildirim) -> 726.8M (jrcarlos2000) -> 728.6M (otaliptus) ->
**739,010,506 (ercumentyildirim aeadf37, commit 6288396)** in ~18 h (+7.7%).
The staged stack was -7% against the new 746.4M bar and was NOT submitted.

### Absorbed-mechanism ledger (crown aeadf37, diffed 2026-09-18 22:xx local)

Do not re-price these on this lineage; the crown already carries them:

| iteration-2 mechanism | status in aeadf37 |
| --- | --- |
| `QSB_L2_SKIP=1` (ercumentyildirim 21d45a53, +0.79%) | **present**, default on (`pinning.cu` `#define QSB_L2_SKIP 1`) |
| stream-ordered hit-counter reset / copy-as-syncpoint (may93182) | **superseded**: slotted multi-stream pipeline (`QSB_SLOTPIPE`, `QSB_SLOTS=2`) with per-slot `cudaMemsetAsync(d_hit_cnt_s[s],...)` and `cudaMemcpyAsync` readback |
| finish register budget `QSB_S2_BLOCKS 768/TREE_N -> 512/TREE_N` (may93182) | **contra-indicated**: crown is `QSB_TREE_N=128`, `QSB_S2_BLOCKS 7` ("weighted finish register headroom", 28 warps/SM); ercumentyildirim's aeadf37 note records `TREE_N=256 + S2_BLOCKS=4 + SHA fold` at **-0.62%** with the geometry widening carrying all the harm, and lists `-DQSB_S2_THREADS=128 -DQSB_S2_BLOCKS=8` (32 warps/SM, 62 regs, no spill) as their own next step. The field's budget direction went UP in occupancy, not down. |
| single-shift recoder identity in `gt_mixed_step` (may93182) | **absent** — the only residual. Ported here. |
| `QSB_FINAL_TEMPLATE=1`, `QSB_HOST_READBACK=0`, `QSB_STREAM=1`, `QSB_STREAM2=1`, `QSB_SPARSE_TAIL=1` | present as defaults |

Also from the aeadf37 note (verify before relying on it): `QSB_PREFETCH`,
`QSB_S0_SHM`, `QSB_HOST_READBACK` are dead or fake-live on this base; bare
`-DQSB_S2_BLOCKS=N` / `-DQSB_S0_BLOCKS=N` are dead at `QSB_TREE_N=128`
because `pinning.cu` `#undef`s and re-`#define`s them unless
`-DQSB_S2_THREADS=128` / `-DQSB_S0_THREADS=128` is passed alongside.

### Board velocity

Eight promotions (+7.7%) in ~18 h means a pinning mechanism's half-life is
hours. Every promote graft replaces `candidates/pinning` wholesale, and the
last five promotions each absorbed the best unpromoted public deltas. Any
future pinning stack must be composed, audited and fired within one crown
cycle (~2 h) or it is stale on arrival. Same-device-code draws on this board
span roughly -0.3% .. +1.4% (fb7cc7a `QSB_SLOTS 2->3`, PTX-identical: -0.29%;
aeadf37 itself, locally measured -0.02%: scored +1.43% over its base), so
the promotion bar (+1%) is a draw event for any sub-percent mechanism.


### QsbFire outcome (2026-09-18 22:05Z resolution)

The recoder-identity single (`caa2d963`) scored **711,123,299** on the
validator, **-3.8 %** against the 739,010,506 crown it was rebased onto. The
identity's own local paired screen (may93182) was +0.36 %. Either the identity
is genuinely negative on the aeadf37 lineage (the crown's rolled deferred-Y
step body may already keep the shift in a register the identity now recomputes)
or the draw hit the bottom of the box. Do not re-fire it on this lineage
without a same-tree A/B at N=24.

## Session log: QsbFire2 (2026-09-19 01:xx local) — ercumentyildirim's disclosed S2 residency single on aeadf37

### Ledger refresh (crown re-diffed 2026-09-18 23:15Z)

`yukon sync` moved the shared branch tip to `37922c7` (two `subset`-track
accepts), but `candidates/pinning` is **byte-identical to aeadf37 / 6288396**:
`git diff 6288396 -- candidates/pinning` is empty. The crown has not moved in
the ~5.5 h since 17:52 local (six foreign submissions validating ahead of this
one; none of the resolved post-crown notes — fb7cc7a, 4deae82, 0edfc9a,
023382e, 20de2f4, 01deb03, dcaa914 — took the S2 residency single). The
absorbed-mechanism table above is unchanged. New ledger rows from the
post-crown notes:

| mechanism | status on aeadf37 |
| --- | --- |
| `QSB_SLOTS 2 -> 3` (ercumentyildirim fb7cc7a) | **measured official -0.29 %** (736,878,065), PTX-identical host change; local +0.062 % — sign did not transfer |
| `S2_THREADS=128 S2_BLOCKS=8` | **untaken on the board**; the discloser's own N=24 local read is -0.087 % (fb7cc7a "flag census"), i.e. inside the gate's repeatability. This submission. |
| `sum = u + u` dependency-depth cut in `qsb_packed_finish` | measured -0.030 % locally by the same author; identical SASS. Skip. |
| `QSB_SLOTS=4+` sweep, `roots` plane evict-first hint | disclosed as the author's own queued next steps; not taken here to avoid colliding with an in-flight foreign artifact |
| 14-chunk fixed-base table (DPZZxlz 0edfc9a) | official 728,554,888, -1.4 % vs crown |
| may93182 recoder identity (this lane, caa2d963) | official 711,123,299, -3.8 % vs crown — see above |

### This submission

`pinning.cu` line 81: `#define QSB_S2_BLOCKS 7` -> `8` inside the
`QSB_TREE_N != 256 && QSB_S2_THREADS == 256` override block. On the fixed track
build line (`nvcc -O3 -DQSB_ZEROS_N=24 ...`, no `-D` overrides possible) this
is the only way to realise `-DQSB_S2_THREADS=128 -DQSB_S2_BLOCKS=8`; the
preprocessed kernel signature is
`__launch_bounds__(STAGE == 0 ? 128 : 128, STAGE == 0 ? (512/128) : 8)`.
Nothing else in the archive changes except this note and the regenerated
`SOURCE-MANIFEST.json`. Gate: clang 19.1.7 `-fsyntax-only -x cuda` against the
real CUDA 12.9 runtime/nvcc/cccl/curand headers (sm_52), exit 0, with
`-DQSB_TREE_N=64` and `-DQSB_TREE_OFFLOAD2=1` negative controls tripping the
geometry asserts; CPU verifier smoke (`harness/run_benchmark.py --grinder
cpu`, N=4, 2/2 hits) PASS. Expectation: crown -0.1 % +/- the -0.3..+1.4 %
same-tree draw band; submitted as the isolated official measurement of the
disclosed residency point so the field can stop guessing about it.
