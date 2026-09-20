# Pinning research lead: routes from 789 M toward 1.0 G

Snapshot: 2026-09-20. Research only. This file is deliberately separate from
`WINNING-STRATEGY.md`, `LATER-IDEAS.md`, `AIM-1G.md`, and the active submission
note. It does not authorize a code change or a submission.

## Live baseline

- Promoted record: `dcd0147c`, 789,011,576 verified candidates/s, source
  `66fede0`.
- Promotion threshold: 796,901,692 (+1%). The 1.0 G target is +26.74%.
- Our FKIENE submission: `4ce3607` **rejected 762,268,179** (−3.39%,
  109,172 hits, 90.87 hits/s). Do not retry. Next pinning cut is `RAW_X=1`
  on the promoted decoder (`QSB_FKIENE=0`).
- Subset YOFF: `c29af055` validating on `e876032` / 595,907,916. Do not cancel.
- Completed controls since this section was first written:
  - `69769298` finished at 769,650,967 (-2.45% versus the promoted record).
    Its RP_SQR/recovery bundle is negative and must not be copied wholesale.
  - `cbce5501` finished at 791,077,271 (+0.262% versus the promoted record),
    below the 1% promotion gate. It is weak evidence that one or more of its
    exact lifetime/addressing cuts may compose, not evidence for the bundle.
  - `42b8009c` finished at 773,633,273 and is only host/noise evidence.
- Other live experiments worth watching are `defcba82`, `356fa57d`, and the
  newer `4b28ec4`; inspect their public notes before attributing any terminal
  score to a particular mechanism.

FKIENE is terminal. Do not claim a competitor's in-flight mechanism as new
work. Subset `c29af055` is validating; do not cancel it.

## Ground truth reconstructed from the source

The hot scalar path is not generic ECDSA or generic scalar multiplication. It
is one fixed-base signed-odd comb:

```text
z = SHA256d(preimage)
P = z * A, A = -r^-1 G       15 table terms, zero doublings
Q0,Q1 = P +/- u2R            square-free, batched recovery
gate = first 24 bits of SHA256(compress(Qi))
```

The fixed-base chain costs a 3M+2S affine seed, thirteen deferred-Y XYZZ
mixed additions at 7M+2S, and one final anchor multiplication: 95M+28S.
The 64 MiB table is near the 72 MiB AD102 L2 capacity. The current general
field multiply is already a 73-IMAD 8x32 Solinas implementation. The dynamic
instruction calibration says the kernel is near the integer-issue ceiling, so
ideas that merely exchange arithmetic for cache locality need an explicit
instruction and traffic budget.

## Evidence standard

Every candidate direction below must pass four gates before implementation:

1. Algebra: write the exact identity and all exceptional cases.
2. Cost: count hot-path field products, squares, reductions, table bytes, and
   global bytes per candidate against the current 95M+28S chain.
3. Novelty: search public Yukon notes and the local dead-end catalog for the
   same mechanism.
4. Native shape: require sm_89 registers, spills, loop SASS, and a rollback
   build. Source-level deletion is not a performance result.

## Implementation handoff board (kept current during research)

This is the short index for implementation agents. The derivations and
negative controls remain below; a row marked `prove first` is deliberately not
permission to start a CUDA rewrite.

| Track | Candidate | Exact handoff | Honest band | Status / stop condition |
|---|---|---|---:|---|
| pinning + subset | **C6 fourteen-term orbit comb** | First build a host recoder/auditor for `alpha=618+127*lambda`, all scalar boundaries, random scalars, zero digits, orbit rank and inverse reconstruction. Only after exhaustive coverage, implement a 14-window 96-byte `(x,beta*x,y)` loader with a kill switch. | about **+2--4%** whole kernel if the 68.27 MiB table stays hot | **prove first**; raw cardinality is only 1.000433x the scalar space and is not a coverage proof |
| subset | **offset-ordinate signed table (`YOFF`)** | Port pinning's table post-pass `y'=y+(K-1)/2`, pure-XOR signed load, offset-aware slope sum, and final anchor conversion into both `hit_filter_field_sc.cuh` and exact replay. Keep the existing table-load path at switch 0. | likely **+0.1--0.8%**, compose rather than submit as a 1B claim | **implementation-ready micro-cut**; static SASS must show the low-limb signed-negation correction really disappears without a new spill |
| subset | **candidate-family / SHA-prefix optimizer** | Host-only enumerate `(cut,s,t,selected tails,CTA packing)` and minimize `epoch producer transforms / candidates + first-state producer transforms / candidates + consumer transforms`, subject to at least 1,200 s of unique 9-of-150 work. Emit the exact selected sets and a verifier identity audit. | unknown; ordinary `7+2` is already bounded below 1% whole kernel | **research implementation ready**; do not touch CUDA until the search finds a family below the current ~5.238 first-hash transforms/candidate including producers |
| pinning | **joint last comb term plus `+/-R` recovery** | Symbolically derive two outputs `S+(T+R)` and `S+(T-R)` from one XYZZ accumulator, count shared products and normalization, then compare to one deferred `7M+2S` add plus packed symmetric recovery. | unknown; only worth CUDA if it deletes at least one field product on the total path | **algebra task**, not yet an optimization |
| both | **long-tail exact exception queue** | Census every omitted carry suffix; for each record saved hot instructions, capture instructions, and event rate. Warp-ballot compact flagged candidate indices and replay an exact kernel. | near zero unless a new long rare suffix is found | **conditional**; stop if capture cost is at least the deleted suffix |

Do not duplicate currently validating work: pinning FKIENE/recovery-lifetime
bundles and subset interleaved epoch SHA/negfold/speculative-prepare bundles
must report first. Also do not reopen F16, generic fused X3, ordinary 14-window
tables, grouped GLV, chain unrolling, or four-epoch SHA fission.

## Early findings

### A. The final comb window and fixed recovery point should be studied jointly

The current code completes the fifteenth table addition, updates both XYZZ
scale coordinates, resolves the deferred affine anchor, and only then starts
the `+/-u2R` recovery. Algebraically,

```text
(S + T_last) +/- R = S + (T_last +/- R).
```

Because `T_last` comes from a fixed, finite table and `R` is fixed for the
problem, `T_last+R` and `T_last-R` can be precomputed. This does **not**
automatically win: it replaces one mixed add plus the very efficient symmetric
recovery with two arbitrary final additions, and two output normalizations may
cost more than the work removed. The useful research question is narrower:

> Is there a two-target mixed-add formula sharing the old accumulator's
> `ZZ/ZZZ`, or a scaled batch-inverse contract, whose total cost is lower than
> the current last 7M+2S plus packed symmetric recovery?

This is a real algebraic opening because both target additions share the same
projective accumulator and all target data can be precomputed. It is not yet an
optimization claim. Derive the complete product count before touching CUDA.

### B. Exception repair is more powerful than publication-only gating

`QSB_HOST_GATE` prevents false GPU hits from being published, but it does not
recover false negatives: an approximate point that misses the hash gate is
gone. A more general architecture is an **exact exception queue**:

1. every optimistic field primitive emits a divergence bit for the exact
   predicate that its dropped carry/reduction would have triggered;
2. candidates with no divergence stay on the fast path;
3. flagged candidates are replayed by a separate exact GPU kernel;
4. tentative hits from both paths still pass the host publication gate.

This can make a bounded truncation exact at the candidate-set level and allows
cuts with higher event rates than C31. It is useful only where capturing the
predicate plus replay costs less than the removed serial suffix. Ordinary limb
carry is about 1/2 and cannot use this trick; multi-limb propagation through an
all-zero/all-one suffix is the intended domain. The next research step is a
full carry-chain census with three columns not present in the existing notes:
`saved instructions`, `predicate-capture instructions`, and `events/s at 1G`.

Public-submission cross-check sharpens the design. Exact in-line conditional
carry experiments `8a51e019` (branch), `471ecf07` (conservative result guard),
`c08f0dcb` (predicated tail), and `1b16f011` (zero-carry correction skip) all
failed to promote. They pay branch/predicate overhead at every field operation
and leave fallback instructions in the hot kernel. That does **not** close an
out-of-band exception queue: the proposed shape always executes the short
optimistic path, ORs a divergence bit into one per-candidate flag, and replays
only flagged candidate indices in a separate exact kernel. Its required proof
is stronger than the host gate but straightforward: every omitted arithmetic
effect must have a necessary-and-sufficient captured predicate, and every
flagged candidate must be replayed whether or not its optimistic hash hit.

This is an **exactness architecture worth keeping**, not a ready speed patch.
Before implementation, derive the event budget. At 1.0 G candidates/s, an
event probability of `2^-k` per candidate means approximately
`1e9 / 2^k` exact replays/s: 119/s at k=23, 30.5k/s at k=15, and 977k/s at
k=10. Even the latter is only 0.098% of candidates, so replay cost can be
small; the likely limiting cost is the hot-path flag capture and queue write,
not exact replay. Avoid a global atomic per event: compact one warp's ballot
with one atomic reservation, or append into a per-CTA ring drained between
batches.

Preliminary source census downgrades its likely speed band. The production
stack already omits the long rare suffixes: the multiply odd-fold event is
about `2^-44`, C31's remaining fold/add/sub events are about `2^-31`, and the
square/even-fold RP experiment is about `2^-23`. At 1.0 G/s, 28 square sites
at `2^-23` generate only about 3,338 replay candidates/s, so replay itself is
cheap. But returning and ORing a sticky event bit at every primitive costs at
least one or two hot instructions, while many of the remaining omissions save
only one to three. `b0fbfb1a` already measured the ungated RP bundle at only
+0.26%. An exception queue becomes a >1% candidate only if the census finds a
new **long** serial suffix with a rare entry predicate; it is not itself the
1.0 G jump, and restoring exactness to C31 would be slower than the current
score path without unlocking a larger cut.

### C. Coarse kernel fission is mostly already subsumed; only finer fission is open

A plain SHA/EC kernel split does not remove instructions and adds roughly
64 bytes/candidate of z write/read traffic. More importantly, the production
slot pipeline already overlaps the next candidate's prepare/scalar work with
the current candidate's root/finish work, and `QSB_SHA_FMA_ADD=1` deliberately
maps pubkey-SHA additions onto the FMA-capable path while the scalar chain is
IMAD-heavy. Therefore the original cross-pipe-fission thesis was overstated:
the obvious overlap is already present inside the persistent kernel.

The only remaining experiment is **finer phase fission** that changes register
residency enough to admit more warps or concurrent CTAs without surrendering
that existing overlap. It requires an incremental resource model covering
registers, shared memory, CTA residency, issue-pipe counters, and the new
global traffic. Keep its estimate at 0; it is not a credible 1.0 G thesis by
itself and should be closed if the split cannot cross a residency boundary.

### D. Fourteen-lookup C6 orbit comb is genuinely unimplemented and worth a proof prototype

The public history contains one unusually important negative statement: PR 531
and submission `0c846cf0` both say they **did not implement** a proposed
"fourteen-lookup C6 orbit design." The four scored GLV families instead used
15 or 16 affine terms. This keeps C6-14 open; it must not be mislabeled as a
retry of GLV40/GLV56.

The mechanism uses the six units of the secp256k1 Eisenstein endomorphism
ring. For an affine point `(x,y)`, one stored orbit represents
`+/-P`, `+/-phi(P)`, and `+/-phi^2(P)`, where the three x coordinates are
`x`, `beta*x`, and `beta^2*x`, the y coordinate is shared, and the sign only
negates y. Since `1+beta+beta^2=0`, a 96-byte record containing two x values
and one y reconstructs the third x as `-x0-x1 mod p`. This gives six oriented
choices per 96-byte record, versus two choices per 64-byte signed-point record.

A balanced Eisenstein base with norm `N` supplies approximately `N` digit
choices per lookup. Fourteen lookups need

```text
N^14 >= 2^256, so N >= 2^(256/14) = 319,557.12.
ideal table = 14 * (N/6) * 96 bytes = 14 * 16N bytes
            >= 68.27 MiB.
```

That is just below AD102's nominal 72 MiB L2, whereas an ordinary fourteen-
window signed table is 144 MiB and scored 728.6 M. It removes exactly one
deferred mixed addition, `7M+2S` (about 601 wide multiplies), taking the comb
from `95M+28S` to `88M+26S`. This is an honest roughly 2--4% whole-kernel
upside before recoding and memory costs, not a 26.7% route by itself.

An exact nearby arithmetic candidate exists. In the basis where
`lambda^2+lambda+1=0` and
`Norm(a+b*lambda)=a^2-a*b+b^2`, the smallest prime norm found at or above the
cardinality threshold is

```text
alpha = 618 + 127*lambda
Norm(alpha) = 618^2 - 618*127 + 127^2 = 319,567 (prime, 1 mod 3)
(Norm(alpha)-1)/6 = 53,261 nonzero C6 orbits
14 * 53,261 * 96 = 71,582,784 bytes = 68.2667 MiB
14*log2(319,567) = 256.000625 bits
```

Prime norm is helpful: modulo the prime ideal `(alpha)`, every nonzero residue
has a full six-element unit orbit, so there are no smaller zero-divisor orbits
to bloat the table. The tiny `0.000625`-bit volume margin is also a warning:
cardinality alone is not a coverage proof. A rectangular 128+128 GLV split or
any appreciable guard band exceeds L2. The recoder must use an almost area-
optimal hexagonal/Voronoi representative of the GLV lattice, and it must deal
explicitly with the zero residue (identity entry, rare branch, or a proved
nonzero regular recoding). Starting from the integer scalar directly is also
wrong: fourteen divisions by `|alpha| ~= 565.3` remove only 128 bits of
one-dimensional magnitude. First map `k` to a short two-dimensional GLV/
Eisenstein representative; only that vector has the required ~128-bit radius.

The costs are material. A record is 96 bytes. If two x planes plus one y plane
are loaded conditionally, two orientations consume 64 bytes and the
reconstructed orientation consumes 96 bytes, averaging 74.7 bytes per random
digit under a uniform orientation. Fourteen digits therefore request about
1,045 bytes/candidate versus 960 bytes for the current 15x64-byte comb, about
8.9% more table traffic, and the table leaves only ~3.7 MiB of nominal L2
headroom. The current device's persisting-window cap is also below the whole
table. Any win must come from the deleted dependent arithmetic, not a claim of
less traffic.

The first deliverable should be host math, not CUDA:

1. select an Eisenstein base `alpha=a+b*lambda` with norm just above 319,557;
2. define nearest-cell division and a canonical C6 orbit rank/orientation;
3. prove fourteen digits cover every scalar modulo the group order, including
   Voronoi boundaries and the last digit;
4. audit reconstruction for boundary plus randomized 256-bit scalars;
5. calculate exact orbit counts, exceptional unit orbits, aligned plane sizes,
   and digit-decoder integer operations.

Only then should the implementation agent build the table/loader. Primary
math anchors: Heuberger--Mazzoli, *Symmetric Digit Sets for Elliptic Curve
Scalar Multiplication without Precomputation*, ePrint 2013/705
(`https://eprint.iacr.org/2013/705`), for the six Eisenstein units and minimal-
norm residue systems; EFD XYZZ (`https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz-3.html`)
for the mixed-add cost being removed.

## Directions already closed for the 1.0 G search

- Sequential `+G`, vanity scanners, six-X set membership, and Gray-code nonce
  walks: SHA256d makes every scalar independent.
- Classic/grouped/joint GLV as already submitted (523-723 M), including the
  40/56 MiB fifteen-term families.
- Fourteen ordinary windows (official 728.6 M on a 739.0 M base) and thirteen
  20-bit windows (449.7 M).
- Co-Z on an affine table, batch-affine gECC across fifteen dependent rounds,
  Pippenger/MSM, Edwards/FourQ/isogeny changes, complete 11M formulas.
- 5x52/10x26 CPU radices, Montgomery reduction, RNS, CGBN, Karatsuba, and
  tensor-core slogans without an exact pointwise-bilinear mapping.
- More pinning SHA table/shared-W1 work without same-host evidence.
- F16 as a naive persistent ninth limb: the next 9x9 product grows to about
  twelve limbs; folding before it restores the work supposedly removed.

## Research queue

1. Derive and cost the shared-accumulator two-target final formula above.
2. Produce the exact-exception census from the actual PTX/SASS carry sites.
3. Audit every in-flight exact register-lifetime claim after its official score;
   keep only mechanisms whose emitted loop/register shape supports the story.
4. Search fixed-base literature specifically for endomorphism-orbit tables that
   reduce the **number of additions**, not merely table bytes while retaining
   fifteen terms. Any proposal must be distinguished from the four failed GLV
   submissions.
5. Search short-Weierstrass formulas for a mixed affine add below deferred-Y
   XYZZ 7M+2S under an incomplete/non-exceptional contract.
6. Build a roofline from official/static evidence: integer issue, L2 table
   traffic, HBM streaming state, SHA FMA/ALU split, and phase residency. A 1.0 G
   plan must identify where the extra 26.7% hardware capacity comes from.

## Subset track: scope expansion and live frontier (2026-09-20)

This memo now covers both benchmark tracks. Research stays in this one file;
no subset or pinning CUDA source is changed by the research lane.

The live subset frontier moved while this audit was in progress:

```text
promoted submission  6dfdb6f8-2a23-4814-88fe-a2cd3c724233
promoted commit      e876032f79e6f4f3af2732bbba39403e29f0e227
official score       595,907,916 verified candidates/s
next 1% gate         601,866,996 = ceil(595,907,916 * 1.01)
1.0 G gap            +404,092,084/s = +67.81%
```

The promotion is the exact first-word (`H0`) public-key SHA gate composed onto
the prior 588,762,499 frontier. It scored +7,145,417, or +1.214%, and therefore
promoted. This is stronger evidence than the earlier 593,284,952 H0 attempt:
H0 is now baseline, not an open port.

Four subset evaluations were live at the observation point:

- `3a999cdf`: frontier remeasurement with one carried mechanism;
- `ee7ece52`: speculative preparation + negfold parity chain + H0 composite;
- `da9bfa50`: interleaved epoch scalar hashes over the negated-fold H0 tail;
- `daf3712c`: independent promoted-tree remeasurement.

Do not submit or cancel from this research lane. Their terminal scores are
needed to separate real mechanisms from the roughly 587--595 M host spread.

### Immediate subset consequence

The old priority list is stale in two ways: `QSB_PK_H0` is already promoted,
and the new promotion raises the next gate to about 601.87 M. A one-percent
micro-optimization must now add roughly 5.96 M/s before it is even promotable.
Getting subset to 1.0 G requires an architectural change, not an accumulation
of register renames or another frontier remeasurement.

### E. The advertised 8-lookup / 32 MiB joint-GLV comb is impossible as stated

An older campaign note proposed splitting the scalar into two ~128-bit GLV
coordinates, consuming one pair of 8-bit digits per lookup, and using eight
tables of 65,536 points. This confuses an 8-bit digit in each coordinate with
a 16-bit digit in each coordinate:

```text
8 lookups * log2(65,536 choices/lookup) = 8 * 16 = 128 bits,
```

not 256 bits. Eight lookups over two 128-bit coordinates require about 32 bits
of joint choice per lookup, or roughly `2^32` entries before symmetry. A table
with 65,536 joint entries needs sixteen lookups, which is not an addition-count
improvement over the current fifteen-term comb. The six cheap secp256k1 units
add only `log2(6)` bits per record and do not repair the missing 128 bits.

This is an information-theoretic closure, independent of CUDA quality. Do not
implement the old 8-add/32-MiB sketch. The C6 fourteen-lookup design above is
different and survives the same test: fourteen alphabets of size about 319,567
carry just over 256 bits.

The same bound explains why fourteen C6 lookups are essentially the minimum
under a 72 MiB affine table. With 96-byte `(x,beta*x,y)` orbit records and six
oriented choices, equal-sized digit alphabets minimize total records by AM-GM;
the lower bound is about 68.27 MiB. Thirteen full-coverage lookups need about
168 MiB with 96-byte orbit records. Storing only `(x,y)` lowers that to about
112 MiB, but then an average `2/3` of the thirteen digits need an on-device
`beta` or `beta^2` field multiplication. Since beta is dense in radix 2, this
is not a cheap shift/add constant multiply. It is a secondary memory/arithmetic
Pareto experiment, not the primary C6 implementation.

### F. Subset: moving the omission cut across the next SHA block is only a small win

The promoted short-epoch family uses six omissions below cut 137 and three
among the final thirteen pushes. Its epoch prefix is
`42 + (137-6)*10 = 1,352 = 21*64+8` bytes. The remaining message therefore
uses six first-hash compressions: one first-block class produced out of line,
one per-candidate scheduled block, and four constant blocks. The separate
SHA256d compression and two interleaved pubkey gates follow.

To move one more complete 64-byte block into the epoch prefix, a split with
`s` early omissions and a tail of `K` pushes must satisfy

```text
42 + (150-K-s)*10 >= 22*64
K+s <= 13.
```

Because the total omission count is nine, the largest viable full-run tail
with useful lane multiplicity is `s=7, K=6, t=2`: fifteen tail choices per
early epoch and

```text
C(144,7) * C(6,2) = 3,295,251,994,320 candidates.
```

The superficially nicer `s=6, K=7, t=3` shape has only
`373,767,008,615` candidates, about 627 seconds at 596 M/s, so it exhausts
before a ranked 1,200-second run. The `7+2` family is large enough, but the
fifteen tail messages have essentially fifteen distinct first blocks. The
compression removed from the consumer reappears as about one first-block
compression per candidate; current `6+3` amortizes only 54 distinct first
blocks over 256 candidates (`54/256 = 0.211`). Hierarchical sharing of the
first six early omissions can reduce the new producer overhead, but it cannot
remove the unique first block after the seventh omission.

Conclusion: this cut changes roughly `6.21` scalar-hash compressions per
candidate to a lower bound near `6.0` plus extra hierarchical producer work.
That is at most a few percent of the scalar-SHA phase and likely below one
percent whole-kernel. It is worth a host-side schedule/class prototype only
after the in-flight second-SHA experiment reports; it is not a 1.0 G thesis.

The current choice of 256 out of `C(13,3)=286` tails is already deliberate.
The omitted thirty triples each kill a singleton first-block class, reducing
the class count from 84 to the theoretical minimum 54 for any 30 exclusions.
There is no overlooked better subset that reduces it below 54. First-block
singleton classes and second-block singleton classes are disjoint, so reducing
the latter necessarily gives back at least one first-block class.

### G. Fused square/add/sub is already present in both hot EC paths

Do not open an implementation lane for a generic Longa-style
`R^2 + E - 2Q` sum-of-products primitive. Pinning enables
`QSB_FUSE_SQRADDSUB2=1` by default and calls `_ModSqrAddSub2` from both the
deferred and final XYZZ add paths. It forms the square and injects the addend
and two subtrahends into the same Solinas reduction. This is the exact
operation that a proposed "fused X3" optimization would target.

Subset is not missing the corresponding specialization. Its speculative
filter calls `qsb_filter_seed_x3(T,T,ZZZ3,Q,bad)`, and exact replay calls
`qsb_replay_seed_x3`; both are dedicated guarded implementations of
`R^2 - PPP - 2Q` for their respective field representations. The subset
routine differs in sign convention because its seed variables are arranged
differently, but it has already eliminated the standalone generic reduction.

This also closes the tempting F16 follow-on. Keeping an extra high limb after
the first fold does not compose cleanly with the next multiplication: an
8x32 product first-fold is nine limbs, the subsequent 9x9 product grows to
roughly twelve limbs, and folding before that multiply reinstates the work we
were trying to remove. Existing audits and source comments reached the same
result. Treat fused-SOP and F16 as completed negative research, not open ideas.

### Live-queue update after the subset promotion

The earlier subset frontier remeasurement `3a999cdf-75d8-4390-aff5-e4e45b87bf66`
finished at **595,242,746/s** and was rejected against the new 595,907,916
frontier. It is byte-level/noise evidence, not a mechanism: the promoted score
is 0.112% above that remeasure, and the next qualifying gate remains
601,866,996. Three subset experiments remain validating (`ee7ece52`,
`da9bfa50`, `daf3712c`).

Pinning currently has four live validations: our exact FKIENE composition
`4ce3607f`, plus `defcba82`, `356fa57d`, and the newer `4b28ec4`. `d37e3975`
finished at 761,083,226 and is negative. No conclusions should be drawn from
queue age; wait for official scores and compare every mechanism against
789,011,576 and the 796,901,692 promotion gate.

Subset currently has four live validations: `ee7ece52`, `da9bfa50`,
`daf3712c`, and `f63181cb`. The last is described as an inert-diff frontier
remeasure and should be used as a same-period noise control, not credited as
an optimization.

### H. Whole-field Eisenstein coordinates are a useful closed direction

A tempting extension of the C6 orbit table is to represent every field element
as a short pair `a+b*beta (mod p)`. In that representation multiplication by
`beta` is a cheap coefficient permutation, so an orbit record might store only
one x pair and one y pair (64 bytes rather than `(x,beta*x,y)` at 96 bytes).
The exact secp256k1 field relation is unusually neat. With

```text
beta = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
A = -0x114ca50f7a8e2f3f657c1108d9d44cfd9
B = -0x3086d221a7d46bcde86c90e49284eb16
A + B*beta == 0 (mod p)
A^2 - A*B + B^2 = p
```

the kernel lattice has a 128-bit Eisenstein generator. The analogous group-
order generator uses the familiar GLV constants one unit lower in magnitude:

```text
A_n = 0x114ca50f7a8e2f3f657c1108d9d44cfd8
B_n = 0x3086d221a7d46bcde86c90e49284eb15
A_n^2 - A_n*B_n + B_n^2 = n
A_n + B_n*lambda == 0 (mod n)
```

This does **not** make general field multiplication cheap. A pair product still
needs three 128-by-128 products with Karatsuba, i.e. 48 32-bit wide products,
before reduction. The current canonical 256-bit multiply needs 64 schoolbook
wide products plus only about nine extra products for sparse Solinas folding,
roughly 73 total. Pair form therefore has a budget of only about 25 additional
wide products for its entire reduction and canonicalization. Reduction by the
dense 128-bit Eisenstein generator needs another dense complex multiply (about
three more 128-by-128 products) even for the first fold, already consuming
roughly 48 more wide products before carry/correction work. It loses the
advantage that `p = 2^256 - 2^32 - 977` gives the current representation.

Keeping only table x coordinates in pair form does not rescue the idea: every
oriented digit must be converted to canonical field limbs or the accumulator
must pay the expensive pair arithmetic globally. Storing canonical `(x,y)` and
computing `beta*x` on demand similarly adds a dense constant multiplication on
about two thirds of C6 orientations. Close this route unless someone produces
an emitted-SASS reducer below the present 73-IMAD multiply; the exact lattice
constants above are provided so another researcher can falsify the cost bound
without rediscovering the representation.

### I. Reopen batch-affine only as a CTA-local fixed-comb experiment

The blanket phrase "batch affine needs fifteen round trips" hides two distinct
designs. Global round fission is still closed: spilling `(x,y)` after each of
fourteen dependent additions costs about `14 * 128 = 1,792` bytes of state
read/write per candidate before the 960-byte table stream, which cannot sustain
1.0 G candidates/s on an RTX 4090.

A CTA-local Montgomery tree is different. Keep one affine accumulator per
thread, form all 256 denominators for one comb round, invert their product once,
recover the 256 inverses in shared memory, and repeat for fourteen rounds. An
ordinary affine add then costs, per candidate and excluding the amortized root
inverse, approximately

```text
batch inverse tree   3M per denominator
slope and y update   2M + 1S
total                5M + 1S per add
fourteen adds        70M + 14S + fourteen CTA-root inversions
current chain        95M + 28S
```

The arithmetic ceiling is therefore interesting: before synchronization and
root inversion, it removes 25 multiplies and 14 squares, roughly 32% of the EC
field-operation count. It applies to both tracks because both use the same
fifteen-term fixed-base comb. It is not implementation-ready. Fourteen serial
product trees mean roughly 200 CTA barriers; a whole CTA waits for each root
inverse, and the current projective path has no inter-thread dependency in its
comb. The subset source's four-lane `zi_inverse_quad` makes the root cheaper
than a lane-0 Fermat inverse but does not make its latency disappear.

A balanced affine reduction tree gives four dependent levels (`15 -> 8 -> 4
-> 2 -> 1`) and the same fourteen total affine additions. It can reduce the
number of synchronization waves, but it must retain many intermediate affine
points and reorganize candidates across lanes. With 64 candidates, retaining
eight post-leaf points already costs 32 KiB of shared memory; denominator-tree
storage and scalar/digit staging consume most of the remaining CTA budget.
It also changes the table-access and SHA lane geometry. The worthwhile host/
static prototype is therefore:

1. model both sequential-256 and balanced-32/64 candidate CTA schedules;
2. count root inversions, active lanes, barriers, shared bytes, and x/y traffic;
3. compile a standalone sm_89 synthetic kernel using the existing field and
   `zi_inverse_quad` routines, without modifying either production track;
4. stop unless the synthetic comb is at least 15% faster than the current
   projective comb at equal table accesses and preserves two resident CTAs.

This is a plausible **cross-track architectural experiment**, not a promised
gain. Its best-case arithmetic is large enough to matter; synchronization,
shared-memory residency, and cooperative-inverse latency can still erase it.
