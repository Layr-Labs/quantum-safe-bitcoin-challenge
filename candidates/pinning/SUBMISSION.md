# Pinning: paired recovery, certified raw doubling and productive finish selection

Effort: medium. Independent research and integration used GPT 6 Astra through
Codex. This is a new implementation for remote validation; no local native C++
or CUDA compilation, GPU execution, timing, register report or ranked score is
claimed.

The candidate starts from promoted Pinning source
`7c3609b87b9d8e094a16be148fe846dfd5ac7807`, submission
`22944657-779f-4b1c-b22e-5b89c8d429c9`, at 805,428,058 verified candidates/s.
The latest frontier was refreshed during release preparation. The manifest
requires a 100-basis-point improvement, approximately 813,482,339/s at that
frontier. The official evaluator determines correctness, score and promotion.

## Why this is a different experiment

My previous submission `e9874278-ac6c-4744-9556-4df3ebd0fb3d`, public
[PR 1087](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1087),
combined hierarchical parity replay with a resident top-32 cofactor schedule.
It completed with verified=true and 779,620,115/s, about 3.20% below the frontier.
The isolated public top-32 predecessor also reported a rejected 766,632,513/s.
Those different official runs do not identify each component's causal cost,
but provide no reason to retain top-32. This release restores the promoted
`cofactor_checkpoint.h` exactly and contains no `ResidentCofactor.cuh`.

The earlier negative-MAC/K32/high-parity composition, my submission `244293e3`,
was verified but rejected at 794,678,663/s. None of that arithmetic bundle is
silently carried into this release. The promoted positive-ordinate arithmetic,
wide parity window, scalar point chain, root hierarchy, SHA implementation,
table geometry, batch size and slot count remain the common starting point.

The new hypothesis is that two adjacent threads can divide a candidate's two
recovered keys, shorten each thread's dependency chain and reduce simultaneous
endpoint state. A separately proved raw-arithmetic identity can eliminate the
need to construct both slopes in each thread. The costs are additional warp
exchanges, doubled finish-grid width and exact replay on uncertified inputs.
These are structural changes with uncertain device economics, so the normal
search briefly compares the new finish with the original promoted finish and
then fixes one route.

## Adjacent threads own the two recovered keys

The new hot finish launches 128 threads for 64 candidates. Its original
candidate index is `(blockIdx.x*128 + threadIdx.x) >> 1`. The even thread owns
the first recovery id and forms raw `u=tbar*weighted_inverse`; the odd thread
owns the second and forms raw `v=vbar*root_inverse`. Each reads its own two
16-byte saved-state planes and its own two inverse vectors. The two threads
exchange the resulting four 64-bit limbs with XOR-one shuffles.

The recovery-root index remains the original candidate index divided by 128.
The weighted bank starts at `ceil(actual_batch_size/128)`. Neither the doubled
grid nor the later sparse replay index is used as a substitute for those
indices. Saved-state plane strides use the actual partial-batch size.

The even thread constructs the first slope with the promoted subtraction;
the odd thread constructs the second with the promoted lazy addition. Each
then computes one recovered x coordinate, one original wide parity-window
probe and, if both endpoints are resolved, one compressed-public-key SHA-256.
All lanes, including unused tail pairs, reach every full-mask shuffle. There
is no early return in the paired hot kernel.

An unusable checkpoint is shared across its pair. If either parity is
ambiguous, or the sum identity is uncertified, both endpoints defer before
either endpoint hashes. Only the even thread marks the candidate. If both
hashes qualify, the even thread wins, preserving the original first-recovery-id
nomination. The encoded original index and recovery-id bits are unchanged.

There are four static shuffle sites: one four-u64 exchange loop and three
u32 exchanges for usability, deferral and hit priority. This corresponds to
eleven 32-bit shuffle operations per paired warp, which now represents sixteen
candidates. This is a source-level communication count, not a measured SASS
instruction count.

## A guarded identity for the actual raw add/sub contract

Replacing `(u-v)+(u+v)` with `2u` unconditionally is incorrect for the promoted
C31 low-word correction arithmetic. For example, `u=0,v=1` gives raw p in the
original expression but zero when doubled. This release does not make that
unconditional substitution, remove a new carry or assume that modular
equivalence implies identical raw output.

Let `B=2^256`, `L=2^64`, `T=2^32`, `K=T+977`, and `p=B-K`. Let S be the actual
promoted `_ModSub256`, and A the actual `_ModAddLazy`, with the current
`QSB_C31 && QSB_SHORT_CARRY` configuration. Both first compute full-width
addition/subtraction and then apply the inherited correction only to the low
64-bit word. The old raw sum is `A(S(u,v),A(u,v))`. Define `d=A(u,u)`.

A sufficient certificate is:

```
x = (u-v) mod L;  x >= K
y = (u+v) mod L;  y < L-K
q = 2u mod L;     K <= q < L-2K
K < d < p
```

The first two bounds prevent the correction of S and A from discarding a
low-word borrow or carry. If their full-width borrow/carry are b and c, their
sum's low word is `(q+(c-b)*K) mod L`. The q interval keeps this in `[0,L-K)`
before the final correction, so neither the old expression nor doubling loses
a low-word carry. Both are congruent to 2u modulo p. The final bound makes d
the unique representative of its residue in `[0,B)`: d-p is negative and
d+p exceeds B. The outputs are consequently identical as raw 256-bit values.

The implemented conservative certificate needs four wrapping-u32 comparisons.
With `uh=uint32(u[0]>>32)`, `vh=uint32(v[0]>>32)`, `dh=uh-vh`, `sh=uh+vh`,
`qh=uh<<1`, and `top=uint32(d[3]>>32)`, accept when:

```
dh >= 3 && sh < 0xfffffffd &&
(qh-2) < 0xfffffff9 && (top-1) < 0xfffffffe
```

The low-half borrow makes x's upper half at least dh-1, hence x>=2T>K.
The low-half carry makes y's upper half at most sh+1, hence y<L-K.
The qh interval allows its possible extra bit while retaining
`2T<=q<=L-4T-1`, inside the sufficient q bounds. The top test places d between
`2^224` and `B-2^224-1`, inside `(K,p)`. These bounds apply to arbitrary raw
u and v, including outputs of the inherited approximate multipliers.

`GuardedSum2u.cuh` computes this certificate. On success, the paired finish
forms only its own slope after doubling; it removes the separate l and m
arrays from that thread's source. On failure, the complete original finish
recomputes both slopes and their original sum. `SUM2U-PROOF.md` includes the
interval argument and test limitations.

## Complete replay and unchanged publication

Unresolved candidates use the existing independently developed two-level
bitmap transport. The allocation contains a 1,056-word report prefix, one bit
per candidate and one summary bit per bottom-level word. The prefix leaves
room for the original counter and 1,024 nomination indices. At the default
8,388,608-candidate batch size, the full report is 1,085,568 bytes per slot.

Each deferred candidate performs two atomic ORs. A same-stream cold kernel
scans nonempty summary words and replays the original full wide-product parity
and finish on the original checkpoint. It cannot defer again. The hot and cold
kernels never publish the same candidate. All bottom-level bits are
representable even if every candidate defers; a dense bitmap is potentially
slow but does not overflow a finite work queue or silently discard work.

The paired route clears the full active report before its prepare launch,
then launches hot finish and cold replay after root completion. No reset lies
between hot and cold kernels. The original route clears only its four-byte
counter and launches the exact promoted stage-2 kernel. Both routes share the
enlarged report allocation and contiguous counter/index addresses; that host
allocation detail is a difference from the original separate slotted buffers.
The original control kernel body and launch arguments are preserved exactly.

Both slot streams retain distinct saved states, roots, reports and sequence/
locktime metadata. Copies and completion events follow the selected finish.
The original 1,024 device nomination cap, first-64 host readback, exact OpenSSL
publication gate and output record format remain unchanged. Checking the direct
return from `cudaEventSynchronize` now precedes clearing a slot's busy state
and consuming its completed buffers.

## Finite comparison using productive search work

The public description of Portablelle's in-flight `46fca857` introduced a useful
whole-pipeline comparison approach. I independently implemented a smaller
two-route version at existing sequence-drain boundaries. No donor source,
binary or diagnostic package was imported. Its affine-prefix, negative-Y and
register-retention alternatives are not part of this candidate.

Route 0 is the promoted finish; route 1 is the guarded paired finish plus exact
bitmap replay. There are four productive warmup sequences in ABBA order, then
eight measured sequences in ABBA BAAB order. Every sequence searches its normal
distinct locktime interval, including the partial final batch. A timestamp
starts before sequence SHA preparation. All slots, hit copies and publication
work drain before the ending timestamp. Cost is elapsed monotonic wall time
divided by the actual completed candidate-count difference.

Four adjacent A/B pairs are compared with their order resolved by route id.
Route 1 is selected only if the geometric gain is at least 1.01, at least three
pairs favor it, and no pair is worse than 0.995. Zero work, invalid clocks,
nonpositive/nonfinite timing or invalid costs keep route 0. After twelve
sequences, the selected route is fixed; clocks and comparison reports stop.
Switching happens only with all slots empty.

These twelve sequences represent 14,935,200,000 genuine candidates for the
standard range, about 18.7 seconds if both paths sustain 800 million/s. Slow
alternatives increase that duration and consume real opportunity cost. All
warmup and measured work remains in the normal total and all qualifying hits
remain eligible for publication. No candidates are duplicated, omitted,
invented, timed outside the official run or substituted with reference output.
The benchmark's timing, scoring and search limits are unmodified.

This policy limits persistent selection of a slower route, but does not
guarantee a ranked gain or protect against later clock changes. The extra
compiled kernels can increase JIT/startup cost. The original route still pays
the common enlarged-allocation and startup resource-query costs. The policy
also compares timing rather than independently verifying both routes on each
candidate; correctness rests on the arithmetic/transport reasoning and the
official verifier, not the timing choice.

Build-time `QSB_FINISH_ROUTE=0` or `1` forces a route for independent experiments;
the default is -1, finite comparison in the normal slotted pipeline. A build
with `QSB_SLOTPIPE=0` and the default route uses the promoted finish. Setting
`QSB_SPARSE_PARITY_REPLAY=0` removes the added route and its allocation/selection
machinery. The paired route deliberately targets the ranked FAST_TAIL=true,
single-hash, current lazy-recovery configuration, with compile-time assertions
for those required assumptions.

## Work ledger and risks

On common usable, certified, unambiguous candidates, total full field
multiplications remain four, parity windows remain two and public-key hashes
remain at most two. Saved-checkpoint logical traffic remains 64 bytes per
candidate. Dependency splitting and source live ranges are the intended
benefit, not a claimed reduction in all arithmetic or memory traffic.

The original scalar finish uses seven active field add/sub calls per common
candidate. The initial unguarded-sum paired prototype used ten; the guarded
version uses eight. Complementary even/odd add/sub branches still execute both
paths at warp issue level, so ten-to-eight active calls is not a claim of fewer
warp-issued add/sub paths. Four certificate predicates and the exchanges cost
work. Twice as many finish warps increase constant broadcasts and can increase
root-sector requests even when logical root bytes are unchanged. The pair may
also do work the old first-key-hit short circuit would avoid.

Fewer declared arrays do not prove fewer physical registers or higher achieved
occupancy. Existing launch bounds are retained: 128 threads with the promoted
stage-2 block target for hot finish, and a four-block target for cold replay.
Cold scanning adds one launch per batch and is unattractive if deferrals are
dense. These risks are why the original promoted finish remains the control.

## Checks and reproducibility

Only Python arithmetic/source models and file checks were run on the authoring
machine. No native compilation or GPU run was attempted. The prior tested
paired arithmetic and bitmap code was reused after source-hash and executable-
text identity checks, rather than pretending repeated tests add device evidence.

* Plain paired transport: 68 batches, 11,084 candidates, 10,412 raw endpoint
  comparisons, 3,021 replays and 72 matching simulated nominations; tail lanes,
  original root indexing and both-key hit priority included.
* Certified doubling: 524,288 reduced-width pairs with 153,600 accepted, plus
  80,772 full-width directed/random pairs with 50,480 accepted. Every accepted
  pair had matching raw output and satisfied the sufficient bounds. The
  unguarded u=0,v=1 counterexample was rejected.
* Guarded paired transport: 48 batches, 10,692 candidates, 3,086 replays and
  81 matching simulated nominations, including forced ambiguities, zero roots,
  unusable rows and boundary/tail batches.
* New productive policy: 124 source-tied Python scenarios for gain, regression,
  ties, drift, unequal work, invalid timings, forced routes and finite
  completion; twelve slot/sequence models retained every distinct interval.
* Release checks found the promoted GPU stage-0/stage-2 template body,
  cofactor tree, other shared arithmetic/SHA headers and host-gate body
  unchanged. Candidate increments, sequence/locktime iteration and publication
  statements match the promoted source. `git diff --check` passed.
* The existing host-gate Python test passed 64 midstate cases, binary-layout
  checks and recovery comparisons against the independent verifier.

The transport models use synthetic raw states and a lower simulated hash
difficulty to exercise nominations. Their acceptance and replay fractions are
deliberately boundary-heavy and are not production probabilities. Policy tests
model C++ behavior from source constants and structure; they do not compile it.
The inherited C31 multiplier remains approximate, and the unchanged host gate
can reject false nominations but cannot recover genuine hits omitted by prior
arithmetic. No all-input GPU proof or performance measurement is asserted.

Reproducible CPU checks included in the archive:

```sh
python3 -B candidates/pinning/test_guarded_sum2u.py
python3 -B candidates/pinning/test_productive_finish.py
python3 -B candidates/pinning/test_host_gate.py
```

Remote submission uses the ordinary Yukon Pinning workflow, with the note file,
exact underlying model and Codex harness attribution, and no claimed score.
Only `candidates/pinning` is archived. The protected harness, verifier, generated
problem rules, setup/build commands and scoring policy are unchanged.

One startup query still prints hot/cold `cudaFuncGetAttributes` to stderr.
Inspection of the current protected `gpu_wrap.py` shows raw subprocess output
is captured for parsing but not archived in its summary artifact. Therefore
neither the previous public result nor this note claims an observed remote
register count; route-selection diagnostics may likewise be absent from public
artifacts. No protected logging code was changed to expose them.

## Public research credit and next result

Credit belongs to the promoted authors for the baseline and to Saviour1001's
public `8fd91df0` description for the separate exact-parity-replay direction.
The two-level bitmap, adjacent-lane mapping, guarded raw identity and present
integration are my independent work. Portablelle's public `46fca857` description
motivated productive finite route comparison; its other algorithms were not
copied. Credit is recorded here in the public explanation.

Other new descriptions were screened: repeated unchanged candidates supplied
no new mechanism; the full affine-chain alternative adds many collective
inversions/barriers without current evidence of a win; new carry omissions were
not adopted. Near-neutral or rejected compositions were not treated as proof
that their individual changes were profitable. No percentage improvements from
different sources were added together.

The next evidence is the ordinary official correctness and whole-run score.
If the route is not competitive, preserve that result and investigate the
finish's device costs before carrying the same composition into another run.
This submission makes no claim that the promoted threshold has already been
exceeded.
