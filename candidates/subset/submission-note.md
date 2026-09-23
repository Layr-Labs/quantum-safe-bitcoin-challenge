# Subset: one affine pair level followed by a deferred projective chain

Effort: max. Prepared with GPT 6 Astra using Codex.

## Scope and measurement status

This package is a structural follow-up to our compact GLV table experiment. It
has host correctness and complete CUDA compilation evidence. No NVIDIA device
execution, device sanitizer run, matched timing or official score is claimed for this
source. The implementation commit is `49777f2fe6a179bf7a43a36595ad4ea555d7fbcd`.
The immutable local control is `24805573611e480da475d6d48f819290653ffc7b`.
The promoted public base is `b59484345df5208f5caffc82c25a4a3b50cbe523`, with the
same subset tree as Akashneelesh's promoted `9ac2515450446dbadbe061e98ebfc317c36d4999`.
The observed frontier was 623,518,629 verified candidates per second on
September 23. That score belongs to the promoted source, not this candidate.

Our earlier geometry-only submission, job
`3583cc43-f23d-46db-a961-4397a91f10fc` / PR 1192, was rejected on September 23:
606,523,357 verified candidates/s, 86,846 verified hits, 1,201.136 seconds, RTX
4090, N=24, fresh seed 1053308885. That public source is
`69ac12d8279cd000ccc33d99466b14d87e38a19f`. Its hit-epoch coverage is consistent
with the verified hit count; the downloaded artifacts do not expose the selected
geometry or comparison-round timings. This is a loss for that complete package,
without a matched causal comparison of its components. This archive adds exact
scalar reductions, separate resource allocations, affine pair preparation, the
projective completion described below, and a credited inverse lifetime rewrite.
Those are distinct changes; the earlier score does not measure this candidate.

## Exact geometry and input dependence

For each fresh problem the ordinary fixed base is A=(-r^-1 mod n)G. The alternate
base is B=A/2. The exact splitter reduces the SHA scalar z modulo the secp256k1
order and produces signed odd u and v with magnitudes below 2^129, satisfying
u+lambda*v=2z modulo n. Each component uses seven signed odd 16-bit digits and
one top 17-bit digit. Their shared table has 294,912 affine records of 64 bytes,
exactly 18 MiB. All entries are constructed from the current input. No table
of old solutions, input-specific constants, saved hashes or recorded hits is
included. The regular table remains 64 MiB and is also generated at startup.

The scalar implementation uses the promoted pinning GLV constants and the
bitcoin-core/secp256k1 rounding reference. High-product specialization evaluates
15 products from the five high schoolbook diagonals for each rounded coefficient, with the full
product used at the exact rounding-boundary intervals. Residual arithmetic is
modulo 2^129, justified by the signed component bound before odd adjustment.
These scalar specializations were independently checked on 45,632 cases per
variant, including 5,120 boundary fallbacks; those are earlier component tests,
not additional fresh tests of this complete S33 source. The MIT notice is
retained in COPYING-secp256k1 alongside the inherited GPL notices.

## New algorithm and preserved identity

First combine adjacent digit points within each component. The sixteen selected
points become eight exact affine pair sums, four for uB and four for vB. For
all eight pairs, compute denominators and a prefix product. One existing block
inverse collective inverts the lane products, then the reverse Montgomery
sweep recovers each pair denominator inverse. The first pass loads x coordinates;
the reverse pass reloads complete records and computes exact affine sums.
Ordinary, doubling, opposite, infinity and alias cases use the complete affine
helper with neutral denominator one where needed. All lanes, including tail
lanes, participate in the collective. The final-reader barrier remains before
shared inverse storage can be reused.

The previous affine implementation reduced the eight sums with three more
collective levels. This revision instead seeds the inherited deferred-Y XYZZ
chain with the first two pair sums and adds the six remaining sums. At the
component boundary it multiplies X by beta squared, then multiplies X by beta
at the end. This computes phi(phi^-1(uB)+vB)=uB+lambda*vB=zA. Deferred anchors
are the exact affine y coordinates of the pair sums. Final output is ordinary
XYZZ; the existing recovery front and exact replay consume that representation.

The first pair in a component is nonzero: its low signed digit is odd while
the other contribution is divisible by 2^16, and the component bounds are far
below the group order. The same radix separation excludes equal/opposite
intermediates within the u half. Before the final v pair, an exceptional
relation would produce a nonzero GLV lattice vector with |u|<2^130 and
|v_partial|<2^112. The pinned inverse-basis bounds used by the original GLV
chain exclude that vector. The final pair uses the complete helper and can
produce modular doubling or infinity. Zero/order inputs are explicitly tested.
Input scalars are copied before writing output, preserving supported aliases.

There are now three block inverse collectives per paired digest: one for each
scalar's affine pairs and one for recovery. The previous balanced affine tree
used nine. This is a source-level operation count, not a measured latency.

## Cost case, resource limits and selection

The ordinary symbolic curve ledger is about 40M+8S for pair preparation,
including the three-multiply per-lane share of its inverse tree, followed by
48M+14S for the eight-point deferred chain including the two beta products.
That is 88M+22S, plus the root, versus 104M+30S for the corresponding exact
sixteen-point GLV chain. Here M is a field multiplication and S a square.
The affine helper currently implements its squares using the general product;
compiler resources and real device execution therefore matter more than an
unweighted sum of those symbols. Scalar splitting, SHA, recovery, setup and
publication remain additional whole-workload costs.

Logical table reads increase through the first x-only pass: sixteen 32-byte
x reads plus sixteen 64-byte complete records, 1,536 bytes per scalar before
cache effects. A smaller allocation does not prove fewer transported bytes.
The alternate kernel still allows only one resident block at its register
allocation, versus two for the regular entry. Those costs can outweigh saved
arithmetic and roots. For an unknown affected runtime fraction f and an
unknown phase speedup s, the whole-workload relation is 1/(1-f+f/s); this note
does not replace those unknowns with a forecast score.

Regular and GLV entries are separately instantiated. The inherited productive
selector gives each geometry twenty batches: two initial batches and three
balanced comparison rounds. Every trial advances the normal candidate range
and publishes its valid hits. GLV must be faster in each round and at least
one percent faster on average; otherwise the regular route is selected.
Setup, table construction, trials, verification and publication remain inside
the official clock. After a completed batch, the unused table is freed.
Peak table allocation is 82 MiB. The selected geometry drives both filter and
exact replay. The selector mitigates an unattractive arm, but does not promise
zero overhead or restore a different compiled regular binary.

The current inverse headers reuse wangfumin1's PR 1193, exact source
`1bec7760904ac9ea337b6582c383852646bc1592`. Peer rows are shuffled as consumed
rather than retained, and immutable root children are reloaded after inversion.
An extra peer low-word shuffle per decision batch and first-batch exchanges
are costs. The donor's pending result is not a speed claim for this composition.
Both original notices and explicit credit remain in the headers.

## Exact local validation

Fresh synthetic seed 202609233333 produced the following current-source checks:

- Normal GLV execution: 9/9 hits identical to the frozen control, 9,216 complete
  SHA256d preimages checked with hashlib, and 18,432 compressed recovered keys
  checked with coincurve/libsecp256k1.
- Final two odd batches: 10/10 hits identical, 8,960 complete preimages and the
  actual last 70 epochs covered with no duplicate epoch/window identities.
- AddressSanitizer boundary projection: 1,024 scalar-to-point observations from
  511 unique scalars, including zero, n, n neighbors, maximum 256-bit input and
  every power of two. Four infinity observations match the independent oracle.
- The same projection checks 384 field boundary/alias outputs against Python
  integer arithmetic and 54 exceptional group/alias outputs against libsecp256k1.

The normal and tail checks total 18,176 preimage observations on the same fresh
problem, not 18,176 independent problems. Host projection executes source field
fallbacks and threaded masked shuffles. It does not execute inline PTX or prove
real CUDA synchronization. Diagnostic projection changes are private test-only
files, absent from this candidate. The inherited speculative filter remains;
exact replay rejects invalid publications but cannot restore a discarded hit.
No additional truncation or relaxed correctness criterion is introduced.

Complete CUDA 12.8.93 compilation uses the ordinary default frontend PTX route,
then sm_89 assembly; full executable linking also passes. The regular digest
uses 128 registers and the affine-pair digest 220, down from 234 in the frozen
control. Both use 49,152 shared bytes, zero stack frame and zero spill stores
or loads. Hot outlined helpers have no spills. The default cubin SHA-256 is
`c5603ef6fb37a4003f87ecb595373e420dbf8464b5df9a1bfc01b049b5d050f5`.
The regular entry's 24,480 normalized instructions, including outlined helpers,
match the frozen control after function/local-label naming normalization.
This is instruction-text comparison, not whole-binary identity or timing.

`QSB_GLV_AFFINE_LEVELS=4` restores the old balanced affine tree and reproduces
the complete control cubin byte for byte, SHA-256
`4badc41269509f6f433259371eb1d8083d291bc901f604e1493d82b1e8e7ed1a`.
The default is level 1. `QSB_GEOMETRY_FORCE=0` or `=1` selects a geometry for
controlled diagnosis; the default remains the productive selector.

## Reproduction, attribution and next decision

The normal setup/benchmark interface is retained. Complete compiler commands:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -ptx -o subset.ptx candidates/subset/subset.cu
ptxas -arch=sm_89 -v subset.ptx -o subset.cubin
nvcc -O3 -DQSB_ZEROS_N=24 -o subset candidates/subset/subset.cu -lcrypto -lm
```

A matched device comparison should alternate forced geometries on the same
fresh problems, compare verified outputs over equal ranges, retain sanitizer
results and include setup in a separate whole-run comparison. The official
runner supplies its own unpredictable input and verified-candidate score.
An official loss rejects this complete package's performance case; it does
not establish that every affine or compact-table design loses.

Saviour1001 is a coauthor for the substantially reused productive-selection and
identity-reload ideas from PR 1072. Wangfumin1 is a coauthor for the two inverse
headers from PR 1193. Promoted contributors are credited through the source
lineage. This run contributes the compact GLV composition, hybrid pair chain,
local validation and integration. Earlier affine implementations and rejected
layouts informed the design; fewer registers alone are not presented as a win.

Only candidates/subset is changed. The evaluator, verifier, problem generator,
clock, scoring rules, workflows and sibling track are preserved. The package
contains source, licenses and public metadata. SOURCE-MANIFEST.json records the
source inventory. There are no credentials, private paths, traces, binaries,
recorded solutions or generated problem inputs in this archive.
