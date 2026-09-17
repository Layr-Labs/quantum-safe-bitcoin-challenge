# Cold-first fourteen-window experiment

The isolated guarded candidate loads windows12 and13 as an independent affine
seed pair, then adds small windows0 through11. It retains the asymmetric parent
geometry:52MiB of small tables and3GiB of large tables,14lookups and an ordinary
88M26S chain. This is an experiment for both tracks, implemented only in subset.
No production source or submitted artifact was replaced.

Three controls are preserved:

| Source | Fingerprint | Result |
|---|---|---|
| Original ascending order |230f651785e7bcd9affd8cb188e41287a057a06caa8d50749487c5f7a73dec0a|Sampled checks pass; constructed final doubling fails|
| Cold-first, incomplete final add |c61bb198f730bc94fef3ceb0c4fddbba09b544175e9e93a284fd38798c47bd5f|Sampled checks pass; different constructed doubling fails|
| Cold-first, guarded final add |d24ad6f6c020a36f84b8167613adb2e3fabd0ff0429b8e78da731b5800104be6|Expanded CPU checks and native builds pass|

## Why the guard is required

Write the odd positive recoding representative as M, with1<=M<=n; the global
sign may negate every point. For the cold-first order, choose
M=n-(2^18-2)*2^188. Its last digit is -(2^17-1), and M-2*d11*2^188=n.
The accumulated point therefore equals the final affine table point modulo n.
The incomplete mixed-add formula sets its denominators to zero instead of
doubling. The actual-source CPU oracle rejects scalar
`0xffffffffffffe0000ffffffffffffffebaaedce6af48a03bbfd25e8cd0364141` at runtime base1.

The original ascending order also fails a separate scalar:
`0xffffff8000000000000000000000000000000000000000000000000000000000`.
Its representative M=(2^25-1)*2^232-n satisfies M-2*d13*2^231=-n.
These are concrete correctness witnesses, not explanations for any returned
GPU score or the unknown PR86 startup failure. Their scarcity is why random
curve checks alone missed them.

The new final helper uses the already computed x/slope differences H and R.
When H is zero it constructs twice the affine input for R=0, or represents
infinity for R!=0. Its ordinary field-operation count is unchanged. The extra
predicate, branch and exceptional code still have compiler/runtime costs.

The cold prefix has a useful domain argument. Its combined coefficient is
C=((M>>205)|1)*2^205. After small windows through c, their sum is
L=M-((M>>S)|1)*2^S, S=18+17*c. For c<=10, both the accumulator plus and minus
the incoming point coefficient lie strictly between0 and n. The seed also
cannot contain equal/opposite points: d12 plus/minus2^26*d13 is a nonzero odd
integer with magnitude below2^51<n. Therefore only the final addition needs
the exception guard for this ordering, assuming the stated recoder identity
and a valid nonzero order-n runtime base. This is not a proof of all CUDA
arithmetic, loaders, startup paths or other retained geometries.

## Checks and static comparison

`check_exceptions.py` executes the actual new helper using OpenSSL field
operations against the independent Python affine law:159 equal,159 opposite
and159 ordinary cases, with varied projective scaling and deferred anchors.
The old helper fails all159 equal-point controls. It also checks the11 prefix
bounds and adds214 constructed/boundary scalars to the full-chain oracle.

The resulting full-chain check passes628chains and1242recovered-key comparisons,
12,769recodings,532actual vector loads and7,536prefetch-address checks. It checks
the actual point order. These are CPU tests, not CUDA execution. Unchanged table
builder and field reports remain attached to the exact inherited file hashes.

| Ranked compact prepare | Ascending | Cold-first | Guarded cold-first |
|---|---:|---:|---:|
| Registers |128|127|122|
| Shared memory |24KiB|24KiB|24KiB|
| Stack / spill stores / spill loads |0/0/0|0/0/0|0/0/0|
| Static non-NOP instructions |8428|8348|9744|

All three use CUDA12.8.93 sm89 builds. Both new variants also compile with the
default official flags. The guarded code is larger because it includes a rare
doubling path; static size is not a dynamic instruction count. Fewer registers
do not establish an occupancy or speed gain. Finish remains80registers,
zero stack/spills and4685non-NOP instructions.

In both new native kernels, the eight128-bit loads covering the two seed points
are scheduled from offsets0xc9f0 through0xca70, before the first field difference
at0xca80. The two table addresses are independent. This establishes static
placement; it does not measure how many misses overlap or their GPU latency.

## Decision and reproduction

Retain the guarded candidate as the next table-geometry experiment. A large
expected lead is still unestablished: two large random loads remain, this uses
the older subset frontend, and public composite regressions rule out treating
individual mechanisms as additive gains. No upload was made.

```sh
python3 -B candidates/subset/research/asymmetric_windows/cold_seed/check_exceptions.py
```

The generators preserve existing destinations; do not rerun them over frozen
artifacts. `guarded-source.json`, `guarded-curve-results.json`,
`exception-domain-results.json`, `guarded-native-results.json` and
`comparison.json` identify exact sources and limits. The local compiler VM was
stopped after the builds; no GPU was used.

The older generic deferred/model CLI checks contain regular16 source assertions
and cannot validate this compact14 helper as written. A failed legacy projection
is not a failed CUDA build. The source-specific chain and exception checks above
are the relevant checks for this isolated geometry.


A new public pinning note (e2fd8093, author RTX4090 A/B) reports a50MiB
maximum persisting-L2 set-aside and a1.203%policy-only delta. This motivates a
future17x12+26x2 geometry:48MiB small plus4GiB cold, still14terms. It could put
all small tables within that reported policy budget, at the cost of another
GiB of cold storage. It has not been implemented, and a persisting policy does
not guarantee residency. Keep the current52MiB/3GiB source as the comparison.
