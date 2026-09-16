# Subset: measured runtime choice between small and wide fixed-base tables

Update: PR60 has since been promoted at **451135044** verified candidates/s,
2.47% above the prior frontier440270249. Pinning PR74 remains pending.
The adaptive candidate itself is still unsubmitted and GPU-unmeasured.
Preparation-time pending statements below are historical.

Draft research note, not an uploaded submission. Effort: xhigh. GPT 6 Astra in
Codex implemented and checked this experiment. Grok 4.6 (CLI model metadata
`grok-4.6-build`) and Gemini 3.8 Flash High provided read-only source reviews.
Muse Spark 1.3 Contributor Free in OpenCode searched primary public research
with curated experiment history. Its reviewed suggestions in these rounds
were duplicate or deferred; no novel implemented mechanism is attributed to
them. Model agreement is not correctness or performance evidence.

## Starting point and attribution

This experiment extends our subset PR60, submission
`65fb673d-5014-4ee8-866d-97d972e7b1f0`, source
`f31dcba6b5a3de04a28e9dcf47b336bad8278fa2`. Its production/audit include
closure fingerprint is
`44f9d5808add05e204b169194af8893dc91a9c01d31448362b0e01c667c9fd06`.
PR60 remains pending at preparation time; there is no official score for it.
The current promoted subset comparison is 440270249 verified candidates/s,
submission `41dd77a6-9b18-4e88-a3bf-3cf05f5ef985`, promoted commit
`8c5cd11547d9d3e58b8c303a87cb6c80decb8024`. Refresh before any upload.

PR60 retains substantial public contributions from jacklightChen, DPZZxlz,
nullforest8200, alvaroborras and Meganpark980320, along with the earlier
promoted epoch work. Its public note documents PR36 streamed recoding and
deferred coordinates, PR40 ranked specialization, and the external product
tree/direct XYZZ recovery lineage. Preserve that attribution and the original
GPL notices and COPYING. This experiment does not claim those mechanisms as
new. Reassess promotion status for the required coauthor list at submission.

The wide runtime-base geometry and bounded table builder were transferred
from our pinning PR74, submission `66c031c6-16e4-484e-ad4d-0e7e0ef3f38f`,
source `154b2bd97a9c918ac3d992c31a6cf7901aeab1a0`. Its own public note
preserves the PR24/46/53/64 lineage. It too remains pending. A pinning score
or a different author's GPU measurement would not establish this subset
executable's throughput.

The frozen wide-prefetch donor has fingerprint
`d0174fa761b852b0f30f71246dd3e43b3e1a053dc9c011b8a4508d246eb0af18`.
The complete combined research candidate has fingerprint
`c8daa257dd1d3f3a1483eaa8047aa3c0af7a7b309103823a3ab959d43a46c9d6`.
`provenance.json` records its full production/audit include closure. Only the
isolated candidate was changed; the queued PR60 and PR74 source is preserved.

## Mechanism and expected tradeoff

The small method uses sixteen signed 16-bit windows and 32 MiB of planar
coordinates. The wide method uses six 26-bit and four 25-bit windows with a
16 GiB interleaved coordinate table. Both tables depend on the current
problem's neg_r_inv base, and both recode the live SHA-derived scalar.
No known seed, preimage, hit or benchmark answer is cached.

The wide fixed-base chain performs 60 multiplications and 18 squarings instead
of 102 multiplications and 30 squarings. These are only the point-chain
counts. They exclude message hashing, recovery, inversion, checkpoint traffic
and startup. Logical lookup bytes fall from 1024 to 640 per candidate, but
actual memory traffic may increase because a 16 GiB random table cannot
inherit the small table's potential cache reuse. Fewer field operations alone
therefore do not justify a speedup claim.

The combined executable compares the two methods on the actual GPU. Startup
always builds the small table. In ranked short-epoch mode it also builds wide
when free device memory after small allocation covers 16 GiB plus a 2 GiB
search reserve. The wide builder processes bounded one-million-entry tiles,
uses batched runtime ladders and frees its temporary scratch before search.
Sampled table entries are checked against OpenSSL. The small table uses the
prior builder and spot-check/fallback behavior. Generic modes use small.

The ordinary ranked loop runs six distinct real batches in S,W,W,S,S,W order.
The first two warm up the methods; the remaining W,S,S,W measurements reduce
simple monotone ordering bias. CUDA event intervals cover the epoch producer,
point preparation, checkpoint/root inversion and recovery/hit pipeline on the
same stream. Total GPU milliseconds divided by total candidates supplies each
method's comparison value, including partial batches. Wide is chosen only if
its time per candidate is more than 2% lower. Ties and invalid timings select
small. No events are created when the wide path is unavailable.

These are search batches, not discarded calibration work. Every iteration
advances epoch_base once by its actual block count, includes candidates in the
ordinary counters, and retains hits through the ordinary output path. The
existing 64-record host cap remains unchanged. Host dispatch explicitly chooses
one of two prepare kernels with the appropriate pointer layout. Both use the
same state transport, inverse hierarchy and finish kernel. There is no table
geometry branch inside each candidate's arithmetic chain.

## Validation completed on this Mac

No NVIDIA GPU is available. This is CPU and native compiler evidence only.
Each CPU chain branch executes extracted source with OpenSSL field arithmetic
and independent scalar multiplication as the curve reference. Each passes
12769 recoding cases, 414 curve chains and 822 recovered-key comparisons. The
small actual planar loader passes 608 cases; wide passes 380 interleaved loader
cases, including large offsets. A further 3312 checks compare prefetch targets
with the next actual logical load. Sparse virtual mappings are used for the
large-table loader test, not a resident 16 GiB host allocation.

The actual timing policy, launch-dispatch function and ranked host loop are
projected into a C++ test with mocked CUDA computation/events. Twenty scenarios
exercise both winners, unavailable wide, invalid timing, one/full/multiple/tail
batches, contiguous ranges, dispatch layout and artificial hit output. Ten
policy cases cover ties, strict threshold, candidate-weighting and invalid
inputs. Event handles are checked for cleanup after short and completed runs.
Mocked hits exercise host routing, not real GPU rare-hit computation.

The production and audit translation units compile with native CUDA 12.8.93
in a local ARM Linux VM. The production sm89 and default-flag builds pass;
the audit sm89 build passes. Small prepare uses 128 registers and 24 KiB
shared memory with 8-byte spill stores/loads. Wide prepare uses 128 registers
and 24 KiB shared memory without spills. The common finish uses 80 registers,
24 KiB shared and no spills. These are compiler resources, not stall counters,
measured occupancy, latency hiding, GPU correctness or verified throughput.

Run from the benchmark work directory:

```sh
python3 -B candidates/subset/research/adaptive_tables/check_adaptive.py
python3 -B candidates/subset/research/wide_windows/check_wide.py --source candidates/subset/research/adaptive_tables/candidate --report candidates/subset/research/adaptive_tables/wide-cpu.json
python3 -B candidates/subset/research/wide_windows/check_wide.py --source candidates/subset/research/adaptive_tables/candidate --small --report candidates/subset/research/adaptive_tables/small-cpu.json
```

Source-bound reports are collected in `validation-summary.json`. The generator
verifies donor fingerprints and refuses to overwrite an existing candidate.
The existing real CUDA build helper snapshots the source closure to a clean
temporary tree; old wrapper binaries are not used as evidence. The local
compiler VM is stopped between builds.

## Review corrections and remaining decisions

Grok's review led to an explicit required mode argument and destruction of
timing events on early exit. Both changes are included in the final identity
and rechecked. Gemini's zero-epoch crash scenario does not arise through the
actual ranked guard: n_epochs is C(137,6), rather than a free input of zero.
The host-loop test should not be mistaken for an arbitrary-length public API.

The two reviewed resource issues have been repaired. Only an optional wide
cudaMalloc returning cudaErrorMemoryAllocation falls back to small, after
clearing that recoverable error state; unrelated CUDA errors still fail.
When small wins, wide memory is released after the existing synchronization.
Twenty actual-loop mock scenarios and seven allocation scenarios check this,
and source-bound CPU and native production/audit checks were repeated. Event
handle copies are explicitly prohibited. These repairs do not establish speed.

Runtime selection excludes table construction, host I/O and startup. It cannot
guarantee total process performance at least as good as the small-only control.
Its short measurements can be noisy and may not represent sustained thermal
or cache behavior. The prefetch donor's SASS places hints late, so zero spills
are not proof of sufficient memory overlap. These limits prevent calling the
candidate a measured improvement or expecting leadership solely from counts.

PR60 has completed; preserve pinning PR74. Use official results, the current
public frontier and exact pending mechanisms to decide whether this experiment
is the strongest successor. No score is fabricated and no submission is made
by this draft. Any later upload requires a fresh identity, appropriate checks,
updated public note and an archive within the track's 8 MiB compressed cap.

## New frontier after preparation

PR62 was promoted at477182283, commit106a6826ecf8998e684169f46e8de9dce1cf3f0f.
Its mixed64MiB table is now an evidence-backed alternative. The isolated full
control in ../mixed_windows uses this geometry in our external-inversion
pipeline and passes CPU/native checks, but has no GPU score. This draft remains
unsubmitted: integrate/compare the new control before choosing the successor.
