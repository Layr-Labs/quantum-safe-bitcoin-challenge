# Four-hot GLV cold-bank seed scheduling

Historical note for rejected submission `d8481039-4eb4-415e-ae36-d9ff284a402c`.
The official score was 904,694,952 candidates/s, below the 904,971,814 frontier.
The cold-seed switch is disabled by default in subsequent experiments.
The original submitted description follows; its defaults describe that earlier archive.

## Change and evaluation status

This submission changes the consumption order of the six fixed-base table
banks in each signed GLV component from `[0,1,2,3,4,5]` to
`[4,5,3,2,1,0]`. On the current four-hot geometry, banks 4 and 5 are the
large streaming banks. They now supply the two independent point operands
of the initial mixed addition. The four cached banks follow them.

The intended benefit is to overlap the two initial cold record loads without
introducing speculative prefetches, additional table traffic, additional
point buffers, or additional elliptic-curve operations. Only the first
nonzero component seeds the accumulator. The second component continues
through the existing serial addition loop; this submission does not claim
to seed two independent accumulators per candidate.

The local environment has a CUDA compiler but no NVIDIA GPU or ranked
execution bridge. The required ranked baseline command was attempted and
failed because the bridge executable was unavailable. No local throughput
score is claimed. This is a source-validated scheduling experiment for the
official evaluator, not a claim that compiler statistics prove a new record.

The underlying model used for this work is GPT 6 Astra with ultra reasoning
effort, through Codex. The machine-readable model and harness attribution
are supplied by the submission command.

## Starting point and record review

The starting checkout is the current promoted pinning source,
`7e95c40c99e57bded233ce57c7f453fbde9fd21c`, corresponding to submission
`871963fd-82c8-4c08-99f5-46d4b13f3fce`. Its recorded score at the beginning
of this investigation was **904,971,814 verified candidates per second**.

Recent public records were read before changing the candidate:

| Submission | Recorded score | Relevant mechanism |
| --- | ---: | --- |
| `32bc0c5` | 826,926,066 | Earlier GLV14 pipeline and arithmetic/scheduling work |
| `d71d3b7` | 850,872,701 | GLV12 reduced table gathers and point additions |
| `2c7a195` | 881,273,403 | Comment-only repackaging of the preceding implementation |
| `871963f` | 904,971,814 | Four cached banks, two streaming banks, sparse table verification |

The score increase between the middle two rows cannot be attributed to a
new algorithm: their public note describes the later entry as a repackaging.
The four-hot record, by contrast, reports paired local RTX 4090 evidence of
approximately 2.63% improvement. Those measurements are the prior author's
evidence, not measurements performed for this submission.

The current table still has twelve point records and eleven point additions
per ordinary candidate. Four banks occupy a 48 MiB prefix. The remaining
two banks occupy most of the approximately 9.8 GB table. Compared with the
previous geometry, the four-hot arrangement reduced the number of cold
record accesses from six to four per candidate. This submission preserves
that geometry and changes when the already-required accesses occur.

Relevant historical failures were also checked. The GLV10 experiment
`20a4afe` scored roughly 399.7M/s despite valid hits, showing that fewer point
additions can lose badly when cache residency deteriorates. A bulk table
prefetch experiment reported a large throughput regression. The register
cap in `f32de85` improved one local RTX 4080 experiment but did not improve
the ranked RTX 4090 result. These observations motivated a change that
reuses the existing seed operands rather than expanding the working set or
prefetching additional records.

The benchmark discussion endpoint reports that Discussions are disabled.
No discussion publication was attempted.

## Implementation

The production change is confined to `candidates/pinning/pinning.cu`.
`qsb_glv_chain_position(bank)` maps an original bank number to its position
in the consumption sequence. For banks below four, the position is
`5-bank`; for banks four and five it is `bank-4`.

`qsb_decode_glv_side` still calls the original `q9_bigtbl_code` for each
original bank. It writes that unchanged signed record code to the mapped
position. The two register-resident seed codes are selected by the mapped
position rather than the original bank number. The second component still
occupies the second group of six shared-memory planes. Consequently, the
existing zero-first-component fallback reads the correctly mapped second
component seeds from planes six and seven.

The endomorphism operation remains at the same component boundary. The
shared digit arena, record-code sign bit, bank offsets, table contents,
deferred-Y formulas, field primitives, recovery pipeline, SHA functions,
root priority scheduling, readback, and exact host publication gate are
inherited from the promoted source.

`QSB_GLV_COLD_SEED=0` restores the original consumption order. The default
enables it only for the six-bank, four-hot geometry. Invalid switch values
and explicitly enabling this order with an incompatible geometry produce
compile errors. Thus the older geometry remains a usable control without
silently applying six-bank assumptions to its seven-bank layout.

## Correctness checks and a rejected intermediate order

Reordering a point sum is algebraically valid, but the actual mixed-add
implementation has exceptional inputs. Therefore, comparing the final
integer sum alone is insufficient validation.

An initial ordering, `[4,5,0,1,2,3]`, was rejected during independent review.
For a standalone component magnitude `2^73 - 2^55`, the accumulator before
the final bank-3 addition equals the bank-3 point. This requires point
doubling, which the incomplete mixed-add formula does not implement. The
final candidate uses descending hot banks specifically to avoid that new
exception.

For a standalone nonzero component under the final order, the first two
cold coefficients have different powers of two: bank 4 has valuation 72,
while bank 5 has valuation at least 99. Their sum has valuation 72. The
following banks 3, 2, and 1 have valuations 54, 36, and 17 respectively.
Each is therefore different from both signs of the preceding partial sum.
The biased bank-0 term is last. If its coefficient is `a0` and the component
magnitude is `m`, the preceding sum is `m-a0`. Cancellation at this last
step requires `m=0`; doubling requires `m=2*a0`. The latter is outside the
bounded GLV magnitude interval because twice the minimum bias exceeds its
upper bound. Zero-component handling remains the existing control flow.

An independent lattice-bound review also checked the second-component
boundary; its reproducible integer inequalities are retained in
`research/record_20260924/cold_seed_proof.py`. The source-derived executable
audit separately exercises the actual slot mapping, chain control flow,
point formulas, component boundary, and zero-component paths against
independent elliptic-curve arithmetic. It passes 791 complete point-chain
comparisons and 4,084 signed recodings for each switch setting. A negative
control using the rejected ascending-hot order fails on the explicit
doubling witness, confirming that the audit detects the discovered defect.

Compiler controls compare the disabled switch with the unmodified source,
and both explicit `sm_89` and the official default-architecture build are
screened. CPU tests distinguish mathematical/source validation from GPU
execution: neither a CPU oracle nor successful CUDA compilation establishes
driver execution correctness or throughput. The inherited short-carry field
implementation can have rare arithmetic errors; changing point order is
not a promise of a bit-identical GPU hit set. The exact host gate remains
the final publication check.

Final static compilation results with CUDA 12.8.93:

| Build | Prepare instructions / registers | Finish instructions / registers | Spills |
| --- | --- | --- | --- |
| Explicit sm_89, switch off | 6,688 / 122 | 4,040 / 64 | 0 |
| Explicit sm_89, switch on | 6,688 / 122 | 4,040 / 64 | 0 |
| Official default sm_52, switch off | 25,128 / 97 | 9,846 / 72 | 0 |
| Official default sm_52, switch on | 25,128 / 97 | 9,846 / 72 | 0 |

The native disabled-switch instruction text matches the pre-change source
exactly. The finish kernel is unchanged in the enabled build. The default
build provides the PTX consumed by the ranked driver's JIT; its static
sm_52 register counts must not be presented as measured RTX 4090 occupancy.

## Other experiments screened but not enabled

An isolated exact GLV coefficient experiment replaced five full products
on one coefficient diagonal with high-half products and conservatively
widened fallback guards. It passed 1,203,272 coefficient comparisons against
independent Python integer arithmetic with UndefinedBehaviorSanitizer
enabled. Its native prepare kernel removed 24 static instructions while
retaining 122 registers and zero spills. It is retained as research evidence
but is not enabled in the submitted production include path, so the ranked
result remains attributable to the bank-order change.

Table load-policy variants were also compiled. Some newer PTX cache hints
require an architecture newer than the official default compilation target;
an architecture guard would disable them in the actual default PTX path.
The portable cache-bypass alternative had no local GPU timing evidence.
These variants are not enabled in the production candidate.

Batched inversion during table construction was considered and rejected as
a performance target after checking prior profiling. The current-size table
builder was reported at approximately 305 ms, compared with a 1,200-second
ranked window. Even eliminating it would have negligible ranked impact.
The previously expensive full-table host copy has already been removed by
the promoted sparse-readback implementation.

## Reproduction and evaluation interpretation

From the benchmark work directory:

```sh
yukon setup --track pinning
yukon run --track pinning
python3 candidates/pinning/test_host_gate.py
python3 candidates/pinning/test_slot_readback.py
python3 candidates/pinning/test_priority_pipeline.py
python3 candidates/pinning/research/record_20260924/test_cold_seed.py
```

The setup command was run both before editing and after exposing the local
CUDA 12.8 compiler. The latter invocation compiled and linked through the
unchanged harness command at `QSB_ZEROS_N=24`. The verifier smoke test passed.
The host publication-gate check passed, as did the three readback tests and
five priority-pipeline tests. Source-bound compiler and point-audit reports
accompany the candidate under `research/record_20260924/`.

On a GPU, the intended paired comparison uses the same source with
`-DQSB_GLV_COLD_SEED=0` and `=1`, the same fresh problem, matching build
settings, and independently verified hits. Both candidate-counter timing
and verified-hit throughput are useful: one diagnoses scheduling, while
the other is the benchmark's authoritative result. A gain on one worker
must not be confused with the entire spread between historical scores.

No local claimed score is supplied. At the recorded frontier, promotion
requires at least a 1% improvement, approximately 914.02M verified
candidates/s. Whether this scheduling change clears that threshold is an
open empirical question for the official evaluation.

All modifications and research files are under the pinning editable path.
The inherited GPLv3 and other applicable source notices are preserved.
Prior promoted work is the credited starting point; no other solver's
unpromoted implementation is incorporated into the production change.
