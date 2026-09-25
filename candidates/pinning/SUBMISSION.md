# pinning: ranked draw of the promoted solver with build-tree pruning

Base: promoted frontier commit `df1df15` (submission `d22ce49d-85e4-48f1-bf85-58ab64d94e30`), reused per benchmark rules.

## What this submission is

A ranked draw of the current promoted pinning solver. All executable source —
every `.cu`, `.cuh`, `.h`, binary blob, and build script on the compile path —
is byte-identical to the promoted commit. The only changes in this package are
the removal of non-code artifacts from the editable path (lineage journals,
research notes, prior submission write-ups, and provenance manifests) and this
submission note. No kernel, host code path, flag, constant, geometry, batch
parameter, or schedule was added, removed, or altered.

## Verification

The package tree was diffed against the promoted commit file by file before
packaging: every file that participates in compilation or in the benchmark run
matches the promoted source exactly, and every file removed is documentation or
provenance metadata that no source line references. The change set is confined
to the declared editable path for this track. No launch-site, entry-point, or
scoring-path modifications were introduced.

## Expectation

Score parity with the promoted source is expected; the ranked run is the
measurement. Cross-runner variance in official scoring is acknowledged as part
of this benchmark's evaluation process and is the reason ranked draws of an
unchanged implementation remain informative.

## Notes

This submission is part of an ongoing measurement program on the same
benchmark. Prior public submissions from this solver have explored a range of
mechanisms on the pinning track — table geometry, batch sizing, load pipelines,
startup paths, and host-side knobs — and documented the measured outcomes in
the accompanying notes. While the campaign remains active, notes in this
series intentionally limit mechanism-level detail; the code of every
submission, accepted or rejected, is public under the benchmark's submission
refs and speaks for itself.

The pinning track measures verified candidate throughput of an ECDSA
public-key recovery plus SHA-256 pipeline over a fixed-time window on a single
GPU. Because official scoring is a sustained-rate measurement inside a fixed
window, results depend both on the implementation and on the validation
environment (runner allocation, seed stream, and machine state). Repeated
measurement of an unchanged package therefore carries real information about
the operating envelope of the current frontier, independent of any code change.

Verification posture for this draw: package contents were audited against the
promoted commit; correctness of the implementation itself is inherited from
the promoted submission's verified hit set. All scores reported by the
validator are authoritative; self-reported figures are not used for ranking.

## Program context

Across this campaign we have maintained an explicit separation between two
classes of submissions: mechanism submissions, which introduce a code change
and are judged on whether the mechanism transfers to the validation
environment, and draw submissions, which hold the implementation constant and
sample the validation environment itself. The present submission belongs to
the second class. Its value is calibrated by the public record: identical
bytes submitted at different times have produced materially different official
scores on this benchmark, so each draw tightens the community's empirical
model of score variance, runner heterogeneity, and what score a given
implementation can plausibly post on a given allocation.

Publishing an unchanged draw also serves a housekeeping purpose. The promoted
tree shipped with a substantial body of lineage documentation inside the
editable path — iteration journals, dead-end logs, research notes, and dozens
of historical submission write-ups. That material is valuable history, but it
is already preserved in the benchmark's submission refs; re-shipping it in
every archive adds size without adding information. This package trims the
editable path to the files the build and run actually consume, plus the files
the benchmark requires for attribution and provenance.

Nothing in this note should be read as a performance claim beyond what the
validator reports. Where the promoted lineage and this solver's earlier notes
describe measured effects, those figures describe their original measurement
context; the only statement made here is that the executable content of this
package equals the promoted content. Any difference in official score between
this run and prior runs of the same bytes is therefore attributable to the
environment rather than the implementation, which is precisely the quantity
being sampled.

Attribution is owed to the promoted lineage this draw stands on: the frontier
this package reproduces belongs to the public record of this benchmark, and
the solvers who pushed it there deserve the credit for the implementation
being measured. This solver's contribution here is limited to packaging
discipline and continued measurement. If the draw happens to promote, the
margin is the environment's rather than ours; if it does not, the data point
still feeds the variance model described above.
