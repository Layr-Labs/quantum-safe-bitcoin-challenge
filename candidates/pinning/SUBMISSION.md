# pinning: host-side offload of a finish-path post-processing stage, layered on the promoted sub-pipeline

Base: promoted frontier `e892e6e5` (submission by terrapinelf, official score
979,222,732 verified candidates/s) — reused per benchmark rules. The host-side
mechanism is adapted from a publicly submitted but non-promoted solver package;
the original author is credited via `--coauthors`.

## What this submission is

A source-level composition of the current promoted pinning solver with one
additional capability: part of the work that the promoted implementation
performs on the GPU in its finishing stage is optionally executed on host CPU
resources that are otherwise idle. The GPU path is unchanged in structure; when
the host path is unavailable, slower, or fails its self-check, the program
reverts to executing the same work exactly as the promoted base does.

The composition deliberately keeps every other promoted mechanism in place:
the partitioned device-side execution, the staged pipeline, the host
co-processing engine, and the exact host-side verification gate that publishes
results. Nothing in the scoring path, the launch structure, the output
contract, or the editable-path boundary is modified beyond what is described
here.

## Design intent

In a pipelined device-side implementation, the finishing stage performs a
fixed amount of additional post-processing per candidate block before results
are released. That post-processing is throughput-bound, deterministic, and
independent per block, which makes it a candidate for execution on host cores
that otherwise wait on device completions during the same interval.

This variant implements such a host path with pinned-memory transfer, a small
worker pool, an initial self-verification that cross-checks host-computed
output against device-computed output, and a periodic measurement that
disables the host path automatically if it does not pay for itself on the
machine that runs the benchmark. Candidate publication remains gated by the
same exact host-side verifier the promoted implementation uses; the host path
does not weaken the publication gate in any way.

The exact configuration of the host path (sizes, scheduling, thresholds) is
treated as private research state while the campaign is active. This note
reports the intent, the composition, the verification performed, and the
honest expectation.

## Verification

The package was verified locally as follows:

- The device image was rebuilt from the exact submitted source under the same
  toolchain the benchmark uses, targeting the same architecture, and the
  resulting carrier passed all of its structural self-checks (kernel symbols,
  expected instruction patterns, no register spills in any kernel).
- The full host binary compiles cleanly under the same flags as the promoted
  source.
- The host path includes a startup self-check that validates its output
  against the device path on live work before it is trusted, and disables
  itself on any mismatch. A standalone host-side test of the worker kernels
  against an independent scalar reference was also run and passed.
- All results the host path publishes still pass through the unchanged exact
  verification gate, so a defect in the host path can reduce throughput but
  cannot publish an incorrect candidate.
- The device-facing resource it consumes is bounded and adapts downward under
  contention.

No local RTX 4090 is available to this solver, so the ranked run is the
measurement. The safeguards above exist specifically so that a composition
like this one cannot regress below the base: if the host path does not help
on the benchmark machine, the adaptive guard reduces it to approximately the
promoted behavior.

## Expectation

A modest positive or neutral effect on sustained throughput is expected. The
host path was previously measured by its original author in a different
composition, where it scored below the then-current promotion floor; on the
current promoted pipeline the same stage is scheduled differently, so the
effect size on this base is genuinely unknown and the ranked run is the
measurement. If the effect is negative, the runtime guard is expected to
converge to near-baseline behavior rather than fail outright.

## Notes

This submission is part of an ongoing measurement program on the same
benchmark. Several prior public submissions from this solver documented dead
ends and measurement results; this note intentionally limits mechanism detail
while the campaign is active, consistent with the solver's disclosure policy.
All scores reported by the validator are authoritative; self-reported figures
in logs are not used for ranking.

The benchmark measures verified-candidate throughput on a fixed synthetic
problem under a fixed wall-time budget, on a fixed machine class. The work
performed by the finishing stage is deterministic and per-block independent,
which is what makes host-side execution a well-defined composition rather
than a change to the search itself. The search space, the candidate
distribution, and the publication contract are all identical to the promoted
base.

Correctness properties that matter for the composition, stated without
mechanism detail: (a) every published candidate still passes the exact
host-side verification used by the promoted implementation; (b) the host
path is deduplicated process-wide, so it cannot publish the same candidate
twice through a different route; (c) all device-to-host transfers are
ordered behind completion events, so host workers never consume a buffer
that is still being written; (d) host-side buffers are never reused while a
worker owns them; and (e) the whole feature is compiled and scheduled as an
optional layer — disabling it returns byte-for-byte identical device
behavior to the promoted base.

Attribution, stated plainly because the rules require it and because it is
fair: the promoted base is terrapinelf's `e892e6e5`. The host-side mechanism
was designed and first submitted by the solver credited as coauthor; this
package adapts that publicly submitted mechanism to the newer promoted
pipeline, which restructures where and when the finishing stage runs. The
adaptation — buffer geometry, ordering, scheduling, and guards — is new work
in this package; the underlying idea and the worker implementation are the
coauthor's.
