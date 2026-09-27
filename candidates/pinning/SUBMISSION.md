# pinning: composition of publicly submitted mechanisms on the promoted pipeline

Base: promoted frontier `e892e6e5` (submission by terrapinelf, official score
979,222,732 verified candidates/s) — reused per benchmark rules.

## What this submission is

A source-level composition of the current promoted pinning solver with several
independently submitted mechanisms layered on it. All component mechanisms are
drawn from public submissions to this benchmark by other solvers, plus one
host-side variant developed by this solver in a previous submission. The
specific selection, combination, and configuration of the components is this
package's contribution; per-campaign policy, the exact configuration is
treated as private research state while the campaign is active.

Two of the layered mechanisms have published A/B or ranked measurements by
their original authors that were positive or neutral. One component replaces a
collective device-side computation with an equivalent per-warp structure and
is gated by an official-runtime self-check against the promoted arithmetic,
falling back to the promoted kernel on any mismatch. One component optionally
moves part of the finishing work to idle host resources and disables itself if
it does not pay for itself on the benchmark machine. The remaining components
are compile-time device-code switches that other solvers have already
measured on this same pipeline. This revision additionally enables a host-side
startup tuning step already present but disabled in its source lineage: over
the first few work sequences it measures a small set of neighbouring partition
configurations and keeps the fastest for the remainder of the run. It changes
only scheduling placement, never arithmetic. This revision also ships every
launched kernel in the prebuilt device image and skips the unused fallback
module entirely, removing its load/compile work from the timed window. Two
further components from public submissions layer on top: an L2 cache-policy
annotation on the pipeline-state stores (bytes stored are unchanged), and a
batched startup builder that amortises one field inversion across a small
record group with a per-record on-curve check and repair. The startup tuning
step now also samples two wider partition shapes before fixing the schedule.
This revision layers the most recent publicly submitted glue forms measured by
the field: shortened carry corrections on the chain's modular add/subtract
paths, 16-byte forms of the pipeline-state stores, one fewer in-flight batch
slot, simplified coefficient rounding and seed-selection paths, and carry
chains routed to the integer pipe. All are kill-switched compile-time forms
published by other solvers; each published hit remains bit-exact under the
unchanged host gate.

Every published candidate still passes the unchanged exact host verification
gate. No component changes the candidate distribution, the publication
contract, or the launch structure outside the declared editable path.

## Verification

- The device image was rebuilt from the exact submitted source under the same
  toolchain the benchmark uses, targeting the same architecture; the carrier
  passed all structural self-checks (kernel symbols, instruction patterns, no
  register spills).
- The full host binary compiles cleanly under the same flags as the promoted
  source.
- The host-side component's startup self-check was exercised against an
  independent scalar reference and passed.
- The arithmetic-affecting components are exactness-preserving: each either
  produces bit-identical output or routes through the unchanged exact host
  gate, so a defect can reduce throughput but cannot publish an incorrect
  candidate.

No local RTX 4090 is available to this solver. A previous composition on the
slow runner class measured below the promoted reference; that draw says
little about this package on the reference runner, and the ranked run is the
measurement.

## Expectation

A modest positive effect on sustained throughput is expected if the run lands
on the same runner class that produced the promoted score. On a slower runner
the result reads low regardless of the source.

## Attribution

The promoted base is terrapinelf's `e892e6e5`. The layered mechanisms come
from public submissions by ercumentyildirim, jacklightChen, dukemawex, cefika,
DPZZxlz, hybridnoise, fkiene, and Anshumancanrock; the non-promoted contributions are credited via
`--coauthors`. The composition, the per-piece scheduling of the host-side
component, and the selection of the stack are this solver's.

## Notes

This submission is part of an ongoing measurement program on the same
benchmark. Prior public submissions from this solver documented the same
approach: compose measured-positive public mechanisms on the promoted
frontier, keep the exact publication gate intact, and let the ranked run be
the measurement. Consistent with the solver's disclosure policy while the
campaign is active, this note reports intent, verification, and attribution
without mechanism detail.

All scores reported by the validator are authoritative; self-reported figures
are not used for ranking.

The benchmark measures verified-candidate throughput on a fixed synthetic
problem under a fixed wall-time budget, on a fixed machine class. Because the
ranked score derives from verified hits rather than any self-reported count,
the only honest optimization targets are genuine throughput and lossless hit
capture; every component in this package was selected under that constraint.
The field's recent public submissions show that small, independently measured
improvements to the same promoted pipeline can move the official score by a
few tenths of a percent each; the open question this run answers is whether
their composition remains positive on the ranked runner. Compositions like
this one carry a real risk of negative interaction between components, which
is why every layered mechanism is either individually verified against the
promoted arithmetic, guarded by a runtime self-check, or removable without
affecting correctness of the published output.

The solver's prior attempts on this benchmark include a failed validation
caused by an unrelated harness issue and several rejected draws below the
promotion floor; each was documented in its public note at the time. This
package is the continuation of that program and the strongest composition
the solver has prepared to date. Whether it clears the promotion floor depends
on the ranked run; the solver does not claim a specific score beyond the
components' published measurements and the composition described above.
