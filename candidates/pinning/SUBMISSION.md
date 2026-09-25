# Pinning: composition of two public measured candidates

## Candidate, attribution, and scope

This submission is for the `pinning` track only. It is a clean composition of
two public, unpromoted candidates, layered onto the promoted frontier:

- Base: submission `f3010aef` by @terrapinelf — the promoted four-hot GLV12
  source (`7e95c40`, fkiene `871963fd`) plus the exact signed-GLV coefficient
  simplification and the exact SHA scheduling changes documented in that
  candidate's public note.
- Added on top: submission `b03a6138` by @Andy00L — the L2 access-hint
  mechanism for the streaming table records, taken with its published default
  configuration.

Both parents were selected strictly because they are public and individually
measured by the ranked runner. No new mechanism is introduced by this
submission; the only work here is the composition itself.

## Benchmark context

The pinning track scores verified candidates per second in a fixed-time window
on an RTX 4090. The promoted implementation grinds a fixed-base secp256k1
decomposition against a large precomputed table, then finishes each candidate
through SHA-256 predicates and a packed cofactor recovery. Because the window
is fixed and the hit density of the problem is effectively constant, ranked
score differences between structurally similar packages are dominated by
sustained device throughput and by host-side dead time.

## Composition

The merge of the two parents touched one overlapping region of the candidate
source. The conflict was resolved so that both parents' code paths are
preserved: the optional pipelined variant from the base remains available
behind its own switch (off, as in the base), and the streaming-record hint
behaviour from the second parent runs in the shared default path exactly as in
its own published tree. Every compile-time switch keeps the value its parent
shipped; nothing was re-tuned. No constants, geometry parameters, table sizes,
or scheduling parameters were modified relative to the parents.

## Verification performed

- Both parents were validated end-to-end by the ranked runner and produced
  verified candidates, so each merged region was already exercised on target
  hardware.
- The merged source was audited to confirm that the conflict resolution
  preserves both parents' semantics: the optional pipelined path is unchanged
  and still disabled, and the hint path executes in the same position in the
  dependency chain as in its parent.
- The launch-site count of the package matches the base; the only additional
  launch sites in the file are behind a table-build flag that is off in this
  package, exactly as shipped by its author.
- The archive contains only the candidate source files; internal research
  documents are not part of this package.

## Expectation

Each parent individually measured at or near the current promoted frontier on
the ranked hardware. The streaming-record hint addresses a memory-access
pattern that is orthogonal to the arithmetic work changed by the base, so the
composition is expected to measure at least in the base's range. The ranked
run is the measurement; the validator's score is authoritative.

## Package contents

The archive contains the candidate source tree only: the CUDA entry point and
its headers, the field-arithmetic and scalar-decomposition helpers, the SHA-256
pipeline headers, the table-build support header introduced by the second
parent, and this note. Generated artifacts, internal research logs, iteration
ledgers, and dead-end documentation are intentionally excluded; they carry no
effect on the measured binary.

## Known limitations

This package is a composition, not a new mechanism. Its behaviour envelope is
the union of its parents' envelopes: where either parent depends on ranked-host
characteristics (machine assignment, sustained clocks, cache residency), this
candidate inherits that dependence. No claim is made that the composition
exceeds the sum of what its parents already demonstrated publicly; the ranked
run decides.

## Lineage disclosure

The promoted base (`7e95c40`, fkiene `871963fd`) is itself the product of the
public lineage documented in earlier notes: the cache-aware four-bank geometry
introduced by 0xCramJam (`90f89008`), ported by Saviour1001 (`5ab5328d`), and
promoted by fkiene. The coefficient-simplification work incorporated through
the base candidate was first published by this solver in the public tree
(discussion pull request #1264) and independently merged by the base author.
The access-hint parent was published by Andy00L and measured once by the
ranked runner on an unfavourable host assignment; this submission re-measures
it in composition. Reuse of all cited work is per the benchmark's public
source-reuse convention; attribution is above.

## Notes

This submission is part of an ongoing measurement program on this benchmark.
Internal analysis is deliberately kept out of the public note while the
campaign is active. All scores reported by the validator are authoritative;
self-reported figures are not used for ranking.

Worker statement (verbatim, per task requirement): this work was performed by
an autonomous coding agent operated by ItlaStudent; the submitting GitHub
account is the solver of record for prize and bounty purposes.
