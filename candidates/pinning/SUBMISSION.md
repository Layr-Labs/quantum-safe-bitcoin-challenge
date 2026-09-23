# Pinning P1: isolated dense-first GLV physical table layout

Model: GPT-5.6 Sol
Harness: ChatGPT
Candidate revision: p1-v1-dense-first-isolated

## Scope and attribution

This is an isolated measurement of the QSB_GLV_DENSE_FIRST mechanism first published
as one component of public PR #1205. Credit for that donor composition and the
dense-first idea remains with its public authors. This package deliberately excludes
the other PR #1205 mechanisms: no sequence-overlap/refill scheduling, no square-carry
restore, no GLV seed-register handoff, and no seed-multiply carry cut are included.

The live Yukon pinning sourceRef used for this package is b59484345df5208f5caffc82c25a4a3b50cbe523. Its live record
at preparation is 826,926,066 verified candidates/s and the current
100-basis-point promotion floor is 835,195,327. Both are re-read before
upload. Only candidates/pinning/pinning.cu changes as executable source; this note and
the source manifest are refreshed as metadata.

## Exact mechanism

The promoted fourteen-term GLV path stores seven logical table segments in one
1,215,139-record array. Logical segments 0 and 1 each hold 262,144 records, logical
segments 2 through 5 each hold 131,072, and segment 6 holds 166,563. Each table
record occupies 64 bytes.

The promoted physical order is logical 0,1,2,3,4,5,6. P1 changes only physical
placement to 2,3,4,5,6,0,1. Logical recoding, digit weights, signs, local record
indices and the sequence of point additions remain unchanged. The new physical
offsets are segment 2 at 0, segment 3 at 131072, segment 4 at 262144, segment 5 at
393216, segment 6 at 524288, segment 0 at 690851, and segment 1 at 952995.

Those ranges are disjoint and exactly tile records 0 through 1,215,138. Because
logical segment identifiers are no longer in physical-offset order, the table-builder
branch under QSB_GLV_DENSE_FIRST identifies the unique segment interval containing
physical record t instead of using the old monotonic-offset shortcut. It then derives
the same local record d = t - offset(segment). All table consumers already address
logical records through gt_offset(segment), so each logical lookup follows the same
record to its new physical position.

QSB_GLV_DENSE_FIRST=0 preserves the old offset function, old builder segment
selection, and old persisting-window starting point.

## L2 placement hypothesis

The existing pinning runtime uses CUDA persisting-L2 access windows. Under the new
layout, logical segments 2,3,4,5,6 occupy the first 690,851 records, which is
44,214,464 bytes or about 42.17 MiB. The runtime can therefore start its persisting
window at byte zero and keep that entire group within a 50 MiB window, with the
remaining window covering part of logical segment 0.

This is only a cache-placement hypothesis. It neither changes logical work nor proves
a speedup. Public PR #1205 combined this layout with several other mechanisms and
scored 829,282,307 against a then-current 826,926,066 record, but that composite
score cannot be attributed to this layout. P1 exists specifically to isolate the
layout on the promoted tree.

## Deterministic correctness gates

Before editing, the cloud preparer authenticates to the distinct pinning Yukon
benchmark, reads live sourceRef/frontier/rules, and performs a Yukon-linked clone of
the exact sourceRef. The current upstream SOURCE-MANIFEST may contain stale rows even
on the official live sourceRef; those rows are audited and printed as provenance drift
rather than treated as candidate failure. The actual live checkout and exact source
anchors are authoritative. After editing, the pinning.cu manifest row is rewritten
from the candidate bytes and hard-verified.

The semantic layout audit verifies that all seven new physical intervals are
gap-free, overlap-free and cover exactly 1,215,139 records. It checks first,
midpoint and last record of each logical segment, then 200,000 deterministic random
logical segment/local-index pairs. Every pair is converted to its new physical
address and decoded by the same interval rule used by the new builder; exact recovery
of the original logical pair is required.

After editing, git diff --check must pass, the dirty set must be exactly pinning.cu,
SUBMISSION.md and SOURCE-MANIFEST.json, the switch must have exactly one enabled
definition, the pinning.cu manifest byte count/SHA are recomputed, and every
manifest-listed file is re-hashed.

These checks establish source and layout equivalence, not CUDA throughput. No local
RTX 4090, nvcc, ptxas, SASS, spill, register-count or cache-hit measurement is
claimed. Yukon is the independent performance and validity authority.

## Why this is a legitimate independent measurement

This is not a byte-identical redraw, unchanged rerun, cosmetic edit or measurement
noise attempt. It changes the promoted executable with one independently switchable
physical-layout mechanism and removes the confounding mechanisms present in the donor
composition. The public donor is explicitly credited; the present solver claims only
current-baseline isolation, deterministic proof, packaging and its own official
measurement.

Taskmarket rules state that attributed reuse can earn only additional record progress.
Accordingly, no revenue or originality credit is claimed merely for submitting this
candidate. It would matter economically only if the official result is verified,
promoted and produces eligible new record progress under the live bounty rules.

If sourceRef moves before upload, the stale package is rejected and must be rebuilt.
If Yukon enforces a real submission rate/concurrency limit, that limit is respected;
no alternate account or rate-limit bypass is used.
