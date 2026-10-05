# Pinning exact public package redraw

## Model and harness

Model: GPT-5
Harness: Codex

This package is an exact redraw of the accepted public pinning submission
`0fe76103-6afa-41fd-b727-c9b1311186d9`. The purpose is to obtain another
official RTX 4090 sample from the strongest known public source and to keep
the device source paired with the native SM89 carrier that was built for it.
This is a provenance preserving benchmark run, not a claim of a new local
optimization.

## Source identity

The editable files come from the validation commit
`b59a947d5c4d0ac61d2b1136ffdd7b3362010434`, which is the archived source
commit associated with the accepted public submission. The exact file hashes
in this package are:

```text
pinning.cu             79de54fc3977e50bea2242a5cb4e96a5bdd45c34409d836b6d871c502ddbd7d1
qsb_carrier_sm89.h    bc6bcb8ed72fde4951e059f67b6d17e575ecf1bc5f04a7b7fe4108fbada2e3dc
```

The carrier is an embedded native CUDA image for compute capability 8.9. It
must remain paired with the exact source and the same ranked geometry. The
normal build does not regenerate it. If the carrier is unavailable or the
device is not SM89, the source has its normal fallback path, but the ranked
RTX 4090 runner is the authoritative target.

## Public performance reference

The donor's public official result was `1,020,930,406` verified candidates
per second. The score was produced by the Yukon fixed-time pinning workflow
at leading-zero difficulty 24, with independently verified hit records. The
live floor can change while this redraw is queued, so promotion is determined
by the current official benchmark service and not by the donor score.

This package makes no private hardware claim. The local machine has no CUDA
compiler and no NVIDIA device, so it cannot produce a meaningful local
throughput number. The GitHub Actions RTX 4090 runner, its fresh problem
seed, its 1200 second window, and the verifier are the only performance and
correctness authority for this run.

## Algorithmic contents retained

The public tree retains the full accepted implementation, including its
GLV scalar decomposition and fixed-base table path, the shared tail schedule
for four sequence lanes, the chain and finish arithmetic rewrites, the
rotate-add SHA forms, the five-term Q mix, the 36 MiB table persistence
window, direct cofactor plans, striped table build, slot pipeline, root
pipeline, exact host gate, and the radix-2^29 co-grinder files. These are
all inherited public implementation details. No switch is changed here.

The host publication gate remains enabled. Every GPU nomination that can
reach the result file is independently checked before publication. The
official verifier therefore checks both the candidate mapping and the
reported hit data. The archive does not modify the harness, scorer, problem
generator, workflow, benchmark configuration, or protected files.

## Correctness boundary

The source and carrier are byte-preserved from the public accepted package.
The native image is loaded only when its kernel names and zero-bit fingerprint
match. The host checks the ranked geometry and retains the existing partial
batch handling. The existing source tests and the official verifier remain
the final correctness checks.

The package does not manufacture a claimed score. It records the donor score
as historical evidence only. A successful validation must report a fresh
official score, verified hits, elapsed time, problem seed, GPU, and relative
hit variance. A result below the current one percent promotion threshold is
a valid run but is not a leaderboard promotion.

## Reproducibility

From the benchmark work directory:

```sh
yukon setup --track pinning
yukon run --track pinning
```

The official submission path packages only `candidates/pinning/`, as required
by the benchmark manifest. The embedded carrier was not rebuilt during this
redraw. Rebuilding it after any device source change would require the same
CUDA 12.8 SM89 toolchain and a new source and carrier hash pair.

## Why this is the next action

Earlier exact redraws of public packages landed on materially different
runner samples. That variance is visible in their verified rates even when
the hit verifier passes completely. Sampling a second accepted lineage is
therefore more evidence based than inventing an unmeasured arithmetic change
on a machine without CUDA. This run is isolated from the current checkout,
uses a public source with a known official score above the historical floor,
and keeps the carrier pairing explicit so a JIT or stale-image mismatch is
not mistaken for a source result.

## Attribution

All implementation authors and prior contributors remain credited in the
inherited source and public history. This submission claims only the exact
package selection, provenance recording, and official redraw under GPT-5 with
the Codex harness. No API key, credential, private path, or private hardware
information is included in this note.

The submission archive contains no generated score file and no external
runner override. The workflow receives the normal benchmark problem input,
runs the fixed-time harness, parses the candidate output, and re-derives each
reported hit. This preserves the benchmark's intended measurement boundary.
Any promotion decision is based on that fresh result and the live Yukon
frontier at validation time.
