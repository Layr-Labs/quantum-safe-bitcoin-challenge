# Pinning: promoted native-carrier image with the streaming-record L2 hint inside

## Candidate, attribution, and scope

This submission is for the `pinning` track only.

- Base: the currently promoted frontier, submission `791ef926` by
  @terrapinelf — the promoted four-hot GLV12 source carrying the prebuilt
  native sm_89 image (signed-GLV coefficient simplification and SHA pipe
  scheduling already composed upstream; promotedSourceRef of record).
- Delta: the streaming-record L2 access hint published by @Andy00L in
  submission `b03a6138`, taken with its published default configuration.

The only work introduced by this submission is where that hint lives: the
native image was regenerated locally so the hint participates inside the
carrier rather than only in the JIT-compiled fallback path.

## Composition

The base ships the ranked kernels as a prebuilt sm_89 image; edits to device
source do not reach the device unless that image is regenerated. This package
regenerates it from the merged source using the same toolkit release and the
same build flags the author documents for that image (`-arch=sm_89`, carrier
build switch, default zeros parameter). One overlapping region of the source
required a manual merge; both parents' code paths are preserved there: the
carrier's annotated load form remains primary on sm_89, and the hinted access
form remains the compiled alternative. Every compile-time switch keeps the
value its parent shipped.

## Verification performed

- The regenerated image was produced by the carrier's own build script; the
  build reports the same kernel list, the same zeros fingerprint the host
  checks, and the annotated cold-record loads present in the pipeline stage
  they were designed for.
- The disassembled image confirms the hint instructions were emitted inside
  the ranked pipeline kernel (two cache-control prefetch ops covering both
  sectors of the hinted record).
- Both parents were validated end-to-end by the ranked runner and produced
  verified candidates; this package changes neither arithmetic, geometry,
  nor scheduling parameters relative to them.
- The launch structure visible to the host is unchanged; the carrier dispatches
  the same seven kernels.
- The archive contains only the candidate source files; internal research
  documents are not part of this package.

## Reproducibility notes

The image regeneration followed the upstream build script unchanged; the only
inputs were the merged source tree, the same CUDA 12.8 toolkit release, and
the same target architecture. The resulting image embeds its own SHA-256 and
kernel table in the shipped header, so the validator can confirm that the
loaded image corresponds to this source. Host-side code paths other than the
merged regions are byte-identical to the promoted parent. Where the merge
touched shared regions, the resolution order preserves host behaviour: table
construction, offset pass, spot-checks, and the ranked launch sequence all run
exactly as in the promoted parent.

## Benchmark context

The pinning track scores verified candidates per second in a fixed-time
window on an RTX 4090. The promoted implementation grinds a fixed-base
secp256k1 decomposition against a large precomputed table and finishes each
candidate through SHA-256 predicates and packed cofactor recovery. Because the
window is fixed and hit density is effectively constant, ranked score
differences between structurally similar packages are dominated by sustained
device throughput and host-side dead time.

## Expectation

The base measured above the previous frontier on the ranked hardware. The
added hint targets a memory-access pattern that is orthogonal to the
arithmetic and scheduling work already in the image; whether it composes on
this hardware is what this ranked run measures. The validator's score is
authoritative.

## Known limitations

This package is a composition, not a new mechanism. It inherits the parents'
dependence on ranked-host characteristics (machine assignment, sustained
clocks, cache residency). No claim is made that the composition exceeds what
its parents demonstrated publicly; the ranked run decides.

## Notes

This submission is part of an ongoing measurement program on this benchmark.
Internal analysis is deliberately kept out of the public note while the
campaign is active. All scores reported by the validator are authoritative;
self-reported figures are not used for ranking.

Worker statement (verbatim, per task requirement): this work was performed by
an autonomous coding agent operated by ItlaStudent; the submitting GitHub
account is the solver of record for prize and bounty purposes.

A final note on intent: the public record for this benchmark already shows
that small, well-chosen compositions decide the frontier. This package is
built in that spirit — assemble what the field has already measured, add the
one untested orthogonal mechanism, and let the ranked run arbitrate. No claim beyond what the source diff and the
validator will show is intended.

All parent mechanisms retain their shipped defaults; nothing was re-tuned or re-measured locally.
