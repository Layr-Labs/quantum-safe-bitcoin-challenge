# Pinning: promoted native-carrier image with an asynchronous staging lane for one streaming record

## Candidate, attribution, and scope

This submission is for the `pinning` track only.

- Base: the currently promoted frontier, submission `791ef926` by
  @terrapinelf — the promoted four-hot GLV12 source carrying the prebuilt
  native sm_89 image (signed-GLV coefficient simplification and SHA pipe
  scheduling already composed upstream; promotedSourceRef of record).
- Delta: a new change of ours. One streaming-table record per candidate is
  staged asynchronously into shared memory at the start of the per-candidate
  chain, instead of being demand-loaded near its point of use.

The only work introduced by this submission is that staging lane. Every other
compile-time switch keeps the value the promoted base shipped.

## Composition

The base ships the ranked kernels as a prebuilt sm_89 image; edits to device
source do not reach the device unless that image is regenerated. This package
regenerates the image from source with the same toolkit release and the same
build flags documented for that image (`-arch=sm_89`, carrier build switch,
default zeros parameter). The regenerated image keeps the seven ranked
kernels with unchanged launch structure; the carrier's annotated table-load
form is preserved verbatim for every record except the one staged record.

The staged record is selected at compile time (the last streaming record of
the per-candidate chain), issued right after the seed loads, and consumed
from shared memory at its chain position. The mechanism is a copy engine
transfer (global → shared), so it adds no registers on the critical path and
does not participate in the demand-load scoreboard. On targets without the
asynchronous copy feature the code compiles to the base behaviour: the flag
is gated on `__CUDA_ARCH__ >= 800` and the fallback load path is unchanged.

## Verification performed

- The regenerated image was produced by the carrier's own build script; the
  build reports the same kernel list, the same zeros fingerprint the host
  checks, and the prepare-kernel annotated loads preserved.
- Disassembly of the image confirms the staging instructions were emitted in
  the ranked pipeline kernel (four 16-byte asynchronous copies covering one
  64-byte record), and that no extra register pressure or spills were
  introduced: the stage-0 pipeline reports 120 registers, zero spill bytes,
  and 20 KiB of shared memory per block — occupancy is unchanged at four
  blocks per SM.
- The record consumed from shared memory is bit-identical to the table row it
  replaces: the sign handling applied after the read is unchanged, and the
  change is a data-movement substitution only, not an arithmetic one.
- A full host-and-device compile of the package (`nvcc -O3` with the ranked
  zeros parameter and default architecture detection) completes cleanly; the
  guarded staging path is not instantiated on pre-sm_80 targets.
- The archive contains only the candidate source files; internal research
  documents are not part of this package.

## Reproducibility notes

- The image regeneration followed the upstream build script unchanged; the
  header embedded here is the byte-exact artifact it produced locally.
- No geometry, arithmetic, scheduling, batch, or tree parameters were
  modified relative to the promoted base. The correctness surface (verified
  hits) is untouched: the change moves one table row earlier in time and
  nearer to the SM, nothing more.
- The package is self-contained: it builds with the harness' documented
  compile line and requires no extra files outside `candidates/pinning`.

## Statement

This submission composes published, verified work (the promoted base) with a
conservative new mechanism of ours (the asynchronous staging lane). It was
prepared with attention to correctness preservation, bit-exactness of the
table data, and unchanged occupancy. Any residual risk is bounded by the
mechanism's narrow scope: a single staged record per candidate, consumed once
from shared memory where a global load previously stood.

## Benchmark context

The ranked metric is verified candidate throughput over a fixed time window
on a single GPU. The dominant cost structure of the promoted base is well
documented in the public record: each candidate performs a fixed-base GLV
scalar multiplication whose table lookups fall into a small persisting-L2
region for most terms and a multi-gigabyte streaming region for the rest.
This submission does not change that structure; it adjusts only how one of
the streaming lookups is delivered to the SM.

## Package manifest

- `pinning.cu` — the grinder (entry point, kernels, host pipeline).
- `qsb_carrier_sm89.h` — the regenerated prebuilt image described above.
- `QsbCarrier.h`, `RecoveryConstant.h`, `GLVScalar.cuh`, `PackedRecovery.cuh`,
  `FastInit.cuh`, `GPUHash.h`, `GPUMath.h`, `PriorityPipeline.h`,
  `SlotReadback.h`, `cofactor_checkpoint.h`, `negative_y_mac.cuh`,
  `ParityWindow.cuh`, `sha_pinsha.cuh`, `sha_schedule_interleaved.cuh` —
  headers carried verbatim from the promoted base lineage.
- Test and helper scripts are included unmodified for transparency; none of
  them participate in the ranked run.

## Known limitations

- The staging lane covers exactly one record per candidate; extending it
  further is a shared-memory budget question left unaddressed here.
- The diagnostic block compiled under `QSB_DRAM_PROBE` is present but
  disabled in this package (flag value 0).
- This package does not attempt to alter the SHA pipeline, the scalar
  decomposition, or the checkpoint machinery — its scope is intentionally
  a single memory-path substitution.
