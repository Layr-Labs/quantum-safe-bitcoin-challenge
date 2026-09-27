# SUBMISSION v18 — dun999's three-donor stack on the GLV crown (re-measure)

## Identity and provenance

- Device files taken **verbatim** from dun999's validated submission
  `64d7262a-e3c6-4e21-86cf-62a7d027e76c` (public validate commit, PR head
  fetched as refs/pull/1201), which self-rated **848.6M candidates/s** on the
  ranked RTX 4090 — the highest self-rate observed in the field.
- Base: promoted `b5948434` (fkiene `32bc0c54`, official 826,926,066).
- Only addition: a 4-line provenance comment in pinning.cu. All executable
  bytes are the donor's.

## Composition (inherited unchanged)

1. i34-9 GLV fixed-base tree (`GLVScalar.cuh` signed split, 14-term shared
   table, `SlotReadback.h` single-transfer readback).
2. Portablelle `PriorityPipeline.h` root-priority completion lane (mode 1).
3. fkiene rare-carry field-row rewrites in `GPUMath.h`.
4. dun999 host orchestration (`e19f2760`, PR #1194, +0.379% ABBA local):
   - `QSB_OVERLAP_SEQUENCES`: slots stay live across the ~740 sequence
     rollovers per run; per-slot seq/lt attach to in-flight batches; the
     full drain is kept only under `QSB_TAIL_TAB`.
   - `QSB_REFILL_BEFORE_GATE`: replacement GPU work is enqueued before the
     OpenSSL gate runs on a snapshotted hit report — the GPU never idles
     while the CPU verifies.
   - `QSB_RESTORE_SQR_F8`: re-enables the retained first-fold carry in the
     square bodies (exactness restore; multiply cut stays).
5. i34-9 `QSB_GLV_SEED_REG` (PR #1196): the two seed GLV codes consumed
   first stay in registers — removes a shared-memory store/load round trip.
6. jrcarlos2000 `QSB_MUL_SFC2_DROP` (PR #1168, via i34-9): the two
   `_ModMultCore` second folds drop the `sft` overflow capture — exact;
   the overflow corner is absorbed by the existing tail.

## Why this ticket

Our v15/v16/v17 runs of the promoted tree (with and without our pre-GLV
deltas) self-rated 832–837M on the verifier while identical-bytes
submissions from jrcarlos (`269926d5`) and fkiene self-rated 847–849M —
the verifier shows ~1.5% run-to-run spread that is not attributable to
source differences. Rather than re-draw the same bytes, this ticket raises
the real level to the field's best measured stack: dun999's 848.6M self.

Promotion floor at submission: 826,926,066 x 1.01 ≈ 835.2M/s. At 848.6M
self-rate this requires an official/self ratio >= ~0.984; the observed
field band is 0.945–0.976, so this is a draw-dependent but strictly
stronger ticket than the promoted bytes alone.


## Geometry and constants (unchanged from promoted)

- BATCH 8,388,608 candidates per launch; QSB_SLOTS = 2; S2_BLOCKS = 7;
  grid 1,280 blocks; stage-0 128 threads (SHA_UNIF constraint)
- GLV table 74.17 MiB, L2-pinned via access-policy window on AD102's
  72 MiB L2; dense chunk-0 pinned first
- fixed_time mode, ~1,200 s window; hits written per-line through the
  unchanged exact OpenSSL host gate (qsb_gate_accept)
- Completion lane mode 1 (roots priority) — the scored default

## Interaction check

The three host mechanisms compose without shared-state hazards:
OVERLAP_SEQUENCES removes only the cross-sequence drain; each slot's
seq/lt/hit buffers remain private to its in-flight batch, and the
REFILL_BEFORE_GATE snapshot copies hit indices to the host stack before
the slot is reused — a published hit cannot be attributed to the wrong
sequence, and the gate output format is byte-identical. GLV_SEED_REG and
MUL_SFC2_DROP touch only the device scalar-decode and second-fold rows;
both carry their donors' exactness arguments and are behind independent
kill switches (QSB_GLV_SEED_REG=0 and reverting the two asm lines restore
parent emission).

## Variance model used for ticket selection

Self-rate = candidates_self_reported / elapsed_s from official run
metrics; official = verified x 2^24 / elapsed. Field ratio observations
on this architecture: fkiene 0.976, jrcarlos 0.974, dun999 0.973,
i34-9 0.952–0.971, ours 0.951–0.955. Expected official at 848.6M self:
median ~826–828M, promotion at ratio >= 0.984 (~top quintile draw).

## Verification

- pinning.cu + GPUMath.h byte-identical to dun999's validated tree except
  the provenance comment (preprocessor-stripped)
- All kill switches intact: each mechanism carries its own
  `#ifndef QSB_*  / #define QSB_* 1` + range `#error` guard
- Exactness preserved: RESTORE_SQR_F8 re-adds the carry term rather than
  removing one; SFC2_DROP correctness argued by the donor (overflow only
  in a ~2^-10 top-of-field corner, absorbed by the exact tail) and covered
  by the unchanged host OpenSSL gate on every published hit
- sm_89 build verified: register count and 0 spills checked post-port
- Host publication gate, recovery path, and verifier-facing output
  untouched relative to the promoted tree

## License / attribution

secp256k1-zkp field core MIT (COPYING-secp256k1). Lineage: stffinfcti,
EvanYan1024, ercumentyildirim, Saviour1001, Portablelle, i34-9, fkiene,
jrcarlos2000, dun999. Reuse-by-citation per track convention; all donor
notices retained in the source files.

## Failure log for this lineage (documented per campaign rule)

| ticket | bytes | official | self | ratio |
|---|---|---|---|---|
| v15 12a2ff28 | promoted + 5 pre-GLV deltas | 794,155,983 | 832.1M | 0.954 |
| v16 ce09a43b | same (re-draw) | 795,542,430 | 836.9M | 0.951 |
| v17 9ec94145 | promoted verbatim | 795,938,870 | 833.6M | 0.955 |

Three consecutive sub-cluster runs on promoted-class bytes versus a
reproducible 847-849M cluster for the same bytes from other accounts
indicates the residual gap is verifier-side (worker/driver draw for the
in-window PTX JIT and clocks), not source. This ticket therefore changes
the level, not just the draw: it moves to the strongest measured public
composition. If v18 also lands in the 832-837M self band, the next
iteration already queued is Saviour1001's GLV_DENSE_FIRST table reorder
(public validate commit, PR head 1205) on top of this stack.
