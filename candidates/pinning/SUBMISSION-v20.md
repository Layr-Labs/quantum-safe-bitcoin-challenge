# SUBMISSION v20 — GLV12 big-table + lean split on the union stack

## What this ticket is

A structural candidate, not a re-draw. The device tree takes the newly
promoted GLV12 dense-table architecture (public PR heads 1258/1259,
`QSB_BIGTBL`: six terms per signed GLV component, 22,893,641 records,
1,465,193,024 bytes) and carries it on our full arithmetic stack — the
same stack that produced the field's 851M self-rate measurements before
the big-table design existed.

Concretely: pr1259's promoted package plus `QSB_GLV_LEAN`,
`QSB_GLV_HIGH10_HI`, `QSB_GLV_ROUND_CC` (GLVScalar.cuh) and
`QSB_MUL_SFC2_DROP` (GPUMath.h). Every BIGTBL code path in this tree is
character-identical to the promoted implementation; the only additions
are the lean arithmetic layers the donors published separately.

## Composition

1. **GLV12 dense table** (odinfree `d71d3b7b` / anamdongparkjinhyeong
   `2c7a195e`, public PRs 1258/1259): six chunks at shifts
   [0,18,37,56,80,104], widths [18,19,19,24,24,top<=10659985], physical
   order [0,1,2,5,3,4] placing the three 16 MiB dense segments at offset
   zero inside the persisting-L2 window; one 32-bit code still carries
   absolute record index + Y sign. Two fewer point-add terms per
   component, at the cost of streaming the large segments from DRAM.
2. **QSB_GLV_LEAN** (kaankolcu/Portablelle line, PRs 1229/1256): the
   split's 32x32 products as explicit `mul.wide.u32`/`mad.wide.u32`, and
   the high15 overflow count taken from the add's carry flag. Identical
   values; removes the wasted IADD3 per product and two compares.
3. **QSB_GLV_HIGH10_HI**: diagonal 10 contributes only its five high
   product words; omitted terms bounded below 9*2^352 (g1) / 8*2^352
   (g2), exactness preserved by the widened bit-383 rounding guard.
4. **QSB_GLV_ROUND_CC**: rounded reciprocal fold via carry flag.
5. **QSB_MUL_SFC2_DROP** (jrcarlos2000 PR 1168, via dun999 `64d7262a`):
   omits the multiply-only second-fold sft overflow capture; the rare
   top-of-field corner becomes a miss absorbed by the exact host gate,
   never a false hit.
6. dun999 host orchestration: `QSB_OVERLAP_SEQUENCES`,
   `QSB_REFILL_BEFORE_GATE`, `QSB_RESTORE_SQR_F8`; i34-9
   `QSB_GLV_SEED_REG`; Saviour1001 `QSB_GLV_DENSE_FIRST` +
   `QSB_SEED_MUL_CUT`; Portablelle `PriorityPipeline.h` mode 1; fkiene
   field-row cuts.

## Why this composition

The verifier's promotion rule is mechanical: official score must exceed
floor(frontier x 1.01). With the frontier now at 881,273,403 the floor
is 890,086,137. The promoted GLV12 package self-reported ~871.6M and
~902.8M candidates/s on two worker draws (official 850.9M and 881.3M —
a remarkably consistent ~0.976 hit-implied/self ratio on both).

On the pre-big-table stack this tree's extra arithmetic layers were the
ones that separated our 849-851M self-rates from the donor's 833-847M
band. If that margin carries over proportionally, this composition
sits a few M/s above the promoted package on the same silicon — which
is exactly the gap the floor requires.

## Verification performed

- **Recode exactness**: a Python port of `q9_bigtbl_code` reconstructed
  each component magnitude from the six emitted codes (record index +
  sign) over 400,061 magnitudes x 2 signs — including every shift
  boundary, the odd top bound 10,659,985 corners, and the proven split
  bound |r_i| <= 0xa2a8918ca85bafe22016d0b917e4dd77 — with zero
  mismatches. The telescoping identity
  sum((2^bits-1)*2^(shift-1)) = 2^103 - 2^17 was checked against the
  chunk geometry and the K = 10659986*2^103 - 2^17 bias.
- **Port fidelity**: every BIGTBL function in this tree
  (`q9_bigtbl_entries/offset/shift/code`, the decode/load arms,
  `kernel_build_gtable` hi/lo split, `gt_build_ladders`,
  `gt_table_scalar`, `gt_spot_check`, `compute_gtable`, the guarded
  `cudaMalloc`/sync/`cudaMemcpy` and the L2 window `skip=0`) is
  byte-identical to the promoted public implementation.
- **Additive diff**: `git diff` of pinning.cu vs the previous package
  adds 104 lines and removes 0 — the GLV14 path is preserved verbatim
  under `#else`, recoverable with `-DQSB_BIGTBL=0`.
- **Official-toolchain build** (nvcc -O3, -DQSB_ZEROS_N=24, the harness
  line): clean compile; sm_89 accounting gives the hot kernel
  `kernel_pinning_pipeline<true,0>` 106 registers, 0 spills — unchanged
  from the previous stack, i.e. the 12-term decode fits the same
  register budget. Hot-kernel .text is 384 bytes smaller than GLV14
  (fewer decode terms), all non-GLV kernels bit-identical.
- The GLV_LEAN/HIGH10_HI/ROUND_CC layers were already validated by the
  standalone coefficient oracle (601,636 cases) in this tree; their
  outputs are value-identical, so the BIGTBL top-field bound is
  unaffected.
- Host publication gate, recovery path, hit format, and benchmark
  interface untouched; every mechanism stays behind its kill switch.

## Known negative result folded in (not shipped)

A 64-bit lane-form rewrite of the `_ModSqr`/`_ModSqrAddSub2` merge
(`QSB_SQR_LANE64`, default off) was implemented, proven bit-exact over
~180k randomized and adversarial cases, and then measured: under the
official nvcc/ptxas the resulting cubin is byte-identical to the
32-bit form — ptxas canonicalizes both schedules. It remains in the
tree only as a documented dead end, disabled.

## Geometry and constants (as composed)

- BATCH 8,388,608 candidates/launch; QSB_SLOTS = 2; S2_BLOCKS = 7;
  grid 1,280 blocks; stage-0 128 threads.
- GLV12 table 1,465,193,024 bytes; dense prefix 48 MiB inside the
  persisting-L2 window; record = 64-byte affine (x,y); 12 terms per
  candidate, one 32-bit shared-memory code each (two seed codes in
  registers).
- GPU table build + 216-sample OpenSSL spot check + full-table readback
  verify before the run; wrong table falls back to the host builder.
- fixed_time mode, ~1,200 s window; every hit still passes the
  unchanged exact OpenSSL publication gate before leaving the binary.

## Reproducibility

Donor artifacts are public refs in the shared repository:
`git fetch origin pull/1259/head` (promoted GLV12 package),
`pull/1256` (GLV_LEAN stack), `pull/1205` (union base),
`pull/1201` (SFC2 drop). Rebuild:
`nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm`
(organizer ranked line, CUDA 12.8.93, RTX 4090 target).

## License / attribution

secp256k1-zkp field core MIT (COPYING-secp256k1). Lineage: stffinfcti,
EvanYan1024, ercumentyildirim, Saviour1001, Portablelle, i34-9, fkiene,
jrcarlos2000, dun999, kaankolcu, odinfree, anamdongparkjinhyeong,
DrCleverHans. Reuse-by-citation per established track convention; all
donor notices retained in source.

## If this lands below the floor

The next lever is geometrical: the 24-bit segments (chunks 3/4) are the
DRAM-streaming cost center — a rebalanced split (e.g. narrower top
segments, different dense-prefix ordering, or a hybrid 7-chunk
intermediate geometry) trades table footprint against term count
differently. Measured self-rate on this ticket decides whether the
remaining gap is variance or structure.
