# v23 — GLV10: five chunks per component (−2 point-adds, −2 DRAM loads)

## What this package is

v22's device stack + init bundle, with the fixed-base geometry stepped from
**six chunks to five** (`QSB_GLV10`). Each GLV component decomposes into 5
table terms instead of 6 → **10 serial point-adds per candidate instead of
12**, and 10 random 64-byte table reads instead of 12. On the measured
workload (~72% of SASS in point-adds, table reads streamed from DRAM) this
is the largest remaining structural cut: −16.7% of the serial add chain,
−16.7% of table traffic.

## Geometry (candidate A — minimum-table)

- Shifts `[0,24,50,76,102]`; widths {24 raw, 26,26,26 signed windows}, plus
  bounded top.
- Top bound **T = 42,639,943** (the proven split bound `BOUND>>102 =
  42,639,942` — margin exactly one).
- Bias `K = 42,639,944·2^101 − 2^23` — telescopes the three 26-bit window
  corrections `(2^26−1)(2^23+2^49+2^75) = 2^101−2^23` plus `T·2^101`.
- Entries: 2^24 + 3·2^25 + 21,319,972 = **138,760,484 records =
  8,880,670,976 bytes (~8.27 GiB)** — fits the RTX 4090's 24 GiB with the
  ~1.5 GiB pipeline buffers.
- `GT_HI` 4096→16384: the 26-bit odd-multiple range needs `hi = m>>12`
  up to 16383. The `m>>12`/`m&4095` ladder split is unchanged.
- Physical order is linear `[0,1,2,3,4]` — no segment fits L2, so the
  GLV12 dense-first permutation buys nothing.

## Verification

- `work_tmp/check_glv10_recode.py` — faithful Python port of the flag-gated
  `q9_bigtbl_*` helpers: **400,052 magnitudes ×2 signs, 0 mismatches**,
  including all shift-boundary corners and the top-bound edges; the bias
  telescoping identity is asserted algebraically.
- sm_89 build: `kernel_pinning_pipeline<true,0>` = **106 regs, 0 spills**;
  hot `.text` 0x19a00 (−384 B vs GLV12 — the unrolled decode sheds two
  terms; the steady-state win is the loop trip count).
- `QSB_GLV10=0` produces the byte-identical v22 device image (verified
  against commit ff0d785's cubin).
- New `qsb_geom_stamp` (compile-time `__device__` constant, baked into the
  shipped cubin) is read back in `cubin_init` — a stale cubin from a
  different geometry is refused before it can build a wrong-layout table.
- The OpenSSL spot-check sequence, bias arithmetic and table-scalar
  semantics are updated consistently in `gt_ladder_chunk`,
  `gt_table_scalar`, `gt_spot_check`/`gt_spot_check_dev`, and the fallback
  `compute_gtable` — all driven by the same `q9_bigtbl_*` constants.

## Expected effect

GLV14→GLV12 removed two terms for +6.6%. GLV12→GLV10 removes two more on a
larger base; conservatively +6–7% self-rate (~960M on a fast worker, ~925M
on a mid draw) → ~940M+ official at a ~0.98 yield — comfortably over the
890.09M floor on either worker class, and within reach of the ~925M goal
the field is converging on.

## Provenance

v22 init-cut stack (native sm_89 cubin, two-phase table build, sampled
readback, threaded ladders, cudaMallocAsync, SIGTERM drain) + the promoted
GLV12 union + our LEAN/SFC2 arithmetic. GLV10 geometry derived internally;
algebra verified by the recode oracle.
