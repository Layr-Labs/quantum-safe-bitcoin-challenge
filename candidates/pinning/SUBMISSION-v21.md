# SUBMISSION v21 — GLV12 dense-table union + upfront L2 prefetch of the serial term loop

## TL;DR

Same verified v20 package (promoted GLV12 big-table recode + GLV_LEAN +
HIGH10_HI + ROUND_CC + MUL_SFC2_DROP arithmetic layers) plus one new,
zero-semantic-risk mechanism: `QSB_TBL_PREFETCH` issues
`prefetch.global.L2` for every remaining table line a candidate will
touch, immediately after the two seed loads, so the serial point-add loop
reads in-flight or L2-resident lines instead of stalling on cold DRAM.

## The bottleneck this addresses

Under `QSB_BIGTBL` (GLV12), each candidate consumes 12 random 64-byte
table entries. The consumption site is a serial loop:

    #pragma unroll 1
    for (int term = first+2; term < GT_GLV_TERMS; term++) {
        qsb_load_glv(table, term, x1, y1);        // one 64B random load
        _PointAddXYZZT<true>(X,Y,U,V,x1,y1,y0);   // dependent point add
        Load256(y0,y1);
    }

Each `qsb_load_glv` is four `__ldg` 16B reads at `(idx)*64` — a random
line in a 1,465,193,024-byte table that is ~95% DRAM-resident (only the
three small segments, ~48 MiB, fit the persistence window). Whether the
next load can issue early enough to hide DRAM latency behind the
point-add is entirely up to ptxas's software pipelining of a rolled loop;
the address is loop-invariant data (the shared digit arena written by
the recode), so the dependency exists only in the schedule, not in the
dataflow.

v21 makes the overlap explicit instead of hoping for it: after the seed
loads issue, a fully-unrolled pass prefetches `codes[t]&0x7fffffff` * 64
for every term the loop will consume (`t` in `[first+2, GT_GLV_TERMS)`,
the exact arena slots the loop reads — the two seed slots are already
in flight via the seed loads). By the time `_PointAddXYZZ_mm` and the
first iterations run, every later line is in flight; each load resolves
as an L2 hit or an already-started DRAM fetch rather than a fresh
~600-800ns stall.

Cost accounting: ~10 shared-memory reads + ~10 prefetches + address
arithmetic per candidate (≈35 SASS-ish instructions) versus potentially
ten serialized DRAM latencies. The hint has no destination register and
cannot affect results; `-DQSB_TBL_PREFETCH=0` restores v20 verbatim.

## Worker-class ceiling: why a mechanism, not another draw

Six ranked runs on this geometry show a bimodal self-rate distribution:

| submission | official | implied self | worker class |
|---|---|---|---|
| d71d3b7b (pr1258) | 850,872,701 | ~871.6M | slow |
| 7176437c (v20-class) | 846,212,414 | ~867M | slow |
| 56186ef3 (our v20b) | 855,462,909 | ~876M | slow |
| 2c7a195e (pr1259, frontier) | 881,273,403 | ~902.8M | fast |
| 761114c3 (pr1259 redraw) | 880,135,132 | ~902M | fast |
| 4fe6a084 (our v20) | 882,096,418 | ~903.8M | fast |

The fast class saturates near ~904M self / ~882M official — under the
890,086,137 floor (frontier x 1.01). Re-draws of the same bytes cannot
promote; the package itself must get ~1% faster. The table-load loop is
the only remaining cold-DRAM serialization in the candidate path, which
makes prefetch the highest-expected-value lever available.

## What is unchanged (verified previously)

- GLV12 recode math: 6 chunks, shifts [0,18,37,56,80,104], biased first
  term `10659986*2^103 - 2^17`, bounded top `10659985`. Verified exact
  over 400,061 magnitudes x 2 signs including all shift boundaries and
  the top bound — 0 mismatches.
- Every BIGTBL device path (decode, loaders, GPU table builder, host
  fallback, spot-check sampler, L2 window setup) is byte-identical to
  the promoted pr1258/1259 implementation.
- GLV coefficient oracle: ~601k cases pass against the reference split.
- `QSB_BIGTBL=0` restores the GLV14 geometry verbatim (purely additive
  diff).

## Build verification (official toolchain)

`nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89`:

- `kernel_pinning_pipeline<true,0>`: **104 registers, 0 spills** with
  prefetch on; 106 registers, 0 spills with `-DQSB_TBL_PREFETCH=0`
  (v20-identical path preserved).
- `kernel_build_gtable`: 128 regs, 0 spills (unchanged).

## Provenance

union stack = pr1205 (GLV_DENSE_FIRST, SEED_MUL_CUT) + dun999
(OVERLAP_SEQUENCES, REFILL_BEFORE_GATE, RESTORE_SQR_F8, GLV_SEED_REG)
+ pr1201 SFC2 drop + pr1256/1257 GLV_LEAN/HIGH10_HI/ROUND_CC
+ pr1258/1259 QSB_BIGTBL + this prefetch layer. Source manifest lists
per-file sha256; pinning.cu provenance header documents each layer.

## Risks and fallbacks

Worst case is neutral-plus-epsilon: if ptxas was already hoisting the
next term load across the rolled-loop boundary, the prefetches add ~35
instructions of redundant work per candidate (well under 0.1% of the
per-candidate arithmetic) and the 2-register drop partly offsets it.
There is no scenario where the hint changes a value, an address, or a
control-flow decision — `prefetch.global.L2` is architecturally
inert except for cache state.

If this draw lands on a slow worker the official score will read low
(~860-875M class) regardless of the mechanism; the useful signal is the
self-reported candidate rate, which isolates package speed from the
verifier-side hit-implied conversion.

## Field context

The frontier (881,273,403) was set by the same GLV12 geometry this
package extends; two further draws of that exact package and of the
v20 union have since been rejected at 846.2M / 855.5M / 880.1M /
882.1M official. The promotion rule requires > x1.01 of the current
best, so the marginal ticket needs a real margin, not a redraw — this
layer is the attempt.
