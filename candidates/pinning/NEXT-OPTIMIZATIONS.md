# Pinning: next-optimization backlog after the SAS-z9 / square-RP_SQR submit

Snapshot: 2026-09-21, packaging f7 + `QSB_SAS_Z9SUB_ALL` + production
`_ModSqr` RP_SQR. Scoped to `candidates/pinning/`. Promoted source is
`66fede0` / `dcd0147c` at 789,011,576. Floor 796,901,692.

## What just shipped in this tree

Default-on in `pinning.cu` / `GPUMath.h`:

- f7 schedule (ROT2 / PTR / PAIRLDS / dest-write / host-gate / C31 / RP_SQR).
- `QSB_SAS_Z9SUB_ALL` (official `be352be3` +0.45% on C31).
- Production SHORT_CARRY `_ModSqr` now uses the RP_SQR macros f7 left in
  the restore path.
- C31 `QSB_FOLD_Z2` (`addc.u32 z2` without unused `.cc`).

SHA flags stay 0. `QSB_UNROLL=1`. `_ModAddLazyOff` still keeps `t1`.

## After this official score

1. If it promotes, stop. The next 1% is a new problem.
2. If it is a near-miss above 789 M, do **not** compose exact top-16
   (`eb6d9871` −1.21%) or dest-write occupancy forcing.
3. If it is a large regression, first suspect the production square RP_SQR
   splice (G-table spot-check) then the SAS z9 fold (host-gate hit rate).
4. CUDA graphs / JIT trim remain bundle fillers only.

## Still closed

- GLV / joint comb / radix-373
- chain unroll 2/3/4
- pinning SHA ST flags
- exact complete top-16
- fused one-grid, Karatsuba, L1 prefetch/bypass
- dropping `_ModAddLazyOff` t1
- dest-write / funnel / Y-pingpong as a fresh 1%
- publishing C31 / RP_SQR / SAS-z9 without the gate
