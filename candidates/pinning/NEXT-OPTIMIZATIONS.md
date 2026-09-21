# Pinning: next-optimization backlog after DUALFETCH on f7e4dde

Snapshot: 2026-09-21. Base reset: `f7e4ddef` at 792,667,656. Promoted floor
source still `dcd0147c` at 789,011,576; promotion bar 796,901,692.

## What this tree ships

- Everything in `f7e4ddef` (ROT2, PTR, PAIRLDS, dest-direct, funnel, RP_SQR,
  host gate, C31, 743).
- `QSB_CHAIN_DUALFETCH=1`: both pair table loads before either mixed-add,
  via `yalt` ping-pong. Removes the WAR on `ya` that blocked ROT2's claimed
  load/madd overlap.

## After this official score

1. If it promotes, stop and let the new floor settle.
2. If near-miss above 792.67 M, keep DUALFETCH; try one of: `cp.async` of
   plane c+1, or SAS `g8` only (not bundled with `f8`).
3. If ~0 hits, audit `*off` final ordinate vs CPU fixed-base on ranked seed.
4. If ~−3.7%, Issue #505 host draw.

## Still closed

- FKIENE alone, RAW_X, SHA ST, GLV, unroll 2/3/4, exact top-16,
  `_ModAddLazyOff` t1, C31/RP_SQR without host gate.
