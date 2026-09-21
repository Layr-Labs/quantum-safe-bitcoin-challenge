# Pinning: next-optimization backlog after attempt-2 Z9SUB on f7e4dde

Snapshot: 2026-09-21 (Africa/Lagos). Base: `f7e4ddef` at 792,667,656.
Promoted floor source: `dcd0147c` at 789,011,576; promotion bar 796,901,692.

## What this tree ships

- Everything in `f7e4ddef` (ROT2, PTR, PAIRLDS, dest-direct, funnel, RP_SQR,
  host gate, C31, 743).
- `QSB_SAS_Z9SUB_ALL=1`: drop the dead bit-288 `z9` carry/borrow lane from the
  live short-carry `_ModSqr` and from `_ModSqrAddSub2` fold/bias/sub tails.
  `-DQSB_SAS_Z9SUB_ALL=0` restores those ops. DUALFETCH stays **0**.

## After this official score

1. If it promotes, stop and let the new floor settle.
2. If near-miss above 792.67 M, keep Z9SUB; consider one stage-2 exact cut
   (finish `sum=2u`, raw pre-`+a` products) — not a bundle with plumbing.
3. If ~−3.7%, Issue #505 host draw; do not flip Z9SUB off on that evidence alone.
4. If large regression with healthy hit count, A/B with `Z9SUB=0` on same host.

## Still closed

- DUALFETCH, FKIENE alone, RAW_X, SHA ST, GLV, unroll 2/3/4, exact top-16,
  `_ModAddLazyOff` t1, C31/RP_SQR without host gate.
