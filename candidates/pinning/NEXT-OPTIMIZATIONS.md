# Pinning: next-optimization backlog after the 789 M floor

Post-score implementation list: `LATER-IDEAS.md` at the repo root.
1.0 G aim, closed research, and the submit ladder: `AIM-1G.md`.

Snapshot: 2026-09-20, after `dcd0147c` **promoted** at 789,011,576
(`66fede0`). Next floor 796,901,692. `b0fbfb1a` (`QSB_RP_SQR`) **rejected 789,394,272** (+0.26%).
This tree turns RP_SQR off and adds `QSB_FKIENE`. One switch.

## What just shipped in this tree

Default-on in `pinning.cu` / `GPUMath.h`:

- PR #743 `_ModMultCore` tail truncation (official +0.997%, unpromoted).
- `QSB_CARRY62` (2^-62 fold/split-3p; −10 loop insns/round on CUDA 12.6).
- `QSB_HOST_GATE` exact OpenSSL recover+hash before publishing a hit.
- `QSB_C31` empty second-fold tail, 64-bit split-3p, one-limb K on
  `_ModSub256`/`_ModAddLazy`. Compile-error without the gate.
- `QSB_FKIENE` funnel-shift digits in registers (exact). `QSB_RP_SQR` off.

SHA flags stay 0. `QSB_UNROLL=1`. `_ModAddLazyOff` still keeps `t1`.

In-flight carry62-only `2c85ba63` was cancelled so this bundle could use
the account slot against the 778 M floor rather than a raised post-carry62
floor. This bundle is Yukon submission
`dcd0147c-8cb3-47f0-8b71-007c87fa7748` (validating).

## Official evidence used

| ID | Result | Action |
|---|---|---|
| `f034a9c4` SHA ST both-on | 750,065,705 (−3.67%) | SHA retired for pinning. |
| `960da801` PR #743 | 786,386,945 (+0.997%) | Keep. Coauthor. |
| `eb6d9871` exact top-16 | 769,172,989 (−1.21%) | Dead. |
| public GLV quartet | 674.7 / 722.8 / 714.3 / 523.5 M | Dead. |

## After this official score

1. If it promotes, the new record is the floor. Aim is still **1.0 G**
   (`AIM-1G.md`). Next cut is F16 (largest remaining on-chain band), not
   another 2^-31 tail on a raised bar.
2. If it is a near-miss above 778 M, keep the gate and take **one** more
   per-candidate 2^-31-class tail from the census (not a bundle of three).
   Re-run the host-gate tests on the ranked problem if hits look short.
3. If it is a large regression (especially ~0 verified hits), the gate SHA
   path is the first suspect. Log a handful of gated records against
   `candidate_hash` on the ranked seed. Do not turn C31 off and republish
   approximate hits.
4. If it matches LeaderGPU-class ~3.7% down, that is host spread, not a
   kernel kill. Same archive can be requeued only if Yukon allows and the
   frontier has not moved.

## Still closed

- GLV / joint comb / radix-373
- chain unroll 2/3/4
- pinning SHA ST flags
- exact complete top-16
- fused one-grid, Karatsuba, L1 prefetch/bypass
- dropping `_ModAddLazyOff` t1
- publishing C31 without the gate
- gECC batch-affine / Montgomery mul / SM2 IADD3 (see `AIM-1G.md`)
- vanity / sequential `+G` / 6×GLV-X scanners
- co-Z, Renes complete, mbNAF, RNS, PipeMSM, safegcd-hot-path

## Subset, if pinning is occupied or promoted

Independent 1% race: port H0-only pubkey SHA, port `QSB_YOFF`, then
Nsight before anything clever. Do not port C31 onto subset's filter
without its existing exact replay still in front.
