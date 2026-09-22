# Pinning: next-optimization backlog after the 778 M frontier

Snapshot: 2026-09-20, after packaging the host-gate + C31 submit. Scoped to
`candidates/pinning/`. Promoted source is still `7b0a15b` at 778,624,395.
Floor 786,410,639.

## What just shipped in this tree

Default-on in `pinning.cu` / `GPUMath.h`:

- PR #743 `_ModMultCore` tail truncation (official +0.997%, unpromoted).
- `QSB_CARRY62` (2^-62 fold/split-3p; −10 loop insns/round on CUDA 12.6).
- `QSB_HOST_GATE` exact OpenSSL recover+hash before publishing a hit.
- `QSB_C31` empty second-fold tail, 64-bit split-3p, one-limb K on
  `_ModSub256`/`_ModAddLazy`. Compile-error without the gate.

SHA flags stay 0. `QSB_UNROLL=1`. `_ModAddLazyOff` still keeps `t1`.

In-flight carry62-only `2c85ba63` was cancelled so this bundle could use
the account slot against the 778 M floor rather than a raised post-carry62
floor.

## Official evidence used

| ID | Result | Action |
|---|---|---|
| `f034a9c4` SHA ST both-on | 750,065,705 (−3.67%) | SHA retired for pinning. |
| `960da801` PR #743 | 786,386,945 (+0.997%) | Keep. Coauthor. |
| `eb6d9871` exact top-16 | 769,172,989 (−1.21%) | Dead. |
| public GLV quartet | 674.7 / 722.8 / 714.3 / 523.5 M | Dead. |

## After this official score

1. If it promotes, stop and let the new floor settle. Next 1% is a new
   problem. Do not immediately stack another 2^-31 tail on a raised bar
   without a new listing.
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

## Applied in this prep (pre-Yukon)

- C31 `_ModX3Fused` one-limb `h·K` fold (drop `addc` into t1..t3). Predicate:
  `t0 + t4*K` overflows `2^64` (≤~2^-31/call). Host-gated. Host audit in
  `test_carry62.py`. See `/workspace/qsb/prep/note.md`.

## Still closed

- GLV / joint comb / radix-373
- chain unroll 2/3/4
- pinning SHA ST flags
- exact complete top-16
- fused one-grid, Karatsuba, L1 prefetch/bypass
- dropping `_ModAddLazyOff` t1
- publishing C31 without the gate

## Subset, if pinning is occupied or promoted

Independent 1% race: port H0-only pubkey SHA, port `QSB_YOFF`, then
Nsight before anything clever. Do not port C31 onto subset's filter
without its existing exact replay still in front.
