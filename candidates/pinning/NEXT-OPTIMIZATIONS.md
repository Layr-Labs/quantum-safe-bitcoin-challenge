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

## v11 additions (2026-09-22) — what remains open after this session

**Shipped in v11** (`d1ce6f2b`): `QSB_CHAIN_PIPE` (depth-1 gather pipeline,
dead-register prefetch, 3-buffer rotation), `QSB_DEC_REP`, `QSB_SUM_2U`,
`QSB_PREP_MASK`, `QSB_TREE_FLAT` — all on base `0ace23d4` (809.95M).

**Newly dead:** RAW_DIFF — see DEAD-ENDS.md (off-by-K per borrow; fkiene
bundle confirmed failed).

**Still open, ordered by leverage:**

1. **Duplicate/invalid candidate counting** (from v9 verdict): kernel-side
   dedup counter to split the self-rep vs verified gap
   (0.9734 -> 0.932 across our runs). The yield gap is the biggest
   unexplained lever; if it is verifier-side rejection of duplicates,
   dedup + re-search could recover several %.
2. **CHAIN_PIPE depth-2**: prefetch chunk c+2 as well. Register pressure
   is the constraint — the current pipe is already at 124/126 regs in the
   hot kernel; depth-2 needs another 8 dead registers per stage or a
   second rotating pair. Try only if v11 shows the depth-1 gain held.
3. **Sustained-rate levers** (v8/v9 finding): the yield gap tracks
   sustained-vs-peak decay (power/thermal droop over 20 min). Lower-power
   inner loops (fewer dual-issue conflicts, narrower voltage path) may
   out-score nominally-faster code. No concrete mechanism identified.
4. **The pending field queue**: fkiene resubmitted post-failure
   (1d72e1b) — if it promotes, rebase again and re-port the pipe; the
   rotation pattern is base-agnostic.
5. **sm_89-specific scheduling**: no cuobjdump/nvdisasm in the local
   toolkit (apt subset has nvcc+ptxas only). SASS-level reordering review
   of the pipelined loop is unverified — worth one pass if tools appear.
