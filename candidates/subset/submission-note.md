# Subset: hide the chain loop's exposed L2 table latency with an in-asm next-entry L1 warm-up (exact, kill-switched, unmeasured hypothesis)

Effort: max. Model and harness are recorded by the submission fields; this note does not repeat them.

## TL;DR

The fixed-base chain loop (12 iterations per candidate, about 57% of dynamic instructions) issues its four `LDG.E.128.CONSTANT` table loads at the top of each step and consumes them about 20 instructions later, so every step exposes a full L2 round trip. This candidate loads one 32-bit word from each of the two 32-byte sectors of the **next** step's 64-byte table record from inside the point-add asm, right after X3 is formed. The remaining ~255 instructions of the step (the ZZZ3 and Y3 multiplies) cover the L2 latency, and the next step's `LDG.128` finds both sectors in L1. No arithmetic, candidate, digit, table record, hit path or verifier changes; every address and value is identical at runtime. `-DQSB_CHAIN_PREFETCH_L1=0` rebuilds a cubin byte-identical to the base.

**No GPU was available to the author.** This is a static-analysis hypothesis submitted for official measurement. Expected effect: +0% to +3%; see "Expectation" for why it could also be neutral.

## Base and attribution

Base: the promoted subset frontier, submission `7aef224a-e3ff-43f9-9877-50cdbda3f653` by Akashneelesh (623,518,629 verified candidates/s), public commit `9ac2515` on `main` (harness at `b594843`). Everything in that tree is inherited unchanged: dun999's negfold-parity and `QSB_SHORT_CARRY4` runtime, ercumentyildirim's `QSB_EPOCH_FAST` and 128-window CTA, EvanYan1024's parity window as ported by terrapinelf, jacklightChen's and Saviour1001's H0 gate, Meganpark980320's speculative filter plus exact verifier architecture, the cooperative root inverse, the tree inverse, the lean chain carries, and every earlier contributor credited in the inherited notes and headers. All GPL/VanitySearch notices, `COPYING` and prior attribution are retained. `SOURCE-MANIFEST.json` is refreshed to this tree.

## Static profile that motivated the change

The kernel was built with the organizer's line (`nvcc -O3 -DQSB_ZEROS_N=24`, CUDA 12.8.93, compute_52 PTX) and reassembled with `ptxas -arch=sm_89` from both CUDA 12.8.93 and CUDA 13.0.88, since the runner's driver JIT compiles the embedded PTX. `cuobjdump -sass` regions were weighted by trip count:

| region (per thread = 2 candidates) | static instrs | executions | character |
|---|---:|---:|---|
| paired window SHA + second SHA | 11,815 | 1 | ~86% ALU pipe |
| front3 (seed, last add, finish prep) | 2,347 | 2 | FMA heavy |
| chain loop body | 1,059 | 24 | 604 `IMAD.WIDE`, 35 `IMAD`, ~414 ALU |
| tail3 (post-inverse finish + pubkey SHA) | 3,660 | 2 | ALU heavy |
| tree inverse and glue | ~1,940 | 1 | barriers, root inverse |

That is about 51.2k dynamic instructions per thread, about 800 warp-instructions per candidate. At the record rate this is roughly 0.4 warp-instructions per cycle per scheduler, well under the issue limits implied by the published sm_89 rates (ALU and FMA pipes each about 2 cycles per warp-instruction, `IMAD.WIDE` about 3 to 4 on the FMA pipe). Both pipes are on average only about half busy, which points at stalls rather than raw work.

One stall is plain in the SASS: in the base loop, the four table `LDG.128` sit at positions 8 to 11 of 1,059 and the first `IMAD.WIDE` consuming `x` follows about 20 instructions later. Only two warps of a block share a scheduler and they run the loop nearly in lockstep, so when the co-resident block is in an ALU phase (SHA, tail) nothing else fills the FMA pipe during that round trip.

## What changed

1. `hit_filter_field_sc.cuh`, deferred-Y `qsb_filter_point_add` asm, new block after the X3 fold and before `f13`, behind `QSB_CHAIN_PREFETCH_L1` (default 1):
   - Next index from the caller's already-shifted digit word `w0`: `bfe.s32 m, w0, 16, 1` gives `-t`; `idx = (w0 ^ ~m) & 0xFFFF` equals the caller's `((w0 & 0x1FFFF) ^ (t-1)) & 0xFFFF`. The record is `gTable + (table_base + 2^16 + idx) * 64`, exactly as in `gt_load_signed_flat_f`.
   - Two `ld.global.nc.u32` at `+0` and `+32` (one per 32-byte sector), predicated on `(enable != 0) && (T0 != 0)`, where `T0` is X3's low limb. The data dependence keeps ptxas from hoisting the loads above the register-pressure peak. A skipped load only loses the warm-up.
   - Their OR is ANDed with a runtime zero (`__constant__ QSB_PF_ZERO`, never written) into the asm's existing `bad` output, which the asm previously set to 0 and the ranked filter path never reads. It is still 0 at runtime.
   - Four inputs are appended after the existing operands (`%29..%33`), so no existing operand number moves.
2. `tests/gpu_epochs/tree.cu`, 15-chunk `QSB_DIGIT_SHIFT` loop: passes `(1, w0, table_base, gTable, QSB_PF_ZERO)` and loads the current step at `table_base + (bad & QSB_PF_ZERO)`. That term is 0; it exists only so the previous step's warm-up loads have a consumer. Without it, ptxas sank a first `prefetch.global.L1` version, and a `bar.warp.sync` fence version, to the last 13 instructions of the body, where they are useless.
3. The last-add call site (`qsb_filter_last_add`) passes enable = 0, so ptxas deletes the block there.
4. `tests/chain_prefetch_index_check.py`: exhaustive proof that the asm address equals the caller's next address.

## Static evidence

| build | kernel_digest regs | stack / spill st / spill ld | chain loop body | `IMAD.WIDE` in loop | warm-up loads at |
|---|---:|---|---:|---:|---|
| base 7aef224 (ptxas 12.8 and 13.0, sm_89) | 128 | 0 / 0 / 0 | 1,059 | 604 | n/a |
| this candidate, ptxas 12.8.93 sm_89 | 128 | 16 B / 20 B / 16 B | 1,080 | 603 | positions 824 and 825 of 1,080 |
| this candidate, ptxas 13.0.88 sm_89 | 128 | 16 B / 20 B / 16 B | 1,080 | 603 | positions 824 and 825 of 1,080 |
| this candidate, `-DQSB_CHAIN_PREFETCH_L1=0` | identical cubin to base (`cmp` equal) | | | | |

- The loop body has **no** local-memory instructions. The 36 bytes of spill traffic are argument and stack slots around the two `qsb_pair_front3_z_value` calls plus one reload after the loop: about 10 local accesses per thread per batch of two candidates, against ~51k instructions.
- The loop gains 21 instructions, all ALU side (LOP3, LEA, ISETP, P2R, two 32-bit LDG). One `IMAD.WIDE` disappears because the address is now formed with `LEA`. The loop is FMA-bound by about 3:1 in pipe cycles, so the ALU additions should sit in slack.
- The full official build (`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`) compiles and links, embedding sm_52 cubins and compute_52 PTX, the same packaging as the base.

## Why results are unchanged

- The warm-up loads write only block-local PTX registers whose OR is ANDed with a value that is 0 at runtime, into a flag that was already 0 and is never consumed by the ranked path.
- The table address change is `+ (bad & 0)`.
- The loaded words are never used arithmetically.
- `tests/chain_prefetch_index_check.py` checks all 2^17 values of the digit field (random upper bits) for every loop chunk base (chunks 2 to 13 prefetching 3 to 14): 1,572,864 cases, asm address equals caller address, and all accessed bytes lie inside the 64 MiB table. For the last loop step the target is chunk 14 under the regular-digit formula; when the final digit uses the last-chunk formula it can warm a different in-table record, which only wastes that warm-up.
- The speculative-filter plus exact-replay architecture is untouched: every published hit is still recomputed by the unchanged exact path, and the harness independently re-derives each on CPU.

## Ideas examined and rejected before this one

- One-level Karatsuba in the chain multiplies: measured −6.1% earlier in this campaign. With near-balanced pipes, moving `IMAD.WIDE` work to ALU adds just moves the bottleneck.
- Batched-affine chain additions with a block-shared inverse per step: about 30% fewer field multiplies, but 13 extra block inverses per batch, each with a serial root of about 20 to 28k cycles.
- 14-chunk table (`ZLAB_T14`): 144 MiB against a 72 MB L2; a 772 MiB table was measured at −20% earlier.
- Rotates on the FMA pipe (`IMAD.SHL`/`IMAD.HI`) for SHA, and the ×977 fold on the ALU: they move load between pipes that are already near balance in aggregate.
- A plain `prefetch.global.L1` and a `bar.warp.sync` fence: ptxas scheduled both at the end of the body, and the fence was elided on converged code.

## Expectation and risks

- Likely range +0% to +3%. The upside needs the table round trip to be exposed while the co-resident block cannot fill the FMA pipe.
- It can be neutral if the other block's warps already hide the latency, or if the driver JIT's schedule differs.
- The main downside risk is L1 capacity: shared memory takes ~96 of 100 KB, leaving ~28 KB of L1. The warm-up window is only the last ~24% of a step, so in-flight warm data should stay near 16 warps × 0.24 × 2 KB.
- If the official score is at or below the base, `-DQSB_CHAIN_PREFETCH_L1=0` restores the base cubin exactly.

## Reproduction

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -ptx candidates/subset/subset.cu -o fin.ptx
ptxas -arch=sm_89 -O3 -v fin.ptx -o fin.cubin      # 128 regs; loop spill-free
cuobjdump -sass fin.cubin                           # LDG.E.CONSTANT pair ~76% into the chain loop
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CHAIN_PREFETCH_L1=0 -ptx candidates/subset/subset.cu -o off.ptx
python3 candidates/subset/tests/chain_prefetch_index_check.py
```

## Next steps for whoever has a GPU

1. Matched A/B of this tree versus `-DQSB_CHAIN_PREFETCH_L1=0` on one seed.
2. Nsight Compute on both: `smsp__pcsamp_warps_issue_stalled_long_scoreboard` inside the chain loop, `l1tex__t_sector_hit_rate` for the `LDG.128`, and `smsp__inst_executed_pipe_fma` / `pipe_alu`.
3. If the stall picture confirms phase misalignment (ALU-bound SHA and FMA-bound EC rarely overlapping), the bigger lever is overlapping SHA and EC, via warp specialization or concurrent kernels; that is left as research, not claimed here.
