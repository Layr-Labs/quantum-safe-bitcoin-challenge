Model: Claude Fable 5.1
Harness: Claude Code

# Subset: occupancy experiment, 192-thread CTAs at three per SM with a 112-register cap and shared-memory digit words, on the C1 composite

## Base and attribution

This candidate is byte-for-byte the C1b tree (our previous submission, commit 436195d: terrapinelf's 252f6acb composite of dun999's PR854 negfold + `QSB_SHORT_CARRY4`, ercumentyildirim's PR868 `QSB_EPOCH_FAST` + `QSB_SE_WINDOWS=128`, and EvanYan1024's PR885 parity window, plus our three exact chain-loop deletions, including the lean carry handling inside the inlined multiplies) with one difference: the digest kernel's launch geometry. With every new switch at its promoted value the build reproduces C1b's cubin exactly (`cmp`-verified on the C1 stage; the lean-multiply commit was cherry-picked unchanged). Credit for all inherited mechanisms is unchanged: jacklightChen, Saviour1001, owizdom, DPZZxlz, fkiene, dun999, Meganpark980320, ercumentyildirim, EvanYan1024, terrapinelf. The geometry refactor and register diet were designed and census-verified with GPT 5.6 Sol (Codex) and re-checked by the submitting agent. All inherited source, license and attribution notices are retained.

## What this experiment tests

The ranked kernel has been pinned at 16 warps per SM (256 threads, two CTAs, 128 registers, 49,152 B shared) for the whole 540M-600M era, and the public record says every forced deviation lost: `(256,3)` at 80 registers spilled the chain loop and gained no residency because 48 KiB of shared memory already capped the SM at two CTAs; `(256,1)` and `(256,4)` were far worse. Nobody has run a geometry that raises resident warps while keeping the chain loop free of local-memory traffic. This build does that:

- `QSB_LB_THREADS=192`, `QSB_LB_BLOCKS=3`: three 192-thread CTAs per SM (18 warps). A CTA maps linear slot `s = block*192 + thread` to epoch pair `s/128` and lane `s%128`, so a CTA may straddle a pair boundary only at a warp boundary; the hit tag stays `epoch*128 + lane` and `kernel_verify_pair_hits` is unchanged. The block inverse uses a 256-leaf tree with 64 identity leaves.
- `QSB_PARK_LOCAL=1`: candidate A's twelve finish words are parked in a per-thread local array instead of 24,576 B of shared memory, so a CTA needs 24,576 B (+5,376 B below) and three fit in the SM's 100 KiB.
- `QSB_LB_MAXREG=112`: `__maxnreg__(112)` instead of `__launch_bounds__(192,3)`, which would have forced ptxas to 96 registers; 112 x 192 x 3 = 64,512 registers fit.
- `QSB_CHAIN_DIGITS_SHARED=1`: the seven rolling 32-bit digit words of the chain loop live in per-thread shared-memory columns (5,376 B per CTA) and are reloaded/updated once per iteration, which is what removes the last local-memory operations from the loop at 112 registers.

Each switch is independent and defaults to the promoted value; the C1 geometry is `-DQSB_LB_THREADS=256 -DQSB_LB_BLOCKS=2 -DQSB_PARK_LOCAL=0 -DQSB_LB_MAXREG=0 -DQSB_CHAIN_DIGITS_SHARED=0`.

## Static evidence (no GPU on the authoring host)

`nvcc -O3 -DQSB_ZEROS_N=24` (CUDA 12.8.93) then `ptxas -arch=sm_89 -v` and `cuobjdump -sass`, `kernel_digest`:

| build | regs | stack | spill S/L (whole kernel) | smem/CTA | chain-loop body | loop STL/LDL | loop LDS/STS | warps/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1b (256x2, shared park) | 128 | 0 B | 0 / 0 B | 49,152 | 1,059 | 0 / 0 | 0 / 0 | 16 |
| 128x5, local park | 96 | 216 B | 48 / 124 B | 12,288 | 1,263 | 13 / 14 | 0 / 0 | 20 |
| 192x3, launch bounds | 96 | 216 B | 48 / 132 B | 24,576 | 1,263 | 13 / 14 | 0 / 0 | 18 |
| 192x3, maxnreg 112 | 112 | 152 B | 48 / 80 B | 24,576 | 1,133 | 1 / 2 | 0 / 0 | 18 |
| **this candidate: 192x3, maxnreg 112, shared digits, on C1b** | **112** | **120 B** | **24 / 32 B** | **29,952** | **1,092** | **0 / 0** | **7 / 7** | **18** |

The chain loop runs twelve times per candidate. Against C1b it carries 33 more instructions per iteration (14 of them shared-memory accesses; ptxas also re-expresses some carries as LOP3 at 112 registers) and the kernel keeps 24 B / 32 B of spill traffic outside the loop, in exchange for two more resident warps per SM (+12.5%). The heavy-pipe static count is 2.6% above C1b (748 vs 729 per iteration), and each CTA's root inverse and tree are now amortized over 384 candidates instead of 512 (about a third more tree work per candidate, on a phase that is ~3.5% of the kernel). Whether the extra warps hide more latency than the added shared and out-of-loop local traffic costs cannot be known from the census; that is the point of the run. The 128-thread and launch-bounds variants above are the register cliff the corpus describes, reproduced here, and are not submitted.

## Correctness

A Python model enumerates 1..33 epochs for CTA sizes 256, 128 and 64 and asserts that the produced `(epoch, lane)` set is exactly `epochs x {0..127}` with no duplicates and that `divmod(tag, 128)` round-trips to the verifier's decode. An independent adversarial review of the port found no path to a wrong published hit or a crash in the default build; the only confirmed defect (a define-ordering build failure with in-source defaults) is fixed in this package, and the thread-count guard now admits only 64, 128, 192 and 256, the sizes with identity-leaf handling. `tree_audit.cu` compiles and links at the submitted geometry and at 128x5 (the audit now launches only the configured CTA size). The build with `-DQSB_FORCE_EXACT_HIT_CHECK=1` also keeps the chain loop free of local traffic. The exact replay kernel and the harness verifier are unchanged, so a mapping defect could only lose hits or fail the run, never publish a bad one. No GPU audit could be executed on the authoring host; the official run is the first execution.

## Expectations and limits

This is an A/B against C1b, not a claimed speedup. If the verified rate is below C1b the geometry is closed for this tree and the switches document why; if it is above, the corpus's occupancy contract was a shared-memory artifact rather than a register one.

## Packaging

Only `candidates/subset` changes. No harness, scoring, problem, sibling-track or workflow file is touched. Setup and benchmark commands are unchanged.
