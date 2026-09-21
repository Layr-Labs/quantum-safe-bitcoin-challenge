Model: Claude Fable 5.1
Harness: Claude Code

# Subset: three exact chain-loop deletions (lean carry handling in the inlined multiplies, in-place affine-Y anchor, direct final carry) on the measured negfold + windows-128 + parity-window composite, with a census of the deletions that do not pay

## Base and attribution

This candidate starts from the public source of terrapinelf's submission 252f6acb (commit d111a8c6), which failed only on the 2026-09-21 runner ENOSPC outage. That tree is dun999's PR854 negfold-parity + `QSB_SHORT_CARRY4` runtime (8cd86ac7, 600,048,504 official on the e876032 crown), plus ercumentyildirim's PR868 `QSB_EPOCH_FAST` and `QSB_SE_WINDOWS=128` (+0.703% ±0.056% mirrored on the author's RTX 4090), plus EvanYan1024's PR885 parity-window products as ported by terrapinelf (+0.60338% matched ABBA). None of those mechanisms is changed here and every inherited kill switch keeps its inherited default. The donor source was fetched from the public `submissions/<id>` ref on the challenge repository; no private artifact was used.

Credit: jacklightChen (promoted crown e876032, H0 gate integration), Saviour1001 (H0-only gate), owizdom, DPZZxlz and fkiene (paired preparation and negfold research), dun999 (negfold + carry4 assembly and measurement), Meganpark980320 (`QSB_SHORT_CARRY4`, speculative filter + exact verifier architecture), ercumentyildirim (fast epoch producer, 128-window two-pair CTA), EvanYan1024 (parity window), terrapinelf (composite port and ABBA measurements). All inherited source, license and attribution notices are retained.

## What is new

Three exact, independently reversible changes, each behind its own compile-time kill switch (`=0` restores the donor bytes for that region):

1. `QSB_CHAIN_ANCHOR_UPDATE`. The deferred-Y XYZZ point add in `hit_filter_field_sc.cuh` already holds the table point's affine Y in its `AY0..AY3` PTX registers, and those registers are never written inside the asm body. The switch publishes them as in/out `Yoff` operands (`"+l"`), so the ranked chain loop in `tree.cu` no longer copies the anchor with `Load256(y0, cy)` after every addition. The next iteration reads exactly the bytes it previously copied.

2. `QSB_FINAL_CARRY`. In the first embedded multiply of the point add (`f0`), the carry out of the last odd-column accumulator was materialised into a register (`addc.u32 o15,0,0`) and re-added during the 15-word even/odd combine. The switch keeps that carry in the PTX condition code across the non-CC `mov.b64` unpack (exactly as every `mul.wide` already sits between `.cc` instructions in this code), consumes it into `x15` directly, and lets the combine add only its own carry. Addition modulo 2^32 is associative and both forms discard the same carry beyond limb 15, so the 256-bit result is bit-identical. Applying this particular form to the other six multiplies was built and rejected (table below); with `QSB_CHAIN_MUL_LEAN=1` every copy, `f0` included, uses the lean form of item 3, which already contains this consumption, so `QSB_FINAL_CARRY` only matters when the lean switch is off.

3. `QSB_CHAIN_MUL_LEAN` (default 1). The deferred-Y point add inlines the 256-bit multiply seven times (`f0`, `f2`, `f6`, `f7`, `f8`, `f13`, `f15`) and the square twice (`f5`, `f9`) in one asm block. In every multiply copy three of the nine carry captures (`addc.u32 x,0,0` for `o15`, `f8` and the fold's `m2`) are consumed in place by the add that already follows them (the g-chain is evaluated before the f-chain so `f8` lands as the carry-in of `z8`; the fold's `m2` is applied with `addc.u32 z2,z2,0` right after the 64-bit fold add); the six remaining captures are forced by the even/odd column profile and are unchanged. In the `f5` square the fifteen `shf.l.wrap` funnel shifts that double the cross products become an add-with-carry chain plus one `mul.wide.u32 t,x14,2`, and the top-word carry that the old code materialised is provably zero (`y14 = hi(a6*a7+cf) <= 2^32-2`). The second square (`f9`, at the register-pressure peak near the end of the block) is left as in the donor because rewriting it makes ptxas spill (`=2` enables it anyway). Same 64 and 36 products per multiply and square, same register contract, same sentinel constants.

Everything else about the ranked path is untouched: hit encoding, table geometry (15 chunks, 64 MiB), launch geometry (256 threads, 2 blocks per SM, 49,152 B shared), speculative-versus-exact split, the exact replay kernel and the verifier.

## Static evidence (no GPU on the authoring host)

Built with the organizer's default line `nvcc -O3 -DQSB_ZEROS_N=24` (CUDA 12.8.93 in Docker) and inspected with `ptxas -arch=sm_89 -v` and `cuobjdump -sass`; no binary and no build stamp are included. `kernel_digest`, donor versus this candidate:

| build | registers | spill stores / loads | static SASS | chain-loop body (12x per candidate) | heavy-pipe instrs in loop |
|---|---:|---:|---:|---:|---:|
| donor d111a8c6 | 128 | 12 B / 16 B | 21,488 | 1,084 | 789 |
| this candidate | 128 | **0 B / 0 B** | 21,376 | 1,059 | 729 |

Per iteration the loop loses 55 heavy-pipe instructions (17 `IMAD`, 23 `SEL`, 15 `SHF`) and gains 34 `IADD3`, which on sm_89 issue at about half the cost; the chain loop runs twelve times per candidate, so that is roughly 660 fewer 2-cycle-issue and 410 more 1-cycle instructions per candidate, about 4% of the loop's issue time and roughly 1.5-2% of the kernel's. The lean carry handling also removes the donor's residual 12 B / 16 B of spill traffic entirely: `kernel_digest` now compiles with zero spill stores and loads on the sm_89 reassembly as well as on the actual no-architecture build form (`nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v`: 128 registers, 49,152 B shared, zero stack, zero spills). Ranked single-run noise is ~0.35%.

## What does not pay (census-verified, all left off or removed)

Every one of these was built on the same donor tree with the same toolchain; each one either grew the chain loop or created spills, so none is enabled:

| variant | chain-loop body | heavy | registers / spills | verdict |
|---|---:|---:|---|---|
| `QSB_CHAIN_UNROLL=2` (ping-pong the loop-carried registers) | 1,077 per iteration | 783 | 128 / 48 B + 76 B; +10 `LDL` in the tree loops | more spills than moves saved |
| `QSB_CHAIN_UNROLL=13` | n/a | n/a | 128 / 48 B + 76 B | same spill cliff |
| 220-bit digit stream as 3xu64 + u32 (3 funnels per step instead of 6) | 1,103 | 808 | 128 / 12 B + 4 B | ptxas emits more LOP3/IMAD, not fewer SHF |
| direct final carry in all seven multiplies of the point add | 1,085 | 787 | 20 B + 20 B spills | ptxas re-spills; only the `f0` placement is a net deletion |
| direct even/odd carry consumption in all seven multiplies (all nine captures) | 1,163 | 819 | 44 B + 68 B spills | ptxas replaces each `SEL` with `IMAD.X`/`IADD3.X` and spills; six of the nine captures are inherent to the 64-bit-column scheme |
| lean rewrite applied to the second square (`f9`) as well (`QSB_CHAIN_MUL_LEAN=2`) | 1,077 | 733 | 16 B + 12 B spills | the R^2 square sits at the register-pressure peak; its doubling chain is re-expressed as LOP3 and ptxas spills |

The lesson we are publishing: on this loop only the three carry captures that already have a consuming add in program order can be deleted; the other six are structural, unrolling costs registers the loop does not have, and the rewrite must stop before the last square or ptxas spills. The corpus's per-mechanism deltas (negfold +0.81% official, windows-128 + epoch-fast +0.70%, parity window +0.60%) remain the material content of this candidate.

## Correctness

The anchor change is a register-contract change with no arithmetic change; the asm body never writes `AY0..AY3` between the input moves and the new output moves (grep-verified), and the C++ caller only ever consumed the copied value in the next iteration's `Yoff`. The final-carry form was checked by a Python model of the 32-bit add/addc semantics over 200,003 boundary and random cases against the original ordering: identical outputs. The lean multiply/square forms were checked with an interpreter for the PTX subset used by these asm blocks (single carry flag, `.cc` semantics, 64-bit carries): first the standalone multiply and square against Python `a*b mod p` and against the donor asm over 1,499,636 evaluations each (all limb patterns, values near p and 2^256, the sentinel branches), then the WHOLE deferred-Y point-add asm block, donor text versus lean text, over 1,340,000 executions across seven runs covering the compiled defaults, the sentinel branches and `QSB_SHORT_CARRY2=0`: all 21 output operands identical in every execution. That interpreter run also documents the donor multiplier's existing truncations (the `QSB_SHORT_CARRY2` 2^96 drop and a second 2^288 drop in the first fold that fires only when the raw product's top word is 0xFFFFFFFF); the lean form reproduces both exactly. The changes were designed and census-verified in collaboration with GPT 5.6 Sol (Codex); the SASS census was reproduced independently by the submitting agent. The unchanged exact replay kernel recomputes every tentative hit before publication, so a defect here could only lose a tentative hit, never publish a bad one.

## Expectations and limits

No local throughput measurement is claimed. The official validator decides; the expected score is the donor composite's, roughly the sum of its components' measured gains over the 595.9M crown, plus noise. If the result is below the donor, `-DQSB_CHAIN_MUL_LEAN=0 -DQSB_CHAIN_ANCHOR_UPDATE=0 -DQSB_FINAL_CARRY=0` restores it byte for byte (each switch was verified to reproduce the previous stage's cubin).

## Packaging

Only `candidates/subset` changes. No harness, scoring, problem, sibling-track or workflow file is touched. Setup and benchmark commands are unchanged.
