# Outlined bounded HM43 native review

Source `a36071b7a5b06c22d16172f8f29ec0d86eb1cd90620e3716526b40349a890047` is the stronger bounded native candidate than inline f47. Outlining restores the exact ready3ac ordinary point loop and SHA producer after address normalization. No GPU timing or performance improvement is established.

`review_native.py` reads saved native reports/SASS, checks the exact 19-file production/audit source union, and verifies that the only source change from f47 is `hm43_warp_inverse`'s `__forceinline__` to `__noinline__` declaration. Both outlined entry points compile for sm89 and default flags. Full source/report/SASS hashes and instruction listings are in `native-review.json`.

| Observed metric | Ready3ac | Bounded inline f47 | Bounded outlined a360 |
|---|---:|---:|---:|
| Registers / shared bytes | 80 / 32832 | 80 / 32832 | 80 / 32832 |
| Stack bytes | 352 | 368 | 344 |
| Compiler aggregate spill stores / loads | 552 / 384 | 556 / 344 | 552 / 388 |
| Kernel non-NOP sites | 15455 | 16550 | 16642 |
| Ordinary loop instruction sites | 1427 | 1446 | 1427 |
| Local read / write operand bytes over 13 adds, per active candidate | 0 / 0 | 104 / 104 | 0 / 0 |
| Producer instruction sites | 3535 | 3536 | 3535 |

The outlined ordinary loop is `0x4450–0x9d70`, with the exit at `0x9d60` and backedge at `0x9d70`. Its instructions equal ready3ac after return-target normalization. It retains four 128-bit global table-load sites, 22 64-bit shared-load sites, 12 64-bit shared-store sites and two late prefetch sites per iteration. There are no LDL/STL instructions in this repeated region. This is stronger evidence than interpreting the whole-kernel compiler spill totals.

The SHA producer is `0x29690–0x37370`, 3535 sites and local-memory-free. Its instructions equal ready3ac after relocation. Consumer polling also matches after relocation and stack-offset normalization: one leader atomic at `0x530`, with 8 bytes of local operand reads per leader spin. Producer polling has one leader atomic at `0x35f10` and no local traffic. These facts do not establish polling latency or atomic cost.

The new cooperative root is a real out-of-line call, `0x1ab60 → 0x37690`, reached only by physical threads 64–95. The caller materializes a 40-byte local root array per participating lane: five zeroed 64-bit words, then four leader-only input stores. The callee reads those words and uses the local array during result correction/canonicalization. This is root-stage work per 192-candidate cohort; it is separate from point-loop work per active candidate.

The outer matrix main region `0x37af0–0x39280` has 378 static sites and one 4-byte local context-reload site (`0x38e10`, `[R1+0x9c]`), outside the inner divstep loop. The preceding `0x38e00 @P0 BRA 0x38fc0` bypasses it: only lanes reaching this conditional path execute the load. Four bytes is the operand width, not unconditional traffic for all32 lanes or all batches. The matrix's compiler outline regions `0x3a0d0–0x3b390` have 301 sites and no local-memory instructions. Unlike f47's matrix, the outlined matrix is therefore not entirely local-memory-free. Its exchange helpers additionally contain WARPSYNC/SHFL or WARPSYNC/VOTE instructions. BRA.DIV alternate paths, inner loops and helper calls prevent treating these region counts as a dynamic instruction total per batch.

The native 16-batch test is at `0x37af0`, with cap failure branching to `0x39b10`; `0x39260` increments the batch count and `0x39280` returns to the cap check. The success branch exits when the V ballot is zero. Failure returns to the caller; only EC lane zero reloads the original shared root at `0x1abf0–0x1ac20`, zeros its fifth word at `0x1ac30`, and calls legacy `_ModInv` at `0x1ac80 → 0x3bdc0`. Both outcomes publish the four result words at `0x1ace0–0x1ad10`, followed by the existing warp/publication synchronization. No unconditional 16-batch termination assumption is used.

The caller region has 36/116 static local read/write operand bytes; the entire cooperative callee including its compiler outlines has 276/312. These are counts across instruction sites, including mutually exclusive paths, not bytes dynamically executed per root or per lane. The saved JSON lists each local instruction so those costs are not hidden behind the restored point-loop result.

Prefer a360 over f47 for further qualification because it removes the integration spill regression while retaining the bounded algorithm. Relative to ready3ac, it still adds full-warp inverse arithmetic, collective operations, a call/local-array ABI and fallback control. Static inspection cannot show that this lowers the root critical path or improves total throughput. Equal register/shared resources do not establish achieved occupancy. Candidate sources were not edited, and no build, VM operation or GPU execution was performed by this review.

Reproduce from the repository root:

```sh
python3 -B candidates/subset/research/warp_root/bounded/outlined/review_native.py
```
