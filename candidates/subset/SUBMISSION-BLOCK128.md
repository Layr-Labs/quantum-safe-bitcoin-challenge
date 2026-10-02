# Subset: four smaller independent digest CTAs instead of two large CTAs

Effort: xhigh

## Context and base

This candidate is based on cefika's promoted `fb6f5a8f-b29e-4506-a50c-c79a9c5a2a0e`, commit `ff27a2b66990a3eb554a1d4453e896c0397337ba`. Its promoted score was 728,337,167 verified candidates per second when the board was rechecked immediately before preparing this submission. The preceding promoted submission is kshitij-hash's `e6715658-2270-4be9-8caa-9a7c7e072dd3`. I read both public notes and compared the promoted tree against this development branch rather than assuming this branch's existing defaults were a faithful control.

The development branch had previously enabled split-tail staggering and disabled vectorized shared-memory parking. Those settings were not the promoted kernel. This experiment restores the promoted settings (stagger 0, park 1) and measures a materially different change: halving the digest CTA from 256 threads to 128 threads, and actually halving all of its live shared buffers. The host co-grinder, its contiguous disjoint epoch ranges, its arithmetic, and its SHA implementation are inherited unchanged.

## Hypothesis

The digest kernel uses 128 registers per thread and 48 KiB of shared memory for a 256-thread block. The default launch bound asks for two such CTAs, hence 512 threads and sixteen warps per SM. Much of the kernel is serially dependent point arithmetic and SHA, interrupted by a cooperative product-tree inversion. Splitting the same sixteen warps into four independent CTAs can change barrier and root-inversion scheduling even without increasing the number of resident warps. The hypothesis is not that nominal warp occupancy increases: it does not. The change supplies four independent product trees and four independent roots, and removes one level from each upward and downward sweep, at the cost of doing twice as many root inversions per candidate.

Whether the scheduling benefit outweighs that cost is an empirical question. The local interleaved scored measurements below support this candidate, but do not prove that Ampere and Ada have identical speedups.

## What changes

`subset.cu` selects `QSB_SE_BLOCK=128` unless an experiment wrapper overrides it. `tree.cu` makes that block size a guarded configuration, permitting 128 or 256 threads and requiring complete window sets. The promoted `QSB_SE_WINDOWS=128` is unchanged. A 128-thread CTA therefore covers one pair of epochs rather than two; each thread still recovers and tests the same two candidate points.

The digest launch bound becomes `(QSB_SE_BLOCK, 512/QSB_SE_BLOCK)`. This is `(128,4)` for the candidate and `(256,2)` for the promoted control. The digest's park buffers use the selected block dimension. The level-packed tree's immutable product buffer, file-scope product arena used to park the first candidate's denominator, and downward inverse buffer also use it. Shrinking only the locally declared tree buffer would not suffice: the file-scope product arena is the active default path. An initial incomplete version retained that arena at its original size; its poor result led directly to locating and fixing the missed allocation.

The existing shared-memory LUT variant is not enabled in this submission. Its storage retains its original 832-word capacity; shrinking that table to the size of a 128-leaf inverse array would be incorrect. The default inverse buffer can be smaller because it does not host that table. The PRE3 callback's park pointer type follows the block dimension, but PRE3 itself is off.

The actual field multiplication, point addition, divstep inverse, affine resolution, SHA rounds, rejection filter, exact hit verification, and hit record format are unchanged. The epoch-pair count is derived from block size through the existing `QSB_SE_HALVES` and `QSB_PAIR_MUL`. Allocation sizes, launch counts, descriptor indexing, first-stage builds and the late hit-index reconstruction already use those quantities. No candidate identity or omission-pattern selection is changed.

`stack_ab.sh` now uses an explicit promoted control rather than the modified stagger arm. It holds the shared GPU lock for each benchmark and artifact copy, propagates failed verifier exits, saves a unique directory of logs and scores, and computes the improvement from verified-hit-derived scores. Its order is A/B, B/A, A/B. This prevents the stale-score and wrong-control problems observed in earlier development tooling.

## Correctness reasoning

Every active lane still identifies the same window pattern within an epoch. The block has exactly 128 lanes, so it contains a complete window set. The existing paired epoch multiplier becomes two; successive blocks receive successive disjoint pairs. Hit tags use the same epoch-times-window plus lane expression, including the late special-register reconstruction. Inactive candidates continue to contribute identity factors to the tree and cannot emit hits.

The level-packed inverse algorithm already accepts power-of-two blocks up to 256. Its level offsets, child indices, parent indices and leaf inverse calculation are functions of the actual block dimension. Its active buffers now have room for twice that dimension for products and one dimension for inverses. The root wave still operates on the same top sixteen nodes and uses the same warp-uniform inverse. Existing barriers and warp synchronizations are retained. The early return remains block-uniform, and partial last batches retain all participating lanes through the inverse.

The host stop signal, drain path, CPU worker lifecycle, CUDA error checks, exact host hit reconstruction and OpenSSL verification are inherited. There are no new lookup caches, persistent benchmark inputs, cached hits, scoring changes or external runtime dependencies.

## Exact development commands

The local wrapper defines `QSB_LOCAL_SM86=1` solely because this shared development machine is an RTX 3090 with a desktop session; the promoted 21-GiB GLV11 table cannot safely coexist with it. Both A and B use the same existing smaller GLV12 geometry. Ranked compilation does not enable that local switch and retains the promoted ranked GLV11/P18 and mixed-Q geometry.

Build each local arm using the unchanged harness compiler convention:

```sh
cd candidates/subset
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o n24L_promoted n24L_promoted.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o n24L_block128 n24L_block128.cu -lcrypto -lm
cd ../..
bash candidates/subset/stack_ab.sh block128 3 promoted 120
```

The runner uses `harness/gpu_wrap.py` and the manifest's `benchmark.sh subset`, with `QSB_ZEROS_N=24`, `QSB_MODE=fixed_time`, `QSB_SECONDS=120`, `QSB_MAX_REL_VAR=none` and problem seed 1789110211. Every GPU invocation holds `flock /tmp/angel-gpu.lock`. Disabling the short-run variance cap is a development setting only; all scored A/B arms had roughly 4,500--5,000 independently verified hits and the ordinary exact hit verifier passed.

Regenerate the ranked image using the pre-existing builder and CUDA 12.8:

```sh
bash candidates/subset/build_carrier.sh 24
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/subset/subset candidates/subset/subset.cu -lcrypto -lm
```

The development toolchain initially lacked cuobjdump and nvdisasm. Installing matching CUDA 12.8 binary tools resolved this build-tool blocker. The builder checked all required kernels and globals, and found the same four LTC64B digest loads as the inherited image. These tools are development-only, not submission runtime dependencies.

## Measured results

The complete alternating A/B, in millions of verified candidates per second:

| Pair | Promoted control, 256 threads | Candidate, 128 threads | Order |
|---|---:|---:|---|
| 1 | 317.379041 | 344.324586 | control, candidate |
| 2 | 310.195276 | 335.292389 | candidate, control |
| 3 | 315.001493 | 327.267847 | control, candidate |
| Mean | **314.191937** | **335.628274** | |

The mean candidate/control improvement is **+6.822689%**, exceeding the standing local submission gate of +4%. All six benchmark invocations returned PASS with every emitted hit verified. The three candidate hit counts were 4,955, 4,818 and 4,706, respectively. The three control hit counts were 4,586, 4,470 and 4,547. An earlier fully shrunk candidate smoke run also passed with 5,181/5,181 verified hits and 360.768248 M/s. That isolated smoke result is not used to claim the speedup; only the completed alternating comparison is.

CUDA 12.8's ranked sm_89 compile reports 128 registers, 24,576 bytes shared memory, zero stack frame, zero spill stores and zero spill loads for `kernel_digest`. The local promoted control reports 128 registers and 49,152 bytes shared memory. The generated candidate cubin is 473,376 bytes, SHA-256 `d938cf8a0d9f28dfe9af5ebf9565d0723009fb3a17a169a6520ac72455832f80`. The fixed default-target host executable compilation also succeeded.

## Failures and caveats

The first lever requested for this iteration was the stack of OUTER_LITK, staggering and PRE3_ROOT. It passed the exact verifier but scored 338.397645 M/s against an isolated promoted control at 360.127130 M/s, about -6.0%. I did not submit it. This is consistent with the existing PRE3 spill/regression evidence, and motivated changing directions instead of collecting more marginal stack combinations.

The first block128 version retained the file-scope product arena. It compiled at 32 KiB shared, passed correctness, but scored only 282.926067 M/s. The winning revision shrinks that actual active arena and compiles at 24 KiB. That distinction is essential; changing only the launch dimensions is not the measured candidate.

The local machine is RTX 3090, not the ranked RTX 4090, despite the inherited artifact's GPU label saying RTX_4090. Local tables are deliberately smaller for both arms. The machine is shared and thermally constrained; absolute results drifted during the comparison. Hit-derived scores have sampling noise. The repeated interleaved comparison reduces these effects but cannot remove architecture and geometry differences. No official score is claimed in this note; the ranked harness must establish it.

The inherited candidate counter can undercount the aggregate CPU-plus-GPU work, triggering the harness's non-rejecting Poisson-band warning in several runs. The submitted change does not adjust counters or scoring. All improvements here are computed from the benchmark's verified hit estimator, not those counters.

## Attribution and next work

Cefika supplied the promoted base and contiguous host epoch walk. Kshitij-hash supplied the preceding record's composition and device configuration. The inherited lineage includes terrapinelf, i34-9, ercumentyildirim, HyeokxC, jacklightChen, fkiene, kaankolcu, newjordan, Meganpark980320, Ryun1, RealAdii and contributors named in the promoted notes. All existing notices are retained. Device arithmetic derives from VanitySearch under GPLv3; the host field/scalar lineage includes libsecp256k1 under MIT.

The new contribution is the dimensioned four-CTA digest layout, including the active shared product arena; faithful promoted controls; failure-propagating, artifact-preserving alternating scored comparisons; and the regenerated native image for this configuration. Local ELF experiment binaries and large board snapshots are excluded from the submission. The source wrappers and compact measurement records remain for reproducibility.

Next experiments will measure launch fill and register/CTA tradeoffs separately, while this best verified candidate is undergoing ranked validation. They will not silently replace the in-flight submission. If Ada reverses this result, the official failure will be recorded and the next direction will target a different bottleneck rather than treating the local gain as portable proof.
