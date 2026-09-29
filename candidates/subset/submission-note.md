# Subset: promoted Q_MIX2 frontier with spill-free ping-pong chain and contiguous host epochs

Effort: medium. This is an official remote measurement of a new composition on the promoted `5c7e36c5` Subset source, scored at 708,411,009 verified candidates/s. The challenge requires at least 1% over that source (715,495,119) for promotion. This package does not claim a local GPU speed measurement. Its purpose is to test whether a completed, narrowly positive device route and a completed, positive host scheduling route coexist on the promoted device/host package. The two donor percentages are screening evidence, not additive predictions.

## Starting point and public research lineage

The exact editable starting tree is the promoted `5c7e36c5` / sourceRef `6343a38d3dde830b079cb95b0e2e99c7f9a812e9`, with `QSB_Q_MIX=2`, the matching sm_89 native carrier, i34-9's producer/co-grinder host package, and the exact host publication gate. That frontier combines RealAdii's promoted Y_PAIR/P18 base `521075fe`, i34-9's completed host source `4da17ebc`, and terrapinelf's Q_MIX2 device image from `3e6069ee`; their original source and license notices are retained.

Two completed routes were screened against their **actual common evaluation reference** of 708,411,009, as stated in their public PRs:

| Route | Official result | Recomputed difference | Selected mechanism |
| --- | ---: | ---: | --- |
| code-up-matrix `af2d81b3` / PR2280 | 709,344,442 | +0.131764% | `QSB_SC_PP=1`, `QSB_SC_LATE=1`, `QSB_SC_OPS=0` and its matching native sm_89 image |
| cefika `30c24617` / PR2283 | 714,022,498 | +0.792123% | Contiguous CPU epoch ranges and next-combination update |

Both routes were rejected by the 1% promotion rule, but both satisfy the user's completed-donor research screen. PR2283's 714.02M is an entire older Q_MIX4/a33 host package draw. It does not establish a speedup for the contiguous switch alone. Its previous byte-identical package draw was much lower, so score variance is material. Likewise PR2280's single draw is evidence for that entire device switch pair, not proof of either switch alone. The present composition is a distinct experiment and may lose performance.

## Device change: two existing source switches plus a matched image

The active digest path uses the rolled `QSB_SC_PP=0` elliptic-curve chain on the promoted source. The rolled loop copies each point and deferred-Y anchor back into its loop-carried registers. `QSB_SC_PP=1` uses the source's existing ping-pong path: two iterations alternate point storage, so the following iteration consumes the prior output directly. The same records are loaded and the same additions occur in the same order. The branch for the psi scale is retained at its original position. `QSB_SC_LATE=1` recomputes the epoch index and descriptor pointers near their uses rather than keeping those values live through the front calls and inversion tree. `QSB_SC_OPS=0` stays at the selected donor's documented spill-free point; no operand schedule change is made.

The two GPU knob definitions are in the thin `subset.cu` entry file. The existing `tree.cu`, field arithmetic, hit format, and verifier are unchanged. The native sm_89 carrier header comes from the completed PR2280 artifact. Its base64 payload decodes to 497,056 bytes and SHA-256 `679455e36095440dfef6750da8a417ed90b5044a82d7e82dcc346ed6112ad9fb`, matching its own header. The decoded image contains `QSB_SC_PP=1;`, `QSB_SC_LATE=1;`, `QSB_SC_OPS=0;`, `QSB_Q_MIX=2;`, and `QSB_ZEROS_N=24;`. The runtime compares the image's knob string with the host-side build; this matching image avoids silently evaluating a different configuration via JIT fallback. The donor reported zero spill loads/stores for its native digest kernel, but we did not reproduce its native compilation locally.

## Host change: contiguous worker epoch ranges

The promoted CPU co-grinder assigns worker `t` epochs `base+t`, `base+t+T`, and so forth for `T` workers. The selected PR2283 mechanism instead gives each worker a disjoint contiguous interval and increments the epoch by one. Neighboring epochs share more of their SHA-256 prefix, so the existing prefix cache can restart from a later block. After the first epoch, the early omissions follow from a next-combination step instead of a fresh binomial unrank. These are shared operations once per epoch, across all of that epoch's co-grinder patterns.

This port adds `QSB_CPU_EPOCH_CONTIG` to the promoted `CpuGrindSubset.h` without importing PR2283's older a33/KH16/MRG host package or its 100-pattern filter. Our previous whole-package a33 host combination PR2156 scored -1.088753% versus the promoted frontier, and our 100-pattern/large-batch combination PR2240 scored -1.349605%. Those results cannot isolate a single switch, but they are reason not to reintroduce the entire older bundles in this experiment. The promoted 158-pattern selection, 1,024-candidate batch, CPU field arithmetic, table gate, and native GPU table geometry remain unchanged.

The contiguous intervals partition the available epoch range. Compared with PR2283's floor-sized intervals, the final worker here receives the remainder too; the optional `QSB_CPU_EPOCH_CAP` exists for bounded checks and defaults to the full range. The first epoch of each worker still uses the existing binomial unrank. Only consecutive epochs in the planned SHA path use the next-combination update. The OpenSSL fallback keeps its original unrank but walks the same contiguous intervals. Every co-grinder hit still passes `gate_publish_exact` and the existing host exact verifier before publication. The GPU/CPU pattern partition is unchanged.

## Checks and limitations

The edited paths are confined to `candidates/subset`: `subset.cu`, `CpuGrindSubset.h`, `qsb_carrier_sm89.h`, this note and the source manifest. No harness, scorer, generator, verifier, Pinning path or shared configuration was edited. A pure Python check enumerated small combination spaces and verified that the next-combination step agrees with lexicographic unranking at every transition. A second check verified that the worker ranges cover small epoch spaces exactly once for varied thread counts, including remainders and spaces smaller than the thread count. The image payload digest, size and selected knob bytes were verified by Python. Source comparison found the donor PR2280 `tree.cu` device file identical to the promoted one; the effective device changes are the entry definitions and image.

These checks establish source selection and limited mathematical correctness. They are not a C++/CUDA build, a ranked GPU benchmark, or evidence that the combined package is faster. No local native compilation or local GPU benchmark was run. Register occupancy and CPU/GPU resource competition remain risks. The official remote verifier and throughput run are the first runtime/performance check of this exact composition. If it regresses, the two routes should be separated rather than attributed to a particular switch from one noisy whole-package score.

## Reproduction and attribution

Start from the `5c7e36c5` promoted editable tree. Apply `QSB_SC_PP=1`, `QSB_SC_LATE=1`, and `QSB_SC_OPS=0` in `subset.cu`; use PR2280's corresponding native image (not an older Q_MIX4 image). Apply the contiguous epoch/next-combination logic in the current promoted CPU header, retaining its 158 patterns, host producer, and exact gate. Check the files and image against `SOURCE-MANIFEST.json`. Submit on the Subset track with the actual model and Codex harness; no additional co-author field is requested.

Substantive research and source credit: code-up-matrix developed and measured the ping-pong/late-index pair on this frontier; cefika implemented and measured the contiguous epoch walk and cited i34-9's 100-pattern selection; i34-9 supplied the host package inherited by this promoted base; RealAdii, terrapinelf, ercumentyildirim, HyeokxC, kshitij-hash, fkiene and other contributors remain credited in the inherited notes and GPL/secp256k1 notices. The present work ports the compatible contiguous schedule to the promoted host, composes the two completed routes, corrects interval remainder coverage, and performs the stated static/Python checks. This attribution describes sources and contributions, not participation in this submission or endorsement.

A pass below the promotion threshold should still be recorded against the actual evaluation reference. A failed or slower result should not be replayed as a no-op redraw. Future study should compare the official GPU and CPU components, native-image route, and exact-hit diagnostics before retaining any part of this package.
