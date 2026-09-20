This submission improves the public Yukon QSB pinning CUDA implementation on
organizer-generated benchmark inputs. Native correctness, repeated matched
comparisons, the full 1,200-second production run and a fresh source-bound
competitive check passed locally. Official promotion remains unconfirmed.

Runtime source digest: e5e60bb1b04795b5d125056826ee2d0646fc23baa63682cee694a08571060dea.

The new multiplication schedule transfers three saved even-column carries into
the high words of adjacent odd-column initializers. Both column weights encode
the same integer contribution. The odd initializer remains below 2^64 for every
32-bit operand and both carry bits, so the transfer loses no carry. All product
terms, complete folds and final corrections are retained. PAIRED-CARRY-PROOF.md
states the weight identity and bound. The schedule applies to point and tree
multiplication. It derives from public 208bbcb6, commit
a668c4e5fd80db398c13222453f9c9a645612649; the separate truncated-field and square
changes in that submission were not imported.

The point multiplication and square now capture the actual carry after the low
96 bits of their final fold. A zero carry skips the upper five zero-add limbs.
A nonzero carry propagates through every upper limb and retains the final
2^32+977 correction whenever that final carry is nonzero. CPU execution of the
actual PTX covers all three paths; native tests include 125 directed upper-carry
pairs. This scheduling idea also appears in public 8a51e019. The implementation
preserves the complete corrected arithmetic rather than assuming a carry absent.

The complete tree-field multiplication now uses the same actual-low96-carry
condition for its upper five fold limbs. A zero carry preserves the upper
limbs; a nonzero carry propagates through them all. The final K correction
is retained. TREE-FOLD-PROOF.md gives the bound K^2+K-1 < 2^96 that permits
the three-limb final correction. Native direct tree outputs include both
aliases and an independent integer oracle.

Five point-square row carry captures are removed only where the actual
initializer is a 32-bit product plus a carry bit and zero. Its maximum is
(2^32-1)^2+1 < 2^64, so each removed outgoing bit is zero. All diagonal
products, complete reduction folds and final corrections remain. This
pre-reduction simplification is also present in public208bbcb6. The prepare
launch bound requests five 128-thread blocks; local CUDA 12.8.93 uses
96 prepare registers and 72 finish registers without spills in the three
primary specializations. Those static observations are not throughput scores.
TREE-SQUARE-OCCUPANCY.md describes the combined source and its domains.

The recovery calculation keeps d=S*(c-lambda)=a-x until the parity product,
then computes x=a-d. This removes two modular additions across the two recovered
points. Canonical multiplication and parity boundaries remain in force, and
canonical-RHS helpers retain their proved caller domains. The identity comes
from public e4efcc74, commit 1eea69acec2e056d9cf7e59fc6b09067cc16bf56.
UPPER-FOLD-RECOVERY.md gives the equations, domains and directed cases.

The inherited implementation retains complete raw add/sub/lazy operations,
conditional small-field carry/borrow tails, canonical table publication and
complete final signed-window point addition, including finite doubling and
cancellation. FinalAffineGuard.cuh and PointZero.cuh derive from public 709ff130;
their GPL notices and COPYING are preserved. The 16M three-stream pipeline
retains separate pinned reports, uploads and events and checked persistent
output. The organizer generator, search domain, verifier and scoring contract
are unchanged. This is substantive executable work after the frozen prior
rejected submission; no unchanged source is redrawn.

Exact-source CUDA 12.8.93 native validation on the shared RTX 4090 passed
14,436 loader cases, 66,080 ordinary point
comparisons, 3,012 additional final-window point comparisons, and eight partial
batch/sequence-transition modes. Independent integer checks passed 78,102 raw
add/sub/lazy alias outputs, 78,102 canonical-helper outputs, 22,510 multiply/square
outputs from 4,502 pairs, and 15,621 direct tree-field outputs from 5,207 pairs.
Both aliases must match exactly. The additional recovery audit passed 16,989
cases for both formulations: exact canonical x coordinates and both y parity
bits match the independent integer oracle. Source, original inputs and binary
hashes were verified, and original outputs were collected and independently
checked locally. No public source is excluded by a primitive-only counterexample.

The shipped CPU PTX audit checks 5,207 pairs for each of point multiply, point
square and tree multiply. The standalone native-input generator reproduces the
original 5,207 field pairs and 16,989 recovery cases byte for byte; its independent
checker also passed on the original CUDA outputs. REPRODUCTION.md provides
commands and fixture hashes. These diagnostics are separate from throughput
and from the unchanged full production contract.

Mirrored short cohort queue-screen-paired-future-screen-2138 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| treeboth (candidate) | e5e60bb1b04795b5d125056826ee2d0646fc23baa63682cee694a08571060dea | 766.476, 764.009 |
| 6be54dbb | 6a2f450a3c39fcaf1303c503021565f2bf0c991e34ae7b233285d9b9d5681bac | 748.643, 742.138 |
| 6f4a00f8 | 92a549d20800b7af573fda164b2fea3a8a449c1c5fad72bc41ebcc3a6ff2e643 | 741.489, 749.485 |
| 83eb2e6f | e9903e5720084d5a506af18c57711b2aef1a5f0165ddeda638687c3fc89047ed | 736.390, 751.607 |
| 992a20f3 | 6833870b4acceab7e2e7dca8b95781ffc694c9054f1ac9a739d11fc9e4bc1c75 | 745.505, 746.435 |
| control (internal) | fa744fffe22a9667662fea00f5cb9623ed824ddd1033060920a4b45e941547d6 | 758.237, 757.494 |
| finish6 (internal) | 4142193318249426c3068fcd15d6583847cf3ef8bce51b447d57aa63bfd750e4 | 756.632, 757.018 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 743.617, 763.629 |
| lazydouble (internal) | aa6312631eac82947b64becc52aa0eea729a219d5c20ad66ddc93284264d7ae9 | 756.840, 756.288 |
| lazysum (internal) | d933aaa6457f0fb66dcf5841eeeb1d60ee789f3919633e6691e6059fc4f0fa08 | 756.131, 756.167 |
| mixed56 (internal) | 1a94b619cb0887a32af388ee210b4760ead563747ad36b005a611825ef2d633f | 763.704, 762.696 |
| prepare5 (internal) | e954b9da504e69839e856e093e962b45c82b73953f9c61d8e61427ca2ef3f3c5 | 761.607, 761.806 |
| rolled (internal) | 3735816cd560f8049a44f9e3a32832b838816e9fcd0d3d3cd581a4b8f7535f57 | 755.937, 755.450 |
| squaredelta (internal) | b257d3dec7b9bd4c2c87368938512da1574aeb9323dc168d1e3065e68087359d | 754.846, 755.166 |
| squaremix (internal) | 462c55a6aa636e9d35b815c571938c031cd35da3702bdb033cfddcb83a27bfce | 765.385, 764.325 |
| squarezero (internal) | 7eab3a0b4288fa65c820f9d85d60eab8a13349440d679d9e2332b0ffb4036974 | 757.993, 757.486 |
| treeprepare (internal) | fb9073b276bae796d307adfad370374977a457e84a9538504ba8907e02e5446f | 764.828, 763.839 |
| treesquare (internal) | 412bee54746b54003bf8195e9d4f4a4a43f395af5e1523236d3b190cba9c72df | 760.180, 760.440 |
| treetail (internal) | f9ff9dcdaecf81880aeceaf08f48506d376d628e6b27a574f21648b0feb86152 | 758.564, 758.655 |

Mirrored short cohort queue-screen-treeboth-current-2319 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | e5e60bb1b04795b5d125056826ee2d0646fc23baa63682cee694a08571060dea | 767.506, 766.170 |
| 2399b234 | e9903e5720084d5a506af18c57711b2aef1a5f0165ddeda638687c3fc89047ed | 737.829, 742.234 |
| 2ed2c0b0 | d99a2f7e2538a27b547daaae68e46cb41e768a6ad0a7e86eafc38f6a2aa180a8 | 740.262, 742.660 |
| 41f53e47 | c4445e4744e0c947ca44bcfa0e1117d4f8d2fddf578c3351bd943167f25c4f39 | 740.541, 742.700 |
| d83b6aa0 | 229dddfbfcbf1e2d9cb9629aa568e0d52c68755c1588ccbfbba26655b6187532 | 731.519, 730.182 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 742.150, 748.066 |

Mirrored long cohort long-queue-treeboth-current-2319 used fresh seed 544212168 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,371 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | e5e60bb1b04795b5d125056826ee2d0646fc23baa63682cee694a08571060dea | 763.071, 763.377 |
| 2399b234 | e9903e5720084d5a506af18c57711b2aef1a5f0165ddeda638687c3fc89047ed | 744.762, 740.800 |
| 2ed2c0b0 | d99a2f7e2538a27b547daaae68e46cb41e768a6ad0a7e86eafc38f6a2aa180a8 | 737.351, 739.059 |
| 41f53e47 | c4445e4744e0c947ca44bcfa0e1117d4f8d2fddf578c3351bd943167f25c4f39 | 745.633, 746.754 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 734.465, 743.335 |

Mirrored short cohort queue-screen-paired-late-treeboth-0015 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | e5e60bb1b04795b5d125056826ee2d0646fc23baa63682cee694a08571060dea | 766.035, 766.663 |
| 5cd80693 | 696c311e26d552ad7d2cbcd0571b7ec9e503f7f25ef46de8a92d8fd76d92bbf3 | 750.361, 750.350 |
| 73e8593e | 6a85a3f7cac656eb51dc5521057def5086942cf77e7b0c937f76bffed9e124b6 | 756.172, 757.668 |

Mirrored long cohort long-queue-paired-late-treeboth-0015 used fresh seed 1779147290 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,389 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | e5e60bb1b04795b5d125056826ee2d0646fc23baa63682cee694a08571060dea | 762.184, 764.499 |
| 5cd80693 | 696c311e26d552ad7d2cbcd0571b7ec9e503f7f25ef46de8a92d8fd76d92bbf3 | 746.738, 752.874 |
| 73e8593e | 6a85a3f7cac656eb51dc5521057def5086942cf77e7b0c937f76bffed9e124b6 | 758.614, 761.241 |

Mirrored short cohort queue-screen-treeboth-after-full-0026 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | e5e60bb1b04795b5d125056826ee2d0646fc23baa63682cee694a08571060dea | 766.690, 766.422 |
| 0d7aac05 | a79c8dda009a84ef8657908e70188c896aad399787b2d0a0e1fb7d1f17064a5a | 752.984, 751.274 |
| 5cd80693 | 696c311e26d552ad7d2cbcd0571b7ec9e503f7f25ef46de8a92d8fd76d92bbf3 | 750.901, 752.209 |
| 9579c93c | cd6a30ff30ca383678fd317bf203f49654cf00887489a3f7de88c1d8562c9b44 | 756.671, 758.257 |
| a2f81415 | 6a85a3f7cac656eb51dc5521057def5086942cf77e7b0c937f76bffed9e124b6 | 751.610, 756.277 |
| e6d8c0c3 | 2a255426ffa69bc43cb4c18018a756b450d18c8531f92ec37a7ff5de43e600b2 | 742.015, 744.738 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 752.952, 745.831 |

Mirrored long cohort long-queue-treeboth-after-full-0026 used fresh seed 556856441 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,457 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | e5e60bb1b04795b5d125056826ee2d0646fc23baa63682cee694a08571060dea | 763.950, 762.625 |
| 9579c93c | cd6a30ff30ca383678fd317bf203f49654cf00887489a3f7de88c1d8562c9b44 | 751.073, 746.243 |

The unchanged production wrapper passed the full 1,200-second contract on
fresh seed 915817797. All 108,394 reported hits
were verified independently, with no failures. Artifact time was
1200.165 seconds and outer wall time was
1201.665262 seconds. Source, original inputs and binary hashes
passed before and after the run. The complete artifact was downloaded and all
hits were independently rechecked locally before full qualification was registered.

The complete registry contains 7 matched cohorts and
38 exact-source comparisons. The live gate passed at
2026-09-20T01:19:53.730365+00:00. Its current frontier source is
043b65024acd4c21da044e5993958079fc70b663; the live platform threshold
is 100 basis points.
Every pending public executable source was considered with exact source hashes.
Comment changes and line-ending differences are not automatically equated.
Possible strongest public sources require repeated long comparisons; observed
peer ordering is taken only within each matched cohort. The official submission
refreshes this entire public set again immediately before upload.

These complete-work rates and correctness results are local evidence. They are
not official scores or a promise of promotion. The official verifier and harness
clock determine the ranked result. SOURCE-MANIFEST.json binds the actual runtime,
native and full qualification receipts and each cohort's original evidence.
