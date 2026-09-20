This submission improves the public Yukon QSB pinning CUDA implementation on
organizer-generated benchmark inputs. Native correctness, repeated matched
comparisons, the full 1,200-second production run and a fresh source-bound
competitive check passed locally. Official promotion remains unconfirmed.

Runtime source digest: 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d.

This candidate additionally reschedules the two complete XYZZ coordinate products
in _PointAddXYZZYCore. It computes R*(V-X3) before the existing ZZ1*=P^2 and
ZZZ1*=P^3 updates. These operations use independent frame arrays; neither moved
product is read by the intervening X/Y work. Both coordinate updates still finish
before deferred or immediate Y recovery. The operation multiset, intermediate
expression DAG, modular primitives, all reduction carries, canonical boundaries
and complete final-window point guards remain unchanged. A source-extracted CPU
symbolic audit checks both QSB_LAZY paths and rejects dependency-breaking negative
reorderings. Fresh native and full checks below qualify this exact source.
YCORE-SCHEDULE.md gives the dependency argument. The scheduling idea was reviewed
in public26e1823e, commit5e1d5bdfa85139a6618e8f79c0188fb187379df6; its unrelated
field arithmetic was not imported. This candidate has no external device module.

The parent submission ce4311f0 was officially rejected at 724,567,912 verified
candidates/s, with all 103,755 reported hits valid. Its exact payload and
original evidence remain frozen. Local matched rates below did not predict
that official score and are not used to adjust it. This is a new executable
source with its own complete qualification.

This source retains the merged parent's two arithmetic and tree changes below,
and adds the dependency-preserving coordinate schedule described above.
First, the second multiplication fold packs its low words
into a 64-bit addition while retaining the actual carry. For B=2^256,
K=2^32+977, the first fold high part h is at most K. Writing h=z8+2^32*z9
proves z9<=1 and, when z9=1, z8<=977; thus z8+977*z9 fits in32bits.
The packed operation is exactly z0+977*z8+2^32*(z8+977*z9), with its carry
and z9 retained into the next limb. The complete upper propagation and final
K correction remain unchanged. CPU interpretation verifies exact raw equality
with the parent for point multiply, point square and tree multiply.

Second, the tree combines its top two levels: lanes zero through three form
the four exclusions using the same ordered multiplication operands as the
original traversal, while lane four forms the unchanged root. Its warp barrier
publishes those four exclusions before the count-eight stage resumes. The
small-tree N=2,4,8 paths and later cross-warp barriers remain. An ordered
expression-DAG proof checks ten powers of two from2through1024. The selected
source passed fresh native point, tree and pipeline checks. The ideas were
reviewed in public f2977f08057f31af3150cc46cf461d2d26f92447; truncated carries
and unrelated public arithmetic changes were not imported.
MERGED-LOW-FOLD-TREE.md gives the full bounds, indices and synchronization.

The retained multiplication schedule transfers three saved even-column carries into
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
96 prepare registers and 70 finish registers without spills in the three
primary specializations when compiled offline for sm89. Those static
observations are not production default-target throughput scores.
TREE-SQUARE-OCCUPANCY.md describes the parent optimization and its domains;
MERGED-LOW-FOLD-TREE.md describes this source's subsequent changes.

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

Mirrored short cohort queue-screen-mergedtop2-next-research-0406 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| latez (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 772.246, 772.968 |
| adaptive (internal) | b64e9df6300dac15991bb2b52cb4414ff20faddd684592b84d472c6de0e43caa | 771.265, 771.280 |
| addtail (internal) | 85e581246415a3384d416468851dab9ae6d668f90b16be170d862d895754c097 | 771.944, 771.097 |
| cold (internal) | 7541125dc3f701fb9f059eef5c3af083f66a1cb46c4e6fb7279ebcdc4a98f7fa | 771.399, 771.303 |
| control (internal) | 5efcdd046a2360ec0ab3994bd4e6abd796bfe550be08f728f0d0deaa04ed15d9 | 768.314, 771.086 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 740.081, 746.207 |
| latezzz (internal) | 05627ce3aca9aea381f3d20e516d35921fcb3e83700b53b67c9db54c19c22c1d | 769.296, 766.617 |

Mirrored long cohort long-queue-mergedtop2-research-finalists-0537 used fresh seed 28957617 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,297 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| latez (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 767.906, 768.697 |
| adaptive (internal) | b64e9df6300dac15991bb2b52cb4414ff20faddd684592b84d472c6de0e43caa | 766.843, 766.802 |
| addtail (internal) | 85e581246415a3384d416468851dab9ae6d668f90b16be170d862d895754c097 | 767.440, 767.719 |
| control (internal) | 5efcdd046a2360ec0ab3994bd4e6abd796bfe550be08f728f0d0deaa04ed15d9 | 766.527, 766.338 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 745.289, 741.056 |
| native (internal) | 170c3170c21fcbb68a9a754f713069701914772ad3c40454f23c8bf803ade6b5 | 765.359, 767.197 |

Mirrored short cohort queue-screen-latez-public-close-0643 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 771.991, 770.979 |
| 736a5884 | 8ace01dd58e206d39373469a7ccc8a239b252f3464facb9286faa542f7fb110d | 757.814, 754.598 |
| 753cf574 | 1f376fcfa53c56c5939ef6e7a2d6af1d3940e9ae80b7914cd73d86a818de784c | 749.126, 744.802 |
| a296916a | 3a18ed3fd00a957f1f840e20a9a465e0af83c496a3620c03a61b50e5430f59ca | 769.239, 766.275 |
| c7a8eaa8 | 02e90625d023763b857ed0bf024fd88c11491606be44c4d7fa80eb5f9b01cb7a | 747.207, 744.192 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 739.265, 744.943 |
| qualified (internal) | 5efcdd046a2360ec0ab3994bd4e6abd796bfe550be08f728f0d0deaa04ed15d9 | 771.177, 771.344 |

This cohort does not establish a strict timing lead over qualified. The original observations remain in the registry. The final live gate requires resolved comparisons for the current frontier and all currently pending public sources; it does not treat terminal submissions as pending or exclude their source as incorrect.

Mirrored long cohort long-queue-latez-public-close-0643 used fresh seed 1674237305 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,422 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 767.940, 768.471 |
| a296916a | 3a18ed3fd00a957f1f840e20a9a465e0af83c496a3620c03a61b50e5430f59ca | 760.628, 763.184 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 741.025, 737.707 |

Mirrored short cohort queue-screen-latez-compact-current-0815 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 769.440, 769.869 |
| 1660605f | af018fb8a34b1a2adc5f057df63f42f90c059cf52076a2b93e840ec3bb261820 | 750.046, 756.206 |
| 4ae03b7a | e5e1510e2575e92dfd900bfb2d07ac46cce01f3ac34c253e0e76bde120b514f8 | 750.466, 750.105 |
| f1d902a6 | 38191c87114cfc8515dcb8e0d6538a06e43cf1e3e2f0cc1d7da8bfcdcceb1465 | 761.084, 754.872 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 751.348, 752.208 |

Mirrored long cohort long-queue-latez-compact-current-0815 used fresh seed 1733832386 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,326 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 770.077, 768.345 |
| 1660605f | af018fb8a34b1a2adc5f057df63f42f90c059cf52076a2b93e840ec3bb261820 | 753.307, 748.648 |
| 4ae03b7a | e5e1510e2575e92dfd900bfb2d07ac46cce01f3ac34c253e0e76bde120b514f8 | 750.659, 741.125 |
| f1d902a6 | 38191c87114cfc8515dcb8e0d6538a06e43cf1e3e2f0cc1d7da8bfcdcceb1465 | 753.527, 747.273 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 747.730, 742.489 |

Mirrored short cohort queue-screen-paired-late-latez-dda-8d5-0833 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 770.905, 771.898 |
| 8d5c1b77 | 8ace01dd58e206d39373469a7ccc8a239b252f3464facb9286faa542f7fb110d | 758.318, 762.859 |
| dda70024 | 261f82645f28335616673ca010ad1f9a61dd1286f0d113bb9bdd35107ecc16ef | 753.286, 760.895 |

Mirrored long cohort long-queue-paired-late-latez-dda-8d5-0833 used fresh seed 232899966 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,574 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 768.000, 768.245 |
| 8d5c1b77 | 8ace01dd58e206d39373469a7ccc8a239b252f3464facb9286faa542f7fb110d | 750.385, 752.636 |
| dda70024 | 261f82645f28335616673ca010ad1f9a61dd1286f0d113bb9bdd35107ecc16ef | 754.203, 753.041 |

Mirrored short cohort queue-screen-paired-late-latez-7f85-0901 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 770.204, 772.147 |
| 7f85dc65 | 4ac3207107cf7dc05eb8a7e8744217506c497c5630b3464c1d3efafbcf06ffa6 | 766.342, 757.322 |

Mirrored long cohort long-queue-paired-late-latez-7f85-0901 used fresh seed 1736220301 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,425 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 768.429, 769.916 |
| 7f85dc65 | 4ac3207107cf7dc05eb8a7e8744217506c497c5630b3464c1d3efafbcf06ffa6 | 756.416, 755.629 |

Mirrored short cohort queue-screen-latez-remaining-0921 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 775.008, 773.247 |
| 224f1d38 | 469fbd3924123223c0db2bde7069600b0859fe18bdc419297940911f3f8702ad | 751.426, 749.848 |
| 2e0c663c | 5cd74a3cc555c4d438d9864e3a5657d6dc941a8535a1b45304cc83bb102d4841 | 760.960, 754.593 |
| f4dc95e6 | 1fb0dc313441339275a2f05808649aecf97b27f33bacba84ca69b2ba78f03126 | 732.979, 725.593 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 747.850, 747.034 |

Mirrored long cohort long-queue-latez-remaining-0921 used fresh seed 1422582745 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,517 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 769.641, 768.842 |
| 2e0c663c | 5cd74a3cc555c4d438d9864e3a5657d6dc941a8535a1b45304cc83bb102d4841 | 749.006, 749.140 |

Mirrored short cohort queue-screen-paired-late-latez-358-387-0937 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 774.608, 771.990 |
| 358b5b93 | fd51aff80807ae7b4dd350a54c10ef6c4cfb82aa0ed1d4298c83fb77c64117ad | 751.035, 759.664 |
| 38720e9a | 35bcf6f14c2a8d16305e3c2ec7ea1f5c9e353585325c7ee44f89ed46cd1152fb | 764.007, 760.028 |

Mirrored long cohort long-queue-paired-late-latez-358-387-0937 used fresh seed 138387901 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,242 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 769.775, 768.473 |
| 358b5b93 | fd51aff80807ae7b4dd350a54c10ef6c4cfb82aa0ed1d4298c83fb77c64117ad | 748.046, 753.496 |
| 38720e9a | 35bcf6f14c2a8d16305e3c2ec7ea1f5c9e353585325c7ee44f89ed46cd1152fb | 759.193, 752.405 |

Mirrored short cohort queue-screen-paired-late-latez-316-5d-0955 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 774.238, 772.582 |
| 316b6c6e | 6916b0c1e8dd24726d1c4e6a9f5935bad99d05ebf0564da424b23ea9b3971a9a | 736.617, 727.795 |
| 5d094a84 | 3c4d50d200340b5b3ce08d1480a9f8c283bff851cf5f9e82a2d74199e5dfb93a | 749.125, 749.737 |

Mirrored long cohort long-queue-paired-late-latez-316-5d-0955 used fresh seed 740711852 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,251 independently verified tuples. Rates below use identical completed work and timing boundaries within this cohort, in forward and reverse order. Internal arms are explicitly labeled.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 768.429, 769.420 |
| 316b6c6e | 6916b0c1e8dd24726d1c4e6a9f5935bad99d05ebf0564da424b23ea9b3971a9a | 724.384, 730.964 |
| 5d094a84 | 3c4d50d200340b5b3ce08d1480a9f8c283bff851cf5f9e82a2d74199e5dfb93a | 744.115, 745.177 |

The unchanged production wrapper passed the full 1,200-second contract on
fresh seed 737462380. All 109,255 reported hits
were verified independently, with no failures. Artifact time was
1200.143 seconds and outer wall time was
1201.530536 seconds. Source, original inputs and binary hashes
passed before and after the run. The complete artifact was downloaded and all
hits were independently rechecked locally before full qualification was registered.

The complete registry contains 16 matched cohorts and
46 exact-source comparisons. The live gate passed at
2026-09-20T10:27:09.627754+00:00. Its current frontier source is
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
