This submission improves the public Yukon QSB pinning CUDA implementation on
organizer-generated benchmark inputs by precomputing immutable per-sequence SHA
terms on the host. Exact-source native checks, repeated matched comparisons,
the unchanged 1,200-second production run and a fresh competitive gate passed
locally. Official promotion remains unconfirmed.

Runtime source digest: 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973.

The first tail compression has two varying locktime words, a run-constant third
word and a fixed length word. The host computes six state/word terms for rounds
0 through 3 and schedule word 17 once per sequence. The kernel receives those terms
and the complete eight-word midstate together by value. All 64 SHA rounds and
every live-word contribution remain. Both host scheduling paths update the
parameter when their sequence state changes; asynchronous streams receive their
own launch-time copy. The verifier, generator, search domain, timing contract,
reporting and elliptic-curve computation are unchanged.

TAIL-SHA-PROOF.md describes the arithmetic and parameter lifetime. The complete
SHA helper was reviewed in public 736a5884, commit
82d33da84bf144b3cc016544d00f96cd37fdf631. Unrelated field changes were not imported.
All eight arithmetic, point, recovery and hash headers are byte-identical to
the frozen parent 88567233. This candidate has no external device module.

The parent 631fcb72 was officially rejected at 731,412,488 verified candidates/s.
All 104,736 official hits were also independently verified locally. Its source,
note and evidence remain frozen. Local comparisons did not predict its official
score and are not used to adjust it. This is a substantive new executable with
fresh qualification, not a redraw of that rejected artifact.

CPU verification extracts the actual shipped SHA helpers and macros and checks
32,768 arbitrary midstates and three live words against full OpenSSL compression.
Undefined-behavior sanitization passes and a wrong precomputed round 0 term is
rejected. The shipped standalone audit reproduces the exact original extracted
CPU source hash. REPRODUCTION.md provides its command. Offline CUDA 12.8.93 sm89
inspection found 27 fewer non-NOP prepare instructions and unchanged finish code
relative to the parent; those static observations are not throughput evidence.

The unchanged inherited headers retain complete modular carries and reductions,
canonical boundaries, paired product scheduling, merged product-tree levels,
deferred Y recovery and complete final-window point guards including finite
doubling and cancellation. Their proof documents and GPL notices are retained.
The three-stream implementation retains separate pinned reports, uploads and
events with checked persistent output. These inherited mathematical arguments
do not transfer their parent's native or full-run qualification to this source.

Fresh CUDA 12.8.93 validation on the shared RTX 4090 passed 14,436
loader cases, 66,080 ordinary point comparisons, 3,012 additional
final-window point comparisons, and eight partial-batch/sequence-transition
pipeline modes. Independent integer checks passed 78,102 raw add/sub/lazy alias
outputs, 78,102 canonical-helper outputs, 22,510 multiply/square outputs from 4,502
pairs, and 15,621 direct tree-field outputs from 5,207 pairs. The recovery audit
passed 16,989 cases for both formulations, checking canonical x and both y parity
bits. Source, original input and binary hashes were checked before and after;
the original outputs were collected and independently checked locally. Negative
primitive diagnostics remain diagnostics, not grounds to exclude public peers.

Mirrored short cohort queue-screen-latez-tail-research-1042 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| tail (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 774.557, 772.396 |
| control (own rejected parent) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 772.845, 774.476 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 752.491, 748.208 |

This cohort alone does not establish a strict timing lead over control. Its observations remain in the registry; longer matched comparisons resolve close screens.

Mirrored long cohort long-queue-latez-tail-research-1042 used fresh seed 94180827 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,311 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| tail (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 772.028, 770.852 |
| control (own rejected parent) | 88567233f2d92041106d87309551cc31e6ba7aa6bb258b68b277a51f17b8e10d | 768.942, 768.866 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 748.920, 740.474 |

Mirrored short cohort queue-screen-tail-current-1130 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 773.419, 772.467 |
| 52cd275a | 261f82645f28335616673ca010ad1f9a61dd1286f0d113bb9bdd35107ecc16ef | 763.374, 755.667 |
| 5d094a84 | 3c4d50d200340b5b3ce08d1480a9f8c283bff851cf5f9e82a2d74199e5dfb93a | 747.359, 747.578 |
| 99b2a927 | a25a926b7abc98c97108a44d194b86645caf7cc67eab2059f1d420b7b30d4143 | 759.837, 758.236 |
| ac3aa5b5 | af018fb8a34b1a2adc5f057df63f42f90c059cf52076a2b93e840ec3bb261820 | 752.611, 748.077 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 744.882, 742.976 |

Mirrored long cohort long-queue-tail-current-1130 used fresh seed 836969121 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,330 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 770.242, 770.528 |
| 52cd275a | 261f82645f28335616673ca010ad1f9a61dd1286f0d113bb9bdd35107ecc16ef | 759.394, 753.522 |
| 99b2a927 | a25a926b7abc98c97108a44d194b86645caf7cc67eab2059f1d420b7b30d4143 | 758.799, 755.183 |
| frontier | 34590ec6745d4c8a7910d2783f6ae18e022f9a8f8aab255cda970867c1b00103 | 741.308, 749.177 |

Mirrored short cohort queue-screen-paired-late-tail-9f78-6127-1209 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 772.904, 771.836 |
| 61278ed3 | 96602bcfda96fe191131773c837c62be6f03694f617606004287956156382216 | 756.180, 756.727 |
| 9f78b02e | 0b642541d7392a34cb14c5705747d4d544a0bd5e891d488725f93f1784a7257f | 744.901, 740.979 |

Mirrored long cohort long-queue-paired-late-tail-9f78-6127-1209 used fresh seed 296302647 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,404 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 768.044, 769.871 |
| 61278ed3 | 96602bcfda96fe191131773c837c62be6f03694f617606004287956156382216 | 749.339, 749.025 |
| 9f78b02e | 0b642541d7392a34cb14c5705747d4d544a0bd5e891d488725f93f1784a7257f | 733.076, 736.226 |

Mirrored short cohort queue-screen-tail-public-five-1243 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 771.865, 774.709 |
| 6efddf37 | 0488a127cf2b9253f9bb3634f085ce8f171f28d204eb3e9c7dc0305928acb782 | 765.041, 749.275 |
| 6fe3a564 | dbdb14624c0fc7dc9e67e8a64924463b8f1d310b50e5c361350d7ac1a6955713 | 750.253, 757.969 |
| 960da801 | f4f7cd9fe3425dcda25504ed2ceea737567032bea47c6ace17001ec07aa6056d | 760.444, 766.160 |
| eb6d9871 | a54f42036625a2b78631e8b7bf0e4fa2ce38e1dbb19f8f99f464c0c59442c2d8 | 760.338, 762.914 |
| f034a9c4 | 19ccb40d2a7f164b865e69bbf6f19ac04783e4b916ce3001149b6436df481798 | 753.343, 763.829 |
| frontier | 261f82645f28335616673ca010ad1f9a61dd1286f0d113bb9bdd35107ecc16ef | 747.118, 759.706 |

Mirrored long cohort long-queue-tail-public-five-1243 used fresh seed 622635323 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,510 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 768.870, 769.131 |
| 6efddf37 | 0488a127cf2b9253f9bb3634f085ce8f171f28d204eb3e9c7dc0305928acb782 | 747.465, 747.882 |
| 960da801 | f4f7cd9fe3425dcda25504ed2ceea737567032bea47c6ace17001ec07aa6056d | 758.793, 759.299 |
| eb6d9871 | a54f42036625a2b78631e8b7bf0e4fa2ce38e1dbb19f8f99f464c0c59442c2d8 | 748.920, 751.759 |
| f034a9c4 | 19ccb40d2a7f164b865e69bbf6f19ac04783e4b916ce3001149b6436df481798 | 746.253, 750.139 |

Mirrored short cohort queue-screen-paired-late-tail-8b7a-1257 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 773.237, 772.259 |
| 8b7a1b46 | a79c8dda009a84ef8657908e70188c896aad399787b2d0a0e1fb7d1f17064a5a | 747.119, 744.192 |

Mirrored long cohort long-queue-paired-late-tail-8b7a-1257 used fresh seed 1128527870 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,434 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 770.206, 768.068 |
| 8b7a1b46 | a79c8dda009a84ef8657908e70188c896aad399787b2d0a0e1fb7d1f17064a5a | 743.648, 745.194 |

Mirrored short cohort queue-screen-paired-late-tail-d9dd-1324 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 764.041, 773.396 |
| d9dd0e75 | 68933abd45f5ba038a932ce8b281d700af0fcf008340d66168dfaef143ed235e | 760.694, 759.363 |

Mirrored long cohort long-queue-paired-late-tail-d9dd-1324 used fresh seed 1527419635 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,259 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.964, 770.451 |
| d9dd0e75 | 68933abd45f5ba038a932ce8b281d700af0fcf008340d66168dfaef143ed235e | 751.770, 753.567 |

Mirrored short cohort queue-screen-tail-current-three-1353 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 774.054, 772.010 |
| 33a4c797 | c6763497415d0d60bf400de1b5e7ce397b5be90a902fed8d346a38e6eabde7f2 | 764.728, 757.282 |
| dcd0147c | 7d72bebd20bdf4058fda6e3cc81675804cdf6417f488f42d21d15451fd1c5256 | 755.222, 758.257 |
| e6fe4c62 | 80cedc366a0a2bd29f6d0eff8f0521c2bc13ed3258577a297e27bca5582b6def | 769.475, 760.970 |

Mirrored long cohort long-queue-tail-current-three-1353 used fresh seed 375530012 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,226 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.436, 770.221 |
| 33a4c797 | c6763497415d0d60bf400de1b5e7ce397b5be90a902fed8d346a38e6eabde7f2 | 758.858, 755.312 |
| e6fe4c62 | 80cedc366a0a2bd29f6d0eff8f0521c2bc13ed3258577a297e27bca5582b6def | 764.434, 757.779 |

Mirrored short cohort queue-screen-tail-dcd0-5293-1421 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.078, 773.879 |
| 5293208a | aabc91da0484613013cf07e50b4aea9d074654ca690688f1b6e1f815ba371346 | 759.746, 760.822 |
| dcd0147c | 7d72bebd20bdf4058fda6e3cc81675804cdf6417f488f42d21d15451fd1c5256 | 754.603, 760.978 |

Mirrored long cohort long-queue-tail-dcd0-5293-1421 used fresh seed 1909639922 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,429 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 770.236, 767.651 |
| 5293208a | aabc91da0484613013cf07e50b4aea9d074654ca690688f1b6e1f815ba371346 | 753.263, 753.727 |
| dcd0147c | 7d72bebd20bdf4058fda6e3cc81675804cdf6417f488f42d21d15451fd1c5256 | 746.624, 749.739 |

Mirrored short cohort queue-screen-tail-top16-private-1322 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 772.406, 773.037 |
| frontier | 261f82645f28335616673ca010ad1f9a61dd1286f0d113bb9bdd35107ecc16ef | 774.201, 761.232 |
| top16 (internal) | af5499356e75894680c2a6717fc327b6af70ea3c35266347ce2658484e091de8 | 774.406, 771.941 |

This cohort alone does not establish a strict timing lead over frontier, top16. Its observations remain in the registry; longer matched comparisons resolve close screens.

Mirrored short cohort queue-screen-paired-late-tail-3fcd-1432 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 775.866, 775.120 |
| 3fcda0f6 | 6175aa23d63a143ecbc2a5e2fe28d5825d32ed08a7b77dd349e2139ae5d021f4 | 756.188, 758.183 |

Mirrored long cohort long-queue-paired-late-tail-3fcd-1432 used fresh seed 120261951 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,421 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.527, 769.802 |
| 3fcda0f6 | 6175aa23d63a143ecbc2a5e2fe28d5825d32ed08a7b77dd349e2139ae5d021f4 | 755.807, 756.431 |

Mirrored short cohort queue-screen-paired-late-tail-8e54-1450 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 774.034, 772.787 |
| 8e54223c | ddd6b54af68f6fde72521ee6311a976963b1a4ce5dad0a3e16d6f62946111d82 | 761.857, 751.262 |

Mirrored long cohort long-queue-paired-late-tail-8e54-1450 used fresh seed 851813992 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,429 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 770.274, 769.102 |
| 8e54223c | ddd6b54af68f6fde72521ee6311a976963b1a4ce5dad0a3e16d6f62946111d82 | 755.904, 754.284 |

Mirrored long cohort long-queue-tail-top16-private-1425 used fresh seed 279374400 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,382 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.787, 771.013 |
| frontier | 261f82645f28335616673ca010ad1f9a61dd1286f0d113bb9bdd35107ecc16ef | 761.805, 754.061 |
| top16 (internal) | af5499356e75894680c2a6717fc327b6af70ea3c35266347ce2658484e091de8 | 768.679, 769.039 |

Mirrored short cohort queue-screen-tail-public-two-1604 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 773.441, 774.042 |
| 993294ca | c267e4f890ee194f1f1536eb3e0b94a184fdfd8d237b1a942c5577bfe440a0c4 | 757.592, 759.280 |
| b0fbfb1a | c38cc14742f711a0a66dd7b51ce6671fe37252d231016667b36f37ba636b3863 | 765.291, 760.459 |

Mirrored long cohort long-queue-tail-public-research-long-1604 used fresh seed 82273228 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,309 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.748, 769.692 |
| 993294ca | c267e4f890ee194f1f1536eb3e0b94a184fdfd8d237b1a942c5577bfe440a0c4 | 754.623, 751.185 |
| b0fbfb1a | c38cc14742f711a0a66dd7b51ce6671fe37252d231016667b36f37ba636b3863 | 760.573, 755.514 |
| explicit (internal) | 02f82f4181072b813a5f00602bab73298f0140d1c959d856f4d9cce9e994936a | 770.938, 769.554 |

This cohort alone does not establish a strict timing lead over explicit. Its observations remain in the registry; longer matched comparisons resolve close screens.

Mirrored short cohort queue-screen-tail-top16-explicit-private-1453 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 773.968, 772.550 |
| explicit (internal) | 02f82f4181072b813a5f00602bab73298f0140d1c959d856f4d9cce9e994936a | 774.919, 771.724 |
| frontier | 7d72bebd20bdf4058fda6e3cc81675804cdf6417f488f42d21d15451fd1c5256 | 766.135, 759.412 |

This cohort alone does not establish a strict timing lead over explicit. Its observations remain in the registry; longer matched comparisons resolve close screens.

Mirrored short cohort queue-screen-paired-late-tail-42b8-1610 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 774.981, 774.086 |
| 42b8009c | abf54fcc3e0024659ef70b596300f9642229bddce8befe835b45454b094f2d4c | 762.740, 760.322 |

Mirrored long cohort long-queue-paired-late-tail-42b8-1610 used fresh seed 1197319976 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,646 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 770.004, 767.498 |
| 42b8009c | abf54fcc3e0024659ef70b596300f9642229bddce8befe835b45454b094f2d4c | 758.852, 752.665 |

Mirrored short cohort queue-screen-paired-late-tail-cbce-1631 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 771.168, 774.230 |
| cbce5501 | 8095e3acf2192bde6348e29d7856e1f8cdba0c8787e4662ce6d1294dfccef81a | 771.438, 766.523 |

This cohort alone does not establish a strict timing lead over cbce5501. Its observations remain in the registry; longer matched comparisons resolve close screens.

Mirrored long cohort long-queue-paired-late-tail-cbce-1631 used fresh seed 1033247801 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,451 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.053, 769.333 |
| cbce5501 | 8095e3acf2192bde6348e29d7856e1f8cdba0c8787e4662ce6d1294dfccef81a | 762.836, 753.416 |

Mirrored short cohort queue-screen-tail-6976-4ce3-1641 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 771.024, 772.683 |
| 4ce3607f | cb6fa52e70dafd407b852f1c98c93a103ffba76763cdfa73a5165c0b04177898 | 747.017, 752.967 |
| 69769298 | 615106d750300c411be277f1ff8098d07e5ef4c5a256008bb3cd7bd75a11a8de | 749.958, 751.138 |

Mirrored long cohort long-queue-tail-6976-4ce3-1641 used fresh seed 147191010 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,379 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 768.349, 769.985 |
| 4ce3607f | cb6fa52e70dafd407b852f1c98c93a103ffba76763cdfa73a5165c0b04177898 | 742.022, 747.256 |
| 69769298 | 615106d750300c411be277f1ff8098d07e5ef4c5a256008bb3cd7bd75a11a8de | 757.288, 753.959 |

Mirrored short cohort queue-screen-paired-late-tail-d37e-defc-1703 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 773.841, 772.384 |
| d37e3975 | 65aad9d9c98248bf5bd9e3d67e68286f099db54fc6cbe9f65c6bbd973e418964 | 758.756, 755.939 |
| defcba82 | 42fed8743f794fb22a3d52a6c9af0a2fd5a0c4405a65ff8347ac6298ecf7d36f | 757.037, 763.413 |

Mirrored long cohort long-queue-paired-late-tail-d37e-defc-1703 used fresh seed 2132899463 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,372 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 768.653, 767.723 |
| d37e3975 | 65aad9d9c98248bf5bd9e3d67e68286f099db54fc6cbe9f65c6bbd973e418964 | 750.268, 752.220 |
| defcba82 | 42fed8743f794fb22a3d52a6c9af0a2fd5a0c4405a65ff8347ac6298ecf7d36f | 755.055, 750.625 |

Mirrored short cohort queue-screen-paired-late-tail-356f-1726 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 772.542, 771.601 |
| 356fa57d | 30226e61808cb28d9ed9a8e97b748dc4ede373df4740b5d0815d0de9652f24bb | 757.722, 753.407 |

Mirrored short cohort queue-screen-tail-4b28-1735 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 773.789, 772.772 |
| 4b28ec4c | 00d273c09b090bb369517ad9aa2be46ad13d6bc81fb29fbce723ae0115a2fb03 | 763.967, 762.690 |

Mirrored long cohort long-queue-tail-4b28-1735 used fresh seed 1745491323 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,483 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 768.219, 768.361 |
| 4b28ec4c | 00d273c09b090bb369517ad9aa2be46ad13d6bc81fb29fbce723ae0115a2fb03 | 751.109, 754.508 |

Mirrored short cohort queue-screen-paired-late-tail-d0f9-b251-1759 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 773.248, 770.499 |
| b251535b | 4a3b6d2dc482d86ae7573a4fbf3360e8a30cc2fdd8ca0e247c6364cc71b6103b | 761.343, 763.960 |
| d0f90bc2 | e99cf890b25c1c50e91266780048032d47890b3bdc156069c864445d76baa7cb | 762.640, 760.250 |

Mirrored long cohort long-queue-paired-late-tail-d0f9-b251-1759 used fresh seed 697490415 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,416 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.290, 768.831 |
| b251535b | 4a3b6d2dc482d86ae7573a4fbf3360e8a30cc2fdd8ca0e247c6364cc71b6103b | 750.458, 752.804 |
| d0f90bc2 | e99cf890b25c1c50e91266780048032d47890b3bdc156069c864445d76baa7cb | 755.521, 756.785 |

Mirrored short cohort queue-screen-tail-bdbf-3fe1-1923 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.731, 773.674 |
| 3fe1c58f | 508bbe2d39c4f3d9a5e17571074ead504470f330152d8d3c0e71c26963ae102a | 764.252, 761.371 |
| bdbf5d40 | da5fbbe3e3d808199db3eb2528898238eec73e9a79acf18d5ac40a81e1f6ee55 | 752.785, 763.840 |

Mirrored long cohort long-queue-tail-bdbf-3fe1-1923 used fresh seed 1579187237 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,453 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 768.347, 769.861 |
| 3fe1c58f | 508bbe2d39c4f3d9a5e17571074ead504470f330152d8d3c0e71c26963ae102a | 764.874, 756.871 |
| bdbf5d40 | da5fbbe3e3d808199db3eb2528898238eec73e9a79acf18d5ac40a81e1f6ee55 | 755.992, 753.084 |

Mirrored short cohort queue-screen-paired-late-tail-07b6-1942 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 773.637, 773.872 |
| 07b61e21 | fa9d92771659563b7b75873f6f440167a75f4ee86a2e759587da285c5818c6d0 | 764.342, 753.013 |

Mirrored long cohort long-queue-paired-late-tail-07b6-1942 used fresh seed 1063981968 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,352 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.348, 770.440 |
| 07b61e21 | fa9d92771659563b7b75873f6f440167a75f4ee86a2e759587da285c5818c6d0 | 752.387, 763.844 |

Mirrored short cohort queue-screen-paired-late-tail-23cc-1953 and completed 4,978,400,000 timed candidates per observation. Every invocation matched 858 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 774.754, 773.315 |
| 23cca647 | d1f6eb904cbdb1c00c202eb99bed173ee705007e8ac451741e1eeac457649fa2 | 764.080, 758.548 |

Mirrored long cohort long-queue-paired-late-tail-23cc-1953 used fresh seed 1583356724 and completed 74,676,000,000 timed candidates per observation. Every invocation matched 10,212 independently verified tuples. Rates use identical completed work and timing boundaries within each cohort, in forward and reverse order.

| Arm | Source digest | Two rates (M/s) |
| --- | --- | --- |
| control (candidate) | 6048058e853dceb2d90f1a8fe59654a803e3fde91823f525e11498e763c50973 | 769.611, 769.918 |
| 23cca647 | d1f6eb904cbdb1c00c202eb99bed173ee705007e8ac451741e1eeac457649fa2 | 756.918, 756.035 |

An additional long comparison against public 356fa57d, runtime
30226e61808cb28d9ed9a8e97b748dc4ede373df4740b5d0815d0de9652f24bb, stopped on seed
1291225207 because the reported hit sets differed. The candidate returned
10,487 valid tuples and that peer returned 10,486; all reported tuples from both
were independently verified. The missing peer tuple is recorded in the preserved
failure investigation. This incomplete cohort supplies no long throughput
comparison, and no correctness exclusion was registered from this diagnostic.
Its successful short cohort remains in the evidence above. Original failed
inputs, binaries and logs are preserved, with no fabricated completion check or
work normalization. The unchanged final gate still checks the complete current
public set before upload.

A separate short screen against public db0e2371, runtime
bef824a9f3a1238df4a715fc6211d6d023c3dacdfa9829bfa26a74d29d8f1b62, also stopped because
its reported hit set differed. On seed 201030808, the candidate
returned 858 valid tuples and the peer returned 419 valid tuples, a subset
missing 439 of the candidate's results. Both returned sets were independently
verified. No complete short or long comparison was registered for this peer;
its long run never started. No production invalid-output exclusion was inferred
from missing reports. These failed original inputs, binaries and logs are also
preserved separately from the completed comparison registry.

The unchanged production wrapper passed the full 1,200-second contract on fresh
seed 210323393. All 109,445 reported hits were
independently verified, with no failures. Artifact time was 1200.171
seconds and outer wall time was 1201.711836 seconds. Original inputs,
source and binary hashes passed before and after. All original hits were also
downloaded and independently rechecked locally before qualification registration.

The registry contains 44 actual matched cohorts and 81
exact-source comparisons. The live gate passed at 2026-09-20T20:10:05.171663+00:00.
Its frontier source is e876032f79e6f4f3af2732bbba39403e29f0e227; the platform
threshold is 100 basis points.
Every pending public source was considered using exact recursive source hashes.
Comment or line-ending differences are not automatically equated. Strongest
public sources require repeated long comparisons. Submission refreshes the
entire public set again immediately before upload.

These are local correctness and completed-work measurements, not official scores
or a promise of promotion. SOURCE-MANIFEST.json binds the runtime, native and full
receipts, CPU proof and every cohort's original evidence. The official verifier
and harness clock determine the ranked outcome.
