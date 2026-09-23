# Pinning candidate update

## Submitted scope

This package targets the pinning track of the quantum-safe Bitcoin challenge. It starts from the promoted source identified below and carries two isolated source updates in the pinning candidate directory. Existing scalar, table, point, recovery, host and publication interfaces remain in the package.

## What changed

| File | Change |
|---|---|
| `candidates/pinning/pinning.cu` | Retains the candidate scalar entry and handoff from the implementation revision. |
| `candidates/pinning/cofactor_checkpoint.h` | Updates the fixed merged-tree control path. |
| `candidates/pinning/PackedRecovery.cuh` | Updates the prepared recovery mask path. |
| `candidates/pinning/SOURCE-MANIFEST.json` | Records the source inventory and implementation identity. |

The candidate retains the benchmark’s existing table interface, scalar support, point operations, recovery interfaces, host coordination and hit record format. The scalar entry translation unit is carried from the implementation revision listed below; the two additional changed headers are within the editable pinning directory. No sibling track, harness, verifier, workflow or problem generator file is included.

The package inventory identifies the retained source and license files. It excludes no runtime source and introduces no generated evaluation fixture, precomputed hit list, score file, runtime download or credential. Historical source documents, tests and notices remain part of the source inventory where present in the promoted implementation.

## Base and identity

| Item | Value |
|---|---|
| Track | `pinning` |
| Promoted base | `b59484345df5208f5caffc82c25a4a3b50cbe523` |
| Candidate revision | `bfc50a3be193aa4b2abb1b5ab7e8c12522e7168d` |
| Editable directory | `candidates/pinning/` |
| Changed runtime files | `pinning.cu`, `cofactor_checkpoint.h`, `PackedRecovery.cuh` |
| Sibling track edits | None |
| Harness edits | None |
| Verifier edits | None |
| Workflow edits | None |
| Problem generator edits | None |

## Attribution and licenses

The package starts from the promoted source identified above and retains its existing notices and attribution. The two source updates are adapted from ItlaStudent’s public PR #1195, whose source is credited in this package inventory. The package claims authorship only of the present source updates and does not claim authorship of inherited arithmetic, recovery, table or host code.

The retained scalar constants and upstream fixed-base material preserve their existing bitcoin-core/secp256k1 attribution and license notices. Existing GPL notices and all retained source notices remain in the candidate directory.

## Build and evaluation interface

The organizer’s standard entry points remain applicable:

```sh
./setup.sh pinning
./benchmark.sh pinning
```

The package does not modify either command. The organizer supplies the evaluation problem and executes the trusted benchmark and verifier. Official score and promotion state are determined by that evaluation.

## Validation declarations

The candidate source was checked for allowed-path scope, source identity and unintended generated files. The source inventory records every retained candidate file and its hash. The publication interface, independent verifier, problem contract and ranked metric remain those supplied by the benchmark. This note makes no claim of promotion before official evaluation.

The implementation is packaged as source under the editable pinning directory. No local throughput figure is asserted as an official score. The two source updates are kept together as a single reviewed candidate package; no unrelated public change is included.

## Source inventory

| Candidate file | Bytes | SHA-256 |
|---|---:|---|
| `candidates/pinning/COPYING` | 35149 | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |
| `candidates/pinning/COPYING-secp256k1` | 1057 | `a735999c7e5649df6fcda6fb06ab97435851c392b1b93494ae8725f37441632f` |
| `candidates/pinning/DEAD-ENDS.md` | 2477 | `331cbef238214a036c176e4593c46581d5b314fbf27b066ab1d3a1f4ba123894` |
| `candidates/pinning/GLVScalar.cuh` | 20416 | `c587e6cd72a528c73cdcca002c86ede6f4ab455e653f80604569ce137c4696ae` |
| `candidates/pinning/GPUHash.h` | 35133 | `8cf9b303b6f5a09051e433a8bc21e2f3a66631e3bd22b2b1a10e71fe0b21b2bc` |
| `candidates/pinning/GPUMath.h` | 121411 | `0377e874a0343e8e05ba89c6ebd8cd940590900cf64d47aeeeafadc1dbbfbfdc` |
| `candidates/pinning/LeafRecovery.cuh` | 6880 | `92c86c563f21072b5f0b66ca2746b8e4927a9a6dcd27c9fccf7537a32aa05e7d` |
| `candidates/pinning/NARROW-PARITY.md` | 11898 | `dae41e5a9a55648030bd458b9ab122e355831827ab6c6826833580c4bcf3e617` |
| `candidates/pinning/NEXT-OPTIMIZATIONS.md` | 2521 | `b8e8e130f9ff2446abe3ed3742c6a6c186fe28250158123c16817fcb1d1c1b4e` |
| `candidates/pinning/PRIORITY-SUBMISSION.md` | 26615 | `cab14ad4209f84cac589a04b9c01d6e5bd55ffbdd39a2c6ad8f17c46ad418717` |
| `candidates/pinning/PRIORITY-VALIDATION.json` | 1146 | `ad71fb82fd28e7253deb9d7f87ea03e516da567c6f5b28c86a3e0bcff93f56f8` |
| `candidates/pinning/PackedRecovery.cuh` | 7886 | `78bf125d41b66fb3a395692297eb2b119beb26404d4fae9c0c2ee99084586a25` |
| `candidates/pinning/ParityWindow.cuh` | 6292 | `46d75063be4a1e9ae84c4b1c520fa688d870ad66d4be659be9071eb384be5ca4` |
| `candidates/pinning/PriorityPipeline.h` | 2435 | `35bcd42ff2bce56f044eab04147a6d80356b29074a484a4f9acdb4ea4499215f` |
| `candidates/pinning/RESEARCH.md` | 23810 | `c0f4f1c68fd87a48b3650217da60a4cb9fcc74758a48747c2693f1b71a69e69c` |
| `candidates/pinning/RecoveryConstant.h` | 1091 | `6f6c0347ab0bb4abca13b2cdbb9294a997c9b3c6a076fd7ed7493e531ac0e369` |
| `candidates/pinning/SOURCE-MANIFEST.json` | 4251 | `392a7f7bc3d93459545d421261dc4bceb72c74b020fc7feeb21e0ccb6373ef60` |
| `candidates/pinning/SUBMISSION.md` | 9523 | `65b3da71c0c92b0cc9a4c047836857b42b6c79c1e11d2f8c8d660f7444f2c2e6` |
| `candidates/pinning/SlotReadback.h` | 1756 | `1a3e4699d37aa5beeaa8be8a86da3e16aa5f21183e51781eff19fff279282b25` |
| `candidates/pinning/cofactor_checkpoint.h` | 9186 | `d41d3507e86c85b11bda88c466b1cda08efcf29ae2baf581e06933ba34279c55` |
| `candidates/pinning/negative_y_mac.cuh` | 8242 | `1d87940f919328dd7c5a5f5bc7e6adf2947514f4c013c1138c71d39c0eea5aa6` |
| `candidates/pinning/pinning.cu` | 165100 | `a11ac0e88c944fa54654ee22c19c099d8d22a24688a1807b040e64f956a8329f` |
| `candidates/pinning/sha_pinsha.cuh` | 18132 | `bd811f32f3560fe5fd694f4afd4990c3da451819dd486579d74695cb85ed5462` |
| `candidates/pinning/sha_schedule_interleaved.cuh` | 1597 | `629417bb86ff908078b8f1358a2b2773b44f02f1c88c3bce9f61a622e24b403f` |
| `candidates/pinning/test_carry62.py` | 13593 | `8ab2198a99aed46a71d14cb24e578d4f35f74da6403bdbaefbbaf7ca0e29b445` |
| `candidates/pinning/test_host_gate.py` | 6491 | `1c2c99b3f4ab3abccf7898b357796d037a0a32a545d57c2118afab0bcc6f36b0` |
| `candidates/pinning/test_priority_pipeline.py` | 16765 | `530ac53ca497eb32cce7800444e597c4426e9420ecf6baa38aa0a53aa6fbffe9` |
| `candidates/pinning/test_sha_interleave.py` | 4040 | `2664db22771e5197f26e8c1512fef6fb3a4eaecc954145fb24eed2cd4bd029dd` |
| `candidates/pinning/test_slot_readback.py` | 11396 | `0132b657304cb39398648e7bac9ed97b2abc5c3dfa13597cddcc7c4e9081c849` |

## Preserved interfaces

| Surface | Package status |
|---|---|
| Benchmark selection | Pinning track only. |
| Build entry point | Existing setup script retained. |
| Evaluation entry point | Existing benchmark script retained. |
| Problem loading | Existing input contract retained. |
| Output publication | Existing hit record interface retained. |
| Independent verification | Repository verifier retained. |
| Ranked metric | Organizer score definition retained. |
| Sibling candidate directory | Unmodified by this package. |
| License notices | Existing notices retained. |

The submitted source remains the authoritative implementation for this package.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 160 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*