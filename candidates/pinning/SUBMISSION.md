# Pinning candidate update

## Submitted scope

This package targets the pinning track of the quantum-safe Bitcoin challenge. It starts from the promoted source identified below and restores the scalar seed handoff in the candidate translation unit. The promoted implementation, supporting headers, host coordination and publication interfaces are retained.

This note describes the present package. The historical research documents retained in the candidate directory are inherited material; their earlier proposals, measurements and descriptions are not claims made by this submission.

## What changed

| File | Change |
|---|---|
| `candidates/pinning/pinning.cu` | Restores the scalar seed handoff and its compile-time selector. |
| `candidates/pinning/SUBMISSION.md` | Supplies this package description. |
| `candidates/pinning/SOURCE-MANIFEST.json` | Records the current source inventory and implementation identity. |

The scalar entry point includes the register seed handoff and the existing zero-component handling. Its table interface, scalar support, point operations and recovery interfaces remain those of the promoted implementation.

The package removes an inherited generated Python cache and two superseded priority-submission metadata files. License notices, source files and retained test files are included in the inventory. The remaining historical notes retain their original text under an explicit inherited-document provenance label.

The implementation and support files remain within the editable pinning directory. The note and manifest are packaging metadata. Inventory rows exclude those two metadata files so the note does not attempt to hash itself.

## Base and package identity

| Item | Value |
|---|---|
| Track | `pinning` |
| Promoted base | `b59484345df5208f5caffc82c25a4a3b50cbe523` |
| Implementation revision | `9867b6b5c83103ec895dac3d1b7fd3254194a064` |
| Editable directory | `candidates/pinning/` |
| Candidate translation unit | `candidates/pinning/pinning.cu` |
| Sibling track edits | None |
| Harness edits | None |
| Verifier edits | None |
| Workflow edits | None |
| Problem generator edits | None |

## Attribution and licenses

The package retains fkiene’s promoted public submission `32bc0c54-29a6-4e3f-9f97-7d6be69915f3`, source revision `b59484345df5208f5caffc82c25a4a3b50cbe523`. fkiene is credited as a coauthor. The promoted source and its preserved notices identify the inherited arithmetic, host coordination and upstream contributions.

The restored scalar seed handoff adapts the corresponding portion of DrCleverHans’s unpromoted public submission `a149d7b4-4ddf-408c-b6c1-1f4e15bfd41d`, published in PR #1164 at source revision `c3ac7754a0050e22d661e69969d3968ad57d8061`. DrCleverHans and fkiene are credited as coauthors for that donor contribution.

The inherited scalar support and grouped fixed-base work draw on may93182’s unpromoted public submission `5ffaef34-a958-4e80-887d-893f6b933605`. may93182 is credited as a coauthor. The package retains the promoted implementation and source notices accompanying that material.

The inherited host transfer support substantially adapts Portablelle’s unpromoted public submission `b67219a6-4dab-4e67-a2c0-04e698006919`, published in PR #1154. The host coordination support also draws on Portablelle’s unpromoted public submission `59b3693f-fb93-4203-a77f-1e286266dede`, source revision `f60940f15ce04bb641e893ff801e295d36eaad3f`. Portablelle is credited as a coauthor for those retained contributions.

The scalar constants and associated upstream material retain the bitcoin-core/secp256k1 attribution to Pieter Wuille and its MIT license notice. The candidate’s existing GPL license and other source notices are retained. This package claims authorship only of its own modifications, not of the inherited arithmetic library, promoted parent or donor implementation.

## Build and execution interface

The organizer’s standard entry points remain applicable:

```sh
./setup.sh pinning
./benchmark.sh pinning
```

The package does not require changes to either script. The candidate is compiled from source using the benchmark’s configured CUDA build command and existing library dependencies. The normal input file, command-line interface and hit record format remain in place. No external service, runtime source download or additional credential is introduced.

The organizer supplies the evaluation problem and executes the trusted benchmark and verifier. The archive does not supply a stored solution, generated evaluation fixture, precomputed hit list or replacement score file. The official result and any promotion are determined by that evaluation.

## Validation declarations

The package inventory identifies the frozen implementation bytes and their hashes. The source archive was checked for allowed-path scope and unintended generated files. The inventory identifies retained files as well as changed files; inclusion does not mean every file was modified. No local throughput figure is asserted as an official score, and this note makes no claim of promotion before the official result.

The compile-time alternative retains the promoted scalar-entry source. The publication interface and independent verifier remain applicable to the restored handoff. Package metadata does not select a different compiler command, problem, candidate stream or scoring definition.

## Source inventory

| Candidate file | Bytes | SHA-256 |
|---|---:|---|
| `candidates/pinning/COPYING` | 35149 | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |
| `candidates/pinning/COPYING-secp256k1` | 1057 | `a735999c7e5649df6fcda6fb06ab97435851c392b1b93494ae8725f37441632f` |
| `candidates/pinning/DEAD-ENDS.md` | 2713 | `e3c19e389c99c12f02b0384f8364df08a6c30eaa0d282eb3c6682cada2c86a4e` |
| `candidates/pinning/GLVScalar.cuh` | 20416 | `c587e6cd72a528c73cdcca002c86ede6f4ab455e653f80604569ce137c4696ae` |
| `candidates/pinning/GPUHash.h` | 35133 | `8cf9b303b6f5a09051e433a8bc21e2f3a66631e3bd22b2b1a10e71fe0b21b2bc` |
| `candidates/pinning/GPUMath.h` | 121411 | `0377e874a0343e8e05ba89c6ebd8cd940590900cf64d47aeeeafadc1dbbfbfdc` |
| `candidates/pinning/LeafRecovery.cuh` | 6880 | `92c86c563f21072b5f0b66ca2746b8e4927a9a6dcd27c9fccf7537a32aa05e7d` |
| `candidates/pinning/NARROW-PARITY.md` | 12134 | `e11e375198795fe95cc839057bebcf77d13ea0b79f17b97a28c69d68bfc417c6` |
| `candidates/pinning/NEXT-OPTIMIZATIONS.md` | 2757 | `1c0900ccc2e2f07a6f175b5418f1d97dad8b063eb6879038313a63b114e52411` |
| `candidates/pinning/PackedRecovery.cuh` | 7733 | `47e16d8a2e3e6d193bd864c681ef8a9af8743332b01beb5ae29e9bb946c34afe` |
| `candidates/pinning/ParityWindow.cuh` | 6292 | `46d75063be4a1e9ae84c4b1c520fa688d870ad66d4be659be9071eb384be5ca4` |
| `candidates/pinning/PriorityPipeline.h` | 2435 | `35bcd42ff2bce56f044eab04147a6d80356b29074a484a4f9acdb4ea4499215f` |
| `candidates/pinning/RESEARCH.md` | 24046 | `3a6e23d2df9cc55700c1e9ba762eb0e33dd15c3f6336eba1741759dbb9ce9bfd` |
| `candidates/pinning/RecoveryConstant.h` | 1091 | `6f6c0347ab0bb4abca13b2cdbb9294a997c9b3c6a076fd7ed7493e531ac0e369` |
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

## Preserved interface inventory

| Surface | Package status |
|---|---|
| Benchmark selection | Pinning track only. |
| Build entry point | Existing setup script retained. |
| Evaluation entry point | Existing benchmark script retained. |
| Problem loading | Existing input contract retained. |
| Output publication | Existing hit record interface retained. |
| Independent verification | Repository verifier retained. |
| Ranked metric | Organizer’s score definition retained. |
| Submission packaging | Editable candidate source only. |
| External dependencies | Existing benchmark dependencies retained. |
| Sibling candidate directory | Unmodified by this package. |
| License notices | Existing notices retained, supplemental upstream notice included. |

The note and manifest are packaging metadata. Retained research documents identify their own historical context. Neither the inventory nor those documents changes the benchmark’s execution or scoring definition. The submitted source remains the authoritative implementation for this package.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 160 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*