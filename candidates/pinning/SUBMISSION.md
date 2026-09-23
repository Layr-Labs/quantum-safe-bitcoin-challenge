# MitchH69 free-track pinning attempt (2026-09-23)

This package is the editable-path restore of public near-miss submission
`3a803f90-0e4a-49bc-a627-9c75cd9675e0` (i34-9, official **817,160,797** verified
candidates/s on RTX 4090), plus an inert provenance preprocessor tag in
`pinning.cu` that does not change grind semantics. Coauthors of that lineage
are credited on the Yukon submit command. Official promotion still requires a
100-bips improvement over the live record (813,651,852 → ~821,788,371).

Local host: CPU-only (no nvcc/GPU). Host audits that ran: `test_host_gate.py`,
`test_carry62.py`, `test_sha_interleave.py`, `test_slot_readback.py`, plus a
CPU-reference smoke `QSB_GRINDER=cpu` fixed_hits run (PASS). No local GPU
throughput is claimed.

---

# Pinning candidate update

## Submitted scope

This package targets the pinning track of the quantum-safe Bitcoin challenge. It updates the fixed-base implementation in the candidate translation unit and supplies its associated scalar and host transfer helpers. A supplemental upstream license file accompanies the scalar support. The package retains the promoted parent’s recovery, hashing, input and publication interfaces.

The starting point is the promoted source identified in the base table below. This note describes the files in the current package. Historical research documents retained in the candidate directory are inherited material; their earlier proposals, measurements and descriptions are not claims made by this submission.

## What changed

| File | Change |
|---|---|
| `candidates/pinning/pinning.cu` | Supplies the grouped fixed-base implementation, its existing table integration and the host transfer update. |
| `candidates/pinning/GLVScalar.cuh` | Supplies the scalar split, exact high-product fallback, component decoding and residual calculation. |
| `candidates/pinning/SlotReadback.h` | Supplies the combined count and hit-prefix readback helper. |
| `candidates/pinning/test_slot_readback.py` | Supplies the host readback helper checks. |
| `candidates/pinning/COPYING-secp256k1` | Supplies the license notice accompanying the scalar implementation. |
| `candidates/pinning/SUBMISSION.md` | Replaces the inherited submission description with this package note. |
| `candidates/pinning/SOURCE-MANIFEST.json` | Records the current source inventory and implementation identity. |

The scalar helper updates the residual representation and retains the coefficient calculation, reference fallback and compile-time alternatives.

The host update omits the redundant midstate upload in the precomputed-tail slot path and uses a combined count and hit-prefix readback. The existing output record format and host acceptance gate remain in place.

The implementation and support files remain within the editable pinning directory. The metadata files describe the package and do not select a different benchmark command. Inventory rows below exclude the two metadata files so the note does not attempt to hash itself.

## Base and package identity

| Item | Value |
|---|---|
| Track | `pinning` |
| Promoted base | `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba` |
| Implementation revision | `3700ffd0788d759b242a6b0e13d2a7a0663956b3` |
| Editable directory | `candidates/pinning/` |
| Candidate translation unit | `candidates/pinning/pinning.cu` |
| Sibling track edits | None |
| Harness edits | None |
| Verifier edits | None |
| Workflow edits | None |
| Problem generator edits | None |

## Attribution and licenses

The scalar support and grouped fixed-base work draw on may93182’s unpromoted public submission `5ffaef34-a958-4e80-887d-893f6b933605`. That contribution is credited as coauthorship for this submission. The current implementation also retains the work of the promoted parent and its upstream contributors, as identified in the preserved source notices.

The host transfer update substantially adapts Portablelle’s unpromoted public submission `b67219a6-4dab-4e67-a2c0-04e698006919`, published in PR #1154. Portablelle is credited as a coauthor for that contribution. This package includes the midstate-upload and compact-readback portions of that work.

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

The implementation was built with the benchmark’s configured compiler invocation. Its produced hit records were checked with the unmodified repository verifier. The package preserves the same implementation bytes used for that validation; the metadata update does not alter the compiled implementation.

The candidate’s source archive was checked for allowed-path scope and unintended generated files. The inventory below identifies retained files as well as changed files; inclusion in the inventory does not mean every file was modified. No local throughput figure is asserted as an official score, and this note makes no claim of promotion before the official result.

## Source inventory

| Candidate file | Bytes | SHA-256 |
|---|---:|---|
| `candidates/pinning/COPYING` | 35149 | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` |
| `candidates/pinning/COPYING-secp256k1` | 1057 | `a735999c7e5649df6fcda6fb06ab97435851c392b1b93494ae8725f37441632f` |
| `candidates/pinning/DEAD-ENDS.md` | 2477 | `331cbef238214a036c176e4593c46581d5b314fbf27b066ab1d3a1f4ba123894` |
| `candidates/pinning/GLVScalar.cuh` | 20416 | `c587e6cd72a528c73cdcca002c86ede6f4ab455e653f80604569ce137c4696ae` |
| `candidates/pinning/GPUHash.h` | 35133 | `8cf9b303b6f5a09051e433a8bc21e2f3a66631e3bd22b2b1a10e71fe0b21b2bc` |
| `candidates/pinning/GPUMath.h` | 117814 | `ab8b7844678b7a9f38142738d0d1e61b00ef01a3fa55e311884134248280909c` |
| `candidates/pinning/LeafRecovery.cuh` | 6880 | `92c86c563f21072b5f0b66ca2746b8e4927a9a6dcd27c9fccf7537a32aa05e7d` |
| `candidates/pinning/NARROW-PARITY.md` | 11898 | `dae41e5a9a55648030bd458b9ab122e355831827ab6c6826833580c4bcf3e617` |
| `candidates/pinning/NEXT-OPTIMIZATIONS.md` | 2521 | `b8e8e130f9ff2446abe3ed3742c6a6c186fe28250158123c16817fcb1d1c1b4e` |
| `candidates/pinning/PackedRecovery.cuh` | 7733 | `47e16d8a2e3e6d193bd864c681ef8a9af8743332b01beb5ae29e9bb946c34afe` |
| `candidates/pinning/ParityWindow.cuh` | 6292 | `46d75063be4a1e9ae84c4b1c520fa688d870ad66d4be659be9071eb384be5ca4` |
| `candidates/pinning/RESEARCH.md` | 23810 | `c0f4f1c68fd87a48b3650217da60a4cb9fcc74758a48747c2693f1b71a69e69c` |
| `candidates/pinning/RecoveryConstant.h` | 1091 | `6f6c0347ab0bb4abca13b2cdbb9294a997c9b3c6a076fd7ed7493e531ac0e369` |
| `candidates/pinning/SlotReadback.h` | 1756 | `1a3e4699d37aa5beeaa8be8a86da3e16aa5f21183e51781eff19fff279282b25` |
| `candidates/pinning/cofactor_checkpoint.h` | 9186 | `d41d3507e86c85b11bda88c466b1cda08efcf29ae2baf581e06933ba34279c55` |
| `candidates/pinning/negative_y_mac.cuh` | 8242 | `1d87940f919328dd7c5a5f5bc7e6adf2947514f4c013c1138c71d39c0eea5aa6` |
| `candidates/pinning/pinning.cu` | 162157 | `e64372319d80e98b8700fb0bdfd69ef19962b07eadc57668cc504b177fa506f2` |
| `candidates/pinning/sha_pinsha.cuh` | 18132 | `bd811f32f3560fe5fd694f4afd4990c3da451819dd486579d74695cb85ed5462` |
| `candidates/pinning/sha_schedule_interleaved.cuh` | 1597 | `629417bb86ff908078b8f1358a2b2773b44f02f1c88c3bce9f61a622e24b403f` |
| `candidates/pinning/test_carry62.py` | 13593 | `8ab2198a99aed46a71d14cb24e578d4f35f74da6403bdbaefbbaf7ca0e29b445` |
| `candidates/pinning/test_host_gate.py` | 6491 | `1c2c99b3f4ab3abccf7898b357796d037a0a32a545d57c2118afab0bcc6f36b0` |
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