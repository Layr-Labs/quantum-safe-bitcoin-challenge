# Priority dispatch: PR1160 GLV14 exact public source

This package preserves the executable pinning source from public PR #1160, head `b06492451cf014301f0946ba53c42dc6224719f0`, and adds only this dispatch note plus an integrity manifest. The compiled candidate bytes are unchanged from that public source. The PR credits i34-9 for the integration, may93182 for the grouped fixed-base and GLV contribution, and Portablelle for the compact host transfer contribution; all three are included as submission coauthors.

## Independent local validation before dispatch

The candidate and the promoted `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba` control were compiled with CUDA 12.8 and run A/B/B/A on the same official pinning problem with a fixed 64-sequence work cap. Every arm searched exactly `79,654,400,000` positions and produced the same 9,449-hit canonical set (`957140874e232cec747c5d44d602504f6dd5a9bfcad5801929c01714151838f6`). The unmodified verifier independently accepted 9,449/9,449 control hits with zero failures.

| Arm | Source | Search seconds | Wall seconds |
|---|---|---:|---:|
| A1 | promoted control | 96.980354646 | 100.492626125 |
| B1 | PR1160 | 95.017962833 | 96.824782694 |
| B2 | PR1160 | 95.212803516 | 97.152207904 |
| A2 | promoted control | 96.582241062 | 98.648783260 |

The two adjacent search gains were +2.0653% and +1.4383%; pooled search gain was +1.7515%, and pooled wall gain including table setup was +2.6624%. This is a local calibration, not an official score or promotion claim. Static builds for sm_89 and sm_52 had no spills; stage-0 registers fell from 124 to 122 on sm_89 and from 101 to 97 on sm_52, while stage-2 resources were unchanged. The repository setup/verifier smoke test, slot-readback tests, SHA tests, carry tests, and host-gate tests passed.

The live record at preparation was 813,651,852 with a 100-bips threshold, so the strict promotion floor was 821,788,371. Yukon remains authoritative for the result.

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