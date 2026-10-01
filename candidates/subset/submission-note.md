# SUBSET: host input scratch and descriptor update

This exploratory SUBSET package retains the promoted GPU implementation and updates the host candidate source. The CPU header includes the previously selected arithmetic-boundary and key-message preparation edits together with a worker-local input scratch update. The host producer and tree source include the descriptor update. The candidate controls for those edits are enabled in the submitted source. No local or official performance gain is claimed.

## Base and attribution

The promoted base is cefika's submission fb6f5a8f-b29e-4506-a50c-c79a9c5a2a0e, source [ff27a2b66990a3eb554a1d4453e896c0397337ba](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/ff27a2b66990a3eb554a1d4453e896c0397337ba). The entry wrapper, paired arithmetic, GPU hash helpers and packaged native image are retained from that base.

The selected CPU arithmetic-boundary edits are attributed to jacklightChen's public submission b1c5e58e-210d-4353-addf-edc4474b8f33, immutable [CPU donor source843c82817bf2de27e7794bb6317d1d223d6fa1fd](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/843c82817bf2de27e7794bb6317d1d223d6fa1fd). The additional key-message preparation, input scratch and descriptor edits are team additions. Only the selected host edits from the donor are included.

The base promotion records acknowledgments to kshitij-hash, terrapinelf, i34-9, ercumentyildirim, HyeokxC, jacklightChen, fkiene, kaankolcu, newjordan and Meganpark980320. Those names are retained here as base provenance. The submission owner is i34-9. This is a solo submission; donor credit is recorded in this note. Existing VanitySearch and third-party notices remain in the candidate sources.

## Updated candidate surfaces

| File | Package contents |
|---|---|
| CpuGrindSubset.h | The selected host arithmetic, key-message preparation and worker-local input scratch edits. |
| tests/gpu_epochs/host_producers_v3.h | The selected host descriptor edit, enabled in the candidate source. |
| tests/gpu_epochs/tree.cu | The corresponding host descriptor interface update. |
| submission-note.md | This scope, attribution and packaging declaration. |

The CPU input scratch edit is confined to the existing planned worker route. The existing CPU digest-output interface is retained. Ordinary field operations, table entries, scalar recovery code, key-message layout and publication interfaces remain present. The package retains the existing fallback routes and host feature checks. It supplies the candidate directly to the ordinary track build and does not add a separately invoked executable or dynamically loaded host component.

The host descriptor update is included in the existing producer and tree source. The existing launch interface, startup self-check, stream ownership and output collection remain in the source. The change does not replace the benchmark's input, verification code or scoring code. The existing fallback producer remains available in the candidate.

## Preserved device and library surfaces

| Surface | Package declaration |
|---|---|
| subset.cu | Retained entry wrapper and candidate arguments. |
| qsb_carrier_sm89.h | Retained packaged native image and metadata. |
| QsbCarrier.h | Retained carrier-selection and fallback interface. |
| GPUHash.h | Retained GPU hash source. |
| GPUMath.h | Retained GPU field arithmetic source. |
| GLVScalar.cuh | Retained scalar helper source. |
| sha_gate_fma.cuh | Retained GPU hash helper and selected addition setting. |
| square32.cuh | Retained field square source. |
| pair_shared.cuh | Retained paired arithmetic and recovery interface. |
| filter_tail_sc.cuh | Retained device filtering source. |
| inverse_limbs.cuh | Retained inverse source and interface. |
| window_schedule_shared.cuh | Retained GPU input hash scheduling source. |
| epoch_groups.cuh | Retained epoch grouping declarations. |
| host_producers_v1.h | Retained alternative host producer source. |
| host_producers.h | Retained legacy producer source. |
| qsb_host_verify.h | Retained candidate-side verification source. |

The packaged native image is unchanged from the base. This is a content declaration about the archive, separate from the image that a remote runtime ultimately selects. The existing carrier checks remain in the package. GPU field operations, hash operations, geometry, public-key output layout and record format are retained. The tree's changed host lines belong to the descriptor update listed above.

## Scope and packaging declarations

| Item | Source declaration |
|---|---|
| Selected competition track | SUBSET. |
| Editable surface | candidates/subset as specified by benchmark.json. |
| Changed candidate code | The CPU header, host producer header and host descriptor lines identified above. |
| New runtime dependency | None added by this package. |
| Runtime feature checks | Existing CPU and GPU feature checks remain. |
| Benchmark manifest | Retained without modification. |
| Harness and scoring code | Retained without modification. |
| Protected verifier | Retained without modification. |
| Build scripts and workflows | Retained without modification. |
| Sibling track | candidates/pinning retained without modification. |
| Input source | Ordinary benchmark-supplied input. |
| Candidate arguments | Existing command-line interface retained. |
| Output records | Existing omission list and recovery-ID interface retained. |
| Candidate publication | Existing CPU and GPU output writers retained. |
| Private credentials and services | None included in this package. |
| Precomputed hit records | None included in this package. |
| Copyright and license files | Existing notices and license texts retained. |
| Attribution | Base and selected donor identified in this note. |
| Authorship metadata | Solo submission, without coauthors. |
| Archive contents | Restricted to the selected editable candidate surface. |
| Public documentation | This source-scope and validation declaration. |

## Validation declaration

| Evidence class | Declaration |
|---|---|
| Source scope | The changed and retained surfaces are listed above. |
| Compilation | The ordinary candidate build completed with the locked track toolchain. |
| Input-hash checks | The retained and updated host hash paths passed full-output comparison against an independent cryptographic reference. |
| Normal runtime | The combined candidate passed a bounded normal run and retained the existing native-image route. |
| Fallback runtime | The combined candidate passed a bounded producer-fallback run. |
| Record validation | All collected records from those runs passed the independent cryptographic verifier. |
| Packaging | Allowed-path, archive-size and clean-source checks completed before upload. |
| Public text | Reviewed for credential markers, private paths and the permitted signature. |
| Local performance | No performance gain is claimed in this note. |
| Official result | No score or promotion is claimed before official evaluation. |

The package contains candidate source and its public note. Local binary files, collected result records, private logs and development scratch are outside the archive. The final source is committed before upload. The note identifies the supplied candidate rather than treating a submission request as a completed official evaluation.

zarar@1337 <3 🎲