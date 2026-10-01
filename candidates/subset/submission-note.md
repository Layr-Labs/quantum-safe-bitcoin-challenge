# SUBSET: root inverse source and native image update

This is an exploratory SUBSET package. It retains the current CPU and host producer source and selects an updated GPU root inverse implementation, arithmetic helpers and inverse lookup source. The corresponding source defaults and native-image fingerprint are part of the package. The ordinary candidate entry point and output interfaces are retained.

The selected package consists of candidate source, its matching native image and this scope declaration. The source inventory below identifies the additions and retained surfaces. The release declarations distinguish candidate validation from the official evaluation status.

## Base and attribution

The promoted base is cefika's submission fb6f5a8f-b29e-4506-a50c-c79a9c5a2a0e, source [ff27a2b66990a3eb554a1d4453e896c0397337ba](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/ff27a2b66990a3eb554a1d4453e896c0397337ba). The entry wrapper, paired arithmetic, GPU hash helpers, host interfaces and existing source notices are retained from that lineage. The native image supplied with this package corresponds to the newly selected inverse source.

The retained CPU arithmetic-boundary edits are attributed to jacklightChen's public submission b1c5e58e-210d-4353-addf-edc4474b8f33, immutable [CPU donor source843c82817bf2de27e7794bb6317d1d223d6fa1fd](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/843c82817bf2de27e7794bb6317d1d223d6fa1fd). Only the previously selected host edits from that source are retained. The current CPU key-message preparation, worker-local input scratch and host descriptor edits are also retained in this package.

The root64 header, associated arithmetic helpers and inverse lookup source are zarar@1337 additions. No new external donor is claimed for these inverse additions. Attribution to cefika and jacklightChen identifies the retained base and host source; it does not attribute the new inverse files to either donor.

The base promotion records acknowledgments to kshitij-hash, terrapinelf, i34-9, ercumentyildirim, HyeokxC, jacklightChen, fkiene, kaankolcu, newjordan and Meganpark980320. Those names are retained here as base provenance. The submission owner is i34-9. This is a solo submission. Existing VanitySearch and third-party notices remain in the candidate sources.

## Updated candidate surfaces

| File | Package contents |
|---|---|
| tests/gpu_epochs/inverse_limbs.cuh | The root inverse selection, source default and retained original inverse interface. |
| tests/gpu_epochs/root64.cuh | The selected root inverse source and helper interface. |
| tests/gpu_epochs/root64_products.cuh | Arithmetic helper declarations used by the selected root header. |
| tests/gpu_epochs/zinv32.cuh | The selected inverse lookup declarations. The original lookup declarations remain present. |
| tests/gpu_epochs/tree.cu | The native-image fingerprint entries corresponding to the selected inverse defaults. Existing host descriptor lines are retained. |
| qsb_carrier_sm89.h | The packaged native image and metadata matching the selected candidate source. |
| submission-note.md | This source-scope, attribution and release declaration. |

The candidate preserves the existing inverse caller and output interface. The selected defaults are included in the matching native image. Existing carrier and fallback interfaces remain present.

## Retained CPU and host surfaces

| Surface | Package declaration |
|---|---|
| CpuGrindSubset.h | Retained current CPU arithmetic-boundary, key-message preparation and worker-local input scratch source. |
| host_producers_v3.h | Retained current descriptor source and selected default. |
| host_producers_v1.h | Retained alternative host producer source. |
| host_producers.h | Retained producer interface and feature checks. |
| qsb_host_verify.h | Retained candidate-side record verification source. |
| CPU worker input and digest output | Existing planned route and digest-word interface retained. |
| CPU table and scalar recovery | Existing declarations and runtime selection retained. |
| Producer fallback | Existing fallback route remains available. |
| Stream ownership and collection | Existing launch-slot ownership, record collection and drain interfaces retained. |
| Candidate publication | Existing CPU and GPU output writers retained. |

These retained host surfaces include the selected input scratch and descriptor edits from the current package. The GPU inverse additions do not replace the host input domain, omission records, recovery-ID interface or key-message layout. The existing CPU feature checks, producer selection and startup self-check remain present.

## Retained device and library surfaces

| Surface | Package declaration |
|---|---|
| subset.cu | Retained entry wrapper and candidate arguments. |
| QsbCarrier.h | Retained carrier loader, selection checks and fallback interface. |
| build_carrier.sh | Retained native-image build utility. |
| GPUHash.h | Retained GPU hash source and existing notices. |
| GPUMath.h | Retained general GPU field arithmetic source. |
| GLVScalar.cuh | Retained scalar helper source. |
| sha_gate_fma.cuh | Retained GPU hash helper and selected addition setting. |
| square32.cuh | Retained general field square source. |
| pair_shared.cuh | Retained paired arithmetic and recovery interface. |
| tree_inverse.cuh | Retained tree organization and inverse caller. |
| filter_tail_sc.cuh | Retained device filtering source. |
| window_schedule_shared.cuh | Retained GPU input hash scheduling source. |
| epoch_groups.cuh | Retained epoch grouping declarations. |

The candidate retains the ordinary GPU geometry, paired candidate work, field and hash consumers, public-key output layout and record format. The inverse headers, selected defaults, fingerprint entries and matching native image are the updated device surfaces identified above. Existing copyright and license texts remain with their source files.

## Scope and packaging declarations

| Item | Package declaration |
|---|---|
| Selected competition track | SUBSET. |
| Editable surface | candidates/subset as specified by benchmark.json. |
| Changed candidate code | The inverse headers, source defaults and fingerprint entries listed above. |
| Native artifact | Candidate image matching the selected inverse configuration. |
| Runtime dependencies | Existing candidate dependencies retained. |
| Runtime feature checks | Existing CPU and GPU feature checks retained. |
| Benchmark manifest | Retained without modification. |
| Protected benchmark and scoring code | Retained without modification. |
| Protected verifier | Retained without modification. |
| Repository workflows | Retained without modification. |
| Sibling track | candidates/pinning retained without modification. |
| Input source | Ordinary benchmark-supplied input. |
| Candidate arguments | Existing command-line interface retained. |
| Output records | Existing omission-list and recovery-ID interface retained. |
| Archive contents | Restricted to the selected editable candidate surface. |
| Source provenance | Base, selected host donor and owned inverse additions identified above. |
| Authorship metadata | Solo submission. |
| Public documentation | This candidate scope and release declaration. |

The public package contains candidate source and the candidate's packaged native image. Collected hit records, local executable files, private logs, credentials, service configuration and development scratch are outside the archive. The final archived source and public note are subject to the ordinary allowed-path and packaging checks.

## Release declarations

| Release item | Declaration |
|---|---|
| Source scope | The archived changed and retained surfaces match this inventory. |
| Ordinary compilation | The candidate builds with its selected defaults under the locked track toolchain. |
| Native image | The packaged image, build fingerprint, executable body and caller interface match the archived source. |
| Inverse outputs | The selected inverse and lookup composition satisfies the independent full-output checks. |
| Native runtime | The combined ordinary candidate passes the normal native route and record checks. |
| Fallback runtime | The combined candidate passes the normal fallback route and record checks. |
| Record validation | All collected release-gate records satisfy the independent cryptographic verifier. |
| Packaging | Allowed-path, archive-size, clean-source and public-text checks are complete. |
| Public text | Attribution, scope and the selected solo signature are present. |
| Official result | Official evaluation remains the authority for a score or promotion. |

The declarations above apply to the supplied source and collected release records. The official evaluation determines the score and promotion status. This note records package scope and provenance; it does not report an official result.

zarar@1337 <3 🎲