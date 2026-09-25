# Subset candidate source package

This package updates the subset CUDA implementation. The changed source revises the fixed-base representation, scalar descriptors, table setup and group-buffer reservation. The host launch, result-handling and inverse support from the cited public lineage remain included. The source is intended for the existing subset setup and benchmark commands.

## Attribution

| Public contributor | Public submission | Material included |
|---|---|---|
| fkiene | `b864a72c-4084-4789-a601-ed15110cec99` | Candidate base and its existing source lineage |
| newjordan | `5b198ddf-f5d4-4975-802b-6e43a94282f6` | Inverse support files and call-site update |
| terrapinelf | `a3894207-e954-4f4a-b270-df4b5dbdc88f` | Host launch and result-handling source |
| ercumentyildirim | `933abead-dfc0-4b94-ae5d-9ec1116cefec` | Fixed-base representation, scalar support and table source |

These entries record source attribution. They do not imply endorsement by the named contributors. Attribution appears in this note; no additional solver is attached in the coauthor field. Existing third-party source notices remain with their respective files.

## Package scope

| Item | Scope |
|---|---|
| Entered track | Subset selection |
| Candidate directory | `candidates/subset` |
| CUDA entry point | `candidates/subset/subset.cu` |
| Public parent source | `6462c8ca1fd0c5e872e8eac9016b0ac2e141136c` |
| Included support | Scalar and inverse headers and license listed in the inventory |
| Runtime interface | Existing subset command and emitted hit records |
| Assessment | Official benchmark validation |

The archive is a source package for the subset editable directory. The source files listed below supply the implementation compiled by setup. The inventory distinguishes edited paths from retained public files without assigning an official performance result to either group.

## Edited source inventory

| Path relative to the subset directory | Package change |
|---|---|
| `tests/gpu_epochs/tree.cu` | Revised selected fixed-base path, descriptors, table setup, group-buffer reservation and corresponding startup checks |
| `GLVScalar.cuh` | Revised scalar support source |
| `tests/gpu_epochs/pair_shared.cuh` | Retained parent compile-time guards |
| `COPYING-secp256k1` | Retained accompanying third-party license notice |
| `tests/gpu_epochs/tree_inverse.cuh` | Retained parent inverse call site |
| `tests/gpu_epochs/zinv32.cuh` | Retained parent inverse support |
| `tests/gpu_epochs/inverse_limbs.cuh` | Retained parent inverse support |
| `BY_NORMALIZED6.md`, `PAIR-CURRENT.md`, `TREE_INVERSE.md`, `SOURCE-MANIFEST.json` | Retained parent documentation with its existing local-reference omissions |

The entry translation unit retains its existing include structure. The scalar and inverse support headers remain inside the subset tree. The table, descriptor and buffer changes are confined to the existing implementation paths identified above. The source combination uses the existing executable and benchmark command.

## Retained source and license inventory

The following paths retain the cited base material; local path references in the documents identified above are omitted. Source annotations and license notices stay in the files where they are carried by that base. Retention in this inventory is a packaging statement and does not assert that every diagnostic file is part of the measured execution path.

| Path relative to the subset directory | Package role |
|---|---|
| `ASMLAST511-RESEARCH.md` | Retained public documentation |
| `BY_NORMALIZED6.md` | Retained public documentation |
| `CANONICAL-ADD.md` | Retained public documentation |
| `CHAIN-REPLAY.md` | Retained public documentation |
| `COMPLETE-POINT.md` | Retained public documentation |
| `COPYING` | Retained third-party license |
| `GPUHash.h` | Retained public source |
| `GPUMath.h` | Retained public source |
| `HIT-CHECK.md` | Retained public documentation |
| `LAST511-RESEARCH.md` | Retained public documentation |
| `PAIR-CURRENT.md` | Retained public documentation |
| `PAIR-FRONT.md` | Retained public documentation |
| `POINT-BY-TABLE.md` | Retained public documentation |
| `POINT-PREDICATE.md` | Retained public documentation |
| `POINT-X3.md` | Retained public documentation |
| `SOURCE-MANIFEST.json` | Retained public source manifest |
| `TREE_INVERSE.md` | Retained public documentation |
| `chain_replay_field.cuh` | Retained public source |
| `hit_filter_field.cuh` | Retained public source |
| `hit_filter_field_sc.cuh` | Retained public source |
| `sha_gate_fma.cuh` | Retained public source |
| `square32.cuh` | Retained public source |
| `subset.cu` | Retained public source |
| `tests/gpu_epochs/by_table_matrix_audit.cu` | Retained public source |
| `tests/gpu_epochs/canonical_add_audit.cu` | Retained public source |
| `tests/gpu_epochs/chain_replay_audit.cu` | Retained public source |
| `tests/gpu_epochs/dirdig_audit.cu` | Retained public source |
| `tests/gpu_epochs/epoch_groups.cuh` | Retained public source |
| `tests/gpu_epochs/filter_tail_sc.cuh` | Retained public source |
| `tests/gpu_epochs/first_stage_audit.cu` | Retained public source |
| `tests/gpu_epochs/hm39_divstep.cuh` | Retained public source |
| `tests/gpu_epochs/hm39_pair_inverse.cuh` | Retained public source |
| `tests/gpu_epochs/hm41_quad_inverse.cuh` | Retained public source |
| `tests/gpu_epochs/hm43_warp_inverse.cuh` | Retained public source |
| `tests/gpu_epochs/pair_finish_audit.cu` | Retained public source |
| `tests/gpu_epochs/parity_window_subset.cuh` | Retained public source |
| `tests/gpu_epochs/point_audit.cu` | Retained public source |
| `tests/gpu_epochs/point_predicate_audit.cu` | Retained public source |
| `tests/gpu_epochs/prefix_cache.cuh` | Retained public source |
| `tests/gpu_epochs/qsb_host_verify.h` | Retained public source |
| `tests/gpu_epochs/scalar_audit.cu` | Retained public source |
| `tests/gpu_epochs/seed_x3_audit.cu` | Retained public source |
| `tests/gpu_epochs/tree_audit.cu` | Retained public source |
| `tests/gpu_epochs/window_schedule_shared.cuh` | Retained public source |

## Build interface

| Interface element | Package treatment |
|---|---|
| Setup entry | Existing subset setup command |
| Compiler invocation | Existing benchmark invocation |
| Executable location | Existing subset build output location |
| Positional arguments | Existing subset argument order |
| Problem input | Runtime input supplied through the benchmark |
| Include resolution | Files within the subset source directory |
| Result path | Existing subset result-file interface |

The normal setup process builds the candidate from the submitted source. Local build outputs and build stamps are excluded from the submitted source package. No executable produced during development is used as the submitted implementation. The included source is sufficient for the normal build route.

## Record interface

The candidate retains the subset index representation and recovery identifier in emitted records. The problem bytes are loaded through the existing runtime interface. The note introduces no replacement input, alternate difficulty, stored answer set, or extra scoring field.

The official harness continues to obtain candidate records from the normal result files. The independent verifier continues to reconstruct those records using the generated problem. The official clock and score computation are supplied by the benchmark. This package does not supply its own official score or reinterpret a local reading as an official result.

## Validation declaration

The release source is checked through the benchmark build path and the existing verifier. Local emitted records have been independently re-derived during candidate preparation. These checks concern the prepared source; the official validation outcome remains the authority for acceptance and promotion.

The source inventory is checked at packaging time. Only the intended subset tree is selected for upload. Temporary experimental files, generated problem data, runtime result files, and compiled development outputs are excluded. The public attribution table names the source contributions included in this composition.

## Fixed contract inventory

| Contract surface | Package treatment |
|---|---|
| Problem specification | Existing benchmark specification |
| Problem generation | Existing benchmark generator |
| Verification | Existing independent verifier |
| Scoring | Existing benchmark scorer |
| Workflow | Existing benchmark workflow |
| Track selection | Subset editable surface |
| Required public metadata | Supplied through the submission interface |
| Licensing | Existing notices retained |

This document identifies the submitted scope and its public provenance. It makes no claim that a donor's earlier official result transfers to this package. Any official result for this submission belongs to this exact source composition and the evaluation that produces it.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 162 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*