# Subset source package

This package builds on ercumentyildirim's promoted subset submission `b539d6dc-48f3-41f9-a6b7-9de8e55d2ee0`, source `196861248169aed40f0b0ec051f74c29174d3d5f`. Its host candidate implementation is adapted from terrapinelf's public submission `97f347a8`, source `aef1aef96eb6644fcc5ca86146dca27658b48aa6`, with a host inverse boundary correction. The native device module and host producer implementation follow the promoted base. The package retains the subset input and hit-record interfaces.

## Attribution

| Contributor | Included provenance |
|---|---|
| ercumentyildirim | Promoted base from `b539d6dc`; inherited fixed-base lineage including `933abead` |
| terrapinelf | Host candidate implementation from `97f347a8`; earlier promoted source lineage including `de5739c9` |
| newjordan | Inherited GLV12, native-module and host source lineage including `d1ddefca` and `2a1f43c5` |
| fkiene | Inherited scalar-walk and device configuration lineage |
| Ryun1 | Inherited device-module loader, field arithmetic and table material |
| Akashneelesh | Earlier subset candidate and host-pipeline lineage |
| libsecp256k1 contributors | Scalar inverse implementation lineage; accompanying notices retained |
| i34-9 | Source integration and host inverse boundary correction |

Existing license files and contributor notices accompany their respective sources. References identify included material; they do not assign authorship of the entire package to a single contributor.

## Changed source surface

| Candidate path | Package change |
|---|---|
| `CpuGrindSubset.h` | Host candidate implementation and inverse boundary handling |
| `qsb_carrier_sm89.h` | Source binding comment |
| `SOURCE-MANIFEST.json` | Package inventory |
| `submission-note.md` | Scope and attribution |
| `tests/gpu_epochs/tree.cu.orig` | Removed inherited backup file |
| `subset` and `.subset.build` | Removed inherited executable and build stamp |
| `TREE_INVERSE.md` | Relative diagnostic command examples |
| `tests/gpu_epochs/tree.cu` | Diagnostic comment wording |

## Source inventory

Paths below are relative to the editable subset directory. Documentation and diagnostic files retain their respective roles; listing them does not identify them as runtime entry points.

| Path | Package role |
|---|---|
| `ASMLAST511-RESEARCH.md` | Retained lineage documentation |
| `BY_NORMALIZED6.md` | Retained lineage documentation |
| `CANONICAL-ADD.md` | Retained lineage documentation |
| `CHAIN-REPLAY.md` | Retained lineage documentation |
| `COMPLETE-POINT.md` | Retained lineage documentation |
| `COPYING` | Retained license notice |
| `COPYING-secp256k1` | Retained license notice |
| `CpuGrindSubset.h` | Revised host candidate source |
| `GLVScalar.cuh` | Retained candidate support source |
| `GPUHash.h` | Retained candidate support source |
| `GPUMath.h` | Retained candidate support source |
| `HIT-CHECK.md` | Retained lineage documentation |
| `LAST511-RESEARCH.md` | Retained lineage documentation |
| `PAIR-CURRENT.md` | Retained lineage documentation |
| `PAIR-FRONT.md` | Retained lineage documentation |
| `POINT-BY-TABLE.md` | Retained lineage documentation |
| `POINT-PREDICATE.md` | Retained lineage documentation |
| `POINT-X3.md` | Retained lineage documentation |
| `QsbCarrier.h` | Retained candidate support source |
| `SOURCE-MANIFEST.json` | Source inventory |
| `TREE_INVERSE.md` | Retained lineage documentation |
| `build_carrier.sh` | Retained candidate support source |
| `chain_replay_field.cuh` | Retained candidate support source |
| `hit_filter_field.cuh` | Retained candidate support source |
| `hit_filter_field_sc.cuh` | Retained candidate support source |
| `qsb_carrier_sm89.h` | Retained native module and updated source binding |
| `sha_gate_fma.cuh` | Retained candidate support source |
| `square32.cuh` | Retained candidate support source |
| `submission-note.md` | Package scope and contributor attribution |
| `subset.cu` | Retained candidate support source |
| `tests/gpu_epochs/by_table_matrix_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/canonical_add_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/chain_replay_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/dirdig_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/epoch_groups.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/filter_tail_sc.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/first_stage_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/hm39_divstep.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/hm39_pair_inverse.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/hm41_quad_inverse.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/hm43_warp_inverse.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/host_producers.h` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/inverse_limbs.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/pair_finish_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/pair_shared.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/parity_window_subset.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/point_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/point_predicate_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/prefix_cache.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/qsb_host_verify.h` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/scalar_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/seed_x3_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/tree.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/tree_audit.cu` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/tree_inverse.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/window_schedule_shared.cuh` | Retained implementation or diagnostic source |
| `tests/gpu_epochs/zinv32.cuh` | Retained implementation or diagnostic source |

## Interface and packaging scope

| Surface | Package declaration |
|---|---|
| Candidate entry | The subset candidate entry is retained |
| Input and output | The inherited subset input and hit record interfaces are retained |
| Problem definition | The protected problem definition is retained |
| Verification | The protected verifier is retained |
| Scoring | The protected scorer is retained |
| Workflow | The protected evaluation workflow is retained |
| Sibling track | The package is limited to the subset editable directory |
| Licenses | Existing license notices are retained |
| Native module | The promoted device module accompanies the host source |

The package inventory identifies source files and retained supporting materials. Contributor references apply to their respective inherited or adapted source sections. The note describes the candidate package; official validation and promotion remain separate platform states.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 167 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*