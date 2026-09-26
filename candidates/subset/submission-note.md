# Subset source package

This package builds on terrapinelf's promoted subset submission `de5739c9-7c49-4f41-8d9c-cf5210ff88f6`, source `48fdd7e95a864ab74b178d1149c7ad80a56725b6`. It includes selected SHA source forms and revised host field-output, host table and host hash and host vector-copy representations, with corresponding source and native-module bindings.

## Attribution

| Contributor | Included provenance |
|---|---|
| terrapinelf | Promoted subset composition, host vector arithmetic and hash implementation |
| fkiene | Selected SHA source forms from public submission `73224391-0ba1-4510-9e32-ec79004d9c09` and inherited scalar-walk material |
| newjordan | Inherited GLV12 candidate and native-module lineage, including `d1ddefca`; host table representation provenance from public submission `e55d31ef-05a5-4a37-ab06-8a49f6371c61`, source `30cc6915b1be3e0d3208869df2956837d1cd4e13` |
| ercumentyildirim | Inherited fixed-base representation and point-table source, including `933abead`; host hash and vector-copy source from `889742ab-710f-433a-accb-4303af020824`, source `acdf549db2d32cfa7f58eac50e6dfa73140fee30` |
| Ryun1 | Inherited device-module loader, field arithmetic and table material |
| Akashneelesh | Earlier subset candidate and host-pipeline lineage |
| Meganpark980320 | Retained public source lineage referenced by the base |
| i34-9 | Source integration and revised host field-output operation |

Existing license files and contributor notices accompany their respective sources. References identify included material; they do not assign authorship of the entire package to a single contributor.

## Changed source surface

| Candidate path | Included change |
|---|---|
| `sha_gate_fma.cuh` | Selected SHA source form and its source selector |
| `CpuGrindSubset.h` | Host field-output operation and table representation |
| `tests/gpu_epochs/tree.cu` | Native configuration binding |
| `qsb_carrier_sm89.h` | Encoded native image and source binding |
| `SOURCE-MANIFEST.json` | Current package inventory |
| `submission-note.md` | Scope and attribution |
| `tests/gpu_epochs/tree.cu.orig` | Removed backup file |

## Source inventory

Paths below are relative to the editable subset directory. Documentation and diagnostic files retain their respective roles; listing them does not identify them as runtime entry points.

| Path | Package role |
|---|---|
| `ASMLAST511-RESEARCH.md` | Retained public documentation |
| `BY_NORMALIZED6.md` | Retained public documentation |
| `CANONICAL-ADD.md` | Retained public documentation |
| `CHAIN-REPLAY.md` | Retained public documentation |
| `COMPLETE-POINT.md` | Retained public documentation |
| `COPYING` | Retained license notice |
| `COPYING-secp256k1` | Retained license notice |
| `CpuGrindSubset.h` | Host candidate implementation with revised field-output operation and table representation |
| `GLVScalar.cuh` | Included candidate implementation support |
| `GPUHash.h` | Included candidate implementation support |
| `GPUMath.h` | Included candidate implementation support |
| `HIT-CHECK.md` | Retained public documentation |
| `LAST511-RESEARCH.md` | Retained public documentation |
| `PAIR-CURRENT.md` | Retained public documentation |
| `PAIR-FRONT.md` | Retained public documentation |
| `POINT-BY-TABLE.md` | Retained public documentation |
| `POINT-PREDICATE.md` | Retained public documentation |
| `POINT-X3.md` | Retained public documentation |
| `QsbCarrier.h` | Included candidate implementation support |
| `SOURCE-MANIFEST.json` | Current source and image inventory |
| `TREE_INVERSE.md` | Retained public documentation |
| `build_carrier.sh` | Included candidate implementation support |
| `chain_replay_field.cuh` | Included candidate implementation support |
| `hit_filter_field.cuh` | Included candidate implementation support |
| `hit_filter_field_sc.cuh` | Included candidate implementation support |
| `qsb_carrier_sm89.h` | Encoded device module with source binding |
| `sha_gate_fma.cuh` | Selected SHA source forms |
| `square32.cuh` | Included candidate implementation support |
| `submission-note.md` | Current public package scope and attribution |
| `subset.cu` | Included candidate implementation support |
| `tests/gpu_epochs/by_table_matrix_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/canonical_add_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/chain_replay_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/dirdig_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/epoch_groups.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/filter_tail_sc.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/first_stage_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/hm39_divstep.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/hm39_pair_inverse.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/hm41_quad_inverse.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/hm43_warp_inverse.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/inverse_limbs.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/pair_finish_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/pair_shared.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/parity_window_subset.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/point_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/point_predicate_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/prefix_cache.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/qsb_host_verify.h` | Included candidate implementation support |
| `tests/gpu_epochs/scalar_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/seed_x3_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/tree.cu` | Included candidate implementation support |
| `tests/gpu_epochs/tree_audit.cu` | Retained diagnostic source |
| `tests/gpu_epochs/tree_inverse.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/window_schedule_shared.cuh` | Included candidate implementation support |
| `tests/gpu_epochs/zinv32.cuh` | Included candidate implementation support |

## Package interfaces

| Interface | Included form |
|---|---|
| Track | Subset |
| Editable directory | `candidates/subset` |
| Translation unit | `subset.cu` |
| Build entry | Existing benchmark compiler entry |
| Input | Runtime problem through the existing executable interface |
| Arguments | Existing executable argument order |
| GPU records | Existing subset-index and recovery-identifier representation |
| CPU records | Existing collected host result-file interface |
| Candidate publication | Existing host hit-checking interface |
| Verification | Existing independent benchmark verification interface |
| Scoring | Existing benchmark scoring interface |
| Native module | Encoded module accompanied by source and regeneration support |
| Host dispatch | Retained base dispatch and worker policy |
| Licensing | Accompanying license files and source notices |

The archive contains candidate sources, supporting headers, the encoded module, the source inventory and public documents. The source selectors accompany the corresponding implementation. This note records package scope and provenance; official evaluation determines its competitive result.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 164 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*