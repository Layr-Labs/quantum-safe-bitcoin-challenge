# Subset — current host and device integration update

This candidate combines a scoped host-producer update with an existing gate-hash option on the current promoted Subset implementation. It retains the promoted device layout, CPU co-grinder and candidate ownership policy. The corresponding native device archive accompanies the selected gate-hash option. The changes are confined to the editable Subset candidate surface.

The source, native archive and combined local output checks described below have been completed. Final package review is complete. No official result or promotion is claimed for this candidate. The submitted entry is identified by the commit and archive recorded by the competition service.

Effort: xhigh; independent advisor review: high.

## Entry scope

| Surface | Contents of this candidate |
| --- | --- |
| Candidate entry point | Existing Subset translation unit with the selected inherited gate-hash option |
| Device layout | Layout selected by the immediate promoted foundation |
| Device integration | Existing arithmetic, kernel interfaces and candidate ownership |
| Native image | Regenerated device archive corresponding to this entry's source selection |
| Host producer | Scoped update to the existing producer's handling of its ordinary output buffers |
| Host cleanup | Matching cleanup for the updated producer buffers |
| CPU co-grinder | Current promoted implementation and configuration |
| Candidate records | Existing tuple and recovery-identifier representation |
| Publication | Existing exact candidate checking and publication interface |
| Supporting files | Inherited arithmetic, hash, loader, build and license support |
| Package hygiene | Omission of an unused inherited source backup |
| Documentation | This entry's scope, provenance and validation declaration |

The selected gate-hash option is already present in the inherited source. This entry enables that option on the current foundation and supplies its corresponding native archive. The selection is an integer-arithmetic implementation choice; the candidate continues to compute the prescribed hash function. It does not change the public problem definition or acceptance predicate.

The host change concerns the existing producer path and its buffer handling. The candidate retains the producer's first-batch comparison, ordinary activation checks, completion handling and fallback interfaces. The cleanup adjustment accompanies that same buffer update. The producer's output used by the selected computation remains represented by the existing interface.

The CPU candidate implementation is retained from the immediate promoted package. Its scalar and vector implementations, table-selection policy, candidate ranges and publication connection remain those of that package. No new CPU table policy, worker-placement policy or adaptive selector is introduced by this entry.

## Foundation and attribution

The immediate foundation is **cefika**'s promoted Subset package, including its CPU candidate traversal update. The preceding promoted host and device foundation by **kshitij-hash** is retained. This candidate keeps the current foundation while integrating the scoped producer update and selecting the inherited gate-hash option.

**terrapinelf** is credited for the inherited gate-hash option carried by the source lineage. **newjordan** is credited for the native-carrier foundation carried by that lineage. **jacklightChen** is credited for its incorporation into the preceding promoted Subset composition. These acknowledgments describe the source lineage rather than an import of an earlier package as a whole.

**RealAdii**, **ercumentyildirim**, **cefika**, **fkiene** and **i34-9** retain credit for host, producer, arithmetic, publication and integration work inherited by this candidate. Earlier source acknowledgments naming **HyeokxC**, **Ryun1**, **newjordan**, **Akashneelesh** and **Meganpark980320** remain attached to their respective contributions. The names in this note record provenance; they do not imply that those contributors prepared, reviewed or endorsed this entry.

Entry-specific integration, the scoped producer-buffer update and packaging are prepared by **i34-9**, the **zarar@1337** account. Contributor names are acknowledged in this note. Existing copyright statements, license texts and source notices remain attached to their respective files.

## Source inventory and interface roles

| File or group | Role in the supplied candidate |
| --- | --- |
| `subset.cu` | Existing candidate entry point and selected gate-hash option |
| `tests/gpu_epochs/tree.cu` | Current host/device integration and matching host cleanup |
| `tests/gpu_epochs/host_producers_v3.h` | Existing host producer with the scoped buffer update |
| `qsb_carrier_sm89.h` | Native device archive corresponding to this candidate |
| `QsbCarrier.h` | Retained native loading and dispatch support |
| `CpuGrindSubset.h` | Retained CPU candidate implementation |
| `tests/gpu_epochs/pair_shared.cuh` | Retained paired device arithmetic and gate integration |
| `tests/gpu_epochs/qsb_host_verify.h` | Retained exact checking and publication support |
| `GPUHash.h` and `sha_gate_fma.cuh` | Inherited hash implementations and selected gate support |
| `GLVScalar.cuh` | Retained scalar support |
| Field, point and inverse headers | Retained arithmetic support |
| `build_carrier.sh` | Existing native-image build support |
| `COPYING` and `COPYING-secp256k1` | Retained license texts |
| `submission-note.md` | Entry-specific public scope and attribution note |

These paths describe source files within the candidate package. Supporting research notes and audit sources retain their original context. Their inclusion does not declare an additional active mechanism, an additional completed experiment or an official result for this composition. Historical settings or results in inherited documents must be interpreted in the context of their original source versions.

Package hygiene includes omission of the unused inherited `tests/gpu_epochs/tree.cu.orig` backup. The active integration file remains `tests/gpu_epochs/tree.cu`.

The candidate surface contains the existing Subset implementation and its supporting files. Protected benchmark drivers, the verifier, problem definitions, workflow and scoring configuration are outside the changes described here. The sibling candidate directory is outside this entry's editable surface.

## Completed validation declarations

| Review item | Status |
| --- | --- |
| Source scope | Checked against the immediate promoted foundation |
| Arithmetic premise | Existing integer gate option reviewed for the selected source path |
| Producer compatibility | Scoped producer update and selected gate option reviewed together |
| Native archive | Payload identity and recorded source fingerprint checked |
| Build configuration | Native and host settings checked for agreement |
| Native execution | Confirmed in the completed combined local runs |
| Combined output check | Complete output records matched protected verified reference records; no failures reported |
| Final package review | Release source, archive contents, required scope and note checked |
| Official correctness and score | No result is claimed for this candidate |
| Promotion | No promotion is claimed |

The completed declarations apply to the reviewed candidate source and its corresponding native archive. The output declaration records agreement with protected verified full records. It does not claim a new independent cryptographic verification of every matching record. Source, build and local validation receipts are retained with the preparation record.

Final package review covers the release source, supplied native archive, editable-file inventory, inherited notices and this entry's note. That review is complete for the supplied package. The official service provides the authoritative correctness outcome, score and promotion status for any submitted entry.

## Packaging and licensing

The package retains the repository-defined problem representation, candidate indexing, recovery identifiers and record format. It includes the native archive appropriate to the selected source and preserves the existing loader and fallback support. Changes in candidate implementation do not alter the competition's authoritative scoring or validation rules.

The GPU arithmetic lineage includes VanitySearch-derived material under its inherited notices. The host field and scalar lineage includes libsecp256k1-derived material under its inherited notices. Existing license texts and copyright statements remain supplied with the source. The integration changes, producer update, native archive and this note accompany those notices in the editable candidate surface.

---

zarar@1337 <3 🎲