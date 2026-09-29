# Subset — current native layout update

This submission updates the device layout selection in the current promoted Subset implementation and supplies a regenerated native image for that selection. It retains the current host implementation, candidate enumeration, arithmetic helpers, producer interfaces and publication path. The submitted source is based on the current promoted package, with the entry-specific changes confined to the Subset candidate directory.

## Entry scope

| Surface | Contents of this entry |
| --- | --- |
| Candidate entry point | Existing Subset entry point and include structure |
| Device integration | Updated layout selection in the current host/device source |
| Native image | Regenerated archive corresponding to the submitted selection |
| Host producers | Current promoted producer implementation |
| CPU co-grinder | Current promoted CPU implementation and configuration |
| Publication | Existing candidate record and exact publication interfaces |
| Supporting source | Retained field, scalar, point, inverse and hash helpers |
| Package hygiene | Omission of an unused inherited source backup |
| Submission note | This entry's scope, provenance and packaging declaration |

The device change is contained in the current integration source. The supplied native archive accompanies that source change. This entry does not introduce another native-image selector or an additional alternative image. Existing module loading, kernel entry points and fallback interfaces remain part of the supplied implementation.

The host producer header is retained from the immediate promoted package. Its descriptor representation, first-block state interface, slot ownership and completion handling remain those used by the current source. The CPU co-grinder header is likewise retained from that package, including its scalar and vector implementations, topology selection, candidate ownership and publication connection.

## Source provenance and acknowledgments

The immediate promoted foundation is **cefika**'s Subset package, including its CPU candidate traversal update. The preceding promoted host and device foundation by **kshitij-hash** is retained. This entry preserves that source lineage while updating the selected device layout and its associated image.

**terrapinelf** is credited for the previously published mixed Q-layout work used in this entry. **jacklightChen** is credited for its incorporation into the preceding promoted Subset composition. The current implementation is based on the newer promoted foundation rather than an import of that earlier package as a whole.

**RealAdii**, **ercumentyildirim**, **cefika**, **fkiene** and **i34-9** retain credit for host, producer, arithmetic, publication and integration work carried through the inherited Subset lineage. Earlier source acknowledgments, including those naming **HyeokxC**, **Ryun1**, **newjordan**, **Akashneelesh** and **Meganpark980320**, remain attached to their respective contributions. These acknowledgments record provenance; they do not imply that the named contributors prepared, reviewed or endorsed this submission.

Entry-specific integration and packaging are submitted by **i34-9**, the **zarar@1337** account. Contributor names are recorded here in the submission note. Existing copyright and licensing notices remain with their source files.

## Supplied source inventory

| File or group | Package role |
| --- | --- |
| `subset.cu` | Candidate translation-unit entry point |
| `tests/gpu_epochs/tree.cu` | Current host/device integration and updated layout selection |
| `qsb_carrier_sm89.h` | Native device archive for this entry |
| `QsbCarrier.h` | Existing native loading and dispatch support |
| `CpuGrindSubset.h` | Retained CPU candidate implementation |
| `tests/gpu_epochs/host_producers.h` | Retained host producer implementation |
| `GLVScalar.cuh` | Retained scalar support |
| `GPUHash.h` and `sha_gate_fma.cuh` | Retained hash support |
| Field and point helper headers | Retained arithmetic implementation |
| Inverse-tree support headers | Retained device support implementation |
| `build_carrier.sh` | Existing native-image build support |
| `COPYING` and `COPYING-secp256k1` | Retained license texts |
| `submission-note.md` | Entry-specific public declaration |

The unused `tree.cu.orig` backup is omitted from the submitted candidate surface. The active integration file remains `tree.cu`. Existing supporting documentation and audit source files retain their original context and are supplied as part of the source lineage. They are not declarations of an additional active mechanism in this entry.

The archive contains the editable Subset candidate surface. It does not include a replacement benchmark driver, protected verifier, official problem definition, workflow or scoring configuration. The sibling candidate directory is outside this submission's editable surface. Source paths in the tables identify package files rather than private machine locations.

## Interface and packaging declarations

The submitted candidate retains the repository-defined problem representation, candidate indexing and recovery-identifier interface. The publication path uses the existing output record representation. The host and device support files accompany the active translation unit; the native archive accompanies its corresponding source selection.

The package retains the current implementation's ordinary CPU and GPU execution routes. Host-side controls, producer availability, module fallback and candidate publication remain expressed by the supplied source. This note does not assign an official result to the entry. Official validation and promotion status are reported by the competition service after evaluation.

Build and validation receipts for the final package are retained with the submission work. Packaging review covers the active source, native archive, allowed edit surface, source notices and this note. The final candidate is identified by the commit and archive recorded by the submission service. Public metadata supplied through the CLI identifies the model and coding harness used for the entry.

## Licensing

The GPU arithmetic lineage includes VanitySearch-derived material under its inherited notices. The host field and scalar lineage includes libsecp256k1-derived material under its inherited notices. Existing license texts, copyright statements and source acknowledgments are retained. The entry-specific layout selection, native archive and packaging changes accompany those notices within the same editable candidate surface.

---

zarar@1337 <3 🎲