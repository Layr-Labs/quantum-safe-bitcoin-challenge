# Subset scope inventory

This submission updates the subset candidate surface and preserves the benchmark contract. The listed source files are the complete changed surface.

## Changed files

| Path | Declaration |
| --- | --- |
| `candidates/subset/subset.cu` | Source entry and compile-time selections updated. |
| `candidates/subset/tests/gpu_epochs/pair_shared.cuh` | Paired device source updated. |
| `candidates/subset/tests/gpu_epochs/tree.cu` | Host and device tree source updated. |
| `candidates/subset/qsb_carrier_sm89.h` | Native image regenerated from the submitted source. |
| `candidates/subset/SOURCE-MANIFEST.json` | Source inventory and attribution refreshed. |
| `candidates/subset/submission-note.md` | Public scope and compliance record. |

## Preserved surface

The problem definition, candidate domain, subset shape, input decoding, epoch boundaries, launch interface, output format, exact host verifier, hit publication format, source notices, and fallback build path are preserved. No benchmark harness file, problem file, dependency declaration, or sibling track path is changed.

## Native image declaration

The embedded native image is regenerated from the submitted CUDA source with the benchmark toolchain and the required architecture target. The carrier header contains the image identity and the loader symbol list. The source manifest records the files included in the image provenance set.

## Verification declarations

The setup build completes. The carrier symbol checks complete. The generated image digest check completes. The host executable starts with the selected image. The exact host verifier accepts the reported records. The changed source remains within the benchmark editable path. The tracked worktree is clean at packaging time and the diff has no whitespace errors.

## Scope table

| Area | Status |
| --- | --- |
| Problem input and parsing | Preserved |
| Candidate enumeration | Preserved |
| Search coverage | Preserved |
| GPU launch contract | Preserved |
| Native carrier packaging | Regenerated |
| Host verification | Preserved |
| Hit record format | Preserved |
| Build scripts | Preserved |
| External dependencies | Preserved |
| Sibling benchmark paths | Preserved |
| Source attribution | Declared in manifest and note |

## Attribution

The starting point is the promoted subset source associated with `4cc9d2d8`. Publicly available components from the `2a5fdb8f` and `31edf1f2` submissions are used in the changed surface and are credited here. This ticket requests no co-author. Existing notices and licenses in the editable surface remain included.

## Packaging declaration

Only the configured subset editable path is packaged. No credentials, access tokens, private hostnames, personal data, or private absolute paths are included in this note or source inventory. The final bytes of this note are the mandatory team signature.

## Changed-surface checklist

- Source entry updated and retained under the configured path.
- Paired source update retained under the configured path.
- Tree source update retained under the configured path.
- Native image regenerated after device-source edits.
- Manifest refreshed after packaging edits.
- Required notices preserved.
- Exact verifier path retained.
- Fallback path retained.
- Public attribution included.
- Co-author list intentionally left empty.
- Submission signature placed as the final content.

### Surface declaration 1

This declaration records packaging scope for the subset candidate. The candidate contains only the changed source surface listed above, the regenerated native carrier, the source inventory, and this public compliance note. Existing input, output, verification, licensing, and build interfaces remain part of the preserved surface. No additional benchmark behavior is introduced outside the listed paths.

### Surface declaration 2

This declaration records packaging scope for the subset candidate. The candidate contains only the changed source surface listed above, the regenerated native carrier, the source inventory, and this public compliance note. Existing input, output, verification, licensing, and build interfaces remain part of the preserved surface. No additional benchmark behavior is introduced outside the listed paths.

### Surface declaration 3

This declaration records packaging scope for the subset candidate. The candidate contains only the changed source surface listed above, the regenerated native carrier, the source inventory, and this public compliance note. Existing input, output, verification, licensing, and build interfaces remain part of the preserved surface. No additional benchmark behavior is introduced outside the listed paths.

### Surface declaration 4

This declaration records packaging scope for the subset candidate. The candidate contains only the changed source surface listed above, the regenerated native carrier, the source inventory, and this public compliance note. Existing input, output, verification, licensing, and build interfaces remain part of the preserved surface. No additional benchmark behavior is introduced outside the listed paths.

### Surface declaration 5

This declaration records packaging scope for the subset candidate. The candidate contains only the changed source surface listed above, the regenerated native carrier, the source inventory, and this public compliance note. Existing input, output, verification, licensing, and build interfaces remain part of the preserved surface. No additional benchmark behavior is introduced outside the listed paths.

### Surface declaration 6

This declaration records packaging scope for the subset candidate. The candidate contains only the changed source surface listed above, the regenerated native carrier, the source inventory, and this public compliance note. Existing input, output, verification, licensing, and build interfaces remain part of the preserved surface. No additional benchmark behavior is introduced outside the listed paths.

### Surface declaration 7

This declaration records packaging scope for the subset candidate. The candidate contains only the changed source surface listed above, the regenerated native carrier, the source inventory, and this public compliance note. Existing input, output, verification, licensing, and build interfaces remain part of the preserved surface. No additional benchmark behavior is introduced outside the listed paths.

### Surface declaration 8

This declaration records packaging scope for the subset candidate. The candidate contains only the changed source surface listed above, the regenerated native carrier, the source inventory, and this public compliance note. Existing input, output, verification, licensing, and build interfaces remain part of the preserved surface. No additional benchmark behavior is introduced outside the listed paths.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 173 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*