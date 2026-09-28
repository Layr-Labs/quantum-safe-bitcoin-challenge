# Subset — device hash and scalar-coefficient implementation

This entry starts from the promoted Subset composition of jacklightChen, submission `5c7e36c5`, at source `6343a38d3dde830b079cb95b0e2e99c7f9a812e9`. It updates selected device hash operations, the guarded scalar-coefficient helper, and the corresponding native image. The promoted Q_MIX2 configuration, host producers and co-grinder composition remain the package foundation.

The CPU co-grinder retains its original 158-pattern candidate family. Its separate correctness repair handles a zero product in the lane-inversion helper through the existing independent-lane inversion path. Apart from this boundary repair, the CPU implementation and its configuration are unchanged from the promoted base. The host producer header is retained unchanged.

## Source provenance

| Role | Public source |
| --- | --- |
| Immediate promoted composition | jacklightChen, `5c7e36c5` |
| Immediate source commit | `6343a38d3dde830b079cb95b0e2e99c7f9a812e9` |
| Earlier promoted device foundation | RealAdii, `521075fe` |
| Earlier promoted source | `46b24ebaa033fb69c7335794b54fd6a156359ec8` |
| Host composition carried by the base | i34-9, `4da17ebc` |
| Host composition source | `f391f74dc765352be557efb044f326f0e2190c45` |
| Published Q_MIX2 device source | terrapinelf, `3e6069ee` |
| Q_MIX2 source commit | `0a38640ed29390ecea2c43fe260bef6ab9b61a3e` |
| Co-grinder package lineage | cefika, `bf001729` |
| Host producer package lineage | ercumentyildirim, `a141df2b` |
| Public-key hash head source | fkiene, `8c07297bb79a8340632b1101e5704ac1294f2b13` |

**jacklightChen** is credited for the immediate promoted host/device composition. **terrapinelf** is credited for the published Q_MIX2 device variant and the underlying co-grinder and producer work carried through the inherited packages. **RealAdii** is credited for the earlier promoted source and its completed-record publication integration. **cefika** and **ercumentyildirim** retain credit for the respective co-grinder and producer packages in the host lineage.

**kshitij-hash** and **fkiene** are credited for the inherited device chain, table-layout and promoted Subset foundation. **fkiene** is also credited for the public-key hash head source carried into the device integration. **i34-9** is credited for the retained host composition and the entry-specific integration of the selected device helpers and CPU boundary repair.

The inherited acknowledgments include **HyeokxC**, **Ryun1**, **newjordan**, **Akashneelesh** and **Meganpark980320**, together with contributors named in the source files. These credits identify source provenance. They do not imply that the named contributors prepared, reviewed or endorsed this entry.

## Selected implementation

The device configuration enables the selected SHA addition routing, public-key hash head implementation, guarded scalar-coefficient carry helper and outlined public-key hash helper. These selections are declared at the candidate entry point and included in the native configuration declaration. The Q_MIX2 layout remains selected.

The scalar-coefficient implementation retains its integer fallback and boundary handling. The public-key helper retains the compressed-key input representation and the existing digest-word interface. The point chain, shared parking, inverse-tree interfaces and completed-hit record representation remain present in their inherited forms.

The lane-inversion repair is confined to the CPU helper's zero-product branch. The ordinary nonzero-product path retains its existing operation sequence. The repair uses the existing input values and independent-lane routine. CPU table construction, geometry selection, resource fallbacks, row prefetching, hashing, epoch ownership and publication remain part of the inherited co-grinder.

## Package surfaces

| File or surface | Role in this entry |
| --- | --- |
| `subset.cu` | Candidate entry point and selected device configuration |
| `sha_gate_fma.cuh` | Selected device hash operations and outlined key helper |
| `GLVScalar.cuh` | Guarded coefficient helper and retained scalar recoding |
| `CpuGrindSubset.h` | Original co-grinder with the separate lane-inversion boundary repair |
| `tests/gpu_epochs/host_producers.h` | Retained host producer implementation |
| `tests/gpu_epochs/tree.cu` | Inherited host/device integration and native configuration fields |
| `qsb_carrier_sm89.h` | Refreshed native image matching the selected device configuration |
| `QsbCarrier.h` | Retained native loader and fallback interface |
| `build_carrier.sh` | Retained native image build support |
| `SOURCE-MANIFEST.json` | Entry-specific source inventory and provenance |
| `submission-note.md` | Entry-specific package description |
| `tests/gpu_epochs/tree.cu.orig` | Unused backup removed from the package |
| `COPYING` | Retained license text |
| `COPYING-secp256k1` | Retained secp256k1 license text |

The retained device support includes `GPUHash.h`, `GPUMath.h`, `hit_filter_field.cuh`, `hit_filter_field_sc.cuh`, `chain_replay_field.cuh`, `square32.cuh` and `y_pair_sc.cuh`. The epoch support includes the existing descriptor, prefix-state and window-schedule headers. The inverse support includes the existing tree, limb and root helpers. Their source notices and attributed lineage remain attached to the corresponding files.

The submitted surface is `candidates/subset`. The input format, candidate index ordering, recovery identifier format and exact host publication interface remain the repository-defined interfaces. Device and CPU records retain their existing publication representation. The protected problem definition, verifier, scoring configuration, benchmark driver and workflow remain supplied by the repository base.

Historical documentation and audit sources remain supporting material in the candidate directory. Their presence does not turn earlier narratives into entry-specific claims. The refreshed source manifest describes the supplied package; the native header carries the corresponding image declaration. Temporary executables, private run outputs and local artifact directories are outside the submitted source package.

## Licensing and retained notices

The GPU arithmetic lineage includes VanitySearch-derived material with its inherited GPL notices. The host field and scalar lineage includes libsecp256k1-derived material with its inherited license notices. The package retains both license files, existing copyright statements and source-level acknowledgments. The entry-specific changes retain those notices in the files that carry them.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 172 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*