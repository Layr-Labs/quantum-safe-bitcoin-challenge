# Subset — device implementation and execution selection

This entry updates the device hash implementation and native execution selection. It carries a bound pair of native device images, their configuration declarations, and the corresponding host loader integration. The co-grinder, host producer composition and candidate interfaces are retained from the team's identified predecessor. This note and the source manifests describe this package separately from the earlier entry.

## Public provenance

| Role | Public source |
| --- | --- |
| Current promoted base | RealAdii, `521075fe` |
| Current base commit | `46b24ebaa033fb69c7335794b54fd6a156359ec8` |
| Co-grinder package donor | cefika, `bf001729` |
| Co-grinder donor commit | `79f750e735b6266fc2d694f3d96e8931f02e2dbf` |
| Co-grinder family-selection source | cefika, `4a197f06` |
| Family-selection source commit | `fdf000cc3707fcd7d9cec1ce9d691c1a29c22792` |
| Host producer package donor | ercumentyildirim, `a141df2b` |
| Producer donor commit | `3b4fce1213a41ab5e2e96d7f038fdd88a07f2878` |
| Retained team predecessor | `b67487a1` |
| Public-key hash head source | fkiene, `8c07297bb79a8340632b1101e5704ac1294f2b13` |
| Submitted directory | `candidates/subset` |

The public identifiers above describe the inherited source composition. The entry retains the attributed host implementation and candidate-plan configuration. The device image package and loader integration are updated together. Entry-specific changes are not attributed to the donors merely because their earlier work appears in the same source tree.

## Attribution

**RealAdii** is credited for the promoted source retained in this entry's lineage. **cefika** is credited for the selected co-grinder package, its earlier composition, and the public CPU pattern-family selection incorporated in this entry. **ercumentyildirim** is credited for the selected producer package and inherited host work. **terrapinelf** is credited for the underlying co-grinder and producer implementations carried by those packages. **kshitij-hash** is credited for the device additions and earlier promoted subset foundation retained here. **fkiene** is credited for the public-key hash head implementation used by the owned integration carried into this entry.

The inherited lineage also includes work credited to **HyeokxC**, **fkiene**, **Ryun1**, **newjordan**, **Akashneelesh**, **Meganpark980320**, and **i34-9**. Existing source comments and license notices retain further acknowledgments. Naming these contributors establishes provenance; it does not imply their participation in preparing this entry, their review of this composition, or their endorsement of it.

The source tree retains `COPYING` and `COPYING-secp256k1`. Copyright statements and upstream notices are preserved in the files where they occur. This note replaces the inherited submission narrative. Claims about tools, authorship, experiments, and results in an older narrative are not presented as work performed for this entry.

## Edit scope

| Surface | Disposition |
| --- | --- |
| Candidate entry point | Selected device configuration and native execution integration |
| Device execution implementation | Updated device hash implementation and bound execution variants |
| Device arithmetic headers | Retained field arithmetic; selected guarded scalar-coefficient helper |
| Embedded native images | Bound image pair with separate configuration declarations |
| Native image loader | Entry-specific integration with retained fallback support |
| Launch and completed-record publication | Updated host integration with retained record format |
| Host co-grinder header | Retained attributed package, selection integration and boundary guard |
| Host selection header | Retained from the team predecessor |
| Host producer header | Attributed producer package |
| Submission note | Entry-specific text |
| Source manifest | Entry-specific provenance and file hashes |
| Backup source file | Removed from the submitted directory |
| Licensing files | Retained |
| Historical documentation and audit files | Retained with documentation path cleanup |

The submitted surface is limited to the subset candidate directory. The official verifier, problem generator, benchmark driver, scoring configuration, setup script and workflow remain supplied by the repository base. The other track is outside this submission's scope. No generated executable or private run output is part of the submitted source package.

## Selected implementation

This package updates the selected device hash implementation and native execution support. Existing input/output interfaces, scalar-coefficient implementation and host verification remain present.

## Package inventory

The inventory below distinguishes implementation files, retained support material and entry metadata. A listed file may contain optional code paths; the inventory is not a claim that every path in every file was exercised locally.

| Candidate file | Package role |
| --- | --- |
| `ASMLAST511-RESEARCH.md` | Retained historical documentation |
| `BY_NORMALIZED6.md` | Retained historical documentation |
| `CANONICAL-ADD.md` | Retained historical documentation |
| `CHAIN-REPLAY.md` | Retained historical documentation |
| `COMPLETE-POINT.md` | Retained historical documentation |
| `COPYING` | Retained license notice |
| `COPYING-secp256k1` | Retained license notice |
| `CpuGrindSubset.h` | Attributed host implementation with entry-specific updates |
| `CpuCoupledWorkers.h` | Host selection support |
| `GLVScalar.cuh` | Selected guarded coefficient carry helper and retained scalar recoding |
| `GPUHash.h` | Retained candidate source or support file |
| `GPUMath.h` | Retained candidate source or support file |
| `HIT-CHECK.md` | Retained historical documentation |
| `LAST511-RESEARCH.md` | Retained historical documentation |
| `PAIR-CURRENT.md` | Retained historical documentation |
| `PAIR-FRONT.md` | Retained historical documentation |
| `POINT-BY-TABLE.md` | Retained historical documentation |
| `POINT-PREDICATE.md` | Retained historical documentation |
| `POINT-X3.md` | Retained historical documentation |
| `QsbCarrier.h` | Updated native loader and host integration |
| `QsbNativeSelector.h` | Native execution selection support |
| `QsbNativeFingerprints.h` | Bound image configuration declarations |
| `QsbNativeGlobals.h` | Bound image object declarations |
| `NATIVE_COMPOSITION.json` | Image and source provenance |
| `verify_native_composition.py` | Read-only package consistency check |
| `qsb_carrier_alt_sm89.h` | Bound alternate native image |
| `SOURCE-MANIFEST.json` | Entry metadata |
| `TREE_INVERSE.md` | Retained historical documentation |
| `build_carrier.sh` | Native image build support with composition protection |
| `chain_replay_field.cuh` | Retained candidate source or support file |
| `hit_filter_field.cuh` | Retained candidate source or support file |
| `hit_filter_field_sc.cuh` | Retained candidate source or support file |
| `hit_filter_field_sc_aluz.cuh` | Retained candidate source or support file |
| `qsb_carrier_sm89.h` | Matching updated native image |
| `sha_gate_fma.cuh` | Updated device hash implementation |
| `square32.cuh` | Retained candidate source or support file |
| `submission-note.md` | Entry metadata |
| `subset.cu` | Candidate entry and selected device configuration |
| `tests/gpu_epochs/by_table_matrix_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/canonical_add_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/chain_replay_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/dirdig_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/epoch_groups.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/filter_tail_sc.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/first_stage_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/hm39_divstep.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/hm39_pair_inverse.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/hm41_quad_inverse.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/hm43_warp_inverse.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/host_producers.h` | Attributed producer implementation |
| `tests/gpu_epochs/inverse_limbs.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/pair_finish_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/pair_shared.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/parity_window_subset.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/point_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/point_predicate_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/prefix_cache.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/qsb_host_verify.h` | Retained candidate source or support file |
| `tests/gpu_epochs/scalar_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/seed_x3_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/tree.cu` | Retained device and host integration with matching image configuration fields |
| `tests/gpu_epochs/tree_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/tree_inverse.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/window_schedule_shared.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/zinv32.cuh` | Retained candidate source or support file |
| `y_pair_sc.cuh` | Retained candidate source or support file |

## Validation declarations

The final composition has been built and exercised through the candidate entry point. The native image pair was available during that execution. Ordinary device and co-grinder records were accepted by the repository verifier. The selected device implementation also has retained output-identity checks against its control. Package checks bind these declarations to the supplied source and image pair.

The host execution integration has separate retained checks for its ordinary decision and unavailable-alternative behavior. The image declarations are validated independently. These declarations describe the configurations actually checked; they do not assert that every possible hardware condition or optional source branch was exercised locally.

The boundary-handling change has separate retained checks against an independent arithmetic reference. Those checks include explicit boundary inputs. They establish the scope of the repair without asserting that an ordinary workload exercises every rare input. The source package keeps that handling together with the selected co-grinder implementation.

Validation evidence belongs to this candidate's source and retained artifacts. It is not inferred solely from a donor's note or from a previous official score. Optional configurations remain subject to the resources and capabilities available at execution. A check of one local configuration does not claim that all resource-dependent configurations were executed locally.

The package contains source and its bound native images; private run outputs and temporary test artifacts are excluded. The protected verifier, problem definition, scoring rules and official workflow remain supplied by the repository. File inventory, whitespace, allowed paths, manifest hashes and note formatting are checked for the final package.

This entry makes no guarantee of an official score or promotion. Its official outcome must be read from the remote evaluation associated with this submission. Acceptance of local records and acceptance of an official performance result are distinct declarations. The entry-specific checks do not rename a queued or rejected result as a promotion.

## Result ownership

The official evaluator determines this submission's validity, score and promotion state. A result recorded for the promoted base or a donor package is not represented as a result of this composition. The note identifies the supplied package, retained surfaces and completed pre-submission checks. Historical documentation remains source material and is not a new performance claim by this entry.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 172 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*