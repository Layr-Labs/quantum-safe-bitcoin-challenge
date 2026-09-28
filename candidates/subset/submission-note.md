# Subset — current promoted implementation with an attributed host package

This entry starts at the current promoted subset implementation and updates the host-side candidate package. The device implementation, embedded native image, and current launch/publication code remain from that promoted base. The submitted changes are the host co-grinder header, the host producer header, the package manifest and this note. An inherited backup source file is removed from the package.

## Public provenance

| Role | Public source |
| --- | --- |
| Current promoted base | RealAdii, `521075fe` |
| Current base commit | `46b24ebaa033fb69c7335794b54fd6a156359ec8` |
| Co-grinder package donor | cefika, `bf001729` |
| Co-grinder donor commit | `79f750e735b6266fc2d694f3d96e8931f02e2dbf` |
| Host producer package donor | ercumentyildirim, `a141df2b` |
| Producer donor commit | `3b4fce1213a41ab5e2e96d7f038fdd88a07f2878` |
| Submitted directory | `candidates/subset` |

The co-grinder header retains the current promoted base's prefetch-distance setting. Its other selected host implementation content comes from the attributed donor package. The producer header is retained from its identified donor. This submission does not adopt a different device configuration along with those host files. The public source identifiers above identify the composition without claiming that the components were originally designed for this entry.

## Attribution

**RealAdii** is credited for the promoted source used as this entry's immediate base. **cefika** is credited for the selected co-grinder package and its earlier composition. **ercumentyildirim** is credited for the selected producer package and inherited host work. **terrapinelf** is credited for the underlying co-grinder and producer implementations carried by those packages. **kshitij-hash** is credited for the device additions and earlier promoted subset foundation retained here.

The inherited lineage also includes work credited to **HyeokxC**, **fkiene**, **Ryun1**, **newjordan**, **Akashneelesh**, **Meganpark980320**, and **i34-9**. Existing source comments and license notices retain further acknowledgments. Naming these contributors establishes provenance; it does not imply their participation in preparing this entry, their review of this composition, or their endorsement of it.

The source tree retains `COPYING` and `COPYING-secp256k1`. Copyright statements and upstream notices are preserved in the files where they occur. This note replaces the inherited submission narrative. Claims about tools, authorship, experiments, and results in an older narrative are not presented as work performed for this entry.

## Edit scope

| Surface | Disposition |
| --- | --- |
| Candidate entry point | Current promoted base |
| Device execution implementation | Current promoted base |
| Device arithmetic headers | Current promoted base |
| Embedded native image | Current promoted base |
| Native image loader | Current promoted base |
| Launch and completed-record publication | Current promoted base |
| Host co-grinder header | Attributed package with the base prefetch setting retained |
| Host producer header | Attributed producer package |
| Submission note | Entry-specific text |
| Source manifest | Entry-specific provenance and file hashes |
| Backup source file | Removed from the submitted directory |
| Licensing files | Retained |
| Historical documentation and audit files | Retained |

The submitted surface is limited to the subset candidate directory. The official verifier, problem generator, benchmark driver, scoring configuration, setup script and workflow remain supplied by the repository base. The other track is outside this submission's scope. No generated executable or private run output is part of the submitted source package.

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
| `CpuGrindSubset.h` | Attributed host implementation |
| `GLVScalar.cuh` | Retained candidate source or support file |
| `GPUHash.h` | Retained candidate source or support file |
| `GPUMath.h` | Retained candidate source or support file |
| `HIT-CHECK.md` | Retained historical documentation |
| `LAST511-RESEARCH.md` | Retained historical documentation |
| `PAIR-CURRENT.md` | Retained historical documentation |
| `PAIR-FRONT.md` | Retained historical documentation |
| `POINT-BY-TABLE.md` | Retained historical documentation |
| `POINT-PREDICATE.md` | Retained historical documentation |
| `POINT-X3.md` | Retained historical documentation |
| `QsbCarrier.h` | Retained candidate source or support file |
| `SOURCE-MANIFEST.json` | Entry metadata |
| `TREE_INVERSE.md` | Retained historical documentation |
| `build_carrier.sh` | Retained candidate source or support file |
| `chain_replay_field.cuh` | Retained candidate source or support file |
| `hit_filter_field.cuh` | Retained candidate source or support file |
| `hit_filter_field_sc.cuh` | Retained candidate source or support file |
| `hit_filter_field_sc_aluz.cuh` | Retained candidate source or support file |
| `qsb_carrier_sm89.h` | Retained native image |
| `sha_gate_fma.cuh` | Retained candidate source or support file |
| `square32.cuh` | Retained candidate source or support file |
| `submission-note.md` | Entry metadata |
| `subset.cu` | Retained candidate source or support file |
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
| `tests/gpu_epochs/tree.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/tree_audit.cu` | Retained candidate source or support file |
| `tests/gpu_epochs/tree_inverse.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/window_schedule_shared.cuh` | Retained candidate source or support file |
| `tests/gpu_epochs/zinv32.cuh` | Retained candidate source or support file |
| `y_pair_sc.cuh` | Retained candidate source or support file |

## Validation declarations

The final candidate was built with the repository setup command and the required CUDA toolkit. The setup verifier smoke test passed. Bounded local executions of the composed candidate exercised its GPU and CPU paths, and the repository verifier accepted the emitted records. The producer's startup check also completed. These declarations refer to checks actually run for this composition, not merely to checks described in a donor's note.

The local runs used the retained native image. No device arithmetic or native image was rebuilt as part of this host composition. Source comparison establishes that the device implementation and image remain from the current promoted base. The package manifest records the selected source files and their hashes.

Optional host-table configurations depend on available resources. A local run using one configuration does not establish that every optional configuration has been executed locally. The local validity checks do not stand in for the official remote run. This entry makes no guarantee of an official score or promotion outcome.

The submitted directory was checked for its allowed path scope, patch whitespace, manifest hashes, temporary backup files, archive size and embedded credentials. The submission metadata names the tools used for this entry independently of the metadata in inherited source packages. Attribution is provided in this note, and the final signature is entry-specific.

## Result ownership

The official evaluator determines this submission's validity, score and promotion state. A result recorded for the promoted base or a donor package is not represented as a result of this composition. The note identifies the supplied package, retained surfaces and completed pre-submission checks. Historical documentation remains source material and is not a new performance claim by this entry.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 172 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*