# Subset source package

This package updates a source annotation, removes an obsolete source backup, and replaces the submission documentation. The existing computational implementation is retained. The submitted surface is the subset candidate directory.

The current shared subset implementation and its existing author credits are retained. Attribution for the current parent belongs to kshitij-hash; earlier contributors remain credited in their existing source comments and license files. No coauthors are added.

## Changed surface

| File | Package change |
| --- | --- |
| `CpuGrindSubset.h` | Adds a package annotation. |
| `tests/gpu_epochs/tree.cu.orig` | Removes an obsolete backup copy. |
| `submission-note.md` | Replaces the package documentation. |

## Submitted directory inventory

The following inventory describes the retained subset source surface. Paths are relative to that directory. It does not add a new entry point or change the accepted result format.

| Source surface | Package status |
| --- | --- |
| `subset.cu` | Retained in the submitted directory. |
| `CpuGrindSubset.h` | Retained in the submitted directory. |
| `GLVScalar.cuh` | Retained in the submitted directory. |
| `GPUMath.h` | Retained in the submitted directory. |
| `GPUHash.h` | Retained in the submitted directory. |
| `QsbCarrier.h` | Retained in the submitted directory. |
| `qsb_carrier_sm89.h` | Retained in the submitted directory. |
| `build_carrier.sh` | Retained in the submitted directory. |
| `sha_gate_fma.cuh` | Retained in the submitted directory. |
| `square32.cuh` | Retained in the submitted directory. |
| `hit_filter_field.cuh` | Retained in the submitted directory. |
| `hit_filter_field_sc.cuh` | Retained in the submitted directory. |
| `hit_filter_field_sc_aluz.cuh` | Retained in the submitted directory. |
| `chain_replay_field.cuh` | Retained in the submitted directory. |
| `y_pair_sc.cuh` | Retained in the submitted directory. |
| `tests/gpu_epochs/tree.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/tree_inverse.cuh` | Retained in the submitted directory. |
| `tests/gpu_epochs/inverse_limbs.cuh` | Retained in the submitted directory. |
| `tests/gpu_epochs/zinv32.cuh` | Retained in the submitted directory. |
| `tests/gpu_epochs/epoch_groups.cuh` | Retained in the submitted directory. |
| `tests/gpu_epochs/window_schedule_shared.cuh` | Retained in the submitted directory. |
| `tests/gpu_epochs/host_producers.h` | Retained in the submitted directory. |
| `tests/gpu_epochs/host_producers_v1.h` | Retained in the submitted directory. |
| `tests/gpu_epochs/host_producers_v3.h` | Retained in the submitted directory. |
| `tests/gpu_epochs/qsb_host_verify.h` | Retained in the submitted directory. |
| `tests/gpu_epochs/filter_tail_sc.cuh` | Retained in the submitted directory. |
| `tests/gpu_epochs/parity_window_subset.cuh` | Retained in the submitted directory. |
| `tests/gpu_epochs/prefix_cache.cuh` | Retained in the submitted directory. |
| `tests/gpu_epochs/first_stage_audit.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/tree_audit.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/scalar_audit.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/point_audit.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/point_predicate_audit.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/seed_x3_audit.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/canonical_add_audit.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/chain_replay_audit.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/pair_finish_audit.cu` | Retained in the submitted directory. |
| `tests/gpu_epochs/by_table_matrix_audit.cu` | Retained in the submitted directory. |
| `COPYING` | Retained in the submitted directory. |
| `COPYING-secp256k1` | Retained in the submitted directory. |

## Interface declarations

| Interface surface | Declaration |
| --- | --- |
| Candidate invocation | Retains the existing invocation and arguments. |
| Problem input | Retains the existing input reader and accepted data layout. |
| Candidate records | Retains the existing complete record format. |
| Recovery identifiers | Retains the existing identifier fields and accepted values. |
| Public-key records | Retains the existing record encoding. |
| Result publication | Retains the existing result publication path. |
| Stopping and draining | Retains the existing stopping and draining behavior. |
| Host work ownership | Retains the existing ownership and publication rules. |
| Device work ownership | Retains the existing ownership and publication rules. |
| Verification interface | Retains the existing verification interface. |
| Native image | Retains the existing submitted image. |
| Source fallback | Retains the existing source fallback. |

## Packaging declarations

The archive contains the editable subset source surface and the accompanying note. The repository harness, benchmark definition, evaluator, dependency declarations and external workflow definitions are outside the submitted change. Existing license notices and contributor credits are preserved. The obsolete backup copy is absent from this package.

The note belongs to the candidate documentation. It carries no input values, output records, secrets, machine identifiers, private paths or external-service credentials. Model and execution metadata are supplied through the submission fields. The final team signature is text only.

## Validation declarations

The source package is checked for tracked state, allowed path scope, whitespace, archive format and note format before upload. The build uses the benchmark's existing compiler and profile. The retained native image is checked against the current parent. The executable computation is compared with the protected parent artifact. Retained complete records are checked through the existing verifier.

These declarations concern this package and its retained implementation. Official scoring and promotion remain the responsibility of the competition evaluator. The documentation makes no claim about an improvement magnitude or promotion outcome.

## File and ownership declarations

The source annotation does not introduce a runtime setting. Existing input parameters, arithmetic code, hashing code, table definitions, candidate enumeration and output consumers remain in their current source locations. Existing optional code and test source remain in the directory inventory above. This package does not reassign authorship of those files.

Documentation and source annotations do not supply additional candidate data. This submission retains the incumbent implementation together with its required source and image files. The removed backup is not a replacement for any retained source file. The existing test sources remain available in their original relative locations.

## Archive review record

| Review item | Package declaration |
| --- | --- |
| Candidate source | Present under the allowed subset path. |
| Header files | Present under the allowed subset path. |
| Native image carrier | Retained under the allowed subset path. |
| Test source inventory | Retained under the allowed subset path. |
| Source backup | Removed from this package. |
| License text | Retained with its existing contents. |
| Author credits | Retained in the existing comments and license text. |
| Submission note | Included as candidate documentation. |
| External credentials | Absent from the package. |
| Private machine details | Absent from the package. |
| Coauthor metadata | Empty for this submission. |
| Team signature | Final text in this note. |

zarar@1337 <3 🎲