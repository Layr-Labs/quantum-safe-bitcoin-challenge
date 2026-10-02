# Subset candidate update

## Changed surface

| Surface | State |
|---|---|
| `candidates/subset/tests/gpu_epochs/tree.cu` | Updated subset execution configuration. |
| `candidates/subset/subset` | Rebuilt native candidate executable. |
| `candidates/subset/qsb_carrier_sm89.h` | Regenerated CUDA 12.8 `sm_89` carrier for the submitted source. |

## Preserved surface

| Surface | State |
|---|---|
| `candidates/pinning/` | Preserved. |
| `harness/` | Preserved. |
| `problems/` | Preserved. |
| `spec/` | Preserved. |
| Benchmark configuration and workflow files | Preserved. |
| Verifier and score paths | Preserved. |

## Compliance declarations

- The archive is limited to the benchmark editable surface.
- The submitted tree contains no credentials, tokens, private hostnames, or private paths.
- The native image was regenerated from the submitted source with the runner-compatible CUDA toolchain.
- The candidate executable and generated carrier are included together so the packaged source and native image remain aligned.
- The standard subset setup completed successfully.
- The standard verifier smoke check completed successfully.
- Local fixed-seed execution completed with verifier acceptance for the published hit records.
- No coauthors are requested for this submission.

## Public attribution

This candidate retains the public subset lineage and notices from the accepted `kshitij-hash` subset submission `4cc9d2d8-0b37-4ae2-b460-f70e41c9e35f`. Attribution is stated here only; no additional contributor metadata is requested.

## File inventory

| Path | Action |
|---|---|
| `candidates/subset/tests/gpu_epochs/tree.cu` | Modified. |
| `candidates/subset/subset` | Regenerated. |
| `candidates/subset/qsb_carrier_sm89.h` | Regenerated. |
| Other editable files | Preserved. |

## Validation inventory

| Check | Result |
|---|---|
| Source formatting check | Passed. |
| Locked-toolchain build | Passed. |
| Native carrier regeneration | Passed. |
| Verifier smoke check | Passed. |
| Fixed-seed candidate run | Passed. |
| Hit publication contract | Preserved. |
| Editable-path restriction | Satisfied. |
| Secret scan | No secrets added. |

## Packaging statement

The package contains the updated subset candidate and its matching native image. No harness, verifier, benchmark, workflow, or unrelated track files are included in the change. Existing notices in the inherited source remain available in the submitted tree.


## Release manifest

| Item | Declaration |
|---|---|
| Candidate scope | Subset track only. |
| Runtime entry | Existing subset entry point retained. |
| Input contract | Existing benchmark input contract retained. |
| Output contract | Existing benchmark output contract retained. |
| Hit encoding | Existing hit encoding retained. |
| Exactness gate | Existing exactness gate retained. |
| Host verification | Existing host verification retained. |
| Device image | Regenerated from the submitted source. |
| Toolchain | Runner-compatible CUDA toolchain used. |
| Archive policy | No files outside the editable surface were added. |

## Preserved behavior

| Behavior | Declaration |
|---|---|
| Candidate enumeration | Preserved. |
| Candidate bounds | Preserved. |
| Problem seed handling | Preserved. |
| Difficulty selection | Preserved. |
| Result publication | Preserved. |
| Duplicate handling | Preserved. |
| Host-side acceptance | Preserved. |
| Failure handling | Preserved. |
| Stop handling | Preserved. |
| Track separation | Preserved. |
| Benchmark lifecycle | Preserved. |
| Existing license notices | Preserved. |

## Review record

| Review | Result |
|---|---|
| Editable-path review | Passed. |
| Tracked-diff review | Passed. |
| Generated-image alignment review | Passed. |
| Build artifact review | Passed. |
| Verifier-path review | Passed. |
| Output-format review | Passed. |
| Secret and credential review | Passed. |
| Note-format review | Passed. |
| Attribution review | Passed. |
| Coauthor metadata review | No coauthors requested. |

## Submission contents

The submission contains the updated subset source, the matching generated native image, and the rebuilt executable required by the existing packaging layout. All unrelated benchmark surfaces remain unchanged. The candidate keeps the repository's existing licensing and attribution material. The public attribution above identifies the inherited accepted lineage without adding contributor metadata.

## Final declarations

The package is ready for the standard Yukon validation lifecycle. The source, executable, and native image are aligned. The benchmark harness and verifier remain the existing versions. The submission does not alter the pinning track, benchmark rules, workflow, or score calculation. No private information is present in the package or this note.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 3 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*
