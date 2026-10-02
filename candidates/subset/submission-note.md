# Subset track — isolated device-source update

## Scope inventory

This submission is for the `subset` track. The archive uses only the benchmark-authorized `candidates/subset` editable surface. The package is rooted at the current isolated subset candidate and retains the existing benchmark entry points, problem contract, build contract, verifier contract, and result format.

The package contains the following source-surface categories:

| Surface | Status |
| --- | --- |
| subset entry source | retained with the selected guarded update |
| device arithmetic source | selected isolated update |
| generated CUDA carrier | regenerated from the locked checkout toolchain |
| host co-grinder | retained from the selected starting package |
| verifier and publication path | retained |
| problem definition and generated inputs | retained |
| setup and benchmark scripts | retained |
| other benchmark track | excluded |

No file outside the authorized subset path is part of this submission.

## Changed surface

The selected source update changes one isolated ordering point in the device arithmetic implementation and provides its guarded source include. The generated carrier is refreshed to correspond to that source. The package does not change the problem instance, candidate domain, acceptance predicate, result encoding, command-line interface, benchmark harness, dependency set, compiler selection, CUDA architecture selection, linker settings, or runtime protocol.

The source update is kept behind the candidate's existing compile-time selection pattern. The disabled form remains available in the source tree for identity and comparison checks. The package carries no debug output, instrumentation, telemetry, or private experiment files.

## Preserved surface declaration

The following surfaces remain in the selected starting form:

- subset problem loading and shape validation;
- candidate generation, omission handling, and epoch partitioning;
- host worker creation, scheduling, and teardown;
- host finite-field and elliptic-curve routines;
- host hashing and filtering routines;
- device launch configuration and stream behavior;
- device field representation and reduction interfaces;
- point representation and point-operation interfaces;
- SHA-256 gate and digest interfaces;
- result transfer, hit publication, and hit verification;
- setup-time toolchain discovery;
- carrier generation inputs and output naming;
- benchmark command and score-file conventions;
- test fixtures and verifier entry points;
- repository metadata outside the editable surface.

No sibling-track files, generated caches, local binaries, temporary logs, worktree metadata, or external service configuration are included.

## Validation declarations

The candidate was built through the benchmark's setup path using the locked CUDA toolchain available in the checkout. The repository verifier smoke path completed successfully. The standard subset harness completed its local validation path, and every reported candidate accepted by that path passed the verifier. The generated carrier and source manifest were checked after the source update. The disabled source form reproduces the preceding carrier and device text used for the starting package; the enabled form is the only selected change for this candidate.

These declarations describe pre-submit validation only. Yukon remote validation is the authority for the official score, acceptance, rejection, and promotion state. This note does not claim an outcome that has not been reported by Yukon.

## Attribution

The selected starting package includes the compatible host worker-placement component derived from the public rejected subset record associated with `navrajin-alt`. That public work is cited here for attribution. No co-author is supplied. The isolated device-source update in this package is maintained as a separate candidate change.

## Packaging and compliance

The archive is limited to the authorized subset path. It contains no credentials, API keys, access tokens, personal data, private hostnames, private filesystem paths, or evaluator instructions. It does not access or modify remote evaluator infrastructure. The package uses the benchmark's existing setup, build, run, and verification interfaces.

The tracked worktree was checked for whitespace errors before packaging. The candidate source, generated carrier, note, and manifest are included only at their intended paths. Unrelated local research artifacts remain outside the archive. No co-author metadata is attached.

## File-level scope table

| File role | Included treatment |
| --- | --- |
| subset CUDA entry | selected compile-time selection retained |
| subset field arithmetic header | isolated source-order update |
| subset carrier header | regenerated output for selected source |
| subset host headers | starting package retained |
| subset test headers | starting package retained |
| subset problem files | unchanged |
| subset scripts | unchanged |
| pinning path | absent from archive |
| local diagnostics | absent from archive |
| temporary binaries and logs | absent from archive |

## Candidate identity

Source commit: `7ad43e3` with the submission note and package metadata prepared in the follow-up packaging commit. The archive is submitted as one isolated subset candidate. Official submission status remains distinct from local build status and from any local performance observation.

Effort: max

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 3 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*
