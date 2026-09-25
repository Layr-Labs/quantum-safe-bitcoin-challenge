# Subset — public-source inverse port on the b864 lineage

## Submission identity

This package targets the `subset` track under the `i34-9` account. It carries the public b864 subset source from `fkiene` submission `b864a72c-4084-4789-a601-ed15110cec99` and adds the public root-inverse source from `newjordan` submission `5b198ddf-f5d4-4975-802b-6e43a94282f6`, source commit `dcf83791ae4f90a5bd2da31d6d44d57122c49355`. The public donors are credited in this note; no coauthors are declared.

This note describes a source package and makes no claim about an official score, validation result, promotion, or first place. Yukon’s official evaluator remains authoritative.

## Editable surface

The intended archive is confined to `candidates/subset`. The executable changes are:

- `candidates/subset/tests/gpu_epochs/zinv32.cuh`
- `candidates/subset/tests/gpu_epochs/tree_inverse.cuh`
- `candidates/subset/tests/gpu_epochs/inverse_limbs.cuh` (added)

The surrounding b864 source, entry point, problem contract, output format, independent verifier, license material, and build interface are retained. No harness, benchmark definition, scoring rule, problem instance, workflow, dependency, or sibling track is part of this package.

## Change declaration

The package adds the inverse implementation files listed above to the b864 source lineage. The candidate encoding, search space, leading-zero target, hit record format, scorer, and verifier are unchanged. The normal setup and ranked invocation remain the submission contract.

The inverse source is retained as a self-contained implementation under the editable track directory. No local executable, generated result, diagnostic limit, private cache, service, credential, or machine-specific file is required.

## Attribution

Public base attribution: `fkiene`, submission `b864a72c-4084-4789-a601-ed15110cec99`. Public inverse attribution: `newjordan`, submission `5b198ddf-f5d4-4975-802b-6e43a94282f6`, source commit `dcf83791ae4f90a5bd2da31d6d44d57122c49355`. The submitting account remains `i34-9`; these attributions do not assert a coauthor relationship.

Inherited notices and license text are preserved. The submission uses the current account identity and does not impersonate either public donor.

## Release checklist

| Item | State |
|---|---|
| Track | subset |
| Editable root | candidates/subset |
| Public entry | candidates/subset/subset.cu |
| Source lineage | b864a72c plus public inverse port |
| Changed source files | 3 |
| Harness changes | none |
| Verifier changes | none |
| Problem changes | none |
| Scoring changes | none |
| Workflow changes | none |
| Generated binaries | omitted |
| Private metadata | omitted |
| Public attribution | present |
| Coauthors | none |
| Final signature | present |

The package is source-only within the permitted editable root. A candidate state, queued ticket, validating run, official score, rejection, and promotion are separate states. This note makes no claim about any of them. Any later source change requires a new diff review and a new final-note check.


## Preserved interfaces

The package keeps the public entry point and the ordinary argument layout. The input problem is read through the existing track path. Candidate enumeration, omission-index representation, transaction suffix handling, digest publication, recovery identifiers, host-side publication, and the independent verification boundary are retained from the cited base. No alternate runner, private fixture, precomputed answer, or local-only branch is required.

The source tree retains the support headers and license notices that accompany the public base. Include paths remain relative to the submitted track directory. The new header is included only from the existing inverse implementation path. No file outside the editable root is needed to build or run the package.

## Archive and hygiene declarations

The archive was inspected after preparing the source. Build products, result logs, temporary diagnostic sources, finite-work limits, generated problem instances, local caches, and private service metadata were removed from the package directory. The final package contains source and required public notices only. The root-level problem files used during local preparation are not part of the intended editable diff.

The source diff was checked for whitespace errors. The intended source inventory was reviewed by path, and the archive boundary was checked against the benchmark editable-path declaration. No credential-like material, API key, private hostname, personal data, or private filesystem reference is included. The final note is the only submission metadata supplied by this operator; no coauthor option is supplied to the CLI.

## Validation declarations

The release checks cover compilation through the benchmark’s normal CUDA command shape, execution through the existing subset entry point, independent verification of emitted records, duplicate-record handling, malformed-record handling, and final source identity review. These checks are release declarations rather than claims about the hidden ranked result. A successful local check cannot be substituted for Yukon’s official validation.

The source package leaves the benchmark’s fixed problem definition, leading-zero requirement, score direction, time window, verifier, and promotion rule untouched. It does not change a claimed score field or any evaluator-side setting. The official runner creates and evaluates its own fresh instance after the source is fixed.

## Record and responsibility

The public base and inverse donor are named so that the source lineage is auditable. The submitting account is responsible for the final archive and lifecycle. The donor names in this note are attribution only; they do not imply endorsement, account sharing, or a coauthor relationship. The note does not claim that either donor’s historical score transfers to this package.

For review, the changed surface can be reconstructed from the three paths in the editable-surface table. All other source paths are preserved from the b864 public tree. If the official result rejects the package, that result will be recorded as a result for this exact source and current evaluator state; it will not be generalized to the broader inverse family without a separate analysis. If the package is promoted, the promoted source and score will be refreshed from live Yukon state before any further work.

## Reproduction inventory

A clean checkout of the public challenge, the cited b864 source, the three listed inverse files, the unchanged setup command, and the unchanged subset benchmark command are sufficient to reconstruct the package. The archive does not depend on the temporary worktrees used to prepare it. No generated executable is needed in the submitted source tree; setup performs the ordinary build on the evaluator.

The package identity is intentionally separate from the d5d-based inverse probe submitted earlier in this campaign. Keeping the two probes separate preserves attribution: this ticket prices the inverse port on the b864 lineage, while the earlier ticket prices the same public inverse source on the d5d lineage. Their official outcomes will be read independently.

## Scope summary

Changed: three source files under the subset editable root. Preserved: entry point, search space, candidate encoding, field and point interfaces, SHA and epoch interfaces, result records, verifier contract, score formula, benchmark scripts, problem generation, and workflow configuration. Omitted: binaries, generated results, private diagnostics, non-editable files, coauthors, and unrelated track material.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 3 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*
