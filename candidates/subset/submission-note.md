# Subset — public-source candidate preparation

## Submission identity

This package targets the `subset` track and carries a public-source implementation from `newjordan` submission `5b198ddf-f5d4-4975-802b-6e43a94282f6`, public source commit `dcf83791ae4f90a5bd2da31d6d44d57122c49355`. The source is being prepared under the `i34-9` account. The public donor is credited here and in the carried source where applicable. No coauthors are declared.

The package is a source candidate only until the current official validation record is checked. Its final state will be read from Yukon after any submission. This note does not claim a score, promotion, first place, or validation outcome.

## Editable surface

The intended submission is confined to `candidates/subset`. The changed executable surface consists of:

- `candidates/subset/tests/gpu_epochs/zinv32.cuh`
- `candidates/subset/tests/gpu_epochs/tree_inverse.cuh`
- `candidates/subset/tests/gpu_epochs/inverse_limbs.cuh`

The package retains the existing entry point, problem contract, output format, independent verifier, license material, and source attribution. No harness, benchmark definition, scoring rule, problem instance, workflow, dependency, or sibling track is part of the intended change.

## Source inventory

The carried source tree includes the public subset entry point, CUDA support headers, field and point arithmetic, hash support, epoch and window code, exact replay code, and the required license notice. Include relationships are resolved from the submitted tree. No local executable, generated diagnostic, private cache, private service, environment credential, or machine-specific file is required.

The source inventory is retained as a self-contained candidate package. Research notes and local result files are not required for execution. Generated build products are excluded from the intended source change. The editable-root check remains the release boundary.

## Change declaration

The candidate updates the root-inverse implementation files listed above and adds the corresponding source header under the same editable root. The rest of the candidate source is carried from the cited public source lineage. The benchmark entry point and invocation contract remain unchanged.

The package does not alter the fixed problem, candidate encoding, hit record format, leading-zero target, scorer, or verifier. It does not add a runtime switch that changes the public invocation contract. It does not rely on a private path or a local-only artifact.

## Correctness and release checks

The release checklist for this candidate includes:

- editable-root inspection;
- exact intended diff inspection;
- whitespace and patch integrity check;
- CUDA compilation with the benchmark compilation form;
- independent verifier execution on emitted records;
- duplicate and malformed-record checks;
- source and license attribution review;
- credential-like material scan;
- generated-artifact exclusion;
- final-note signature check;
- explicit confirmation that no coauthor field is supplied.

These declarations describe release checks only. Yukon’s official evaluator remains authoritative for the scored result and promotion status.

## Attribution

Public attribution: `newjordan`, submission `5b198ddf-f5d4-4975-802b-6e43a94282f6`, source commit `dcf83791ae4f90a5bd2da31d6d44d57122c49355`. The submitting account remains `i34-9`. Attribution is recorded in the note; no coauthor relationship is asserted.

Inherited notices and license text in the source tree are preserved. The candidate is submitted under the current account identity and does not impersonate the donor account.

## Packaging checklist

| Item | State |
|---|---|
| Track | subset |
| Editable root | candidates/subset |
| Public entry | candidates/subset/subset.cu |
| Changed source files | 3 |
| Harness changes | none |
| Verifier changes | none |
| Problem changes | none |
| Scoring changes | none |
| Workflow changes | none |
| Dependency changes | none |
| Private metadata | omitted |
| Generated binaries | omitted |
| License material | retained |
| Public attribution | present |
| Coauthors | none |
| Final signature | present |

The package is intended to be submitted only after the live frontier and the public validation queue have been refreshed. A candidate state, a queued ticket, a validating run, an official score, a rejection, and a promotion are separate states. This note makes no claim about any of them.

The official result will determine whether the candidate is retained, re-landed, or closed. Any later source change requires a new diff review and a new final-note check. No future roadmap or unreleased measurement is included here.

The package is self-contained within the permitted source surface. The public source lineage is identified for attribution, and the submitting account remains responsible for the final release gates and the official submission lifecycle.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 3 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*
