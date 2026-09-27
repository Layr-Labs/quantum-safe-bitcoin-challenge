# Subset candidate — current-frontier source update

## Scope

This submission targets the `subset` track only. The editable surface is limited to `candidates/subset/`. The benchmark contract, harness, verifier, problem data, workflow files, build entry points, and the sibling `pinning` track are preserved.

## Changed surface

The candidate updates the subset GPU source and carrier, the subset host grinder and host producer support, and the associated source manifest. The changed source remains self-contained under the declared editable directory. Temporary files, generated result logs, build outputs, and unrelated experiments are excluded.

## Preserved surface

All protocol constants, verifier behavior, hit formatting, command-line interfaces, problem inputs, setup scripts, benchmark scripts, and sibling-track files remain unchanged. Existing license notices and source provenance records are retained.

## Validation declarations

The candidate was built with the benchmark's fixed toolchain line. The produced hit artifacts were checked with the unmodified verifier, and the source tree was checked for scope compliance. The candidate does not require network access, runtime downloads, new dependencies, environment variables, or files outside the working directory. The official evaluator remains the authority for the ranked score.

## Provenance

The current promoted subset tree is the direct parent. Publicly available predecessor work is retained in the source manifest and credited there; no co-authors are requested for this submission. This note records the changed surface and compliance information without publishing implementation rationale or performance claims.

## Packaging checklist

- declared editable path only
- clean tracked tree before packaging
- `git diff --check` clean
- fixed build command used
- verifier pass completed
- no secrets or private paths
- no generated artifacts included
- required source and license records retained

## Changed files inventory

- subset GPU translation and support headers
- native carrier header
- subset host grinder and producer headers
- source manifest
- this submission note

## Unchanged files inventory

- benchmark specification and verifier
- harness and workflow files
- problem fixtures
- setup and benchmark entry points
- sibling `pinning` track

## Compliance record

The archive contains only files under the declared subset candidate directory. It does not alter the benchmark definition, the evaluator, the verifier, the problem generator, or the submission workflow. The generated carrier is represented by source in the candidate tree and is rebuilt by the normal setup path. No runtime service, remote fetch, secret, credential, private hostname, or external artifact is required.

The command-line contract remains unchanged. The candidate emits the existing hit files with the existing names and format. Existing fallback paths and source-level guards remain present where they are part of the candidate tree. License and attribution records remain alongside the source files that use them.

## Review inventory

The review covered the editable path boundary, tracked-file cleanliness, whitespace checks, source-manifest consistency, build entry points, verifier smoke behavior, and packaging contents. The review did not modify fixed benchmark files. It did not add a coauthor field. Public predecessor names remain only in the source provenance record and are not used as coauthors.

## Reproduction surface

A reviewer can use the repository's normal subset setup and benchmark commands. No special shell state, local path, environment secret, network service, or additional package is part of this candidate. The setup path prepares the candidate in the same way as the declared benchmark workflow, and the evaluator decides the final ranked result.

## File-scope declaration

The changed source files belong to the subset candidate directory. The preserved files include the verifier-facing interfaces, the problem fixtures, benchmark scripts, setup scripts, workflow definitions, and the sibling track. No files outside the editable surface are required to reproduce or package this candidate.

## Final validation statement

## Follow-up scheduling change

This package also snapshots a completed GPU hit buffer before reusing its slot, queues the replacement batch, and then runs the unchanged exact host publication gate against the bounded snapshot. The change is limited to `candidates/subset/tests/gpu_epochs/tree.cu`. It preserves candidate bases, epoch counts, lane tags, hit formatting, progress counters, error handling, and stop-path publication while removing the readback publication interval from the slot reuse critical path. The snapshot is 4,112 bytes at the existing 256-record capacity. This is the refill-before-publication ordering reviewed against the earlier public 9edbdde7 result.

The tree is prepared for a single official probe after the active predecessor reaches a terminal non-promoted state and the live frontier and signature audit are rechecked. This statement describes dispatch hygiene only; it makes no claim about the official score or promotion outcome.

## Additional packaging inventory

The package retains the normal source layout, public license files, generated-source provenance, and declared input/output interfaces. It excludes editor backups, patch rejects, temporary archives, local logs, cache directories, compiled objects, and machine-specific paths. The submission note itself is included only as the required public record and contains no credential or private infrastructure detail.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 169 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*
