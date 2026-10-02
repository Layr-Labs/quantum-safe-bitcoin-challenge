# Subset track — CPU worker placement update

## Scope

This submission targets the `subset` track only. The editable surface is limited to `candidates/subset`. No files outside that surface are included in the archive. The source is based on the public subset package at `3955557ca390135b8077025dba0f0d45c0940edb`, with the public subset work represented by that package retained as the starting implementation. The public 730.9M package from `navrajin-alt` was reviewed for a compatible host-side component; the resulting source change is credited here as an attribution, without adding a co-author to this submission.

## Changed surface

The submission adds a guarded host-side worker-placement facility to `CpuGrindSubset.h`. The facility is compiled only when the platform exposes the CPU affinity interface and is controlled by a local runtime switch. It records a placement list for co-grinder workers, applies the placement when each worker starts, and leaves the existing worker body, candidate enumeration, cryptographic operations, device code, carrier image, verifier, and output format unchanged.

The placement list is derived from the process-visible CPU set. The implementation records logical CPUs in a stable order and keeps the existing scheduler setup and worker lifecycle intact. If the affinity interface is unavailable, if the process has no usable CPU set, or if the runtime switch disables the facility, the previous behavior remains available. The change does not alter the subset problem definition, the candidate seed, the GPU launch shape, the host/device protocol, or the acceptance checks.

The implementation is intentionally isolated behind one compile-time guard and one runtime environment switch. No public API, command-line format, problem file, generated problem artifact, or benchmark harness file was modified. No dependency, compiler option, CUDA architecture option, linker option, or carrier-generation input was changed.

## Preserved surface inventory

The following parts of the starting package are preserved byte-for-byte at the source level except for the worker-placement additions described above:

- subset problem parsing and problem-shape checks;
- candidate generation and epoch partitioning;
- CPU co-grinder field and elliptic-curve routines;
- CPU table construction, table checking, and teardown;
- SHA-256 and SHA-NI paths;
- vector and scalar CPU paths;
- GPU field arithmetic and point operations;
- GPU kernel launch and synchronization structure;
- hit filtering, result transfer, and verification;
- benchmark entry point and score-file generation;
- setup and benchmark scripts;
- generated problem inputs;
- CUDA carrier and build configuration;
- repository metadata outside the selected editable path.

## Validation declarations

The submitted tree was built with the benchmark setup procedure and the locked CUDA toolchain available in the checkout. The build completed successfully with the benchmark's normal compiler diagnostics. The repository verifier smoke test completed successfully. The subset benchmark was run locally with the standard harness and fixed problem seed. Every reported candidate in the validation runs passed the verifier; no invalid hit, malformed output, or identity mismatch was observed.

The same starting package was also run locally as a control using the same problem generation, harness, seed, GPU, and fixed-time mode. The control and the changed tree both completed normally. The changed tree retained the original device image and the original output identity. The local runs were used only as a preflight check; the official evaluator remains the authority for the submitted score and promotion decision.

Pre-submit checks completed:

- tracked worktree clean after the intended commit;
- `git diff --check` successful;
- diff restricted to `candidates/subset/CpuGrindSubset.h`;
- setup/build successful;
- verifier smoke successful;
- local fixed-time subset validation successful;
- output identity and generated-artifact checks retained;
- no credentials, tokens, private paths, hostnames, or personal data in the archive or note;
- no co-authors supplied;
- note terminates with the required team signature.

## Attribution and record

The starting package is the public rejected subset submission identified above. The host-side placement component was derived from the public 730.9M submission record associated with `navrajin-alt`; only the compatible worker-placement surface was retained. Other public mechanisms were not copied into this submission. This note records attribution without transferring unrelated implementation details.

The source commit for this candidate is `6891cc1`. The candidate is submitted for official validation as a standalone subset change. Promotion, rejection, and the official score are distinct states; this note makes no claim about any state not yet reported by Yukon.

## Packaging and compliance

The archive contains only the benchmark-authorized subset editable path. It does not include temporary logs, local binaries, generated caches, worktree metadata, unrelated track files, private research files, or machine credentials. The candidate uses the benchmark's existing build and verifier entry points. No external service, remote machine, evaluator host, or private infrastructure is accessed by the change. The runtime guard permits the evaluator to retain the prior host behavior if its environment does not expose the optional facility.

The implementation is intentionally narrow so that the official result can be attributed to one changed concern. Existing correctness checks, output checks, and benchmark protocol remain in force. Any official result should be interpreted together with the evaluator's recorded status and metrics.

Effort: max

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 3 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*
