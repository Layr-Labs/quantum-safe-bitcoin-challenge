# SHA-only split research

This local RTX 3090 experiment separates paired SHA generation from the
promoted point/inverse/finish consumer. The latter retains its original
two-factor inverse tree. A four-word SoA scalar workspace uses 2 GiB at the
tested launch size, versus 8 GiB plus validity bytes for the prepared
four-stage candidate. Exact replay and the unchanged harness verify hits.

The public source was extracted with
`git show HEAD:candidates/subset/tests/gpu_epochs/tree.cu` at commit
`7e95c40c99e57bded233ce57c7f453fbde9fd21c`. The unauthenticated GitHub raw
version was checked byte-for-byte; see `public-source-verification.json`.
No unpublished source was sent to DeepSeek. The experiment remains local.

The first three 60-second runs used seed 2026092501 and N=24 under one
exclusive `/tmp/qsb-gpu.lock`:

| Version | Verified hits | Verified M candidates/s | Completed candidates |
| --- | ---: | ---: | ---: |
| Prepared four-stage, four-factor baseline | 1844 | 257.059952 | 15,233,712,128 |
| SHA split, first use | 1772 | 247.161592 | 14,629,732,352 |
| SHA split, warm | 1850 | 258.086055 | 15,300,820,992 |

All emitted hits passed independent CPU verification. The warm comparison
has exactly the same 1844 hits over the common completed prefix. Gains of
0.3992% in verified score and 0.4405% in completed work do not establish a
1% improvement over the prepared candidate. Keep production unchanged.
This check is not an exhaustive proof of inherited speculative arithmetic.

`frontier.cu` reproduces the committed public tree with only include paths
relocated, retaining its default launch size. Every directly included quoted
header was verified equal to its HEAD version. A separate direct comparison
with this baseline is recorded in `run-frontier-comparison.log`; consult
`frontier-run-handle.json` for status before running another GPU job.

That direct comparison completed successfully: public baseline 1791 verified
hits / 249.789205 M/s / 14,763,950,080 completed candidates; SHA-only candidate
1834 verified hits / 255.848251 M/s / 15,099,494,400 completed candidates.
Verified gain was 2.4257%, completed-work gain 2.2727%. All 1791 hits in the
common completed prefix matched. `public-frontier-qualification.json` records
the machine-readable comparison. This supports a second local candidate;
it does not demonstrate superiority over the prepared four-stage candidate.

`candidate128.cu` and `sha128.cuh` test 128-thread SHA producer blocks while
leaving the point/inverse/finish consumer unchanged. Each producer block
owns exactly two epochs instead of four. The local mapping audit covers
15 small and odd epoch counts; production harness checks remain necessary.
The initial sm_52 resource report has 77 SHA registers and no stack, versus
72 for the 256-thread producer. Actual RTX 3090 execution uses driver JIT.
Consult `sha128-run-handle.json` before starting another GPU run.

The 128-thread experiment is now rejected. The warm run verified 1814 hits
at 253.051544 M/s versus the adjacent 256-thread reference's 1826 hits at
254.702040 M/s. Completed work fell 0.4464%; all 1814 common-prefix hits
matched. Both embedded PTX variants assemble to 124 SHA registers on sm_86
and sm_89 with CUDA 12.8 ptxas. The older sm_52 resource counts therefore
should not be used to infer occupancy on the actual target architectures.
These offline resource reports are not measurements of the driver JIT.

The next hypothesis, `candidate_single.cu`, assigns one scalar to each SHA
thread and calls the existing public single-state hash routine. The paired
point/inverse/finish consumer remains unchanged. Offline sm_86/sm_89 SHA
assembly uses 40 registers with no spill or stack. `audit_single.cu` checks
all scalar words against the original paired SHA routine and an independent
straightforward CPU SHA compression implementation plus OpenSSL for the
second hash. Four input fixtures cover 60 odd/boundary epoch cases and guard
regions; benchmark runs start only if the audit passes. Consult
`single-run-handle.json` for status; compiler evidence alone is not a pass.

DeepSeek via Claude Code reviewed only the public SHA function snippets at
the verified HEAD. `helper-public-sha-provenance.json` records extraction,
public byte equality, selected line ranges, and hashes. Main-agent review
confirmed initialization, round order, and feed-forward, and corrected an
unnecessary input-nonaliasing caveat: read-only first-state inputs may alias
on the odd tail. No unpublished source was included in that helper call.
The helper review is neither a whole-grinder proof nor performance evidence.

The single-state audit passed: 126,976 256-bit scalars match both the paired
GPU implementation and CPU/OpenSSL, including all guards. Its warm benchmark
verified 1826 hits at 254.660437 M/s versus the adjacent paired reference's
1814 hits at 252.835221 M/s. The verified gain was 0.7219%, completed-work
gain 0.4484%, and all 1814 shared-prefix hits match. Retain the experiment
as a validated alternative; the incremental gain is below 1%.

A compile-only experiment disabled `QSB_PAIR_SHA_UNROLL_WINDOW`. It still
used 124 SHA registers on sm_86 and sm_89, so no GPU run was scheduled for
its register-pressure hypothesis. The next configuration, `sha_bound3.cuh`,
instead changes only the paired SHA launch bounds from `(256,2)` to
`(256,3)`. Offline assembly uses 80 registers and no stack or spills on both
targets. Its separate audit checks the paired bounded kernel against the
single-state GPU kernel and independent CPU/OpenSSL oracle before benchmark
runs. Consult `bound3-run-handle.json` before launching another GPU task.

Completed raw runs, inactive source variants, and embedded PTX dumps through
the single-state experiment are preserved in `closed-through-single.tar.xz`.
All 109 archived members were hash-verified before their loose copies were
removed. `closed-through-single-manifest.json` maps original names and hashes
to archived payloads; gzip/xz payloads are stored decompressed. For example:

```sh
tar -xOf closed-through-single.tar.xz output-candidate_single-j/run.json
```

The audit, qualification summaries, active source files, and process records
remain directly accessible. No evidence was discarded by packaging.

The bounded paired SHA audit also passed all 126,976 scalar comparisons.
Its warm run achieved 254.668686 M/s versus 252.964814 M/s for the adjacent
paired reference: +0.6736% verified score, +0.4484% completed work, and all
1814 shared-prefix hits identical. This is insufficient to replace production.

Host-only CUDA event diagnostics then measured 24 batches per variant,
excluding the initial four from steady means. Embedded device PTX matched
the previously tested variants exactly. Actual RTX 3090 JIT SHA resources
were 123 registers / 2 resident blocks for the paired reference, 79 / 3 for
the bounded version, and 40 / 6 for the single-state version. SHA times were
43.2754, 42.9117, and 43.5322 ms respectively. The bounded SHA reduction is
only 0.84%, predicting roughly 0.143% total improvement if other stages were
unchanged. The identical consumer's measured time varied by 1.25% across
these short diagnostic runs; the cause of that variation was not measured.
No sub-1% throughput claim is promoted based on these diagnostics.

A separate, unscored profile of the prepared four-stage production candidate
also matched its device PTX exactly. Its steady batch means were 167.777 ms
point arithmetic (66.54%), 41.765 ms SHA (16.56%), 28.726 ms tail (11.39%),
and 11.656 ms inverse (4.62%). It was not run adjacent to the earlier profiles
and should not be treated as a ranked throughput comparison. Actual point
resources were 128 registers, no local memory, and two resident blocks.

The next point-stage hypothesis is to defer loading the constant recovery
point coordinates until after the fixed-base chain. The current public
`qsb_pair_front3_z_value` accepts eight coordinate limbs by value, though
they are only used in the final preparation. A local alternative can keep
the same chain and arithmetic order and load immutable `QSB_U2R` afterward.
Whether this reduces register pressure is unproven; inspect compiler output
before scheduling a GPU trial, and require packet/validity equivalence plus
the unchanged harness if it proceeds. This is a research hypothesis, not an
implemented or qualified improvement.

The four instrumented source snapshots are hash-verified in
`profile-sources.tar.xz`; the generators and raw diagnostic logs remain
directly available. Finished profiling binaries were removed after hashing.

The completed `candidate.cu`, `candidate_bound3.cu`, and `candidate_single.cu`
source snapshots are now hash-verified in `sha-variant-sources.tar.xz` to
keep the submission tree within its size cap. Before rebuilding those
variants, their audits, or their profiles, restore them in this directory
with `tar -xJf sha-variant-sources.tar.xz`. No live process uses the archived
loose files.

The current official frontier is 623,518,629 candidates/s on an RTX 4090.
These RTX 3090 diagnostics are not official scores. The previously submitted
three-stage version remains tracked separately; this variant was not submitted.
