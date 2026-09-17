# Local experiment selector

`selection.py` is a small offline research triage tool inspired by the useful part of structured kernel experiment selection: make the source, hypothesis, constraints and evidence explicit before choosing another experiment. It does not import KernelBlaster code, call external services, tune compiler flags, edit candidates, run a benchmark, or submit anything.

The result is a conservative **declared-receipt gate**, not a substitute for the checks that generate those receipts. A caller supplies local explicit records and is responsible for their authenticity. The script checks their statuses, identities, completeness and freshness; it does not rehash the referenced artifacts or inspect their contents. Each qualification receipt records an artifact SHA256 so an independent verifier can bind the actual file. Do not describe the selector itself as executing qualification or proving a GPU result.

## Files and use

- `selection.py`: standard-library CLI and `evaluate(data, now)` function.
- `selection-records.json`: concrete initial proposals, intentionally not submission-ready.
- `selection-results.json`: deterministic example output at an explicitly supplied historical time.
- `test_selection.py`: focused gate, evidence-separation and duplicate regression tests.

From the benchmark root:

```
python3 -B candidates/subset/research/frontier_rebase_sep17/selection.py \
  candidates/subset/research/frontier_rebase_sep17/selection-records.json
python3 -B -m unittest discover \
  -s candidates/subset/research/frontier_rebase_sep17 -p test_selection.py -v
```

Use `--output result.json` to save a result. `--now 2026-09-17T15:00:00Z` is for reproducible tests/replays; omit it for live current-time evaluation. A previously allowed result is not reusable as authorization: refresh the official frontier and own-slot receipts and evaluate again immediately before any independently authorized submission. There is no upload function or background process.

## Input contract

Top-level `schema_version` is 1. `context` declares:

- `benchmark_id`, `target`, `frontier_id`, and `accepted_base_id`.
- Optional `accepted_promotion_id` for explanatory provenance.
- `selected_pending_base`, distinct from the accepted base; `pending_frontier_ids` lists explicitly reviewed/pending labels, and `reviewed_pending_heads` maps reviewed IDs to their exact source heads. An unreviewed pending head cannot become an eligible base.
- `current_source_fingerprint`, a full 64-character lower-case SHA256 for the selected closure; `already_submitted_fingerprints` prevents unchanged-source repeats.
- `frontier_receipt`: an explicit receipt `id`, `observed_at` with timezone, matching `frontier_id`, `accepted_base_id`, `pending_frontier_ids` and `reviewed_pending_heads`.
- `own_slot`: `status` (`free` is the only permitted submission state), `observed_at` and `frontier_receipt_id`, tying it to that same snapshot.

Both live receipts must be from the past, at most **120 seconds old**. Future times, missing receipts, different snapshots or a validating own slot block submission. The equality checks are intentionally exact, including the supplied pending-list order: copy a single snapshot consistently rather than merging loosely related observations.

Each record supplies `id`, `base_id`, `frontier_id`, `mechanism_id`, `mechanism_revision`, `hypothesis`, `benchmark_id`, `target`, `correctness` (`pass`, `fail`, `unknown`) and `compatibility` statuses for `target`, `compiler`, and `fixed_flags`. A source fingerprint may be absent for incompatible/unimplemented proposals; it is mandatory for source-bound evidence and submission. A selected pending base also needs a `base_head` equal to the reviewed head.

Evidence entries have `kind`: `hypothesis`, `cpu`, `native`, `gpu_controlled` or `gpu_official`. Non-hypothesis evidence counts only with a passing status, artifact reference and exact record fingerprint. GPU evidence additionally requires `measurement: measured`, matching benchmark/target, `comparison_id` and `run_id`. Static instruction reductions, CPU checks and predicted speedups remain their original evidence types. Numeric magnitudes do not affect ranking: a claimed 100-fold static improvement cannot become a measured GPU result.

`prior_failures` remain attached to records. An unresolved prior failure filters the proposal. To resolve one, supply `resolved: true` and a `resolution` with an artifact and matching source fingerprint. Merely changing that boolean does not clear it. If the previous failure was performance-related, its relevance and the changed mechanism need an explicit reviewer decision; the selector cannot infer that from prose.

`qualification` contains optional additional `required` check names plus `receipts` keyed by name. The minimum set cannot be removed by the input:

```
source_closure, cpu_correctness, native_default, native_sm89,
audit_default, audit_sm89, startup_policy, package, public_note
```

Every required receipt must have `status: pass`, the selected record's exact `source_fingerprint`, an `artifact` reference, and a 64-character `artifact_sha256`. This checks receipt structure/binding, not the underlying file. Add mechanism-specific checks to `required`; for example protocol, exceptional-domain and inherited-body comparisons when applicable. **GPU timing is deliberately not a mandatory check.** The user's willingness to enter a credible CPU/native-qualified candidate without GPU access is preserved.

## Filtering, ordering and limits

Known-invalid records, incompatible or unresolved target/compiler/flag states, changed frontiers, nonselected bases, unreviewed pending heads, unchanged submitted sources and unresolved failures are filtered. Duplicate detection uses either the same source fingerprint or the same `(base_id, mechanism_id, mechanism_revision)`. Those identifiers are explicit reviewer-maintained mechanism labels; arbitrary renamed prose is not automatically recognized. A corrected, materially changed mechanism gets a new revision and retains the prior failure record with its resolution evidence.

Within eligible experiments, evidence categories are ordinal: official GPU, controlled GPU, native, CPU, hypothesis. The selector does not compare heterogeneous throughput numbers, calculate predicted gains, or assert that a stronger-evidence candidate is faster. Submission-ready records rank before incomplete research; missing qualification count and ID provide deterministic ties. Duplicate representatives prefer higher evidence and then fewer missing qualifications. The output exposes all filter reasons, missing qualifications, ignored evidence, evidence kinds and submission blockers, not only a winner.

The fixed minimum qualification set is specific to this subset workflow. It does not claim that those nine checks are sufficient for every new architecture. Fresh receipts do not resolve an actual correctness failure or authorize changing the trusted harness. Read the research and validation records before using the ranking to choose work.

## Seeded decisions

The seed context records parent-supplied accepted submission `db248c65`, promotion/frontier `df2fb8b`, selected pending source `1a3af597`, and pending own PR200 separately. These are historical coordination labels, **not a current official queue receipt**. Exact reviewed heads and a fresh frontier receipt are intentionally absent. The example output allows no submission.

1. **CompileIQ / compiler-flag exploration** is filtered because trusted flags are fixed. It is an incompatibility record, not an unsuccessful native experiment.
2. **CUDA13 shared-memory spilling** is filtered because the proposed mechanism needs a newer compiler than the current CUDA12.8 setup. This tool does not replace the trusted compiler.
3. **Rust / cuda-oxide field arithmetic** keeps the existing component reports as inherited references. CPU arithmetic passed, and sm89 PTX assembled, but host integration is incomplete and local-array traffic remains (96-byte stack, 17 LDL/21 STL static sites despite zero reported spills). There is no candidate closure fingerprint or GPU performance result to promote. See `research/oxide_experiment/README.md`.
4. **Pending-source integration** records the promising integration work while awaiting exact reviewed head, frozen closure and qualification. It does not treat a pending author's claims as measurements of our integrated source.

The first two compatibility findings and pending-base labels are supplied by the parent task; this bounded selector work did not fetch new upstream source or run native builds. The Rust observations come from the existing local report. Candidate files, accepted/pending snapshots, production, stages and submissions are untouched.
