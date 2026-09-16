# Small workflow changes with high value

Review date: 2026-09-16. Both our submissions remain validating; there is no
returned GPU result yet from which to infer a speedup or diagnose a failure.

| Change | Evidence from this work | Practical effect |
| --- | --- | --- |
| Check the full include closure, not just the entry point | `subset.cu` now contains one include. The trusted wrapper's build stamp uses only that file's mtime and N. | Prevent a header-only experiment from silently reusing a stale binary. |
| Save test results against a source fingerprint | Our audit printed counts and individual hashes, mixed with C++ diagnostic output. | A separate JSON report identifies the tested source and explicitly labels CPU-only evidence. |
| Refresh the public frontier at experiment-selection and submission decisions | A stronger pending architecture appeared while the compact-table prototype was being evaluated. | Stop an obsolete experiment promptly; cache concise note findings instead of rereading everything. |
| Separate inherited improvement from our additional change | The large expected gain comes mainly from the imported epoch pipeline; the four-lane tree has no GPU A/B measurement. | Avoid attributing another implementation's advantage to our own patch. |
| Spend the next validation effort on GPU evidence | More CPU samples cannot check PTX compilation, GPU synchronization, register spills or actual throughput. | Once CUDA access exists, prioritize a clean build, the included GPU audit, then a same-host unchanged-base comparison. |

The large-improvement preference remains in force for candidate research. The
small changes here improve how experiments are selected and evaluated; they do
not lower that target or justify additional minor-performance submissions.

## Commands

From the benchmark repository:

```sh
python3 candidates/subset/preflight.py \
  --note candidates/subset/submission-epoch-inverse.md
python3 candidates/subset/check_candidate.py --report /tmp/qsb-subset-audit.json
```

Compare `source_fingerprint` in those two JSON reports before citing an older
audit. The aggregate covers all quoted dependencies of production and the GPU
audit, including the header containing our new inverse. If an include leaves
the track or is missing, preflight fails. This is a source/package check, not
proof that CUDA executed successfully.

On a CUDA host, a fresh compile that bypasses every existing build stamp:

```sh
python3 candidates/subset/preflight.py --cuda
```

That command copies the include closure to a temporary directory, compiles the
production entry point and retained GPU audit using the benchmark's N=24 flags,
then removes temporary artifacts. It does not run either executable. For an
actual local benchmark after editing a header, invalidate the generated binary
and stamp in `candidates/subset/` before setup so the unchanged trusted wrapper
also rebuilds. Do not change the trusted wrapper's cache implementation here.

The next useful GPU comparison is the imported `b733504` tree versus the exact
submitted four-lane helper, using the same GPU, seed, difficulty and duration.
Verify hits independently and check registers/spills; do not substitute a
comparison with the much older production baseline for this incremental A/B.
No GPU rental, additional submission or expanded automation is authorized by
this review. Existing interactive authorization remains separate from the
results-only recurring monitor.

## Scope

Added a read-only preflight, a machine-readable option to the existing audit,
and local workflow instructions. Candidate CUDA sources, trusted benchmark
files, queued submissions, global instructions and recurring monitors were not
changed. A future returned failure should add one targeted regression check,
not an indiscriminate expansion of the test suite.
