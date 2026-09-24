# Fixed-base table L2 access-policy experiment

Status: both variants compiled; local 3090 diagnostics in progress.
Base: public Subset frontier 623,518,629 candidates/s at 7e95c40.

Hypothesis: the 64 MiB point table is repeatedly accessed, whereas large
first-state and epoch buffers stream through L2. A bounded persisting policy
may reduce eviction of table entries. The runtime computes the permitted
reservation from device properties, queries the actual limit, and caps the
fraction tagged persisting at reservation/table-size to avoid oversubscribing.
Miss segments retain normal caching. Unsupported hardware keeps the original
path. No scalar, field, SHA, enumeration, or hit-publication code changes.
Default QSB_GTABLE_PERSIST is zero while under experiment.

The implementation uses documented CUDA 12.8 runtime APIs and types from
installed cuda_runtime_api.h and driver_types.h: cudaDeviceSetLimit,
cudaDeviceGetLimit, cudaStreamSetAttribute, cudaAccessPolicyWindow.
The hint applies to the default stream used by all existing kernels.
The 3090 and 4090 have different cache capacities, so local measurements
are diagnostics and cannot establish an official 4090 improvement.

Build: PATH=/usr/local/cuda/bin:$PATH python3
candidates/subset/research/cache_policy/trial.py build
Run: flock -x /tmp/qsb-gpu.lock python3
candidates/subset/research/cache_policy/trial.py baseline --label=-a
(or candidate --label=-b). Each run uses unchanged harness/gpu_wrap.py,
N=24, 60 seconds, independent CPU validation, seed=2026092501. No root
score file is written. Rebuild after any included-source change.

All experiment design and code are by GPT-6 Astra high in Codex. The earlier
DeepSeek/Claude Code structural audit attempt timed out; it supplied no result.
No unpublished source was sent to that service.

## Outcome

Rejected locally: baseline 1,758/1,758 verified hits, 245.241197 M/s;
candidate 1,658/1,658 verified hits, 231.363644 M/s (-5.66%).
Self-reported rates 243.5 versus 231.8 M/s agree with the regression.
Device API probe confirms the policy was available: L2 6,291,456 bytes,
reservation 4,325,376 bytes, hit ratio 0.064453, no API errors.
This 3090 result does not prove a 4090 outcome. Not submitted; the experimental
block was removed from production and retained in cache-policy-snippet.txt.
For reproduction insert that snippet immediately before /* Upload params */
in tree.cu before rebuilding the wrappers. Rebuildable binaries were removed for source-only submission; their hashes are in ../split_pipeline/removed-build-artifacts.json.
