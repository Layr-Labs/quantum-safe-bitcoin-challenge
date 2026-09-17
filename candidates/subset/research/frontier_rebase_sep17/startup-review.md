# Current-frontier startup scope and audit command

The selected guard-only source is fingerprint `24e96b567aa1662aa941e582ca4928cf393c9fcaf22161ee3a2b6355b0827199`. `guard-startup-results.json` binds the complete production/audit include closure. The prior weak candidate's separate `startup-results.json` remains bound to `5f4bc59e84820d9c114f249ec09a853f67610842270e1e561983cd2fec3d3093`; that source was rejected by the parent native screen, and is not silently relabeled.

## Policy applicability

The new frontier source has no `l2_policy.cuh`, no `cudaStreamSetAttribute` call and no persisting-L2-limit configuration. Its inline host table startup and selected geometry are byte-identical to the exact local pending217 snapshot. The main `preflight.py` package check requires the previous 180-capacity/30-error policy report **only when `tests/gpu_epochs/l2_policy.cuh` belongs to the source closure**. Therefore that particular policy check is not applicable here. There is no actual after-build policy call to execute or certify. Neither old policy receipts nor their case counts have been copied into the new qualification.

Instead, `check_startup.py` executes the actual selected host table-startup block with mocked allocation/CUDA/curve-builder APIs. The only executable syntax replacement is the CUDA kernel launch, replaced with a mock that checks the exact4096-block/256-thread launch geometry. Host control flow remains the actual source. Independently expected geometry is15 windows `[18]+[17]*14`,64MiB output,245760-byte low ladder,983040-byte high ladder and252 host spot samples. No table arithmetic or GPU execution is performed.

Fifteen cases passed: four kernel-error/spot-result combinations, three host allocation failures, three CUDA allocation-status failures, three CUDA copy-status failures on the usual path, a fallback-upload failure, and a synchronization-status failure. **Eight of these document inherited ignored error statuses, not successful error recovery.** The source checks the later kernel error and curve spot-check outcome, but ignores the immediate `cudaMalloc`, `cudaMemcpy` and `cudaDeviceSynchronize` return values at this startup site. Artificially independent mock outcomes expose that control-flow limitation; they are not a claim about a real GPU failure's subsequent behavior. The parent has been notified.

Two compiled projection-only negative controls fail: removing the host fallback and changing the first window width. They prove the test exercises the extracted startup branch and geometry, rather than only an independent reimplementation. Each run confirms the source identity is unchanged before/after testing.

Reproduce from the benchmark root:

```
python3 -B candidates/subset/research/frontier_rebase_sep17/check_startup.py \
  --source candidates/subset/research/frontier_rebase_sep17/guard-candidate \
  --base candidates/subset/research/frontier_rebase_sep17/pending217 \
  --report /tmp/guard-startup-results.json
```

In a lean submission checkout use `--source candidates/subset` and retain the pending217 snapshot required by `--base`. This produces a truthful nonapplicability/startup record, not the older L2-policy report schema's `PASS`/180/30 assertion. The generic selector's startup receipt may point to it with an explicit scope note; package preflight itself does not demand this absent-policy receipt.

## Audit compilation

`subset.cu` only includes `tests/gpu_epochs/tree.cu`; there are no entry-only defines that need to be duplicated for the audit. The audit includes that same tree and renames its host main. From the benchmark root, set SRC to the exact source directory and compile to a fresh external output:

```
SRC=candidates/subset/research/frontier_rebase_sep17/guard-candidate
nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/qsb-frontier-audit \
  "$SRC/tests/gpu_epochs/tree_audit.cu" -lcrypto -lm
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -o /tmp/qsb-frontier-audit-sm89 \
  "$SRC/tests/gpu_epochs/tree_audit.cu" -lcrypto -lm
```

The first command preserves the trusted default virtual target. These are verified source/command recipes, **not builds executed by this startup checker**. Parent-owned production/audit reports must supply compile results. GPU execution of either audit is a separate validation level. Main preflight's CLI binds its own directory; for isolated snapshots use its `source_identity(root)` API, or place the unchanged preflight script in the actual staged `candidates/subset` before the package command. Calling the main production preflight while merely naming a different working directory would check the wrong candidate.

The selector seeds now retain the weak ordinary-region native failure (`+38` static common-path sites, registers100→119, parent review) and a separate canonical pending217 final-guard proposal. This does not claim measured speed or final package readiness. Current source, base snapshots and stages were not modified by this work.
