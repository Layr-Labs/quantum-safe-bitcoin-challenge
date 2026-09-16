# Adaptive subset table experiment

Update: PR60 has since been promoted at **451135044** verified candidates/s,
2.47% above the prior frontier440270249. Pinning PR74 remains pending.
The adaptive candidate itself is still unsubmitted and GPU-unmeasured.
Preparation-time pending statements below are historical.

Prepared 2026-09-16; isolated research, not submitted. Production remains the
PR60 closure `44f9d5808add05e204b169194af8893dc91a9c01d31448362b0e01c667c9fd06`.
This candidate is
`c8daa257dd1d3f3a1483eaa8047aa3c0af7a7b309103823a3ab959d43a46c9d6`.

The candidate carries both the established 32 MiB planar table and the 16 GiB
interleaved table. The wide path cuts its fixed-base chain from 102M+30S to
60M+18S, but its cache behavior could outweigh those savings. The runtime
comparison addresses that uncertainty on the actual device. It does not make
the wide path, or this combined executable, a proven improvement.

## Behavior

Startup always builds the small runtime-base table. The wide table is built
only for the ranked short-epoch mode when at least its 16 GiB plus a 2 GiB
search reserve remains free. All tables depend on the current problem.
Generic modes use the small table. The wide builder and optional rolling L2
prefetch come from the frozen wide experiment; the small chain, loader and
builder bodies are namespace-renamed from the frozen PR60 control.

Six normal search batches use S, W, W, S, S, W. The first two warm up each
path; the remaining four measure the complete GPU producer/inverse/recovery
pipeline using CUDA events. Each time is weighted by the actual candidate
count. Wide is selected only if its measured time per candidate is more than
2% lower. Invalid timing locks the small path. If wide was unavailable, no
timing events are created. Event handles are cleaned up on short runs too.

Every trial uses a distinct ordinary epoch range. Trial hits follow the usual
output path and count toward the run; none are discarded to time the method.
The inherited host output cap of 64 hit records per batch is unchanged.
The mode is an explicit host dispatch argument between two prepare kernels;
both feed the same checkpoint/inverse/finish implementation.

## Evidence and limits

`validation-summary.json` binds all current reports to the candidate fingerprint.

| Check | Result |
| --- | --- |
| Small chain | 12,769 recodings, 414 curve chains, 822 recovered keys |
| Wide chain | Same counts; 3,312 next-load prefetch address comparisons |
| Actual small/wide loaders | 608 planar and 380 interleaved cases |
| Extracted host control flow | 20 mocked scenarios, 10 timing-policy cases |
| Native production | CUDA 12.8.93, sm89 and default flags: pass |
| Native audit | CUDA 12.8.93, sm89: pass |
| Small prepare resources | 128 registers, 24 KiB shared, 8-byte spill store/load |
| Wide prepare resources | 128 registers, 24 KiB shared, no spills |
| Shared finish resources | 80 registers, 24 KiB shared, no spills |

The curve checks execute extracted chain code with OpenSSL field operations
and sparse oracle tables. Host tests execute the real policy, dispatch and
ranked while-loop with mocked kernels/events and two artificial hits per batch.
They cover full/tail batches, both winners, disabled wide and invalid times.
These checks do not execute CUDA or establish actual rare-hit behavior.
Native builds run in a local ARM Linux compiler VM, not on an NVIDIA GPU.

The timing excludes table construction, host I/O and other startup costs.
Selecting small now frees the wide allocation after the existing GPU sync.
It does not recover construction time and cannot guarantee total process
performance at least as good as PR60. Short/noisy samples, thermal
changes, cache differences and final partial batches remain timing risks.

## Reviewed external feedback

Grok 4.6 (`grok-4.6-build`) reviewed the initial source. Its explicit-dispatch
and early-event-cleanup suggestions were applied and checks rebuilt. Gemini
3.8 Flash High reviewed the final source. Its zero-epoch crash example is not
reachable through this ranked entry: that guard sets n_epochs to C(137,6).
The isolated loop alone is not a general zero-length API.

Gemini identified retained wide memory and fatal optional allocation failure.
Both are now repaired: only cudaErrorMemoryAllocation selects the fallback,
its error state is cleared, and other CUDA errors still fail. A small winner
releases wide after synchronization; seven allocation scenarios and the host
flow check cover fallback, unrelated errors and exactly-once release. Timing
handle copies are explicitly deleted. The review's fixed680MiB headroom claim
was unsupported. External reviews preceded these source-bound local repairs.

Both reviewers accepted the range/dispatch flow on source inspection. That is
not independent GPU validation. The report is `external-review.json`.

## Reproduction and next decision

From the benchmark repository:

```sh
python3 -B candidates/subset/research/adaptive_tables/check_adaptive.py
python3 -B candidates/subset/research/wide_windows/check_wide.py --source candidates/subset/research/adaptive_tables/candidate --report candidates/subset/research/adaptive_tables/wide-cpu.json
python3 -B candidates/subset/research/wide_windows/check_wide.py --source candidates/subset/research/adaptive_tables/candidate --small --report candidates/subset/research/adaptive_tables/small-cpu.json
```

`prepare.py` verifies the two donor identities and refuses to overwrite an
existing candidate. `provenance.json` records both parents and all source files.
The candidate is a full source closure, not a patch to the submitted tree.
The native compiler helper is documented in the sibling pinning research
directory; run it read-only with reports directed here while subset is selected.

PR60 is promoted; preserve pending PR74. PR62 has since raised the frontier
to477182283 with a64MiB table. The next comparison must include the independently
checked mixed64 control in ../mixed_windows, not only the older32MiB control. A public draft note accompanies this experiment. It is not a claim of
submission readiness, an uploaded artifact, or measured speed.
