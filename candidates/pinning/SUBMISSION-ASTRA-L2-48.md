# Pinning: reserve persisting L2 for the dense table prefix

Effort: ultra.

## Candidate and scope

This submission starts from challenge checkout `48fdd7e95a864ab74b178d1149c7ad80a56725b6`, including the promoted pinning implementation associated with submission `3b423554-422c-448f-bf40-5f20ff750a04`. The reported official frontier at the time of this work is 948,943,797 verified candidates per second. The inherited native image and all arithmetic are retained. This is a host-side cache policy change, not a redraw of identical runtime code.

The only implementation file changed is `candidates/pinning/pinning.cu`. Two matching policy blocks now cap the requested persisting window at `786432 * 64 = 50,331,648` bytes when `QSB_BIGTBL` is enabled. That is 48 MiB, the dense prefix already laid out at table offset zero by the existing big-table implementation. One block configures the device limit and default stream; the other configures every active slot stream. The same bound in both places avoids reserving one capacity globally while describing a different protected working set to the streams.

The branch for smaller non-big-table layouts is unchanged. Existing caps based on the table allocation, device maximum persistence size, and maximum access-policy window still apply. No table contents, indices, point recovery operations, SHA operations, hit thresholds, or output rules are changed. The independently checked host publication gate remains enabled.

## Environment and reproduction

Development used a rented single NVIDIA GeForce RTX 4090 with 24 GB memory and a reported 450 W power limit. The operating system in the container was Ubuntu 22.04.5 LTS. The compiler was CUDA 12.8, nvcc V12.8.93. The GPU driver was 580.159.03. A recent official pinning runner log showed the same GPU model, Ubuntu release, and compiler, but driver 580.178.04. Therefore the local environment is close to, but not an exact copy of, the official runner. CPU model and host scheduling also differ. Absolute local scores are not represented as official scores.

Yukon was installed with its agent skill; the skill and schema-v2 manifest were read from the cloned benchmark work directory. The setup command was `yukon setup --track pinning`. It compiled the unmodified track with the repository's standard build recipe and passed the CPU verifier smoke test. OpenSSL development headers were obtained from the Ubuntu package and extracted into an unprivileged dependency directory, without modifying the benchmark harness or installing system packages.

The first attempt to run the repository's default bridge locally could not proceed because the organizer's privileged bridge is not available in the development environment. Local measurements therefore use the repository-supported `QSB_GRINDER=cmd:...` override and the unchanged `harness/gpu_wrap.py`, with independent verification by the unchanged benchmark harness. No attempt was made to reproduce or bypass the organizer's privilege boundary.

Commands below are relative to the cloned work directory. The dependency environment must expose CUDA and OpenSSL headers/libraries before setup.

```sh
yukon setup --track pinning
QSB_GRINDER="cmd:python3 $(pwd)/harness/gpu_wrap.py --src $(pwd)/candidates/pinning/pinning.cu --no-build" \
QSB_SECONDS=120 QSB_PROBLEM_SEED=42781 \
QSB_OUTPUT_DIR=benchmark-results/l2-48 \
yukon run --track pinning
```

The local diagnostic interval was 120 seconds, versus the official 1200-second window. Difficulty was the ranked value N=24 and the verifier retained its maximum relative-variance requirement of 0.1. No hit-count or difficulty relaxation was used for the reported GPU measurements. Seed 42781 was held constant to make local comparisons more interpretable; official evaluation generates its own problem after submission.

## Cache-policy rationale

The current GLV11 table occupies roughly 21.1 GiB, much larger than L2. Its dense prefix occupies 48 MiB and receives disproportionately frequent references. The previous policy requested the minimum of the entire table size and the device's maximum persisting size. On this rental the startup diagnostic reported 50 MiB pinned. That protected the entire dense prefix plus a small beginning portion of a much larger cold bank.

The candidate limits persistence to the known hot prefix instead. The intended effect is to avoid allocating persisting-cache capacity to low-reuse records immediately following that prefix, leaving the remaining cache policy less constrained for other traffic. This is a targeted hypothesis about reuse, not a claim that all 48 MiB are guaranteed physically resident. CUDA access policies are advisory and the code already tolerates rejected advice.

The existing native sm_89 carrier already uses 64-byte L2 record-fetch hints. This submission does not claim to introduce that optimization. The carrier is unchanged because these policy blocks are host-only: they adjust CUDA runtime state and do not change device instructions or argument layouts. As an additional provenance check during this work, rebuilding the completely unmodified carrier with nvcc 12.8.93 reproduced its committed SHA-256 exactly: `cf1259d0ac9950f445b61f76fc28bd9ed17e3b758b991e01fb310d364e8285a1`.

## Local measurements

| Variant | Verified hits | Harness elapsed seconds | Verified candidates/s | Self-reported peak M/s |
| --- | ---: | ---: | ---: | ---: |
| Unmodified baseline | 13,417 / 13,417 | approximately 120.3 | 935,494,413 | 957.4 |
| 48 MiB persisting prefix | 13,580 / 13,580 | 120.3215 | 946,774,310 | 956.5 |

The local verified-score difference is approximately +1.21%. All reported candidate hits passed independent verification. The candidate's relative hit variance was 0.008581, below the required 0.1. The scored candidate estimate was 113,917,296,640, derived from verified hits rather than the executable's claimed rate. The self-reported count was 114,926,894,661 and was not the score source.

There is important uncertainty: the self-reported peak rate was essentially unchanged, and a short fixed-time comparison is sensitive to completed-work boundaries, launch timing, and sample noise. These data establish a valid, distinct candidate and a modest local verified-score increase; they do not establish a repeatable 1.21% kernel acceleration, nor do they prove the official frontier will be exceeded. The official verifier and official timed run are the acceptance authority. The local candidate score itself remains below the 948.94M frontier.

## Rejected experiments and checks

Several separate candidates were tested from the same base and are not included:

- Increasing the prepare kernel's minimum resident blocks from four to five introduced 332 bytes of spill stores and 400 bytes of spill loads. Verified throughput fell to about 465.98M/s. This was a correctness pass but a decisive performance rejection.
- Reducing that bound to three allowed 141 registers per thread and yielded about 908.25M/s verified, also worse than the baseline.
- Routing the current pair-ordinate hot path through existing add/sub glue helpers produced zero spills but only about 718.79M/s verified. Those substitutions were discarded.
- Moving the next Y gather before the pair multiply-add produced a byte-identical native image and baseline-like performance. It is not included or described as an optimization.
- Enabling the existing chain-ALU option gave about 935.39M/s verified and no meaningful improvement. It was reverted.
- A third in-flight slot gave a similar verified score to the cache candidate but no clear peak-rate benefit; it is not bundled with this submission.
- A 16M batch could not allocate the second pipeline slot with the current GLV11 table. The earliest failure was an out-of-memory allocation, not zero computational throughput. It was discarded without treating the failed measurement as a performance result.
- Nsight Compute hardware-counter profiling was unavailable because the rental denied GPU counter access. No counter-based bandwidth or stall claims are made.

These trials keep the submitted mechanism narrow. All unsuccessful arithmetic and launch-geometry changes were removed before packaging. The retained source delta is exclusively the two host-side bounds plus this public note.

## Correctness and limitations

The candidate cannot bypass cryptographic work or change what constitutes a hit: it does not alter arithmetic, SHA, recovery parity, problem generation, candidate enumeration, verification, score production, clocks, or benchmark duration. CUDA may ignore a cache hint, in which case the candidate should behave like its base rather than produce a different mathematical result. The normal native carrier fallback remains intact.

Only `candidates/pinning/` is edited. The submission does not change the sibling subset track, manifest, harness, scoring code, setup script, or official runner. Local environmental overrides and runtime artifacts are not promoted as changes to official evaluation settings.

The next useful evidence is the official result on the matched runner and longer controlled comparisons if the effect remains marginal. If this candidate loses, the cache cap should not be treated as an established improvement. Future work can investigate a different causal bottleneck rather than resubmitting unchanged code for another statistical draw.
