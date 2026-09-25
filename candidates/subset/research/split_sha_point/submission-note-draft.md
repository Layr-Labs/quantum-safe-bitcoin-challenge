# Separate paired SHA, point preparation, inverse tree, and finish

This candidate separates the promoted Subset consumer into four kernels while
preserving candidate enumeration, the existing paired SHA helper, point
formulas, speculative field arithmetic, inverse-tree topology, hit gate, and
independent exact replay. The local warmed RTX 3090 comparison measured
248.735239 M/s against 243.494215 M/s for our preceding three-stage pipeline,
a 2.1524% improvement. All 1,784 candidate hits passed the unchanged independent
verifier, and the candidate contains all 1,747 hits from the comparison run.
These are local qualification results; no official score or promotion is
claimed. The official frontier at preparation remains 623,518,629 candidates/s.

The primary developer is GPT-6 Astra, high reasoning effort, using Codex.
DeepSeek V4.1 Flash [1m] through Claude Code, with tools disabled, performed
bounded arithmetic checks of sanitized timing/resource numbers and the two
rejected follow-up experiments. Those calculations were independently checked.
It did not design or implement the kernels. An earlier call using verified
public committed source timed out and contributed no findings. No unpublished
local source or credentials were sent to that helper. Passing emitted-hit
checks is evidence for those runs, not a proof of correctness for every input.

## Public starting point and official state

The public base is commit `7e95c40c99e57bded233ce57c7f453fbde9fd21c` in
[the QSB repository](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge).
Its Subset files are byte-identical to the promoted candidate at `9ac2515`:
`git diff 9ac2515 HEAD -- candidates/subset` was empty before development.
That promoted submission is `7aef224a-e3ff-43f9-9877-50cdbda3f653`, by
Akashneelesh, with score 623,518,629 candidates/s. Later base-branch commits
changed the sibling track. Existing author and license notices remain intact;
the inherited point, field, SHA, tree and replay algorithms are not claimed
as original contributions here.

Our preceding three-stage version was submitted as
`fd16dfa4-4db2-4aee-ab1f-519c564a53b8` and is still validating at the time of
this note. This submission is a further implementation improvement, not a
claim that the preceding version was accepted. Its original upload,
`f8448bc8-19b7-43a5-8f70-e285d1634aae`, was rejected before evaluation for
exceeding the expanded archive cap. That packaging problem was repaired by
removing rebuildable executables and preserving large run records compressed.
The present archive likewise contains source and compressed research evidence;
its expanded size is checked against the track's 8 MiB limit before upload.

## Environment and reproducible measurements

Local hardware is an RTX 3090 with CUDA 12.8. The official ranked runner is an
RTX 4090. The local harness artifact's static `RTX_4090` field is not hardware
detection and is not used to relabel these measurements. Local GPU experiments
are serialized using the shared GPU lock. The benchmark, verifier, seed rules,
scoring rules, setup, sibling track and official workflow are unchanged.

The normal source compilation remains:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o candidate candidate.cu -lcrypto -lm
```

The diagnostic scripts use the unchanged `harness/run_benchmark.py` and
`harness/gpu_wrap.py` with `--bench subset --N 24 --mode fixed_time --seconds 60
--max-rel-var none --seed 2026092501`. A research capture wrapper saves GPU
stdout while delegating the ordinary harness behavior. Independent CPU hit
verification is unchanged. No diagnostic root `score-subset.json` is submitted,
and no local score is passed as an official claimed result.

Current trial scripts and raw evidence are in
`candidates/subset/research/split_sha_point/`. `trial.py` builds and runs named
variants; `run_pair.py` runs the warm candidate and the preceding split baseline
consecutively under one externally held lock:

```sh
flock -x /tmp/qsb-gpu.lock python3 candidates/subset/research/split_sha_point/run_pair.py
```

Large historical `run.json` files are retained losslessly as `run.json.gz`;
decompress them before using readers expecting the uncompressed name. Removed
executables are rebuildable and their sizes and SHA256 hashes are recorded.
The archive may package older research directories as a verified compressed
research archive; extract that archive at its recorded location to reproduce
those older experiments. Active production source remains directly available.

## Why the consumer was split

An initial CUDA-event profile of 24 ordinary batches put about 99.17% of kernel
time in the monolithic consumer. First-state generation was about 0.25%, so
producer-only changes were unlikely to clear a 1% improvement requirement.
A separate sampled `clock64` diagnostic suggested roughly 62% combined point
front, 19% SHA, 9% tree, and 10% finish. Those block samples include scheduling
and instrumentation effects; the asymmetric point-front timings were not
interpreted as exact isolated operation costs.

The first useful change separated point preparation, the inverse tree, and
the finish. On the local device, its warm run verified 1,772 hits at
246.977743 M/s against 1,738 hits and 242.367982 M/s for a fresh promoted
monolithic baseline (+1.90%). Completed-work counters improved by about 1.87%.
The warm candidate's hit set matched the baseline over the common completed
range. This qualified the three-stage candidate for official evaluation.

The three-stage front still coupled paired SHA and two point calculations in
one kernel with 128 registers. Simply forcing its launch bound from (256,2)
to (256,3) reduced the register count to 80 but introduced a 152-byte stack and
regressed steady throughput from about 246.0 to 168.2 M/s. Its 1,135 emitted
hits verified, but it failed performance qualification. A streaming-workspace
cache experiment also failed: 243.579542 M/s versus 245.078939 M/s after warmup,
about -0.61%, with all 1,747 warm hits verified. Neither experiment is enabled.

The present version separates SHA from point preparation, shortening the live
state within each kernel instead of forcing a smaller register budget. Paired
SHA remains paired and uses its existing helper. Each point thread handles one
candidate and no longer holds another candidate's SHA value across a point
calculation. The ordinary point kernel compiles with 118 registers and no stack;
the SHA kernel uses 72 registers and no stack. A compact point variant still
required a 144-byte stack at 80 registers and was not selected for GPU trials.

## Implementation and dependency ordering

`subset.cu` enables `QSB_SPLIT_PIPELINE=1` and selects
`ZLAB_LAUNCH_BLOCKS=131072`. The split definitions are in
`tests/gpu_epochs/split_pipeline.cuh`; the allocation and launch selection are
in `tests/gpu_epochs/tree.cu`. The promoted arithmetic headers and window/tree
schedule remain unchanged. Disabling the split switch restores the monolithic
path; restoring the prior launch count gives the prior baseline geometry.

The workspace has sixteen 64-bit fields per candidate in a structure-of-arrays
layout and one validity byte per candidate. Adjacent lanes access adjacent
words. For a full launch, 131,072 paired SHA blocks represent 524,288 epochs and
67,108,864 candidates. The field workspace is 8 GiB and validity adds 64 MiB.
The SHA/point refinement adds no allocation compared with the three-stage
version: it temporarily reuses the first four fields, which will later hold
denominators and then inverse denominators.

1. The SHA kernel uses the original epoch-pair/lane mapping and
   `qsb_pair_epoch_z_value`. It writes each four-limb z into the first four
   fields. It emits A for each active epoch and B only when the second epoch
   exists. An absent B uses A's first-state pointer for the inherited paired
   helper but its result is not stored or consumed.
2. The point kernel has one thread per candidate. It reads all four z limbs
   and passes them by value to the unchanged point-front helper. It then writes
   the same sixteen-field front result and validity byte as the preceding
   pipeline. Invalid denominators become multiplicative identity, with the
   remaining fields zero and validity cleared.
3. The inverse kernel preserves the original 256-thread topology and A/B
   pairing. It loads both denominators before multiplying their leaf product,
   runs the same block inverse tree, and overwrites denominator fields with
   the two inverses. Inactive candidates contribute identity factors; all
   required threads participate in the collective tree operations.
4. The finish kernel reads twelve finish fields and four inverse fields, skips
   invalid/out-of-range candidates, calls the unchanged tail/gate, and emits
   the original tentative record format. The existing exact replay kernel
   recomputes each tentative hit before host publication.

These launches use the same default stream. SHA completes before point reads;
point completes before tree reads; tree completes before finish reads. Each
point thread loads its own four z inputs before overwriting only its own record.
Each inverse thread owns its A/B records, and different blocks own disjoint
candidate ranges. Neither reuse creates a cross-thread overwrite of live input.
The final point and finish blocks guard indices before accessing the workspace.
Actual candidate count is the stride for partial batches; allocation covers
the maximum batch. The inverse kernel retains identity factors for inactive
entries and does not skip required barriers.

The extra SHA stage costs one more launch and four 64-bit stores plus four
64-bit loads per candidate. The complete split also incurs the original
front/tree/finish workspace traffic. These costs are deliberate tradeoffs for
independent resource usage, not changes to the search or scoring workload.

## Correctness evidence and inherited limits

All 1,694 hits from the first SHA/point trial passed the unchanged verifier.
The warm candidate verified all 1,784 hits. The fresh three-stage baseline
verified all 1,747, and every baseline hit, including its recovery identifier,
appears identically in the candidate set. Earlier three-stage runs verified
1,649, 1,683 and 1,772 hits, including exact common-prefix comparisons against
the promoted baseline. These tests support ordinary end-to-end behavior for
the tested problem; they do not prove exhaustive correctness for every input.

Earlier tree geometry audits exposed an inherited distinction: the production
carry-truncated speculative multiplier fails some adversarial limb tests in
both altered geometry and the original 256-thread baseline. The canonical
multiplier and root inverse passed; geometry tests using exact multiplication
passed at 128 and 512 threads with partial and boundary counts, including
8,191 elements. The current change preserves the promoted speculative
arithmetic, inverse pairing and tree order, and preserves exact replay before
publication. It does not describe that speculative arithmetic as universally
exact, or claim that emitted-hit verification proves absence of all misses.

## Current local result and decision

| Variant | Verified hits | Wall seconds | Verified-hit M/s |
|---|---:|---:|---:|
| SHA/point split, first use | 1,694 | 60.1922 | 236.082224 |
| SHA/point split, warm | 1,784 | 60.1655 | 248.735239 |
| Fresh three-stage split baseline | 1,747 | 60.1858 | 243.494215 |

The first-use wall result includes startup/JIT and is not the qualification
comparison. Its accumulated diagnostic search rate was about 247.2 M/s at
46 seconds. The consecutive warm comparison gives +2.1524% from verified
hits. This gain is relative to the earlier three-stage version; it is not
multiplied by the older +1.90% experiment to invent a combined official gain.
The unchanged official frontier is the target, and the RTX 4090 evaluation is
authoritative. Short local measurements retain thermal, startup and sampling
limits. No acceptance or promotion is claimed before that evaluation.

The final production build is checked against the tested research build by
normalized PTX SHA256. Source hashes, PTX evidence, resource reports, raw hit
artifacts, rejected experiments and remote verdict tracking are retained under
the editable Subset tree. The next decision is the official verdict: an
accepted/promoted score above the prior frontier constitutes success; a
rejection or insufficient score requires investigation and further work.
