# Split the paired consumer into point, inverse, and finish kernels

Draft: awaiting the warm local qualification result. No official acceptance
or improvement is claimed by this document.

This candidate separates the promoted Subset paired consumer into three GPU
kernels, retaining the original candidate enumeration, paired SHA computation,
point formulas, speculative arithmetic, inverse-tree topology, hit gate, and
independent exact replay. Its purpose is to let the point front, inverse tree,
and finish run with separate register/shared-memory requirements. The cost is
an additional global-memory workspace and two kernel launches per batch.

The starting public source is commit
`7e95c40c99e57bded233ce57c7f453fbde9fd21c` of
[the QSB repository](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge).
The official Subset frontier observed during this work is 623,518,629 verified
candidates per second. All earlier author and license notices remain in the
candidate files. This work builds on the promoted composite; it does not claim
that the underlying field, point, SHA, or inverse algorithms originated here.

Primary development used GPT-6 Astra with high reasoning effort through Codex.
DeepSeek V4.1 Flash [1m], accessed through Claude Code with tools disabled,
performed one bounded arithmetic check of sanitized stage timings and resource
counts. Its numerical results were independently checked. It did not design
or implement this pipeline. An earlier public-source structural-audit call
timed out without a result; that call is not credited with findings. No
unpublished local source or credentials were sent to the helper.

## Measurement environment and scope

Local diagnostics use an RTX 3090 under CUDA 12.8. They are not RTX 4090 ranked
measurements. The harness artifact's static `RTX_4090` label is not hardware
detection. Local GPU work is serialized with the shared GPU lock; no clock,
power, harness, verifier, seed-generation, scoring, or sibling-track changes
are part of the candidate. CPU compilation may run concurrently with GPU work.

The ordinary build form is retained:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o candidate candidate.cu -lcrypto -lm
```

The unchanged `harness/run_benchmark.py` and `harness/gpu_wrap.py` perform
fixed-time diagnostics at N=24 with synthetic seed 2026092501. A research
wrapper saves the kernel stdout otherwise discarded by `gpu_wrap.py`; it
forwards the original behavior and leaves independent hit verification intact.
Research outputs live under the Subset editable tree. No local root-level
score file is supplied as official evidence or used to prefilter submission.

## Evidence used to choose the change

A CUDA-event profile of 24 ordinary batches put approximately 99.17% of GPU
kernel time in the monolithic digest consumer. Group construction, epoch
construction, first-block state generation, and exact replay together consumed
less than 1% on this local device. In particular, first-state generation was
about 0.25%, making producer-only changes an unattractive route to a 1% gain.

A separate sampled `clock64` diagnostic inside the consumer suggested roughly
62% combined point-front time, 19% SHA, 9% tree, and 10% tail. These are block
samples, not isolated-stage timings: compiler scheduling and the probes can
affect attribution, and the two point-front intervals are asymmetric. They
serve only as prioritization evidence, not as a ranked performance claim.

Several preceding experiments were rejected locally. Compacting first-state
slots from sixteen to eight passed exact compression and hit-set checks but
showed no meaningful gain. Persisting-cache hints for the fixed-base table
regressed. Monolithic 128-thread and 512-thread geometries also regressed;
the latter successfully used 96 KiB opt-in dynamic shared memory. Those
experiments and their reproduction material are preserved in the research
folders and are not enabled in this submission.

## Implementation

The split front kernel uses the same paired SHA helper and point-front helper
as the original consumer. For each candidate it writes sixteen 64-bit fields
and one validity byte. The fields use a structure-of-arrays layout so adjacent
lanes access adjacent words. Four fields initially contain the denominator;
the other twelve contain the finish numerators/state. An invalid denominator
is replaced with the multiplicative identity and its validity byte is cleared.

The inverse kernel uses the same 256-thread blocks, 128-window halves, A/B
pairing, leaf multiplication, and product-tree order as the promoted consumer.
For each thread, A and B are 128 candidate indices apart. Both denominator
factors are loaded before the inverse calculation. The two resulting inverses
replace the four denominator slots in the shared workspace. Each thread owns
both of its output records; blocks have disjoint candidate ranges, so these
writes cannot destroy another thread's input factors.

The tail kernel reads the existing twelve finish fields and the four inverse
fields, calls the unchanged finish and gate, and emits the unchanged tentative
record format. The original exact replay kernel then recomputes every tentative
hit before the host can publish it. All kernels use the same default stream,
which orders producer completion before consumers. The existing blocking host
readback and publication behavior remain in place.

The allocation covers the maximum batch; each launch indexes it with the actual
candidate count as its stride. Partial and odd epoch tails retain identity
factors for inactive entries. The front and tail skip invalid indices, while
all inverse-kernel lanes remain present for collective tree operations. Thus
an incomplete final batch cannot skip required barriers or report an inactive
candidate. There is no change to which omission combinations are searched.

The larger trial uses 131,072 physical paired-front blocks, yielding 524,288
epochs and 67,108,864 candidates per full launch. Its sixteen-field workspace
is 8 GiB, with a further 64 MiB for validity bytes. The smaller initial trial
used 32,768 blocks and a 2 GiB field workspace. These are batch-size choices,
not changes to the scoring window or mathematical workload.

The split code is isolated in `tests/gpu_epochs/split_pipeline.cuh`, with a
`QSB_SPLIT_PIPELINE` switch in the host/launch code. Trimmed split builds omit
the unused monolithic consumer from the module to avoid unnecessary JIT work.
Non-trimmed legacy builds retain that consumer. Compiling the organized
production form produced exactly the same emitted PTX as the larger tested
research form; the normalized PTX SHA256 is
`31140b45f9c7045542e6f8ad51353a15ff77f5255d883a9028d3cf848820354c`.

## Correctness and limits of the checks

The first split trial independently verified all 1,649 emitted hits. Its hit
set exactly equals the baseline hit set restricted to the same 106,168,320
completed epochs. The larger initial trial independently verified all 1,683
hits. A warm repeat is pending at the time of this draft. These checks cover
ordinary end-to-end behavior; they are not a proof that every inherited
speculative field operation is exact for every adversarial limb pattern.

During the geometry experiments, the inherited edge-case tree audit failed
both the new geometry and the original 256-thread baseline. The standalone
canonical multiplier and root inverse passed. Inspection confirmed that the
production tree uses the inherited speculative carry-truncated multiplier.
Exact-multiplier geometry audits passed at 128 and 512 threads with boundary
and partial-tail counts, including 8,191-element runs. This distinction is
recorded explicitly: production speculative arithmetic is not being relabeled
as universally exact. The pipeline preserves it and preserves the separate
exact replay before publication.

## Local performance record

| Variant | Verified hits | Wall seconds | Hit-derived M/s | Peak diagnostic M/s |
|---|---:|---:|---:|---:|
| initial split, 32,768 blocks | 1,649 | 60.1285 | 230.054242 | 246.7 |
| fresh promoted baseline | 1,738 | 60.1540 | 242.367982 | 243.6 |
| larger split, first use | 1,683 | 60.2085 | 234.485810 | 249.0 |

The first-use wall scores include startup/JIT and do not establish a win.
The larger trial's approximately 45-second accumulated search rate was
246.7 M/s versus the fresh baseline's 242.5 M/s at its nearest progress report.
Those counters motivate a warm repeat but are not a substitute for verified
throughput or the official long RTX 4090 evaluation. The final note must add
that repeat's result and the current pre-submission frontier before submission.

Reproduce the diagnostic builds and runs with the saved scripts under
`candidates/subset/research/split_pipeline/`. The independent benchmark uses
`--bench subset --N 24 --mode fixed_time --seconds 60 --max-rel-var none`
and the fixed diagnostic seed above. Run one GPU process at a time with
`flock -x /tmp/qsb-gpu.lock`. The official runner remains responsible for the
fresh unpredictable problem, full scoring window, verification, and verdict.
