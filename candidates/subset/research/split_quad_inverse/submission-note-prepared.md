# Split consumer stages and amortize inverse trees over four factors

This candidate separates paired SHA, point preparation, inversion, and finish,
and groups four denominators per inverse-tree leaf. It preserves candidate
enumeration, the existing SHA and point helpers, the inherited field arithmetic,
the finish gate, and independent exact replay before publication. The tree still
has 256 threads, but its leaves cover four candidates per thread instead of two.
That reassociation is intentional and is tested separately from kernel splitting.

The latest local RTX 3090 comparison measured 254.586866 M/s against
251.017691 M/s for our preceding four-stage, two-factor pipeline: +1.4219% from
verified hits and +1.3575% in completed candidates. All 1,826 candidate hits
verified, and all 1,800 baseline hits match within the common completed epoch
range. These are local qualification results, not official RTX 4090 results.
No acceptance or promotion is claimed. The official frontier at preparation is
623,518,629 candidates/s.

Primary development used GPT-6 Astra, high reasoning effort, through Codex.
DeepSeek V4.1 Flash [1m] through Claude Code, with tools disabled, performed
bounded arithmetic checks of sanitized stage/resource numbers and rejected
experiment metrics. Those calculations were independently checked. It did not
design or implement the kernels. An earlier call using verified public committed
source timed out and contributed no findings. No unpublished local source or
credentials were sent to that helper.

## Public base and preceding work

The base is `7e95c40c99e57bded233ce57c7f453fbde9fd21c` in
[the QSB repository](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge).
Its Subset files are byte-identical to the promoted candidate `9ac2515`:
`git diff 9ac2515 HEAD -- candidates/subset` was empty before development.
The promoted submission is `7aef224a-e3ff-43f9-9877-50cdbda3f653`, by
Akashneelesh, scoring 623,518,629 candidates/s. Existing author and license
notices remain intact; the inherited field, point, SHA, inverse and replay
algorithms are not claimed as original contributions here.

Our earlier three-stage pipeline is submission
`fd16dfa4-4db2-4aee-ab1f-519c564a53b8`, still validating at preparation time.
This note does not describe that submission as accepted. Its first upload,
`f8448bc8-19b7-43a5-8f70-e285d1634aae`, was rejected before evaluation because
the archive exceeded the expanded 8 MiB cap. Rebuildable binaries were removed
and historical evidence compressed to repair packaging.

The local development sequence was:

* Separate the monolithic consumer into front, inverse and finish. The warm
  version measured 246.977743 versus 242.367982 M/s for a fresh promoted
  monolithic baseline (+1.90%), with about +1.87% completed work and matching
  common-prefix hits. That version was submitted for official evaluation.
* Separate paired SHA from point preparation. A warm comparison measured
  248.735239 versus 243.494215 M/s (+2.15%), with +1.86% completed work and
  identical hits in the common 112,721,920-epoch prefix. This four-stage version
  is the baseline for the current experiment, not a new official frontier.
* Group four denominators per thread before the inverse tree. The current
  comparison below isolates this change against the preceding four-stage form.

Those percentages come from different local comparisons and are not multiplied
together to invent an official combined gain. The RTX 4090 evaluation remains
authoritative.

## Environment, measurements, and reproduction

Local diagnostics use an RTX 3090 with CUDA 12.8. The harness artifact's static
`RTX_4090` label is not hardware detection. GPU runs are serialized with
`flock -x /tmp/qsb-gpu.lock`; CPU compilation and analysis may proceed separately.
The benchmark, verifier, seed rules, scoring, setup, sibling track, and official
workflow remain unchanged. No diagnostic root `score-subset.json` is supplied
as official evidence or used as a claimed score.

The normal production build remains:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/subset/subset candidates/subset/subset.cu -lcrypto -lm
```

Local scripts invoke the unchanged `harness/run_benchmark.py` and
`harness/gpu_wrap.py` with `--bench subset --N 24 --mode fixed_time --seconds 60
--max-rel-var none --seed 2026092501`. A research capture wrapper retains kernel
stdout while forwarding ordinary harness behavior. Independent CPU hit
verification remains intact.

Current material is under `candidates/subset/research/split_quad_inverse/`.
`trial.py build` builds the production-arithmetic baseline and candidate. The
separate `audit.cu` uses exact multiplication and OpenSSL as an independent
oracle; build it with `nvcc -O3 -DQSB_ZEROS_N=24 ... -lcrypto -lm`.
`run_qualified.py` first runs that audit and stops on failure; only a passing
audit permits the baseline and candidate diagnostics. Run it under the GPU lock.
Then run `trial.py candidate --label=-c` under the same lock for the warm repeat.
The baseline-a, first candidate-b, and warm candidate-c artifacts are retained.

Large run records are preserved losslessly as `run.json.gz`; decompress before
using readers that expect the original name. Older experiments are in
`research/history-before-four-stage.tar.xz`; extract it from the research
directory to restore their paths. Its manifest records each materialized file
hash. Every archived member was verified before removing the original copy.
Rebuildable executables are removed after recording their hashes. Sources,
logs, hit records, failed audits, and decisions are preserved.

## Why these stages and this batch size

A CUDA-event profile of 24 ordinary batches attributed about 99.17% of kernel
time to the original monolithic consumer; first-state generation was only
about 0.25%. A separate sampled clock diagnostic suggested approximately 62%
combined point-front time, 19% SHA, 9% tree, and 10% finish. Those are sampled,
instrumented block intervals, not exact isolated operation costs. They guided
prioritization rather than serving as a ranked result.

Splitting lets SHA, point preparation, the collective inverse tree, and finish
use different register/shared-memory budgets. In the current compiled form,
SHA uses 72 registers, point preparation 118, four-factor inversion 128, and
finish 76. All report zero stack. The inverse stage uses 24 KiB of shared
memory; the other split stages use none. The earlier two-factor inverse used
90 registers with the same shared allocation. Thus the new grouping trades a
larger register footprint for more candidates per collective tree and root
inverse.

The default has 128 windows, 256 threads, and 131,072 paired SHA blocks per
full batch: 524,288 epochs and 67,108,864 candidates. The sixteen-word workspace
is 8 GiB, plus 64 MiB for validity. Point and finish use one thread per candidate.
The four-factor inverse uses 65,536 blocks, covering eight epochs per block.
Production derives that epoch count as `4 * (QSB_SE_BLOCK / QSB_SE_WINDOWS)`;
the tested default equals eight. The GPU audit and performance qualification
cover the 128-window configuration.

Splitting adds global workspace traffic and launches compared with the
monolithic baseline. The four-factor refinement changes no workspace allocation
or number of pipeline stages. It amortizes tree synchronization and root inverse
work over twice as many candidates per block. Merely reducing launch overhead
or reporting fewer candidates is not the intended gain.

## Implementation and data dependencies

`subset.cu` enables `QSB_SPLIT_PIPELINE=1` and selects
`ZLAB_LAUNCH_BLOCKS=131072`. The split kernels are in
`tests/gpu_epochs/split_pipeline.cuh`; allocation and launch selection are in
`tests/gpu_epochs/tree.cu`. Promoted arithmetic and schedule headers are intact.
Disabling the split switch restores the monolithic path; its prior launch count
restores the preceding promoted geometry.

The workspace is sixteen 64-bit structure-of-arrays fields per candidate, with
one validity byte. Adjacent lanes access adjacent words. The first four fields
are reused in order for SHA z, point denominator, and inverse denominator.
The remaining twelve contain finish state.

1. Paired SHA preserves the original epoch-pair/lane mapping and
   `qsb_pair_epoch_z_value`. It stores four z limbs for each active candidate.
   An absent second epoch is not stored or consumed.
2. One point thread reads its own four z limbs before calling the unchanged
   front helper and overwriting its own sixteen fields. Invalid denominators
   become multiplicative identity and have validity cleared; inactive entries
   are not accessed.
3. Each inverse thread owns four records separated by the window count. With
   factors a,b,c,d, it forms left=a*b, right=c*d, and leaf=left*right. The same
   256-thread block inverse tree returns 1/leaf. Multiplying that by right and
   left yields 1/(a*b) and 1/(c*d); multiplication by sibling factors produces
   1/a, 1/b, 1/c, and 1/d. Absent factors are identity. All lanes still enter the
   collective tree. Only present records receive output.
4. Finish reads the existing twelve state fields and four inverse fields,
   invokes the unchanged gate, and emits the original tentative record format.
   The existing exact replay kernel recomputes tentative hits before the host
   can publish them.

Every stage uses the same default stream, so producers complete before
consumers read. All four denominators are loaded before a thread writes any
inverse. No other thread owns or reads those records in this inverse launch;
blocks cover disjoint epoch groups. Global overwrites therefore cannot destroy
another thread's live input. Partial batches use actual candidate count as
stride; allocations cover the maximum batch. Point and finish guard out-of-range
indices, while inverse padding preserves barrier participation. The host readback
and publication path are unchanged.

## Correctness checks and arithmetic limits

The new grouping was first audited with `QSB_SHORT_CARRY3=0` and
`QSB_K2S_PARITY_WINDOW=0`, enabling exact multiplication. It covers epoch counts
1,2,3,4,5,7,8,9,15,16,17,31,32,33,65: 31,744 factors in total, including unit
factors, p-1, p-2, limb boundaries, deterministic wide inputs, and partial blocks.
Both the original two-factor kernel and new four-factor kernel are compared
independently with OpenSSL modular inverses.

The initial test incorrectly required canonical limb representation and reported
1,874 raw differences on unit-factor cases. Its original source and failure log
are retained. Diagnostic output showed both kernels returning p+1 rather than
1 in the printed unit cases; these represent the same field element. Correcting
the oracle to compare values modulo p produces zero modular failures for either
kernel over all 31,744 factors. This is a correction to the research oracle,
not a change to the benchmark verifier or production arithmetic.

The production candidate retains the inherited carry-truncated speculative
multiplier. Earlier edge-case audits failed both the original baseline and
alternative geometries in that mode; exact-multiplier audits passed. Changing
factor grouping is not claimed to make speculative arithmetic universally
exact. Exact replay still gates every published hit. The normal production
runs also independently verify every emitted hit and compare hit identities
over their common completed range. These checks support the tested workload;
they do not prove absence of every possible miss on adversarial inputs.

The first four-factor run verified all 1,729 hits and exactly matched the
baseline over 111,673,344 completed epochs. The warm run verified all 1,826 hits.
Its first 115,867,648 completed epochs contain exactly the baseline's 1,800 hits,
including recovery identifiers, with no missing or extra records. Completed
counts come from the kernel's retained shutdown summary, not extrapolated peak
rates. Earlier split stages also passed independent verification and matching
common-prefix checks.

The normalized PTX SHA256 of both the tested research binary and final production
binary is `6924c12d9c27e1240c044c75a1a3ec5e66f7025e660ac90bd150666e14362417`.
The host launch expression was separately reviewed: the derived default divisor
is eight and positive epoch counts are bounded by 524,288, so ceiling division
matches the tested expression without overflow. Source hashes and this comparison
are retained in the manifest and research evidence.

## Current local performance and rejected alternatives

| Variant | Verified hits | Wall seconds | Verified-hit M/s | Completed candidates |
|---|---:|---:|---:|---:|
| Four-stage, two-factor baseline | 1,800 | 60.1531 | 251.017691 | 14,831,058,944 |
| Four-factor inverse, first use | 1,729 | 60.1711 | 241.044199 | 14,294,188,032 |
| Four-factor inverse, warm | 1,826 | 60.1665 | 254.586866 | 15,032,385,536 |

The warm comparison gives +1.4219% verified-hit throughput and +1.3575% completed
work. Its diagnostic accumulated rate near 45 seconds was 253.8 M/s versus
251.0 M/s for the baseline. The first-use wall result includes startup/JIT and
is not treated as a win. Short local runs retain startup, thermal and sampling
limits; the official long RTX 4090 test must establish the ranked result.

Other hypotheses were rejected and are not enabled: compact first-state slots
showed no meaningful gain; persisting table-cache hints regressed; monolithic
128/512-thread geometries regressed; streaming split-workspace hints lost about
0.61%; forcing the paired front to 80 registers introduced a 152-byte stack and
lost about 31.6% in steady diagnostics. Fusing inverse and finish had identical
completed work and identical 1,765-hit sets, with only a 0.029% wall difference,
so it did not qualify. These outcomes and reproduction sources remain available.

The next decision is the official verdict. A local improvement or a validating
submission is not completion. Acceptance/promotion and a verified score above
the prior official frontier must be checked on the public QSB page or its linked
evaluation; otherwise the error or insufficient gain requires further work.
