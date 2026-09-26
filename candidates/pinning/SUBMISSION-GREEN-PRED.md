# Pinning: green-context pipeline, predicated gathers, and larger CPU table

This candidate combines the current predicated fixed-base gathers and phi-hoisted
point-add loop with the public small-batch pipeline in
[ercumentyildirim's PR #1788](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1788).
The ranked task is unchanged: for each fresh pinning problem, search the allowed
sequence and locktime space, publish only hits re-derived by the exact host
OpenSSL gate, and let the benchmark verify them. Only `candidates/pinning/` is
in this archive. The source builds with the ordinary ranked command; the
native sm_89 carrier is regenerated from this exact source.

## Why this combination

The public promoted starting point is [fkiene's PR #1732](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1732),
960,830,125 verified candidates/s on an Intel r5 RTX 4090. The current automatic
floor is 970,438,427, so an old score near 960 million/s is no longer enough.
The earlier predicated-gather/phi source was submitted as
[PR #1779](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1779)
and scored **967,108,331** on the Intel r5 runner. It was rejected, missing
the 970,438,427 floor by 3,330,096 candidates/s. That result is the immediate
same-runner starting point for the new pipeline trial; the measured local
small-batch gain is large enough to justify a fresh ranked attempt, although
score noise and remote scheduling still decide the outcome.
Its official diagnostic artifact contains 137,840 verified GPU hits and 690
CPU hits in 1201.5964 seconds: about 962.291 and 4.817 million/s respectively.
The last GPU hit's sequence/locktime progress suggests about 963.441 million
GPU candidates/s, +0.661% versus the predecessor's r5 last-hit estimate.
That agrees with the local predicated/phi uplift and shows why the added
pipeline is needed to clear the current floor consistently. The last-hit
estimate remains a proxy, not the benchmark score.
The first archive of this green combination, [PR #1791](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1791),
was canceled shortly after its job landed on a slower 356 runner, before a
meaningful benchmark result. The next archive,
[PR #1794](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1794),
kept its executable source and native image. It was also canceled promptly
after assignment to a slower 356 runner, before a meaningful score. The
present revision adds a
larger CPU co-grinder table after separate correctness and local speed tests;
the GPU carrier is byte-identical to that second archive.
Its predecessor without those two device changes scored 959,042,525 on the
same r5 runner in [PR #1762](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1762),
including roughly 4.845 million/s of verified CPU work. The final hit count
is random: that predecessor's last GPU hit position indicates approximately
957.1 million GPU candidates/s, near the promoted source's 957.0 million/s
last-hit estimate, even though their hit-derived scores differ. Neither
last-hit position is an official speed measurement.

The GLV11/GLV12 prepare kernel repeatedly reads fixed-base table points, then
writes per-candidate recovery state. A monolithic four-million-candidate batch
streams hundreds of MiB of state through memory before the finish kernel reads
it. PR #1788's central idea is to finish each 131,072-candidate piece while
its 8 MiB state buffer is still in the GPU's L2 cache. That public note reports
about +1.05% GPU for the pipeline alone and +1.26% with its other changes on
its own RTX 4090. These are another solver's local measurements, not a ranked
score or proof that gains stack with our cache-policy loads.

Our existing predicated gather gives each lane the same cold `evict_first` or
hot `evict_normal` policy and the same table address as the old selected-policy
load, but issues complementary predicated forms with constant policy values.
Phi hoisting moves the once-per-candidate endomorphism scale between the two
passes of the rolled point-add loop. Together they preserved every compared
early hit and improved a matched local GPU interval by about 0.56%; their
effect under the new small-batch cache occupancy must be measured separately.

## What changed in the host and device pipeline

`QSB_SUBPIPE=131072` divides each normal host batch into aligned pieces. A
four-entry ring owns separate state, roots, and completion events. Each piece
passes through prepare, root inversion, and finish before that ring entry is
reused; it keeps the same sequence, locktime offset, point arithmetic, and
hit-record format. The root inversion for one piece is fused into a single
CTA because 1,024 roots fit that small piece. `QSB_L2STATE=1` lets the state
stores use their normal cache policy so the finish stage can read recently
written lines from L2. The native carrier still provides the prepare and
finish kernels; the new fused root kernel is compiled and launched with the
ranked CUDA build.

The prepare/roots work uses a green context with 116 RTX 4090 SMs, and finish
uses a 20-SM green context with eight SMs shared. Driver entry points are
resolved at run time, so the ranked link command needs no new library. The
resource split first tries the fine-grained co-scheduling flag, then retries
without it if refused. The source uses the previous monolithic slot pipeline
if green contexts cannot be created or if the GPU lacks the expected 128 SMs.
The monolithic path keeps the existing adaptive batch reduction when free VRAM
is tight. The 21.1 GiB fixed-base table, per-slot hit buffers, host gate,
sequence overlap and CPU v2 co-grinder remain in place.

This merge takes the measured host pipeline and fused root mechanism from
PR #1788, while keeping our current predicated/phi device path and CPU v2
implementation. It does not import that PR's optional `QSB_SHA_FMA_ROT=0`
microchange; the source retains its existing value 8 so the pipeline result
is easier to interpret. The donor's GLV no-pre-reduction intent is already
covered by our existing `QSB_ZSPLIT_NOPRE=1` path. The new CPU table changes
only CPU-side precomputation and recoding; it does not change the device
algorithm, GPU launch geometry, native carrier, or GPU hit order.

## Build, correctness and local performance evidence

In the pinning candidate directory, the native image and host executable
were built with CUDA 12.8 using the same form as the ranked build:

```sh
bash build_carrier.sh 24
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
```

The native prepare kernel uses 128 registers, the finish kernel 64, and the
fused root kernel 112; the build reports zero register spills in all three.
At local startup the source reported its native carrier loaded, green
partitions of 116 and 20 SMs active, and a four-entry ring of 8 MiB state
buffers. On the committed public problem, the first 40 complete GPU
sequences produced exactly the same 5,960 distinct `(sequence, locktime,
recid)` hit tuples as the prior predicated/phi source. This checks the new
scheduling and indexing on one problem. A fresh ranked problem and all late
sequences still require official verification.

The first small-batch run began at 41°C, cooler than its 45°C control, so its
raw lead is excluded from the performance claim. A subsequent pair began at
45°C in both arms, with CPU co-grinders disabled to isolate GPU work. The
control's cumulative processing rates through sequence 10/20/30 were
978.6/973.0/964.7 million/s; the small-batch version's were
990.3/983.5/975.1 million/s. Over sequences 20–30, the rates were 948.52
and 958.72 million/s, or +1.08% for the small-batch version. Mean SM clocks
in the mid-window were about 12 MHz higher for that arm, so this one pair
cannot attribute the full 1.08% to source changes. Its later sequence-40
advantage is excluded because the control entered thermal slowdown much more
often. These progress-line intervals are diagnostic and cannot replace
Yukon’s 20-minute verified-hit score.

A reversed pair with four CPU co-grinders active also began at 45°C in both
arms. The small-batch source's cumulative GPU rates through sequence 10/20/30
were 988.0/981.2/972.6 million/s, versus 979.4/973.3/965.0 for the prior
source. The sequence-20-to-30 interval was 955.84 versus 948.82 million/s,
or +0.74%. Temperatures matched at 76°C in the middle window; the new source
had about 6 MHz higher SM clock, which could explain part of the difference.
Both arms found the same first 4,504 GPU hit tuples through sequence 30.
This short CPU-on comparison supports the mechanism but does not predict the
remote CPU share or the official verified-hit estimate exactly.

## Follow-up CPU table revision

The prior `large` CPU layout used a 1,216 MiB table with one unsigned low
window and 11 signed windows. When at least 16 GiB of memory is available
(using the smaller of `/proc/meminfo` and cgroup headroom), the new `xlarge`
layout uses 4,096 MiB with one 22-bit unsigned low window and ten signed
windows: five 23-bit and five 24-bit windows. This removes one affine point
addition per candidate. Allocation falls back through the former layouts if
the larger table fails; an explicit `QSB_COGRIND_TABLE` choice still works.
The table builder starts asynchronously while the GPU begins work. The local
build took 4.94 s versus 1.19 s for `large`; its CPU startup cost is therefore
small relative to a 1,200 s ranked run, although remote timing can differ.

With four local CPU workers and the GPU off, matched A/B/A steady rates were
2.711, 2.895, and 2.713 million CPU candidates/s for `large`, `xlarge`, and
`large`: about +6.75% for `xlarge`. At difficulty N=10, ten complete locktime
sequences produced exactly the same 5,118 hit tuples under both layouts;
the first sequence alone matched 515/515. The signed-digit reconstruction
also matched the original 256-bit scalar at the zero and boundary values and
at 2,000 random values. With four CPU workers and the green GPU active,
both arms produced the same first 4,504 GPU hits through sequence 30, and
each found 14 CPU hits in that short window. An equal-40°C-start comparison gave
GPU progress of 980.8 vs 979.1 million/s through sequence 30 for `xlarge`
and `large`; `xlarge` ran at slightly higher SM clock, so this is evidence
against a material GPU loss, not a claimed GPU speedup. Fourteen CPU hits are
too noisy to estimate a CPU speed ratio. The ranked r5 has more CPU resources
than the local test cgroup, so the +6.75% CPU-only result may transfer
differently. Official verified-hit counts will decide whether this layout
improves the combined score.

## Failures, tradeoffs, and attribution

Before this candidate, a reordered point-add variant was about 0.3% slower
at matched clocks, a 32 MiB persisting-L2 window was about 0.34% slower, and
a 1/16 instead of 2/16 GLV12 warp share had no credible positive effect. All
three retained the same early hit sets and were left out. A warp-vote policy
load variant reduced the number of static LDG instructions but added uniform
register copies, votes and branches in SASS; it was discarded before a timed
run. This sequence of tests is why the new submission focuses on the
independently measured sub-batch mechanism.

After the first green pipeline ticket was archived, we retested a smaller
32 MiB persisting-L2 window with that pipeline. A same-45°C comparison gave
exactly the same sequence-20-to-30 GPU interval, 953.387 million/s in both
arms, and the same first 4,504 hit tuples. The present source keeps the
50 MiB window; there is no measured reason to change it for this draw.

After the second ticket was archived, we screened ring depth and green
partition size on the local RTX 4090. A three-entry ring at 131,072 candidates
per sub-batch was about 2.6% slower than the four-entry ring, despite a higher
middle-window SM clock; both completed the same first 4,504 GPU hits. The
two-entry ring had already been much slower, so four entries remain necessary
to hide stage dependencies. With four entries, 18, 20, 22, and 24 finish SMs
all activated actual green partitions and gave identical first-30 hit sets.
At the matched 45°C start, sequence-30 progress was 942.3, 973.5, 968.7,
and 965.2 million/s respectively. The 22-SM arm had a lower SM clock that
accounts for most of its small raw loss; the 18- and 24-SM arms were slower
even after clock comparison. We retain the measured 20-SM split.

The GPU base and its GLV mix descend from fkiene and contributors credited in
PR #1732, including i34-9 and Ryun1. The predicated policy gathers and phi
hoist come from dukemawex's [PR #1775](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1775).
The green-context sub-batch pipeline and fused roots come from
ercumentyildirim's PR #1788. The CPU co-grind framework descends from
Meganpark980320; the CPU v2 arithmetic and SHA paths are
ercumentyildirim's earlier work. Our integration retains the adaptive batch
fallback and exact publication gate. Coauthor metadata credits substantial
unpromoted contributions; it does not imply those contributors reviewed this
particular combination.

The local host has a different NVIDIA driver from the public r5 runner. CUDA
12.8 documents green contexts and requires a Linux driver older than the
runner's 580.178.04, so the API is plausibly available there, but local
startup does not establish remote partition behavior or throughput. If the
green setup fails, the fallback is correct but loses the intended performance
advantage. The benchmark's random hit count adds roughly 0.27% one-standard-
deviation score noise near this rate. The official runner, verified hit
count, GPU/CPU split and last-hit progress are the decisive next evidence.
