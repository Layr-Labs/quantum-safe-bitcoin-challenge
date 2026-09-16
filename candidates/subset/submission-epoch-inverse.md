# Subset GPU epochs with a four-lane inverse tree

## Context, objective and attribution

This candidate targets a large architectural gain over the production
Quantum Safe Bitcoin subset frontier. The initial promoted baseline was
Meganpark980320's `b7bdbf1c-a819-4c02-ad21-660c1f16f0da`, landed at
`a040c21c12306610bf53c718304ad1c93c360b65`, with 126688029 verified candidates
per second. The final pre-submit scan found their projective follow-up
`c691d3da-07dd-4372-a6bc-1d39d3d2035a` promoted at `df765fe`, raising the
current bar to 129574439. This updated number is the submission baseline.
The ranked contract is RTX 4090, N=24, fixed_time,
1200 seconds, both recovery IDs, with the ordinary verifier and score logic.
The production minimum improvement is 100 basis points.

This work was performed with GPT 6 Astra, effort xhigh, through Codex. The
workstation is an Apple Silicon Mac with no NVIDIA GPU and no CUDA compiler.
The source starts from the public unpromoted subset port by nullforest8200:
submission `873ed724-9815-4e13-a02f-072f21e3f992`, source commit
[`b733504088873a409baff306a2633ec63e6e5762`](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/b733504088873a409baff306a2633ec63e6e5762).
That substantial unpromoted contribution is credited with `--coauthors
@nullforest8200`. The source already contains the main GPU-epoch architecture
and arithmetic optimizations. We claim the four-lane inverse-tree change and
the new host reference audit, not authorship of the inherited pipeline.

The imported public note reports that its implementation is a port of earlier
development promotion `a33f042b-78d5-4ef6-b125-70c1618373c4`, commit `7c1e716`,
which reportedly achieved 442824991 verified candidates per second. The old
development repository and Yukon benchmark were unavailable during this work,
so this historical number is attributed to that public note, not independently
certified here and not used as a claimed local score. The source is available
in the current public candidate and was inspected directly. Its architecture
provides a credible large-step candidate even without treating a historical
number as a prediction for the current runner.

The retained `TREE_INVERSE.md` is the upstream historical technical note. It
credits odinfree's unpromoted GPU-epoch precursor `0db6e203` and earlier
promoted work for signed fixed-base tables, XYZZ accumulators, runtime scalar
folding, canonical inverse-tree multiplication, specialized squaring, shared
SHA schedules, and shared-denominator recovery. Those credits are preserved.
Its old measurements refer to its original tree, not to our modified helper.

## Research selection

The initial scan examined the current promoted subset prefix cache and the
pending projective normalization improvement from Meganpark980320,
`c691d3da-07dd-4372-a6bc-1d39d3d2035a`, source `56fc682`. It also examined
jacklightChen's pending compact runtime recovery table on the pinning track,
`783bdbdf-d836-4444-820c-ce76c435b4b7`, source `5200fbc`. An initial subset
prototype combined those ideas and passed CPU algebra and SHA checks, but had
no GPU measurement. Before submitting, another frontier scan revealed the
more advanced GPU-epoch port. That prototype was superseded, and none of its
code is in this candidate's production include closure. A smaller table alone
did not justify discarding the port's established XYZZ and batch-inverse design.

The selected source already folds `neg_r_inv` into its fixed base, retains
projective coordinates, eliminates per-candidate scalar modular multiplication,
uses a signed 32 MiB table, and shares one field inversion across a block.
It also computes repeated SHA prefixes once per epoch and shares schedules for
the remaining fixed suffix. These mechanisms address substantially larger
costs than launch-size or instruction-level tuning of the production baseline.

The new inverse helper extends the source's work-efficient product tree. Its
lowest two levels operate on four neighboring lanes using warp shuffles. Only
the upper tree is placed in shared memory. This changes the communication
layout without increasing the number of field multiplications or inversions.
Its benefit beyond the imported source is a hypothesis for remote evaluation;
the large-step rationale relative to 126688029 includes the whole imported
pipeline, not an unsupported claim that four fewer barriers alone yield a
particular percentage improvement. The final scan also read `7fdc8ed`
(welttowelt), `b2ede54` (fkiene), `dfb66e9` (newjordan), `ef833e4`
(pepedesigner), and `54918fb` (anamdongparkjinhyeong). Their notes describe
projective recovery, broadcast/read-only data, larger batches and reduced
summary-file synchronization. Projective recovery and amortized launch/I/O
work are already present in the selected architecture; its scheduled hash
uses shared or constant schedules. Their smaller resource-tuning changes
were not blindly grafted onto this different kernel. No source from those
five candidates was imported.

## Implementation and correctness argument

The production entry point remains the one-line include of
`tests/gpu_epochs/tree.cu`. Relative to imported commit `b733504`, every imported
file is byte-identical except `tests/gpu_epochs/tree_inverse.cuh`. The unchanged
GPL license text and notices are retained. Added files document provenance and
provide a reproducible CPU reference audit. Nothing outside `candidates/subset/`
is part of this change; trusted harness, scoring and problem files are unchanged.

Let a four-lane group hold nonzero denominators a, b, c, d. The two even lanes
compute p=a*b and q=c*d. A shuffle makes the opposite pair product available
and lane zero computes p*q. The upper shared tree now has one leaf per group,
64 leaves for a production 256-thread block. A standard up sweep obtains the
block product, one modular inverse inverts that product, and the down sweep
returns the inverse of every group product. The even lanes then compute
1/p=(1/(p*q))*q and 1/q=(1/(p*q))*p. A final broadcast and multiplication by
the neighboring original factor yields the inverse of each original lane.

All shuffles are reached by all 32 lanes with the same full-warp mask. Source
lanes remain within each group of four. Conditional arithmetic surrounds no
conditional shuffle. The supported block sizes are 32, 64, 128 and 256; all
production launches use 256. Inactive tail lanes keep identity factors and
remain present through every shuffle and block barrier. The inherited caller
only permits whole-block early exit before the helper. The helper's final
canonical multiplication clears the fifth limb, as the predecessor did.

For n=256 the old tree had 256 leaves; the new upper tree has m=n/4=64.
The arithmetic is n/2 pair products, n/4 group products, (m-1) upper up-sweep
products, 2*(m-1) upper down-sweep products, n/2 pair inverse products, and n
final lane products. The total is 3*n-3=765 multiplications plus one inverse,
exactly the same as the original work-efficient tree. Shared memory falls
from 4*512*8=16384 bytes to 4*128*8=4096 bytes. Block barriers fall from 18 to
14. The new helper adds twelve 64-bit shuffle operations per lane and retains
pair/sibling values in registers. Occupancy, register spills and instruction
scheduling determine whether that trade is beneficial; no GPU timing is
inferred from source-level counts.

The epoch pipeline remains unchanged. A runtime geometry guard enables the
specialized case: n=150, t=9, 42 prefix-remainder bytes, 218 tail bytes, 44
suffix bytes and a 9906-byte preimage. An epoch fixes six omissions below cut
137. Its 256 consumer lanes select 256 distinct triples from the last 13
pushes. The GPU producer computes the shared prefix from the current problem;
the consumers hash the remaining six blocks, SHA256d, both recovered public
keys and the required gate. Hash schedules and tables are built from runtime
bytes. There is no answer cache or dependence on a known evaluation seed.

## Local validation and limitations

The new `check_candidate.py` extracts actual production functions into a
temporary C++ library. It implements block barriers, warp shuffles and CUDA
symbol copies on the CPU. OpenSSL BIGNUM substitutes for PTX field primitives;
the SHA header's equivalent C sigma macros substitute for assembly sigmas.
These substitutions allow independent reference testing on this workstation,
but do not validate CUDA compilation, GPU races, PTX arithmetic or performance.

The completed checks are:

| Check | Result |
| --- | ---: |
| Actual modified inverse helper vs Python modular inverse | 3840/3840 |
| Actual epoch producer + scheduled window hash vs hashlib on full preimages | 6144/6144 |
| Actual signed recoder vs independent modular reconstruction | 10262/10262 |
| Actual combinadic unranking vs itertools combinations | 8086/8086 |
| Recursive production and GPU-audit quoted includes | 9 files, complete |

Inverse tests run eight blocks at each of 32, 64, 128 and 256 lanes, including
all-identity inputs, p-1 and p-2, random nonzero values and partial active tails.
They assert the exact multiply count, exactly one inverse, equal barrier counts
for every lane and a cleared fifth output limb. CPU thread barriers model the
participation contract; they are explicitly not a replacement for Compute
Sanitizer on the actual GPU.

Hash tests use fresh synthetic problem seeds 20260916 and 679162400, twelve
epoch ranks each, and every one of the 256 window choices. Ranks include the
start, launch boundaries, the last two epochs and random interior values.
Every complete SHA256d is compared with hashlib on the benchmark's independent
full-preimage constructor, and all reported skip sets are sorted and distinct.
The actual schedule builder found 84 distinct first-block classes and 26
second-block classes on both problems. Recoding checks include 0, n-1, n,
n+1, 2^256-1, every single-bit scalar and 10000 random scalars.

`yukon setup --track subset` completed its CPU verifier smoke on this host.
The required `yukon run --track subset` was attempted, including after the
final code change, but could not start the ranked grinder: this Mac lacks the
Linux benchmark bridge, and spawning sudo is blocked in the local environment.
No ranked local score was produced. No CUDA compilation, GPU sanitizer result,
or end-to-end GPU hit verification is claimed for this variant. The inherited
GPU arithmetic audit is retained for execution on a CUDA host.

## Reproduction, archive and evaluation

From the benchmark work directory:

```sh
yukon setup --track subset
python3 candidates/subset/check_candidate.py
yukon run --track subset
```

The host audit requires Python, a C++17 compiler and OpenSSL development
headers. `OPENSSL_PREFIX` and `CXX` may select their installed locations.
It leaves its generated library in a temporary directory that is removed
afterward. The official CUDA compile command remains unchanged:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/subset/subset \
  candidates/subset/subset.cu -lcrypto -lm
```

On a CUDA host, run the retained actual GPU arithmetic audit and synchronization
check before using local performance numbers to select a successor:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 \
  candidates/subset/tests/gpu_epochs/tree_audit.cu \
  -o /tmp/qsb-subset-tree-audit -lcrypto -lm
/tmp/qsb-subset-tree-audit
compute-sanitizer --tool synccheck --error-exitcode 9 /tmp/qsb-subset-tree-audit
```

The unchanged production tree source SHA256 is
`92e9b024cd191b478f5b9d9dcb327883a9bb12650afd2e57642e83165acfdac5`.
The modified inverse helper SHA256 is
`5bba7cc408e048b6c9c22fc3be063100172d0ce17712932df7ca3a7483ef7f8f`.
The archive contains source and documentation only and is well below 8 MiB.
No binary, score artifact, credential or private transcript is included.

Remote evaluation is authoritative. The imported port is itself pending and
may raise the frontier before this evaluation completes. A correct result
that loses to that newer frontier is not a promotion. If the helper regresses,
the first useful evidence is a matched comparison of register count, spills,
shared-memory occupancy and verified throughput against the imported 16 KiB
tree. If validation finds a correctness problem, the failure should be added
to the local audit before another submission. No local score is attached.
