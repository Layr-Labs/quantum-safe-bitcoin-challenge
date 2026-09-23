# Pinning: priority completion, fewer transfers, and overlapped host verification

Effort: high. This implementation and integration use GPT 6 Astra in Codex.
The authoring VPS has no NVIDIA GPU. Local evidence consists of compilation,
source comparison, CPU dependency/data-lifetime tests and an isolated CPU
cost measurement. No local GPU speedup or claimed ranked score is supplied.

## Previous result and the missing improvement

Our previous submission `59b3693f-fb93-4203-a77f-1e286266dede`,
[PR #1139](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1139),
has exact head `f60940f15ce04bb641e893ff801e295d36eaad3f`. It ran the promoted
arithmetic with the three root kernels on a higher-priority stream, then
returned finish and copies to the original ordinary stream.

The official run returned 814,080,739 verified candidates/s, verification
true, 116,603 verified hits and 1,201.5232 seconds, on problem seed
2,013,030,730. It was rejected below the required one-percent improvement.
The current promoted control remains
`9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`, from
[PR #1102](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1102),
at 813,651,852/s. Its integer promotion floor is 821,788,371/s.

The direct score ratio of #1139 to that control is +0.052711 percent, not a
demonstrated one-percent speedup. An additional 0.946790 percent relative
to #1139's observed score would be required to reach the current floor.
A small difference between separate fixed-time runs is not an isolated
measurement of the priority mechanism. We preserve the previous source and
result rather than describing this package as a repeat of a proven gain.

The previously prepared mode-2 alternative motivates the next scheduling
experiment. The new implementation also removes two unnecessary transfer
operations and takes the exact CPU publication gate out of the path before
the next GPU enqueue. These are concrete changes to work and ordering;
their combined GPU effect still needs the remote run.

## Source lineage and scope

The fresh checkout is the current promotion above, with our #1139 host delta
ported onto it. All inherited device arithmetic and notices are preserved,
including the TOP16, narrow-parity, K32, RAW-recovery and negative-Y MAC
composition. In particular the narrow parity header from @Portablelle's
PR #965 is unchanged. Credits to @fkiene, @Saviour1001, @stffinfcti,
@EvanYan1024, @ercumentyildirim, @Ryun1 and other inherited contributors remain
in the source and historical notes. The older single-stream code already
contains a contiguous-readback pattern; this change applies a checked,
asynchronous per-slot version to the active pipeline.

Only `candidates/pinning/` is edited. No sibling track, problem, verifier,
scoring code, bridge, workflow or build command is changed. No external
unpromoted implementation is imported. The priority helper and its previous
tests are our own #1139 source. The transfer and snapshot helpers are new
host code. Existing GPLv3-only notices and `COPYING` are preserved.

The batch remains 8,388,608 candidates with two in-flight slots. The same
candidate ranges, field operations, filters, root association, launch
geometry, table layout and host OpenSSL acceptance rule are used.

## Four active changes

### 1. Keep the full completion chain on the priority stream

`QSB_COMPLETION_MODE=2` now keeps the three root kernels, finish grid and
result readback on the high-priority auxiliary stream. Preparation still
runs on the existing ordinary stream and records a dependency event; the
auxiliary stream waits for it. `slot_done` is recorded on the stream that
actually completes the result copy.

Relative to mode 1, this removes the roots-ready event recording and the
ordinary stream's wait for that event. More significantly, it gives pending
finish work priority as well as the small root chain, potentially returning
a reusable slot sooner. It may instead reduce useful overlap with prepare;
static reasoning cannot establish which effect dominates.

Stream priority is only a scheduling hint. It does not preempt already
running work, and correctness relies on the event dependencies, not on an
assumed scheduling order. The helper uses the priority range reported by
CUDA and also handles devices with no priority differentiation.

### 2. Remove the unused per-batch midstate upload

With `QSB_TAIL_PRE=1`, the active `FAST_TAIL=true` prepare specialization
reads the midstate from `tp.mid`. The host already constructs `cur_tp` for
each sequence and passes it by value in the kernel arguments. The separate
32-byte upload to `d_mid_slot` is therefore unused by this compiled path.

`QSB_SKIP_UNUSED_MIDSTATE=1` omits the host staging copy and H2D enqueue only
when `QSB_TAIL_PRE` is enabled. The original transfer remains compiled when
precomputation is disabled or the optimization switch is zero. The initial
slot allocations and initialization remain valid. No stale midstate is used:
the actual consumed state is the current sequence's kernel argument.

### 3. Read the count and hit prefix with one copy

`QSB_COMPACT_READBACK=1` uses `SlotReadback.h`. Each slot has a device buffer
of 1,025 32-bit words: the count followed by the original capacity of 1,024
indices. Its pinned host mirror contains 65 words: the count and the same
first 64 indices consumed by the previous implementation.

One asynchronous 260-byte D2H copy replaces the previous 4-byte and 256-byte
copies. The helper records `slot_done` only after a successful copy enqueue
and propagates each direct CUDA error. The count pointer and indices pointer
are passed separately to the unchanged kernels. Each index remains naturally
aligned for its existing 32-bit store. Buffer capacity, reporting limit and
record encoding are unchanged.

Together, changes 1-3 remove four CUDA operations enqueued per batch relative
to #1139: the unused H2D, one D2H, one event record and one stream wait.
This count is verifiable from the source; it is not a percentage speedup.

### 4. Enqueue new work before publishing the previous batch

Previously `drain_slot` synchronized the completed slot, performed exact
OpenSSL checks and appended its hits, then returned so the host could enqueue
the next batch. The new `QSB_OVERLAP_PUBLICATION=1` path separates these
operations. After a successful wait, `HitSnapshot` takes ownership of the
count, the first 64 indices, and the old sequence and base locktime.

The host then reuses the slot to enqueue the next batch and its completion
event. Only after that successful enqueue does it verify and publish the
saved snapshot. Thus the GPU can process the new batch while the CPU checks
the previous hits. No new CPU thread or OpenSSL context sharing is introduced.

This snapshot is essential: a sufficiently fast next batch could otherwise
overwrite the pinned report while the CPU was reading it. Copying the
sequence and locktime is equally necessary because slot metadata is changed
at reuse. `HitSnapshot` owns its payload by value and bounds the copy to 64
indices. Empty reports do not read the indices. End-of-sequence drains still
verify and publish synchronously before the sequence advances or a constant
table is replaced. The existing reporting order and exact acceptance test
are retained for a fully completed search range.

At a forced timeout, a saved report may remain unpublished during the brief
enqueue of the next batch. The publication boundary is therefore not promised
to be identical to the old loop at an arbitrary interruption; the official
score includes this effect. Hits that are published retain correct attribution.

An isolated CPU timing of the actual `qsb_host_exact_hit` implementation
motivated this change: three runs of 3,000 calls measured 368.960, 383.934
and 374.496 microseconds per invocation on this VPS. This is CPU cost, not
GPU throughput. The existing two-slot pipeline can already mask part of it,
the remote CPU differs, and some nominations may require two recid checks.
No fixed end-to-end gain is inferred from these timings.
The packaged `gate_cost_probe.py` extracts the actual production loader and
check function; its generated C++ was verified byte-identical to the original
timing probe. Reproduce from the repository root with
`python3 candidates/pinning/gate_cost_probe.py --problem problems/pinning.bin`.

## Correctness boundaries and component controls

The device event wait still precedes every read of the pinned result and
every reuse of device state. A slot becomes busy only after its new result
copy and event recording succeed. The old snapshot is independent of both
the reused device buffer and its pinned destination. The host verifies the
snapshot before advancing to the next iteration; a launch failure returns
an error rather than claiming completed work.

The packed GPU record is decoded with the saved sequence and locktime;
recid handling and the exact host recovery/hash gate are unchanged. The
64-record publication cap and all existing speculative arithmetic limitations
are inherited. This package does not add a carry omission, approximation,
exceptional-point skip or new nomination filter.

The following switches permit component diagnostics:

| Switch | Submitted default | Diagnostic alternative |
| --- | --- | --- |
| `QSB_COMPLETION_MODE` | 2 | 0 original stream, 1 roots-only priority, 3 ordinary-priority split |
| `QSB_SKIP_UNUSED_MIDSTATE` | 1 | 0 retains the per-batch H2D |
| `QSB_COMPACT_READBACK` | 1 | 0 retains separate count/index copies |
| `QSB_OVERLAP_PUBLICATION` | 1 | 0 publishes during the slot drain |

These switches select code paths, not a timing-dependent autotuner. For an
exact historical performance control, use the frozen #1139 head separately;
disabling switches is a component diagnostic, not a claim that all host
instructions or allocations become byte-identical to that older executable.

## Validation and reproducibility

The fresh clone followed Yukon's printed work directory. Setup passed the
independent CPU verifier smoke test. The unmodified `yukon run --track pinning`
was attempted on the baseline, but the organizer's GPU bridge is absent on
this VPS. No local ranked score was generated. Local CUDA 12.8.93 compilation
is available separately from a GPU runtime.

The exact build and CPU test results are recorded in `COMPLETION-VALIDATION.json`
and the source manifest. The checks cover the active default, the historical
scheduling/transfer choices, and the non-precomputed midstate fallback.
Device instruction listings are compared with a clean promoted control to
ensure the orchestration changes did not alter the GPU arithmetic.

All four builds passed: organizer default, native sm89, component switches
disabled with mode 1, and `QSB_TAIL_PRE=0`. The active candidate's seven device
functions have instruction listings identical to the clean promotion:
51,804 instructions for sm52 and 15,864 for sm89. This is compiled-code
equality, not GPU execution or a throughput measurement.

`test_priority_pipeline.py` compiles the actual priority helper with a CUDA
dependency mock and UBSan. It covers the four modes, reuse, partial batches,
sequence rollover, unavailable priority differentiation, injected failures
and deliberately removed waits. It is an ordering model, not a GPU scheduler.
All five tests passed, covering 79 scenarios, 18 injected failures and two
negative controls with missing waits.

`test_slot_readback.py` compiles the actual transfer helper with deferred
copy commands. It checks allocation sizes, the count/index offset, the
64-index prefix, isolation between reused slots and direct API failures.
Failed copies must not be followed by a successful-looking new done event.
Its three tests passed: 36 batches across two reused slots, four injected
errors and an offset mutation that incorrectly overlaps the count.

`test_hit_snapshot.py` compiles the actual snapshot header. Seven boundary
counts include zero, 64, 1,024 and UINT32_MAX. It compares old and reordered
publication across 132 two-slot scenarios, deliberately overwriting both
payload and metadata before publication. These tests check data ownership
and candidate association without assuming that a CPU gate finishes before
the next GPU DMA. The inherited host-gate tests are also run.
Both snapshot tests passed, as did the host-gate test with 64 SHA256d midstate
samples and recovery agreement with the independent verifier.

Example reproduction from the candidate directory:

```sh
python3 -B test_priority_pipeline.py
python3 -B test_slot_readback.py
python3 -B test_hit_snapshot.py
python3 -B test_host_gate.py
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -o pinning-sm89 pinning.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_COMPLETION_MODE=1 \
  -DQSB_COMPACT_READBACK=0 -DQSB_SKIP_UNUSED_MIDSTATE=0 \
  -DQSB_OVERLAP_PUBLICATION=0 -o pinning-components-off pinning.cu -lcrypto -lm
```

## Public alternatives examined

[PR #1151](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1151)
already combines our roots-only mode with a 16M batch. Its priority header
is identical to #1139 and its search source changes only that batch-size
line. It reports a +0.081918 percent equal-work comparison of roots priority
on its 16M configuration, with 2,384 identical hits. It does not evaluate our
new full-tail priority, transfer removal or host-publication overlap.

The latest audit through #1152 did not identify this completion/transfer
combination. Some newer packages change launch geometry or remove arithmetic
carries. `QSB_RP_MUL_F8` has a reported small local gain but omits a real carry;
that change is not included. The exact `QSB_MUL_SFQ` register alias has no
isolated positive timing or native instruction evidence in the public notes
examined and is not used as a filler optimization.

The cache-preference #1129, arithmetic bundle #1133, row-MAC #1135 and 16M
batch #1138 now have official rejected scores of 798,578,822, 811,231,199,
621,477,675 and 781,541,228/s respectively. Those whole-package results do not
isolate every component, but they do not justify importing those packages
without further evidence. No such source is silently combined here.

## What the remote experiment will establish

The official fresh-problem, verified fixed-time score determines whether
this composition crosses the promotion floor. The priority choice may lose
overlap even though fewer host operations are enqueued; CPU publication may
already be mostly hidden. Conversely, better slot turnaround and earlier
enqueue can compound the transfer reductions. Neither outcome is known from
static instruction equality or CPU tests.

A matched GPU follow-up should compare equal completed search ranges and
sorted verified hits, include final slot drains and publication in timing,
and alternate execution order. Peak self-reported rates are not substituted
for completed work or the official score. One Pinning submission is made for
this concrete source, with no promise of a one-percent gain before testing.
