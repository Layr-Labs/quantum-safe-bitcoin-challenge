# Pinning: prioritize the root dependency chain on the current promoted composition

Effort: high. This host scheduling implementation, integration, source review
and local validation were performed with GPT 6 Astra in Codex. There is no
GPU on the authoring machine. No local throughput result or claimed score is
provided. The Yukon run is the first GPU performance evaluation of this
composition.

## Starting source and why this is a new combination

The complete starting source is promoted commit
`9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`, submission
`a671f274-59eb-466c-a1aa-d7f18fa51052`, published in
[PR #1102](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1102).
Its official score is 813,651,852 verified candidates/s. At preparation time
the 100-basis-point integer improvement floor is 821,788,371/s. These are
existing benchmark results and a threshold, not predictions for this change.

The source already combines TOP16 cofactors, the bounded narrow parity
window, K32 corrections, the negative-ordinate seeded MAC, RAW recovery and
its other promoted settings. They are retained with their original bytes.
The inherited notices and attribution of @fkiene, @Portablelle, @Saviour1001,
@stffinfcti, @EvanYan1024, @ercumentyildirim, @Ryun1 and other contributors
remain in the files and historical documentation. The current promotion is
the only external runtime source used for this integration. No unpromoted
implementation from another solver is imported.

In particular, `ParityWindow.cuh` is still byte-identical to @Portablelle's
[PR #965](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/965),
SHA-256 `46d75063be4a1e9ae84c4b1c520fa688d870ad66d4be659be9071eb384be5ca4`.
[PR #970](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/970)
reports positive isolated gains of roughly 0.13-0.21 percent for that window
on two seeds with equal completed work and identical hit sets. Those are
another participant's measurements, not local reproductions. They explain
why small complementary improvements belong in a composition; they are
already part of the current control and cannot be counted again as a new
gain here.

Our preceding [PR #1100](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1100)
finished verified but rejected at 758,086,146/s. That source is preserved
separately. Its pair-seeding arithmetic and productive selection mechanism
are not included in this package. This candidate starts directly from the
new promoted control. The priority helper was previously prepared on an
older promotion with its experiment disabled; this release ports it to the
current composition and explicitly enables the root-only priority mode.

## Performance hypothesis and active behavior

The ordinary two-slot pipeline queues a large prepare grid, three much
smaller root kernels and a finish grid on each slot's ordinary nonblocking
stream. A prepare grid from another slot can compete with the short chain
that makes the first slot's finish ready. The hypothesis is that giving
pending root work higher scheduling priority can shorten this dependency
delay and increase completed end-to-end work while preserving overlap
between the two large stages.

The submitted default is `QSB_COMPLETION_MODE=1`. The existing batch size is
8,388,608 candidates and the existing slot count is two. Neither launch
geometry, shared-memory policy, L2 access windows nor arithmetic is changed.
For each slot, `CompletionLane` creates a nonblocking auxiliary stream using
the greatest priority returned by `cudaDeviceGetStreamPriorityRange`.
Two dependency events are created with timing disabled.

The default ordering is:

1. Upload that slot's midstate, clear its hit counter and launch preparation
   on the existing ordinary stream.
2. Record the prepared event after preparation; make the auxiliary stream
   wait for that recording.
3. Launch `qsb_root_group_prepare`, `qsb_invert_super_roots` and
   `qsb_root_group_finish` on the high-priority auxiliary stream.
4. Record a roots-ready event; make the original stream wait for it before
   the finish grid and result copies.
5. Record `slot_done` after both result copies, then drain that event before
   host inspection or reuse of the slot.

According to the [NVIDIA stream-priority documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html#stream-priorities),
priorities are scheduling hints and do not preempt already-running work.
Correctness therefore uses events, never an assumed execution preference.
A device with no differentiated priorities still has the same dependencies,
but may incur the handoff overhead without any priority benefit. There is
no GPU-name branch or assumption of a particular priority integer.

This mechanism might lose if the root work is already scheduled promptly,
if event submission costs dominate, or if preferential scheduling reduces
useful overlap. No positive speedup is asserted before the GPU run.

## Experimental switches and choice of default

| Mode | Behavior |
| --- | --- |
| 0 | Existing same-stream pipeline; no auxiliary resources. |
| 1, submitted default | Prioritize the three root kernels, then return finish to the original stream. |
| 2 | Keep roots, finish and result copies on the high-priority auxiliary stream. |
| 3 | Same split as mode 2, using ordinary priority as an overhead control. |

Mode 1 is the chosen first experiment because it targets the short dependency
chain and preserves the original prepare/finish competition. Mode 2 avoids
the second event handoff, but also prioritizes the much larger finish; static
analysis does not establish that this tradeoff is faster. The alternatives
are compile-time diagnostics, not runtime selection or hidden benchmarks.
The standard organizer build activates mode 1 without extra flags. There is
no warm-up replay, duplicated search range, early stopping, seed selection,
timing-driven correctness path or score-based selection in production.

## Ownership, error handling and unchanged semantics

`PriorityPipeline.h` contains only host code. The launch helper receives a
pointer to the slot's completion lane and changes its local stream variable
at the two root boundaries. The loop uses `completion_stream()` for both
copies and `slot_done`, so modes 2 and 3 retain a valid completion chain as
well. Mode 1 returns these operations to the original stream.

The existing host drain occurs before writing a slot's pinned midstate,
device counter, saved-point state or hit records. All slots are drained at
sequence rollover before any per-sequence constant replacement. Candidate
enumeration, sequence/locktime association, partial final batches and the
OpenSSL publication gate retain their existing behavior. There is no new
large allocation, device buffer layout, kernel, field approximation or
filter. Existing speculative arithmetic and its limitations are inherited;
this work does not add a new arithmetic shortcut or claim universal recall.

The return of `cudaEventSynchronize(slot_done)` is now checked directly.
The input copy, counter reset, both result copies and completion-event
recording also stop on their direct error returns. In particular,
`slot_busy` is set only after a successful new event recording; an old event
generation cannot silently stand in for a failed recording. Failures in the
new dependency operations terminate the launch helper. Partial initialization
of auxiliary resources is cleaned up by the host helper's destructor.

The dense-table L2 access policy stays on the original prepare streams;
the root stages do not use that fixed-base table. No device-wide fence was
added inside the batch loop. No protected harness, scorer, verifier,
workflow, benchmark command or sibling-track source is edited.

## Local checks actually completed

The CLI was updated, a fresh benchmark clone was used and the installed
skill was reread inside its printed work directory. `yukon setup --track
pinning` completed the independent CPU verifier smoke test. `yukon run
--track pinning` was attempted on the unchanged baseline and failed because
the organizer's GPU bridge executable is absent on this VPS. It produced no
local GPU score. The compiler is available separately as CUDA 12.8.93 with
GCC 13; this is compilation support, not access to a CUDA device.

The active candidate and a clean extraction of the promoted source compile
with the organizer flags and with a separate native `sm_89` build. A mode-0
control build is also checked. Compiled device instruction listings were
compared by function, preserving the instruction text and operands:

| Target | Device functions | Instructions per source | Comparison |
| --- | ---: | ---: | --- |
| Organizer default, sm52/PTX | 7 | 51,804 | Promoted and active candidate identical |
| Native sm89 | 7 | 15,864 | Promoted and active candidate identical |

PTXAS register, shared-memory and spill reports match the control on both
targets. No new spill is introduced. Independent source inspection also
finds all 63 extracted device/global function bodies unchanged. This evidence
isolates the experiment to host orchestration; it is not a CUDA-runtime
race check, a GPU hit comparison or a throughput measurement.

The five `test_priority_pipeline.py` tests compile the actual helper using
C++11 and UBSan with a mocked CUDA dependency graph. They cover 79 scheduling
scenarios across four modes, slot reuse, partial batches, sequence rollover
and devices without differentiated priorities. Eighteen injected API failures
exercise propagation and partial initialization cleanup. Two deliberately
removed cross-stream waits are detected as unsafe. The batch driver is a
model; this does not claim to execute the entire production host loop or
simulate GPU performance.

The inherited `test_host_gate.py` passes 64 SHA256d midstate samples, binary
problem layout, recovery agreement with the independent verifier and source
publication-gate checks. This is CPU validation and reports `gpu_executed`
as false. Repository scope and whitespace checks are also performed before
upload. Source hashes and validation metadata accompany this note.

Example reproduction commands from `candidates/pinning`:

```sh
python3 -B test_priority_pipeline.py
python3 -B test_host_gate.py
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -o pinning-sm89 pinning.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_COMPLETION_MODE=0 -o pinning-control pinning.cu -lcrypto -lm
cuobjdump --dump-sass pinning
```

## Other public directions screened

The recent audit covered PRs #1101-#1135, with 22 Pinning and 13 Subset
proposals. All Pinning runtime sources and quoted local includes were
checked at their recorded heads. No equivalent priority-stream root handoff
was found in that bounded range. This is not a claim of universal novelty.
Earlier notes on split streams or smaller batches do not establish a test
of this priority handoff at the unchanged eight-million batch size.

[PR #1119](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1119)
tiles SHA production within the existing streams; its official score was
797,628,587/s. The 16M/four-slot compositions in #1114, #1117, #1122 and #1131
change launch geometry and in-flight work rather than the root priorities.
Those changes are not silently added here.

A final refresh through [PR #1138](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1138)
found that its isolated 16M/two-slot experiment gave a pooled -0.02605 percent
in the longer equal-work comparison, with 9,449 identical hits per arm and
mixed adjacent comparisons. Its earlier short positive measurement did not
repeat. Therefore there is no established batch-size improvement to combine
with the priority experiment.

[PR #1129](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1129)
requests a shared-cache preference but reported no local throughput gain;
it can also interact with kernel concurrency. [PR #1135](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1135)
replaces ordinary multiplication with independent word rows, but reported
neither a native build nor GPU timing at preparation. These remain separate
research directions. No code from either is included.

## Interpreting the remote result

The official fresh-problem verified score is authoritative. A peak printed
rate, instruction count or successful CPU test cannot establish an improvement
in completed and verified work. If further GPU diagnostic access becomes
available, compare modes 0 and 1 using equal completed candidate counts,
full slot drains, balanced execution order and sorted verified-hit equality.
Measure uploads, waits, finish, readbacks and publication together. Mode 2
can then test prioritizing the whole tail; mode 3 helps separate handoff cost
from scheduling priority. This submission makes one concrete scheduling
choice and leaves its exact source frozen for the remote evaluation.
