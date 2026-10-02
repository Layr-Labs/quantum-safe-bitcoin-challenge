# Iteration 8: first-state producer-scratch lifetime overlay (default off)

## Hypothesis and logical review before execution

This is a capacity repair, not another arithmetic retiming or stride-packing
experiment. `QSB_FIRST_PRODUCER_SCRATCH=1` removes the two independent group-record
and epoch-map allocations in the supported slot pipeline. Each slot instead
uses a disjoint pair of byte ranges at the start of its own first-state buffer.
No device declaration, carrier knob, first-state pitch, producer inputs,
descriptor layout, batch enumeration, or exact publication gate is changed.

State transitions in `sp_launch`:
1. `sp_collect` waits for the old slot's completion event before reuse. Under HPv3,
   `wait_check_reuse(d_ep)` additionally waits for startup checker readback,
   including the producer-failure path. Slot0 and slot1 have separate buffers.
2. The GPU fallback enqueues group production and incremental epoch production
   on the same `st`. Groups and map have separate aligned ranges; the former is
   bounded by `group_capacity`, and map indexes are batch-relative epochs.
3. Incremental epoch production writes the separate descriptor allocation.
   Once that kernel completes, the group/map inputs are dead. First-state
   production reads descriptors alone and writes ordinary first-state outputs.
   Same-stream ordering, including native carrier launches, prevents overlap.
4. Checker capture and digest are enqueued only after first-state production.
   The checker never reads the dead group/map inputs. Host uploads skip all GPU
   producers and write the same ordinary descriptor/first-state outputs.
5. Reuse waits again; the first-state prefix may then become producer scratch.

Capacity/error/empty cases: the overlay is restricted at preprocessing time to
slot pipeline + hitpath + grouped fast epochs. Other shapes retain original
allocations. Byte-size overflow and group/map fit are checked before aliasing;
if fit fails the original separate-allocation path is used. CUDA allocation
failure exits before any aliased pointer is consumed. Slot1 aliases are assigned
only after successful first-buffer allocation. An oversized runtime group span
uses the unchanged direct producer or errors if it was trimmed out. Final partial
batches stay within the full-capacity layout; no empty batch is launched. The
process's existing teardown does not independently cudaFree group/map/first
pointers, so no new double-free ownership path is introduced. Future explicit
cleanup must treat these two pointers as borrowed ranges, not allocation bases.

Ranked production cap: 2,097,152 epochs/slot, 362,053 group records/slot x128 B,
8,388,608 map bytes/slot, 1,073,741,824 first-state bytes/slot. Group range ends
at 46,342,784 bytes; map range ends at 54,731,392. Both fit. Eliminated requests:
109,462,784 B total = 104.391845703125 MiB; first-state allocation unchanged.
Known major-request headroom increases from 458.537048 to 562.928894 MiB.
This is allocation accounting, NOT evidence of ranked runtime/context fit.

## Verification plan and duplicate-cost guard

Use the pre-existing `lab/iter6_ab.sh` with N24, fixed seed1789110211 and120s
for one frozen pair against the unchanged current qualified source. Previous
center-fusion builds/screens tested device arithmetic; this tests different
host pointer lifetimes. Rebuild the local control because its .build stamp is
missing, required by the existing benchmark wrapper and frozen-run preflight.
Do not replay either iteration7 negative probe or stack an invented checker.
Record the memory snapshot during this same pre-existing benchmark, not another
GPU run. A verified pair is a capacity/correctness screen, not a promotion gate.
Current bat414fe58f is validating/watcher-owned; no polling loop or retry submit.

## Completed existing-verifier result

The one planned pair passed: control391.585980M/s (5658/5658hits),
probe394.201799M/s (5725/5725hits), +0.668006%. This is NOT a demonstrated
speedup or promotion gate. Probe remains default-off; production source and
qualified carrier unchanged. NVML samples during these same runs show12244MiB
control versus12136MiB probe (108MiB decrease); requested allocation savings
are104.391846MiB. Device-fatbin snapshots are byte-identical. Existing wrapper
does not retain raw checker stdout, so no additional full-check PASS claim.
All numeric evidence and caveats: `iter8-findings.json`. No stacked verifier.
