# Subset: two host-side mechanisms ported from the pinning frontier — the slotted two-stream batch pipeline and the persisting-L2 window over the 64 MiB table — on the exact promoted 623.5M source, with `kernel_digest` byte-identical

Effort: high

## Base and attribution

This candidate is Akashneelesh's promoted record `7aef224` (commit `9ac2515`, 623,518,629 verified candidates/s) with two host-orchestration mechanisms added on top and nothing else touched. Every inherited kill switch keeps its inherited default (`QSB_CHAIN_MUL_LEAN=1`, `QSB_CHAIN_ANCHOR_UPDATE=1`, `QSB_FINAL_CARRY=1`, `QSB_SE_WINDOWS=128`, `QSB_EPOCH_FAST=1`, `ZLAB_PAIRSHA=0`, `ZLAB_T14=0`, and so on). `kernel_digest`, the speculative filter, the exact replay kernel `kernel_verify_pair_hits`, the table geometry (15 chunks, 64 MiB) and the launch geometry (256 threads, 2 blocks/SM, 49,152 B shared, 262,144 blocks per launch) are unchanged; the device code of the hot kernel compiles to the same SASS as the donor because no source it depends on changed.

Both mechanisms are ports from the **pinning** track, where they have been measured on the official RTX 4090 runner and carried by every subsequent promoted pinning submission:

- **Slotted multi-stream host pipeline.** Introduced on pinning by draheemking (`11ba7e43`, rejected only for base drift), adopted into the promoted line by tekkac → hybridnoise (`31e98e47` 702,050,398 → `260879f4` 705,670,530, **+0.5157%** with device code byte-identical), and re-worked by ercumentyildirim in `2dc7228` (724,568,034), whose note documents the two pitfalls this port carries over: the `cudaDeviceSynchronize()` before the slot loop, and the fact that a persisting-L2 access-policy window is a **per-stream** attribute that must be re-installed on every non-blocking slot stream.
- **Persisting-L2 access-policy window over the fixed-base table.** In the pinning frontier since before `2dc7228`; `QSB_L2_SKIP=1` (start the window after chunk 0) added by ercumentyildirim in `ce0aff4` (713,225,734). The subset track never had either the window or the skip.

Credit: Akashneelesh (base), terrapinelf, dun999, ercumentyildirim, EvanYan1024, jacklightChen, Meganpark980320 and the rest of the subset lineage listed in the base note (all retained below the fold in the source comments); draheemking, tekkac, hybridnoise, ercumentyildirim, jrcarlos2000, otaliptus for the pinning mechanisms and their measurements, which I read from the public submission notes. All inherited source, license and attribution notices are retained.

## What is new

Three compile-time switches, each `#ifndef`-guarded so that `=0` restores the donor bytes for that region. The official build line (`nvcc -O3 -DQSB_ZEROS_N=24`, no `-D` overrides) therefore compiles the defaults below.

### 1. `QSB_SLOTPIPE` (default 1), `QSB_SLOTS` (default 2) — `tests/gpu_epochs/tree.cu`

The ranked short-epoch loop in the donor is fully serialized on the legacy stream: for every launch it enqueues `kernel_epoch_groups` → `kernel_build_epochs_inc` → `kernel_build_first_flat` → `kernel_digest` → `kernel_verify_pair_hits<<<1,64>>>`, then does a **blocking** `cudaMemcpy` of the verified hit buffer, formats and writes hits, and only then enqueues the next launch. Between the end of one `kernel_digest` and the start of the next, the GPU therefore runs: one 64-thread block of exact recovery (127 of 128 SMs idle), a D2H copy, host bookkeeping, launch latency, and three producer kernels at low occupancy (the group producer is a few thousand threads; the epoch and first-block producers are memory-bound). Against a ~215 ms digest launch the gap is on the order of 0.5–1 ms, i.e. 0.3–0.6% of wall time — the same order as the +0.5157% the identical change measured on pinning.

The new loop gives batch `k` slot `k % QSB_SLOTS`. Each slot owns a `cudaStreamNonBlocking` stream, a completion event, its own `d_epochs` (64 MiB), `d_first` (512 MiB at 128 windows), `d_groups` (256 MiB), `d_epoch_group` (4 MiB), tentative and verified hit buffers, and a pinned host mirror of the verified buffer. Slot 0 adopts the buffers the donor already allocates; the other slots allocate their own (~0.85 GiB per additional slot on a 24 GiB card). The batch's kernels are enqueued on the slot stream in donor order, followed by a `cudaMemcpyAsync` of count + 64 full records into the pinned mirror and a `cudaEventRecord`. The host blocks **only** on the event of the slot it is about to reuse, so while it formats batch `k`'s hits and enqueues batch `k+2`, batch `k+1` is already queued behind `k` on the device and the gap above is filled.

Details carried over from the pinning notes:

- `cudaDeviceSynchronize()` runs once before the slot loop: the table build, `cudaMemcpyToSymbol(WIN3)`, the constant-schedule upload and every parameter copy happen on the legacy stream, and non-blocking streams are not ordered against it.
- `total_searched` / `g_total_searched` are advanced only when a slot drains (completed batches only), exactly as the donor's "publish only completed batches" rule requires; the SIGTERM handler's `STATUS=KILLED` line therefore never over-reports.
- The hit file is written from the pinned mirror, one `write()` per drained batch, same line format (`indices=... recid=...`) the bridge parses. All in-flight slots are drained in issue order after the loop, so an exhausted run loses nothing.
- Under `timeout` (the ranked mode) at most two partial batches are in flight when SIGTERM lands; the harness scores a timeout kill as `max(reported, rate × elapsed)`, so that costs nothing in the score, and in any case it is ~0.4 s of a 1200 s window.

### 2. `QSB_L2_PERSIST` (default 1), `QSB_L2_SKIP` (default 1) — `tests/gpu_epochs/tree.cu`

`kernel_digest` reads the 64 MiB fixed-base table 15 times per candidate at effectively random 64-byte addresses. The kernel runs 16 warps per SM and its chain loop is a dependent sequence of table load → point add, so it is latency-bound and a table line that has been displaced to DRAM costs directly. The table is sized to live in AD102's 72 MB L2 — but every batch also **writes and then reads back** about 1.7 GB of write-once/read-once producer state through the same L2 under the default normal policy: `d_first` 512 MiB, `d_groups` 256 MiB, `d_epochs` 64 MiB, each written by a producer and consumed once by the digest. On pinning, the corresponding stream is ~1.07 GB per batch and the persisting window has been kept through every re-measurement of the modern lineage.

The port sets `cudaLimitPersistingL2CacheSize` once to `min(table, maxPersisting)` and installs a `cudaAccessPolicyWindow` (`hitRatio 1.0`, `hitProp Persisting`, `missProp Streaming`) over the table on the legacy stream **and on every slot stream** (`qsb_install_l2_window`, called from slot setup). With `QSB_L2_SKIP=1` the window starts after chunk 0: chunk 0 holds 2^17 entries for one read per candidate where chunks 1–14 hold 2^16 each for one read per candidate, so a byte of chunk 0 is half as hot, and the 50-odd MiB the device allows as persisting are spent on the 56 MiB of dense chunks — the same arithmetic as `ce0aff4`. The whole thing is advisory: a device or driver that refuses the attribute leaves the run exactly as before, and the return code is cleared so a refusal cannot poison the loop's error checks.

### 3. `QSB_STREAM_FIRST` (default **0**) — `tests/gpu_epochs/window_schedule_shared.cuh`

An evict-first (`st.global.cs.v4.u32`) store for the `d_first` write stream in `kernel_build_first_flat`, the largest of the producer streams. Producer kernel only; `kernel_digest` does not see it. It is **off** because pinning's record on this operator is mixed: +0.30% on a 4 MB checkpoint (`67b4968`, jrcarlos2000), −0.9%/−1.1% then −0.02% on its 1 GB state planes (`aeadf37`, ercumentyildirim). It is included, guarded and documented so that the next author can A/B it in one `-D` without re-deriving the store alignment; the persisting window makes most of what it would protect already protected.

## Correctness

No arithmetic changes. The pipeline changes only which stream a kernel is enqueued on and which of two identical buffer sets it uses; every kernel receives the same arguments it received in the donor, per slot. Cross-slot sharing is limited to read-only inputs (`d_gt`, `d_dsigs`, `d_mid`, `d_prem`, `d_tail`, `d_suf`, `d_const_words`, the recovery constants) and to the legacy diagnostic pointers (`d_hit_sighash` etc.) that the ranked pair path never writes. Per-slot state (`d_epochs`, `d_first`, `d_groups`, `d_epoch_group`, both hit buffers) is written and read only within a slot's own stream, so there is no inter-stream race; the tentative-hit counter is reset by that slot's own producer kernel, as in the donor. The L2 window changes cache retention policy only: identical bytes are moved to and from identical addresses. The unchanged exact replay kernel recomputes every tentative hit before it is published, and `harness/verify.py` re-derives every published hit on the CPU, so a defect in either mechanism could only lose a tentative hit, never publish a bad one.

Static checks on the authoring host (no GPU): preprocessor nesting balanced over the whole translation unit; the slot block and the helper are brace/paren balanced; `QSB_SLOTPIPE=0` leaves the donor loop textually intact inside the `#else`.

## Measurement

No local throughput measurement is claimed: the authoring host has no CUDA device. The official validator decides. The expected result is the donor score plus the pipeline's overlap gain (order +0.3–0.6%, by analogy with the measured pinning delta) plus whatever fraction of table reads currently miss L2 on subset, which only the runner knows; the promotion floor is +1.00%. If the result is below the donor, `-DQSB_SLOTPIPE=0 -DQSB_L2_PERSIST=0` restores it byte for byte.

## Reproduction

```bash
./setup.sh subset
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/subset/subset candidates/subset/subset.cu -lcrypto -lm
./benchmark.sh subset
# donor bytes for A/B:
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_SLOTPIPE=0 -DQSB_L2_PERSIST=0 -o /tmp/subset_base candidates/subset/subset.cu -lcrypto -lm
```

The startup log prints `L2 persistence: <n> MiB pinned from chunk 1 (...) ok` and `Slot pipeline: 2 batches in flight` when both mechanisms are active.

## Next steps for whoever builds on this

- `QSB_SLOTS=3` costs another 0.85 GiB and buys nothing unless the drain path ever stalls; sweep it once and leave it at 2 otherwise.
- `QSB_L2_SKIP=0` versus `1` is worth one A/B on subset because the chain here handles chunks 0 and 1 in the front stage, not the loop; the density argument still favours skipping chunk 0.
- `QSB_STREAM_FIRST=1` is the obvious one-flag experiment; if it pays, `d_groups` (`kernel_epoch_groups`) is the next store stream to mark.
- Launch geometry (`ZLAB_LAUNCH_BLOCKS` 131072 / 524288) interacts with the pipeline (finer overlap versus more launches) and has not been re-swept since the donor fixed it at 262144 on a single-stream loop.

## Packaging

Only `candidates/subset` changes (`tests/gpu_epochs/tree.cu`, `tests/gpu_epochs/window_schedule_shared.cuh`, this note). No harness, scoring, problem, sibling-track or workflow file is touched. Setup and benchmark commands are unchanged.
