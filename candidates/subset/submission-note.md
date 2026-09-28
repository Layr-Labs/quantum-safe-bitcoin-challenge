Effort: Claude Opus 5.5 in Claude Code

# Subset: the promoted `5c7e36c5` with the `a33e04c3` co-grinder package (KH16 / MRG / AINL) on top

## Summary

This entry starts from the promoted subset record `5c7e36c5` (source `6343a38`, 708.41 M/s). It adds the complete host package of `a33e04c3` (704.27 M/s), which was the best other completed run on the same device image family. The two runs differ in exactly two places:

| side | `5c7e36c5` (record) | `a33e04c3` | this entry |
|---|---|---|---|
| co-grinder and host producers | `a141df2b` lane as carried by `4da17ebc` | round-9 lane with QSB_CPU_KH16, QSB_CPU_MRG, QSB_CPU_AINL, blocking GPU waits sharing the host core | **`a33e04c3` files** |
| device Q layout | `QSB_Q_MIX` 2, image `f7454842…` | `QSB_Q_MIX` 4, image `003e3d39…` | **`QSB_Q_MIX` 2, image `f7454842…` (the record's)** |

No queued or completed entry carries this combination. The record's device side is kept byte for byte: the `QSB_Y_PAIR` paired single-reduction chain with `QSB_SC_PARK`, the P18 five-term chain, `QSB_Q_MIX` 2, and the completed-slot snapshot that queues the next GPU batch before the exact host gate runs.

This is a **measurement entry**. It gives no point estimate. The reading rules are fixed below, before the run.

## Why this combination

Every ranked subset run publishes its verified hits. Each hit's window pattern shows whether the GPU (the 128 late-window triples) or the co-grinder (the other 158) found it. So each score splits exactly into a GPU part and a CPU part. The GPU part also has a measure with no Poisson term: GPU hits come out in lexicographic epoch order, so the rank of the last GPU hit is the number of epochs the GPU walked in the timed window.

On those two measures:

- **CPU part.** `a33e04c3` drew 64.08 M/s. The same round-9 lane without the three new switches drew 63.09 and 63.09 (`3e6069ee`, `9fa96f74`). The `a141df2b` lane that the record uses drew between 61.83 and 63.58 in six runs, mean about 62.7. So the three switches read about +1.0 M/s from one run, which is about 1.5 standard deviations of the CPU hit count. The author reports +2.4% CPU rate locally.
- **GPU part.** Epochs walked per second for the same image drift by up to about 1% between hours of one day on the ranked machine. Against a line fitted through three byte-identical runs of one tree on the same day, the two `QSB_Q_MIX` values read as follows. `QSB_Q_MIX` 2 read −0.37%, −0.71% and +0.20% in three runs. The record's run and the byte-identical tree `34d84f7e` ran in adjacent slots and differ only in this constant. Their pair reads about +0.14% for `QSB_Q_MIX` 2 once the fitted drift is removed. The pooled difference is small (about −0.3% ± 0.3%) and not established in either direction. This entry keeps the record's value and its native image, so the device side stays byte-identical to the record.

The host package does not change which candidates the GPU searches. The GPU part of `a33e04c3` sat on the same-day line (−0.02%), and the record's host package reads the same way. So the combination is expected to add the CPU switches to the record without moving the GPU. This run tests that expectation.

## What changes, file by file, against `6343a38`

- `CpuGrindSubset.h`: the `a33e04c3` file, byte for byte. Against the record this brings the following.
  - QSB_CPU_KH16: the message schedule of the key hashes for a group's 16 keys runs in AVX-512, and the rounds run four keys at a time with SHA-NI.
  - QSB_CPU_MRG: each 8-lane field product accumulates the low and high partial products of a column in one register.
  - QSB_CPU_AINL: the small field operations are always inlined.
  - The round-9 table rules: QSB_CPU_TAB9_FRAC 0.50, QSB_CPU_TAB9_CAP_MB, and the QSB_CPU_TOUCH9_MAX_S first-touch guard.
  - QSB_CPU_X4PS and QSB_CPU_SHC, which reuse W+K words and skip constant padding words in the schedule.
  - Table-row prefetch distance 8. The record uses 3. The CPU reading above was taken at 8, and the `4da17ebc` lane at 3 read within noise of the distance-8 runs, so this entry keeps the donor's value.
- `tests/gpu_epochs/host_producers.h`: the `a33e04c3` file. It is the record's producers plus QSB_HP_BLOCKSYNC: when the producers do not pin themselves, the main thread's whole core is published for the co-grinder.
- `tests/gpu_epochs/tree.cu`: the record's file with the three host-side hunks of `a33e04c3`. The per-batch event wait sleeps instead of spinning when the producers ask for it. The final `[HP]` statistics line gains the helper-chunk count. The record's `QSB_Q_MIX` 2 is kept (line 408).
- `subset.cu`: the record's bytes apart from the inert, unreferenced redraw tag line.
- `hit_filter_field_sc_aluz.cuh` and `tests/gpu_epochs/tree.cu.orig` are removed, as in `a33e04c3`. Neither is referenced by the default build (`QSB_SC_ALUZ` is 0, and `.orig` is not a source). The image check below shows that the device code is unchanged.
- `qsb_carrier_sm89.h`: regenerated for this tree. The only changed line is the source-sha comment.
- `SOURCE-MANIFEST.json`: file hashes for this tree.

## Correctness

- Every GPU and co-grinder hit still passes the unchanged exact host recovery and gate before it is published, and the harness re-verifies every published hit.
- The host files are the ones that ran in `a33e04c3`, whose ranked run completed with all hits verified. The device side is the record's, whose ranked run also verified.
- Nothing in the host package references `QSB_Q_MIX`, so the Q layout and the co-grinder do not interact in code.
- The co-grinder's candidate space stays disjoint from the GPU's (the 158 other window triples), as in both parent trees.

## Native image check

- The carrier was regenerated with `build_carrier.sh` from a fresh export of this tree. The embedded cubin is byte-identical to the record's: sha256 `f7454842…`, 462,496 bytes, 3 LTC64B loads in the digest kernel. Every line of the header except the toolchain line and the source-sha line is identical to the record's header.
- Positive control for the toolchain: the same procedure on a fresh export of the record's own tree reproduces its image `f7454842…` byte for byte. The rebuilt header was checked to be newly written before the comparison.
- The carrier knob fingerprint is compiled only from `tree.cu` and `subset.cu`. Neither host header defines any of the 95 knobs in the list. As a positive control, the same search finds the knob definitions in `tree.cu` and `subset.cu`. Neither of those two files changes any knob here. So the host knob string equals the image's, and the native image loads with no JIT fallback.
- The tree compiles and links completely with a CUDA 12.8 toolkit. The kernel resource report matches the record's image.

## Purpose of this run, and how it will be read

The record and `a33e04c3` each carry one half of this tree. Each half was measured once. The same `a33e04c3` tree is also queued again as a byte-identical entry. This run adds a different reading: the CPU switches on the record's device image, in a separate slot.

Reading rules, fixed before the run:

- **Co-grinder part** (hits outside the 128 GPU triples, times the score over the hit count). It is compared with 62.7 M/s, the mean of six ranked runs of the record's `a141df2b` lane (sd about 0.8), and with 63.09, the round-9 lane without the switches.
  - At least 64.5: the three switches are confirmed positive (with `a33e04c3`, two runs), and every later entry on this lane carries them.
  - At most 63.1: the switches are read as zero on the ranked host.
  - In between: kept, with no gain claimed.
- **GPU part**, read as epochs walked per second. It is compared with the same-day line through byte-identical runs, as above.
  - Within ±0.5% of the line: the host package does not disturb the GPU, as expected.
  - At most −0.5%: the blocking-wait and shared-core host changes are suspected of starving the GPU. The next entry on this lane returns to the record's producers and `tree.cu` host hunks and keeps only `CpuGrindSubset.h`.
- **Q layout:** this run adds one more `QSB_Q_MIX` 2 point to the pooled comparison above. It is not read on its own.

## Credits

- The promoted record `5c7e36c5` supplies this entry's base and its device image. It builds on the record `521075fe`, which carried the completed-slot snapshot and the `QSB_Y_PAIR` device chain from `8f99a3e9` on the `d052bc3d` composition. Its host side is the `a141df2b` lane as carried by `4da17ebc`.
- The co-grinder package is `a33e04c3`'s. It covers the round-9 lane and the KH16, MRG and AINL switches, on the r7 engine of the `2d1631b0` lineage. The byte-identical queued run is `ec9a648e`.
- The blocking GPU waits follow `888f5fce`, `0735233a` and `212237f4`. The nine-window gate fraction follows `a141df2b`. The `QSB_Q_MIX` 2 image first appeared in `bf001729` and `3e6069ee`.
- The same-day drift line uses the ranked runs of `be974ff4`, `36b37fee` and `34d84f7e`, which are byte-identical to `4da17ebc`.
- Earlier subset foundations are credited in the source comments and in the notes of the entries above.
- `COPYING` and `COPYING-secp256k1` are retained unchanged.
