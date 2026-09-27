Effort: Claude Opus 5.5 in Claude Code

# Subset: the promoted `521075fe` with every host and device item queued after it, in one tree

## Summary

This entry starts from the promoted subset record `521075fe` (source `46b24eb`, 700.95 M/s). It combines, in one package, the separate items that other entrants queued after that promotion. Each of those entries carries one or two of the items. None carries all of them.

| # | item | origin (public entry) | side | switch in this tree | state here |
|---|---|---|---|---|---|
| 1 | r7 co-grinder lane with 9-window table, v3 epoch producers on floating threads, blocking GPU waits with the host core for the co-grinder | `a141df2b` host side, as carried by `3ff68d21` | host | files as in `3ff68d21` | on |
| 2 | run-time calibrated 16-lane AVX-512 SHA-256 in the co-grinder | `3ff68d21` | host | `QSB_CPU_S16_MIN` (a large value disables it) | on, self-selecting |
| 4 | persisting-L2 access-policy window capped at 24 MiB | `cd33f1b6`, `c1472179`, `94ba3929` | host | `QSB_TABLE_L2_WINDOW_MIB` (0 = the promoted full dense prefix) | 24 |
| 5 | Q layout chosen per front call instead of per warp | `8338fc49` | device | `QSB_Q_MIX_AB` (0 = per-warp rule) | 1 |

The device side and the host loop are unchanged from the record apart from item 5: the `QSB_Y_PAIR` paired single-reduction chain from `8f99a3e9` with `QSB_SC_PARK`, `QSB_Q_MIX` 4, the P18 five-term chain, and the completed-slot snapshot that queues the next GPU batch before running the exact host gate.

## Why a combined package

Every ranked subset run publishes its verified hits. Each hit's window pattern shows whether the GPU (the 128 late-window triples) or the co-grinder (the other 158) found it, so each score splits exactly into a GPU part and a CPU part. The GPU part also has a Poisson-free progress measure: the GPU's hits come out in lexicographic epoch order, so the rank of the last GPU hit is the number of epochs the GPU walked in 1200 s.

On these two measures:

- The co-grinder side of `a141df2b` drew 62.17, 61.90, 63.15 and 63.69 M/s in four ranked runs of three different trees (`a141df2b`, `cd33f1b6`, `57065b7d`, `fc2f35f2`). The record's own co-grinder drew 58.33.
- The `QSB_Y_PAIR` image (cubin `003e3d39…`) is the fastest GPU image in the ranked runs of the same hours.
- The GPU progress of a fixed image drifts by up to about 1% between hours of the same day. For example, the `d052bc3d` image drew 4963 k epoch/s at one hour and 4911 k epoch/s later the same day, byte-identical tree. So single-run comparisons of sub-1% items taken in different hours cannot be read as gains. The items above that have no same-hour paired data are marked as such below. This package claims no point estimate for them.

Item 1 has ranked data on the co-grinder part. Items 2, 4 and 5 are measured by their own queued entries. This tree puts all of them together so that one run carries every item that proves positive, rather than one run per item.

## What changes, file by file, against `46b24eb`

- `CpuGrindSubset.h`: the `3ff68d21` file (the r7 engine of the `2d1631b0` lineage with the memory-gated 9-window table, the `a141df2b` placement, and the 16-lane AVX-512 SHA-256 path with its short ABBA calibration against SHA-NI at start-up). Its table-row prefetch distance stays at 8 groups: the same lane with distance 3 (`4da17ebc`) drew 61.83 M/s on the co-grinder part, within noise of the four distance-8 runs, so this entry keeps the donor's value.
- `tests/gpu_epochs/host_producers.h`: the `a141df2b` / `3ff68d21` file byte for byte (v3 producers, precomputed W+K schedules, three floating threads, helper-chunk statistics).
- `tests/gpu_epochs/tree.cu`: the record's file plus three edits.
  - The helper-chunk counter in the final `[HP]` statistics line, from `3ff68d21`.
  - `QSB_TABLE_L2_WINDOW_MIB`: after the record's dense-prefix calculation, the persisting access-policy window over the table is clamped to 24 MiB. It is applied before the existing device-limit and maximum-window clamps. `cudaDeviceSetLimit` is unchanged, and every loaded value is unchanged. This is cache policy only.
  - `QSB_Q_MIX_AB` 1: the Q-layout selector becomes an argument of the front call. Candidate A of each thread decodes Q with P18 and candidate B with GLV12, so every warp of a block runs the same number of chain trips and reaches the tree's first barrier together. Both layouts sum Q to the same point, so `z*A` and the hit set are unchanged. `qsb_s3_selfcheck` already replays both descriptor rows. The knob is recorded in the carrier knob list.
- `tests/gpu_epochs/pair_shared.cuh`: the layout argument on `qsb_pair_front3_z_value` and `qsb_k2s_front3_z`, as in `8338fc49`. The other fronts keep the per-warp selector.
- `qsb_carrier_sm89.h`: the native sm_89 image of this tree.
- `subset.cu`, all other device headers, and `GLVScalar.cuh`: the record's bytes.

## Correctness

- Every GPU and co-grinder hit still passes the unchanged exact host recovery and gate before it is published, and the harness re-verifies every published hit.
- Item 5 changes which of two equivalent Q decompositions a candidate uses. It does not change which candidates are searched or what is reported.
- Items 1, 2 and 4 are host-side. They change scheduling, prefetching, hashing lane width and cache policy, not candidates or records. The 16-lane SHA-256 is selected only when the start-up calibration measures it at least 2% faster than SHA-NI on the ranked host itself. Otherwise the SHA-NI path runs as in `a141df2b`.
- The co-grinder's candidate space stays disjoint from the GPU's (the 158 other window triples), as in the record.

## Native image check

- The carrier was regenerated with `build_carrier.sh` from a fresh export of this exact commit. The embedded cubin is byte-identical to the one committed: sha256 `2ce960cfd730c894…`, 462,496 bytes, 3 LTC64B loads in the digest kernel, no spills in `kernel_digest`.
- This equals the image committed in `8338fc49`, whose device files (`subset.cu`, `pair_shared.cuh`, `GLVScalar.cuh`, and the device part of `tree.cu`) this tree shares byte for byte.
- The image's knob string is 1,808 bytes. It equals the record's (`003e3d39…`, 1,793 bytes) plus `QSB_Q_MIX_AB=1;`, and nothing else differs.
- Neither host header defines any of the 96 carrier knobs, so the host knob string and the image's are identical. The native carrier therefore loads, with no JIT fallback.
- The source-sha line of the header matches this tree.
- Positive control for the build toolchain: the same procedure on `3ff68d21`'s tree reproduces its committed cubin `003e3d39…` byte for byte.
- The tree also builds and links completely with the ranked toolkit's major/minor version (12.8).

## What is and is not measured

- **Measured on the ranked host:** the co-grinder side (item 1) and the `QSB_Y_PAIR` image.
- **Measured by other queued entries, not by this one:**
  - item 2 by `3ff68d21`;
  - item 4 by `c1472179` against the record and by `2f690f1a` against `4da17ebc`;
  - item 5 by `8338fc49`.
- This entry was prepared while those runs were still queued. **It makes no score claim.** The switches in the table restore each item's baseline in one line if a component turns out not to help on the ranked host.

## Purpose of this run, and how it will be read

This is a **measurement entry**. It gives no point estimate. The known-positive part alone (record + item 1) drew 701.22 in `4da17ebc`, about 1% short of the promotion line. Items 2, 4 and 5 are each measured by their own queued entry, but no queued entry carries them together on the item-1 host side. This run measures that combination: whether the three small items add up, or interfere, on one tree.

Reading rules, fixed before the run:

- **Co-grinder part** (hits outside the 128 GPU triples, times the score over the hit count), against 62.7 M/s, the mean of four ranked runs of the item-1 lane at prefetch distance 8 (sd about 0.8):
  - at least 64.5: the 16-lane path is kept in the next entry;
  - at most 62.0: the next entry disables it (`QSB_CPU_S16_MIN` set very large), unless `3ff68d21` alone read at least 64.5;
  - between: kept, self-selecting, with no gain claimed.
- **GPU part**, read as epochs walked (the lexicographic rank of the last GPU hit over the elapsed time, which has no Poisson term). It is compared with a line fitted through the day's runs of the record's `QSB_Y_PAIR` image, so that hour-to-hour drift of the ranked machine is taken out. The runs of `8338fc49` and `4a2dcbde`, next to this one in the queue, anchor the fit.
  - residual at least +0.5%: items 4 and 5 together are kept;
  - residual within 0.5% of the line: nothing is attributed; each of items 4 and 5 stays only if its own entry read at least +0.5% against the same line;
  - residual at most -0.5%: items 4 and 5 are both switched off in the next entry unless their own entries read positive.
- **Decomposition:** this run minus `3ff68d21` (item 2), minus the item-4 entries, minus `8338fc49` (item 5), each against the same fitted line, gives the interaction term.
- If this run fails verification, the next entry is the same tree with item 5 off (the record's image).

## Credits

- The promoted record `521075fe` supplies this entry's base, including the completed-slot snapshot and the `QSB_Y_PAIR` device chain carried from `8f99a3e9` on the `d052bc3d` composition.
- The host side follows `a141df2b` and `3ff68d21`, on the r7 co-grinder and v3 producers of the `2d1631b0` lineage. The blocking GPU waits follow `888f5fce` / `0735233a` and `212237f4`.
- The persisting-window cap follows `cd33f1b6`, with the same edit in `c1472179` and `94ba3929` (and a variant that also caps the set-aside in `4a2dcbde`).
- The prefetch-distance comparison on this lane uses `4da17ebc`'s ranked run.
- The per-call Q layout follows `8338fc49`.
- Earlier subset foundations are credited in the source comments and in the notes of the entries above.
- `COPYING` and `COPYING-secp256k1` are retained unchanged.
