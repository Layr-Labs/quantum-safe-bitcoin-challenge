# Dense-only GLV L2 policy experiment

Source base: pinning candidate `e3e413bb820dc339a11cf30df4de7ade8d179845`,
the source behind PR #1205. That candidate scored 829,282,307 verified
candidates/s on the RTX 4090, below the 835,195,327 promotion floor.

The table's dense physical prefix, segments 2 through 6, is 690,851 records
at 64 bytes each, or 44,214,464 bytes = 42.166 MiB. PR #1205's 50 MiB
persisting access window covers that prefix and the first 128,349 records of
segment 0. This experiment lets the two CUDA L2 controls vary independently:

- `QSB_L2_DENSE_WINDOW=1` ends the access-policy window after segment 6, on
  both the default stream and each slot stream. `0` restores the 50 MiB choice.
- `QSB_L2_DENSE_RESERVE=1` requests only 42.166 MiB via
  `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)`. `0` restores the
  50 MiB request. The request is advisory and must be checked on the actual
  runner. These flags take effect only with `QSB_GLV_DENSE_FIRST=1`.

The four combinations form a 2x2 A/B of the window and the device set-aside.
NVIDIA's L2 guide says normal reads may use unused set-aside capacity, so
shrinking the window alone *may* release effective capacity even though it
does not change the device limit. The independent flags help distinguish it.
The smaller window also removes the persistence hint from about 0.49 segment-0
loads per GLV half, so faster throughput is a hypothesis, not a consequence of
the address model.

Both all-on and all-off versions compiled with CUDA 12.6 for `sm_89`. Stage 0
used 122 registers and zero spills in the all-on build, matching PR #1205's
static resources. `git diff --check` passed. No local NVIDIA GPU is present;
there is no L2 hit/miss, DRAM-byte, or verified-throughput evidence for this
experiment. Keep it out of the ranked queue until one matched GPU comparison
shows a clear end-to-end gain. Measure `lts__t_sector_hit_rate.pct`,
`dram__bytes_read.sum`, table load sectors, and verified candidates/s, with
the same source/problem and alternating policy order.
