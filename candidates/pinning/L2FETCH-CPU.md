# 64 B L2 fetch granularity + host-CPU co-grinding

This change sits on the promoted frontier `7e95c40c`. GPU arithmetic, table geometry, the pipeline and the exact host gate are unchanged.

## 1. Request a 64 B L2 fetch granularity explicitly

Table records are 64 B, and both of their 32 B sectors are always consumed. `gt_load_signed_flat_m` reads each record as four per-thread 16 B loads.

**Measurement.** A standalone probe on rented RTX 4090s (driver 580, CUDA 12.8) read random 64 B records from a 9.1 GiB footprint:

| L2 fetch granularity | 4 warps/SM | 16 warps/SM (stage-0 occupancy) |
|---|---|---|
| not set (driver reads back 64) | 7.6 G records/s | **4.4 G records/s** |
| 64, set explicitly | 7.7 G records/s | **7.8 G records/s** |

**What this shows.**
- At stage-0 occupancy, a record's two sectors reach DRAM as two independent accesses unless 64 B is requested explicitly.
- The frontier needs about 3.6 G cold records/s. That is close to the default ceiling.

**The change.** One `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64)` call right after `cudaSetDevice`. The readback is printed in the log line `L2 fetch granularity: a -> b B`.

**Result.** Paired runs on two rented 4090s (600 s and 400 s runs, same card, frontier as control): **+0.35% to +0.39%**, with sd 0.04–0.09% across 2–3 pairs.

## 2. Host-CPU co-grinding (`CpuGrind.h`)

**Idea.** The runner's CPU is otherwise idle apart from the exact host gate. Host threads grind the same problem over a disjoint sequence range: sequences count down from `0xFFFFFFFE`, while the GPU counts up from `0x80000000`. Each thread owns one sequence and walks locktimes from 0.

**Algorithm per candidate.**
1. SHA256d through OpenSSL.
2. z·A with A = neg_r_inv·G. This uses sixteen unsigned 16-bit windows over a 64 MiB host table, with batch-affine additions (one Montgomery inversion per step across a 4096-candidate batch).
3. Both recids from one shared denominator.
4. The 24-bit compressed-key test.

**Hit safety.** Every candidate hit passes the same exact OpenSSL check as GPU hits (`qsb_host_exact_hit`) before it is written to `results/pinning_hit_cpu.txt`.

**Threads.**
- Worker threads run at `SCHED_IDLE`, so they never delay the GPU host thread.
- The thread count is the affinity mask or the cgroup CPU quota, minus two.

**Local verification.**
- 0 mismatches against OpenSSL over 600k field operations and 400 table entries.
- At N=12, 1,612 of 1,612 CPU hits passed the harness verifier.

**Result.** About 314k candidates/s per thread on an Apple M5 core. Paired GPU+CPU runs on a rented 4090 box measured +0.25% with the first grinder version, at 18 threads of an old Xeon E5-2696 v3.
