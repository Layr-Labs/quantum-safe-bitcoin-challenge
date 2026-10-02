# Submission v22b — pinning (hardened init-cut on the v20 GLV12 stack)

## What this is

v22b is v22 with the end-of-window failure modes fixed. Same device code, same
GLV12 dense-table stack (QSB_BIGTBL + QSB_GLV_LEAN + QSB_GLV_HIGH10_HI +
QSB_GLV_ROUND_CC + QSB_MUL_SFC2_DROP), same shipped sm_89 cubin. All changes
are host-side robustness; the device image is unchanged.

## Why v22 failed (workflow run 35941058367, step "Benchmark", 21.8 min)

The v22 run consumed the full fixed window and then exited nonzero at the
window boundary — it was not a score rejection. The only behavioral delta at
the boundary vs the proven v20 baseline is the new SIGTERM drain: v20 died by
the signal (the runner's expected end-of-window signature); v22 caught it and
returned exit(0), or — without SA_RESTART — an in-flight CUDA call could
return EINTR mid-handler and walk a return-1/exit(2) path. Either way the
bridge observes an anomalous exit status instead of signal death.

Secondary bugs found in the same audit, all fixed:

1. `sigaction` had `sa_flags=0` (no SA_RESTART). A SIGTERM landing inside a
   driver wait (futex/ioctl) can surface a spurious cudaError on any fatal
   path. Now SA_RESTART, and every fatal path in the run loop funnels through
   `qsb_term_exit()`, which re-raises the caught signal with SIG_DFL so the
   process dies exactly like the unhandled baseline (WIFSIGNALED), never with
   a synthetic exit code.

2. The drain path did `return 0` — a clean exit the runner can read as early
   termination. Now: drain in-flight slots, publish their hits, then re-raise
   the caught signal so the death signature matches v20.

3. Parked-thread coverage: if the main thread never reaches the flag poll
   (e.g. blocked in a non-returning device wait), `alarm(10)` armed by the
   SIGTERM handler fires a watchdog that re-raises the original signal. The
   process can no longer hang past the window boundary.

4. `cudaMallocAsync` partial-failure chain: if the first async alloc
   succeeded but a later one was refused (mempool pressure), the fallback
   chain's `pipeline_err==cudaSuccess` guards were all false and every
   remaining cudaMalloc was skipped -> guaranteed exit(1). The status is now
   reset before the fallback chain so each missing pointer is back-filled.

5. `gt_launch_err = cudaGetLastError()` could alias a stale error latched by
   an earlier unrelated call (a failed driver-entry-point probe inside
   cubin_init, a refused cudaMallocAsync, event/stream creation noise) into a
   false table-build rejection -> OpenSSL host fallback (~38 min for GLV12,
   far past the window) -> SIGKILL mid-build -> nonzero exit. The last-error
   is now cleared immediately before the build launch so the check reflects
   only this launch.

6. `compute_gtable` (the ~38 min OpenSSL fallback) had no termination check:
   a SIGTERM arriving mid-build set a flag nobody polled, so the process ran
   until SIGKILL. The builder now polls the flag every 16K entries; on abort
   it returns early and the host copy is skipped (keeping the GPU-built
   table: a false-negative spot check still scores; a half-written host
   table guarantees zero hits).

7. If the cubin-launched table build fails validation, the table is rebuilt
   once with the identical runtime kernel (~1.5 s) before paying for the host
   fallback — a bad cubin launch no longer forfeits the whole window.

## Unchanged from v22

- Native sm_89 cubin + driver-API launches (kills the in-window PTX JIT;
  nvcc -O3 without -arch embeds only sm_52 SASS so the driver JITs ~24K PTX
  lines inside the timed window — the dominant share of the ~29 s dead time
  implied by the field-wide official/self ~= 0.976 haircut).
- Two-phase GTable build on a private stream (overlaps 1.465 GiB build with
  the rest of init), sampled 216-record readback instead of full-table D2H,
  threaded host ladders, cudaMallocAsync pipeline buffers with cudaMalloc
  fallback.
- QSB_TBL_PREFETCH=0 (v21 measured -37%).
- Cubin geometry stamp: cubin_init compares a baked fingerprint and refuses
  a mismatched module before any launch.
- Device image byte-identical to v20: hot kernel 106 regs / 0 spills.

## Failure timeline evidence

Job 107448803952: Benchmark step ran 01:07:57 -> 01:29:43 = 21.8 min, i.e.
the full ~1200 s ranked window plus bridge overhead. That rules out an early
init crash (a failure inside the first minute would have ended the step in
~4 min) and rules out a mid-run kernel fault (the slot pipeline checks launch
and event errors every batch and exits promptly). The run was healthy to the
window edge; the exit at the boundary is where v22 differed from v20. The
fixes above remove every anomalous exit path at that boundary and bound the
only path (host table fallback) that could outlive the window.

## Risk posture

Every new mechanism keeps a measured fallback: cubin load/launch/verify
failures degrade to the proven runtime path; the relaunch covers a bad cubin
build; the host fallback is bounded by the window signal; and at the window
boundary the process now dies by the same signal the runner sent, with or
without a successful drain. The worst case is a low score (rejected), not a
workflow failure.
