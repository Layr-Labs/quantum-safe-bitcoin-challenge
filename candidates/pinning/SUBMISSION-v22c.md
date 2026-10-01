# Submission v22c — pinning (init-cut with the ordinate-offset pass restored)

## What this is

v22c is v22b plus one line of correctness: the `qsb_table_offset_y` post-pass
is launched again. v22 and v22b both failed the Benchmark step after running
the full fixed window; the root cause turned out to be a dropped kernel launch
in the two-phase table-build restructure, not the signal handling that v22b
hardened. All v22b hardening (SIGTERM re-raise, SA_RESTART, watchdog,
mallocAsync chain fix, stale-error pre-clear, bounded host fallback, cubin
relaunch) is retained. Device code is unchanged — the offset kernel already
existed; only its launch site was restored.

## Root cause of both failures (runs 35941058367, 35947077862)

The scoring harness (`run_benchmark.py`) exits nonzero unless the artifact
verifies: `ok = no failures AND >=1 hit AND K>=100 verified hits
(hit_relative_variance <= 0.1)`. Both runs executed the complete 1200 s window
and then failed — consistent with an artifact containing ~zero verifiable
hits, not with a crash.

`QSB_YOFF=1`: the gtable stores each ordinate as `y + (K-1)/2` so the search
kernels can decode with a pure XOR. In v20 the post-build offset kernel
`qsb_table_offset_y` ran after `kernel_build_gtable` and the spot check.
During v22's two-phase restructure (build moved onto a private stream, sync
deferred to phase 2), the offset launch was dropped along with the old block
and never re-added. The table then held raw ordinates while the search
kernels decoded offset ordinates: every reconstructed public key was wrong,
every kernel-reported candidate failed the exact OpenSSL gate, and the run
published zero hits for the entire window — deterministically, in both
submissions.

The sampled spot check did not catch it because it correctly verifies the raw
table contents (scalar -> point -> limbs vs record); the offset is a consumer
encoding convention applied after verification, exactly as in v20. A launch-site
audit of every `<<<...>>>` in v20 vs v22 confirms `qsb_table_offset_y` was the
only lost launch.

## The fix

`qsb_table_offset_y<<<(GT_TOTAL_ENTRIES+255)/256,256>>>(d_gt)` now runs at the
end of phase 2 — after the event sync, after the sampled spot check, after
the optional cubin-rejected runtime relaunch, and after the host-fallback
copy — so whichever path produced the final table contents gets the offset
applied exactly once, preserving v20 ordering semantics (verify raw table,
then offset). The launch error is checked synchronously and is fatal. The
kernel launches through the runtime path (one small JIT, ~ms, in-window).

## Why this changes the outcome

v20 scored 882.10M with this exact table pipeline. The v22 restructure kept
every byte of device math identical; the only semantic regression was the
missing offset pass. With it restored the kernels decode ordinates correctly
again, hits pass the OpenSSL gate, and the artifact accumulates verified hits
as in v20 — now with the init dead-time elimination (cubin module, overlapped
table build, sampled readback, threaded ladders, cudaMallocAsync) still in
place, which is where the expected +2.3% yield-ratio recovery lives.

## Failure-mode forensics (why this is the cause, not a guess)

The v22b audit had already ruled out the alternatives:

- **Not a bridge/exit-status failure**: v22b re-raises the caught signal with
  SIG_DFL, so the process dies WIFSIGNALED exactly like v20 — yet the run
  still failed after a complete window.
- **Not a timeout**: the Benchmark step ran ~21.4 min (the full window plus
  setup), the same as every healthy field run.
- **Not a stale cubin**: the geometry stamp check existed in v22b and the
  module loads only when stamp, symbols, and readback-verified constants all
  match; a refused module falls back to the runtime path wholesale.
- **Not ladder ordering**: `iso.alpha/beta` and `pp.neg_r_inv` are populated
  before `gt_build_ladders` runs; the threaded build is per-chunk disjoint
  with a serial fallback.
- **Not malformed hit records**: the hit line format is unchanged and the
  parser tolerates partial trailing records.

What remained was the artifact itself: `ok` requires at least one hit and
~100 verified hits. Zero published hits in a healthy 900M/s window means the
searched candidates were all invalid — which is precisely what an un-offset
ordinate table produces, because the search kernels' XOR decode yields
garbage points that the host gate (and the verifier's independent
re-derivation) correctly reject. The bug is silent by construction: nothing
crashes, no CUDA error is raised, the table passes its own correctness
check, and the run consumes the full window — matching the observed
signature of both failures exactly.

## Risk notes

- The offset launch is on the default stream after phase 2; the existing
  `cudaDeviceSynchronize()` before the slot loop orders it ahead of all
  search kernels. It cannot race the build (event-synced earlier) nor the
  host copy (synchronous memcpy).
- The spot check still validates the raw table before the offset is applied,
  so a corrupt build still falls back to the host builder — safety order
  preserved.
- `qsb_geom_stamp` remains baked in the shipped cubin and checked at load;
  the cubin itself is unchanged from v22b (host-only edit).
- If the cubin module is absent or fails any gate, the whole run degrades to
  the v20 runtime path — which is the known-scoring configuration.
