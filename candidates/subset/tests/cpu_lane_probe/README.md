# Terrapin CPU lane comparison candidate

Base: combined `228e4ee`. CPU arithmetic and table implementation come from public submission
`0361b3a2-c696-4635-bc15-1dc0c3da3a79`, source commit
`f2c7267f64f8df2d44d0d3b169f9f40818866d55` (terrapinelf, attributed there to
Claude Opus 5.5 / Claude Code and its coauthors). This integration was prepared
by GPT-6 Astra / Codex. Existing licenses and attribution remain in the tree.

The CPU header is replaced, with the conservative placement changes documented in
`../cpu_placement/README.md`. The entry disables QSB_CPU_DIAG_EPOCH, preserving
epoch start zero rather than encoding machine details into candidate enumeration.
Producer, GPU kernels, carrier, exact gate, harness and problem files are unchanged.
The CPU header is an alternative pipeline; it must not be layered as an arithmetic
patch onto Megan's f16 loop. Do not also transplant the public native-image file.

This is a comparison candidate, not a validated performance claim. Public notes
report CPU instruction/prefetch improvements, real correctness checks, and a
proxy-based 10-window estimate, but no measured >=1% total gain over combined228e4ee.
The producer improvements are already in our base. The 10-window proxy is not a
measurement of a resident 17-GiB table on the ranked machine.

## Common CPU screen

The original QSB_CPU_DEVBENCH hook is absent in Terrapin's header. `wrapper.cpp`
therefore times cand increments after the first completed batch, using only fields
both headers share. Compile both the control and candidate with this same wrapper,
without QSB_CPU_DEVBENCH. It waits at most 120 seconds for setup, warms for one
second, measures 1..120 seconds, flushes hit output, then exits. It does not modify
candidate code for timing. No screen was run by the source-review agent.

Example build from repo root:

```
g++ -O3 -std=c++17 -Wno-deprecated-declarations -pthread -DQSB_ZEROS_N=16 \
  candidates/subset/tests/cpu_lane_probe/wrapper.cpp -lcrypto -o /tmp/terrapin-probe
```

Use `-DCPU_HEADER='"/absolute/path/to/control/CpuGrindSubset.h"'` to compile
an otherwise identical control wrapper. Pass `problems/subset.bin 20` as argv.
Use a separate run directory per arm; the wrapper creates its results directory.

Controls required: same problem, same host/core affinity, identical explicit
QSB_CPU_THREADS_ENV, memory envelope, and same ABBA order. Keep the control's PFN=1.
Start with automatic geometry, recording both geometries, memory, huge-page backing,
build completion and fallback messages. A geometry-matched follow-up can use
QSB_CPU_NW=11 for this candidate if the control uses eleven windows; this is a
candidate-specific diagnostic override, not a production default. Do not silently
compare a 3.75-GiB candidate table with a different-memory control and attribute
the result solely to arithmetic.

This staging branch removes the public header's unchecked qhp::g_share_cpu path
and its automatic worker-count expansion. Use an explicit identical thread count
for every controlled CPU or GPU comparison. The conservative placement layer
reserves the observed main CPU(s) and proven physical siblings; this does not enable
our separate optional safe-sharing implementation. Full native GPU/build/all-hit
verification is still required before submission.

The source wrapper predates the parent's measured-process shutdown fix: use its
io-lock-protected flush and `_Exit(0)` copy when launching detached-worker probes;
ordinary global teardown can race detached workers. This placement commit does not
modify that independent wrapper or any active measurement checkout.

## Stop gates

The combined result is +0.4684% total against crown. Reaching +1% needs a further
+0.5291% total relative to combined. At a 7..7.5% CPU share, that is approximately
+7.1..7.6% CPU with the GPU unchanged; require ~10% stable CPU uplift to justify
the next GPU screen. Retire if the proper matched control is negative or gains
are within control drift. Any invalid hit, wrong carrier, table failure, unbounded
setup, or changed enumeration is a stop. Final promotion depends on the live
frontier and the full verified score, never this CPU screen alone.
