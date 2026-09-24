# Stage timing, RTX 3090 diagnostic

The 24-batch CUDA event profile (excluding first batch for averages) measured
0.317 ms groups, 2.610 ms epochs, 1.385 ms first states, 540.854 ms digest,
and 0.190 ms replay. The digest is 99.1745% of measured kernel time.
First-state optimization alone has a measured upper bound near 0.254% here;
producer fusion cannot plausibly produce a 1% overall gain on this device.
This evidence redirects work to the digest kernel's CTA and shared-memory
geometry. Timings do not establish the stage mix on the official RTX 4090.

profile.cu is a diagnostic source copy with six CUDA events around existing
launches and a report for the first 24 batches. capture.py preserves the GPU
stdout discarded by gpu_wrap.py while delegating all original behavior to
that unchanged harness. No harness file is modified. The first attempt exited
before search because a relative problem directory was resolved from the
runner cwd; the successful command uses absolute paths for problem and output.

GPU invocation was serialized with flock -x /tmp/qsb-gpu.lock. The unchanged
harness ran N=24, seed=2026092501 for 25 seconds and independently verified
hits. Local artifacts and kernel timings remain in output/. summary.json
contains compact measurements. Do not interpret the static RTX_4090 label
inside the harness artifact as actual hardware detection.
