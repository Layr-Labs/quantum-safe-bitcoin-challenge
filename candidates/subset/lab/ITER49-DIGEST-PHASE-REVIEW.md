# Iteration49 — sampled repeated digest phase census (diagnostic only)

## Hypothesis and disposition
The startup prefix direction is spent locally (iteration48 bounded the measured
one-time table cost). Iteration28 already censused GPU/host producer routing.
This iteration instead measures boundaries INSIDE the repeated digest kernel:
paired window SHA, two recovery fronts, denominator cross work, block inversion,
and two tails. No phase is replaced and no candidates are omitted.

**No performance candidate, qualification or submission.** The diagnostic adds
branches, timestamps, stores, CSV collection and host synchronization; its score
is not a before/after improvement. Production source selector and embedded
native carrier are unchanged. Default-off sm89 and matched sm86 cubins reproduce
iteration48 production byte-for-byte.

## Implementation and logical review before the verified run
`QSB_DIGEST_PHASE_AUDIT` defaults to 0. Enabled builds must explicitly bypass the
embedded carrier and use the paired, dual-epoch, K2S3M, 128-thread slot pipeline.
Every 8192nd CTA samples the first lane of each of its four warps. Seven clock64
issue timestamps go to a bounded 4096-row buffer for the appropriate slot.
Slot selection compares the exact d_first base pointers; both are distinct in
the measured production layout. Clears and kernel launches use the same stream.
Collection happens only after the existing slot completion event; buffer reuse
happens only after collection. Thus the two in-flight slots cannot overwrite one
another. Empty samples remain zero, the array is capped, and the checked CUDA
allocation/copy failures stop the diagnostic. The results FILE is flushed per
batch (normal benchmark stop uses _exit, bypassing C++ destructors); normal exit
also frees both allocations and host memory. No barrier or arithmetic values,
point association, enumeration, gate or hit publication was changed.

Initial source-anchor assertion failed BEFORE tree.cu was written. A subsequent
build of that uninstrumented source was not run on the GPU. The unique anchors
were corrected, the actual instrumented binary rebuilt and hashed before the
single verified run. These failures are retained, not passed off as measurements.

## Command and existing independent verifier
Build: `nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_86 -Xptxas=-v -o <audit>
 candidates/subset/lab/n24L_iter49audit.cu -lcrypto -lm`.
Run: `flock -x /tmp/angel-gpu.lock env QSB_AUDIT_BINARY=<audit>
 QSB_GRINDER='cmd:python3 candidates/subset/lab/iter49-grinder.py --src
 candidates/subset/lab/n24L_iter49audit.cu' QSB_ZEROS_N=24
 QSB_MODE=fixed_time QSB_SECONDS=90 QSB_MAX_REL_VAR=none
 QSB_PROBLEM_SEED=1789110211 ./benchmark.sh subset`.
The adapter uses the pre-existing gpu_wrap parsers and artifacts and the
pre-existing benchmark.sh/run_benchmark/verify/score path. It bypasses only
kernel_argv's `timeout` wrapper: Popen runs the kernel directly, then sends its
existing graceful SIGTERM stop after the requested window and waits for the
full drain without a kill timeout. Native stdout/stderr and every artifact are
retained. The independent exact verifier is unchanged.

N24 seed1789110211 **4210/4210 exact hits PASS** (3577 GPU + 633 CPU).
Diagnostic verified throughput385.342070M/s, elapsed91.6485s. Self-count
29,796,000,000 is below35,316,039,680 inferred from hits; the existing Poisson
advisory is retained. These numbers cannot qualify an optimization. Actual GPU
is local RTX3090/sm86; the harness artifact's RTX4090 label is not hardware proof.

## New phase evidence
111 complete launches,2097152epochs/launch,512sample rows/launch =56832rows.
All seven timestamps strictly increase; sample IDs0..511 occur exactly once
per batch; slots equal batch parity. After excluding batches0..19,46592rows:

| interval | mean cycles | median cycles | fraction of summed observed intervals |
|---|---:|---:|---:|
| entry through paired SHA |107933.3|94355.0|24.585%|
| recovery front A + parking |111400.2|94713.5|25.374%|
| recovery front B + reload |97586.9|84496.0|22.228%|
| denominator cross + leaf product |4053.0|3405.0|0.923%|
| shared block inversion |58940.7|49930.0|13.425%|
| both tails/gates/publication |59115.2|50698.5|13.465%|

Thus the combined fronts account for47.602% of observed boundary intervals,
paired SHA24.585%, and inversion13.425%, in THIS instrumented local image.
The distribution does not nominate shared inversion as the largest interval;
front arithmetic/gather or paired SHA is the stronger next work-elimination
lead than further tree address/carry tuning. This is an evidence-based priority,
not a proven bottleneck decomposition or a performance candidate.

SASS contains exactly seven CS2R SR_CLOCKLO sites at offsets00e0,12d20,13210,
13550,14b80,1a500,1ac30. The two front intervals each enclose one call to the
same body at2be90; inversion encloses the root call at1ad20; tails enclose two
calls to1d680. Digest14800instructions,128registers,24KiBshared,0stack and
0spill stores/loads. Assembly confirmation validates placement/calls, NOT
complete dependency serialization: timestamps are issue boundaries. Pure
arithmetic can be scheduled across source boundaries. Clock64 includes warp
scheduling, barriers, stalls and the instrumenting branch/store overhead.
Sampled warps have lane divergence. Host collection adds synchronization.
Interval sums are not GPU service time, cannot be added as independently
removable work, and give no Amdahl bound or RTX4090 prediction. One diagnostic
run is not a repeatability or causality claim.

## Production identity and next action
Disabled native cubin:473504B SHA256a724399a04db515f8bebd836b1af866e6e7a1b778a98ce73a384c6f7704de147.
Disabled matchedlocal cubin:473632B SHA25695598e445005e424dbee6feb03f87adeb4c498afe1b07250fa0e169e9edbbfa5.
Both cmp-identical to iteration48. Production default/selector remain untouched.
Board recheck at entry: faf5422a/efef868,753.571538M/s,target761.10725338M/s.
Read faf5422a and d7c57dd4 public notes; efef868 public tree is local git source.

Next distinct necessary experiment: isolate front gather latency vs field
arithmetic, or eliminate work in paired SHA (not outlining/loop-bound/address
redraw). Any actual candidate still needs positive production incrementality,
>=3 alternating pairs against the actual current leader,+4%local gate and
exact PASS before immediate attributed submission. No gate run for diagnostics.

Lossless CSV codec and archival compaction retain every timestamp and earlier
payload; final integrity/cap receipts are recorded separately.
