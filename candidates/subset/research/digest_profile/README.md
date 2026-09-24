# Sampled digest phase clocks, RTX 3090 only

384 thread-zero block samples from batches2-4 give approximately18.64% SHA,
39.48% first point front,22.68% second point front,8.85% product/inverse tree,
and10.35% tail. All244emitted hits passed CPU verification. These are sampled
clock64 intervals within the kernel, NOT isolated-kernel throughput. They
include scheduling/cache effects, and compiler motion or the sampling probes
can affect phase attribution. In particular the asymmetry between point fronts
means the individual values must not be interpreted as intrinsic arithmetic
costs; their combined62.15% is the useful prioritization clue. No score claim.

The diagnostic adds six device timestamp stores for thread0 in the first128
blocks, reports four launches, and retains the ordinary default candidate
arithmetic, launch dimensions and buffers. Compilation still reports128
registers, zero stack and49,152bytes shared for the consumer. Timed run12s,
N=24, seed2026092501, unchanged harness and verification. Cold JIT/startup
substantially affects this short wall-clock score, which is not an optimization
comparison. Source, build log, raw kernel log and summary are preserved here.
