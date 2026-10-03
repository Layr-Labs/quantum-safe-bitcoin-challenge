# Pinning research: optional generator precomputation for the exact joint host gate

Effort: max

## Evaluation scope and evidence

This submission contains an exact host implementation change: the joint OpenSSL publication gate plus one optional generator precomputation on the main thread's group. Source-bound CPU correctness and service tests passed. A normal CPU-enabled GPU execution produced 8,986 nominees on a fresh problem; the unchanged canonical verifier independently accepted all 8,986. A separate healthy-GPU, CPU-co-grinder-disabled comparison retained all eight fixed rows and showed no established whole-process improvement. This is one exploratory evaluation with the ordinary CPU co-grinder enabled. It does not claim a measured 1% score improvement, a promotion, or a speedup from the local qualification.

The source starts from the promoted pinning frontier, submission `0fe76103-6afa-41fd-b727-c9b1311186d9`, source `b59a947d5c4d0ac61d2b1136ffdd7b3362010434`. The selected device pipeline is unchanged. The source set has 39 files, with 38 byte-identical to the current control and only `pinning.cu` changed. The carrier image is byte-identical to the control: SHA-256 `2ae6c12c4693eec356a8d09e35aa3f9459e3dbfcf51d52720ba359b94d3d54d6`. There is no native kernel, parameter ABI, geometry, point arithmetic, field reduction or SHA implementation change.

The earlier JOINT_ONLY candidate is relevant prior evidence. It passed correctness qualification but its official evaluation scored 991,348,161 candidates/s and was rejected against the required 1,031,139,710.06 candidates/s. That result does not establish a causal regression of joint multiplication, because it was not a matched control/candidate comparison. It also supplies no positive whole-process evidence for this new combination. The added group precomputation is a distinct implementation change and must earn its own qualification and measurements.

## Exact host change

The original exact host gate derives the transaction digest for a nominated sequence and locktime, computes the scalar `u1`, forms `P = u1*G`, and adds the signed fixed point `R`. The gate then obtains canonical affine coordinates, constructs the 33-byte compressed public key, computes its SHA-256 and checks the required leading-zero predicate.

The reviewed JOINT_ONLY transformation computes the same point with OpenSSL's two-term multiplication:

```cpp
EC_POINT_mul(grp, Q, u1, R, BN_value_one(), ctx)
```

This represents `u1*G + 1*R`. It removes the temporary `P` allocation, the separate generator multiplication and the following point-add call. The supplied `R` is still duplicated for each call, inverted for recid 1 and freed afterwards. The scalar calculation, digest, canonical coordinate extraction, compressed-key packing and acceptance predicate remain unchanged. There is no cached signed `R` or shared mutable verification context.

The new change invokes `EC_GROUP_precompute_mult` once on the main thread's owned verification group, after successful group/order/constant/point initialization and before nominees are processed. It does not invoke the precompute API per nominee. The group is the same named secp256k1 group used by the existing exact gate, and the API's result and active-precompute state are reported once.

Precomputation is optional. The caller does not add an early return, discard a nominee or weaken verification when this optional call returns failure. The ordinary exact joint gate remains in place. Existing group initialization failures retain their original failure path, and ordinary per-call allocation or arithmetic failures retain the existing rejection behavior. This change introduces no alternate acceptance predicate.

The source journal contains the four literal JOINT edits and the one group-initialization insertion. The JOINT edits are reversed in their original function scope; reversing the initialization insertion and then those edits restores the complete control source byte-for-byte. All other 38 source files, including the embedded carrier header and its build recipe, remain identical.

## CPU service experiment

The CPU comparison uses the complete exact host gate, including suffix/midstate digest handling, scalar calculation, owned `R` duplication and sign handling, point arithmetic, affine extraction, full public-key hash, zero test and object cleanup. The comparison is JOINT_ONLY versus the same JOINT_ONLY function with precomputation applied to its owned group. The two gate function bodies are byte-identical. The precomputation setup cost is recorded separately, as the production proposal calls it once rather than on each nomination.

One fixed ABBA sequence was registered and executed: A1, B1, B2, A2. Each position processed the same 2,418 actual nominee tuples. The runtime and headers both reported OpenSSL 3.0.2. The resulting service times were:

| Position | Implementation | Mean time per nominee |
| --- | --- | --- |
| A1 | JOINT_ONLY | 316.239 microseconds |
| B1 | JOINT_ONLY with group precompute | 78.967 microseconds |
| B2 | JOINT_ONLY with group precompute | 78.993 microseconds |
| A2 | JOINT_ONLY | 315.298 microseconds |

The mean across the two controls was 315.769 microseconds, compared with 78.980 microseconds for the two precomputed positions. The measured reduction was about 75.0% in this standalone gate's service time, or about four times its service capacity. Precomputation took approximately 0.658 milliseconds. The control group's active-precompute state remained zero; the candidate group's state was one before and after the proof.

These figures are CPU gate service observations. They are not a GPU rate, a scored candidate rate or a whole-process gain. The selected host loop already queues replacement GPU work before checking older nominees, so some verification work can overlap the GPU pipeline. CPU quotas, library implementation, nomination arrival rate and that overlap can change the result in the normal benchmark.

An older precompute experiment tested the original generator-only gate and found no service improvement. This new test applied precomputation to the two-term joint gate. The older negative result is retained; it is not being relabeled as a positive result or treated as evidence for the current path.

## Correctness and publication

Before timing, the CPU fixture checked all 2,418 nominees with both the nominated recid and the opposite recid. Across 4,836 pairs, it compared the complete 33-byte compressed public key, all 32 public-key SHA bytes, all 32 transaction-digest bytes and the acceptance flag. All comparisons passed. Every nominated recid was accepted, and the opposite recids in this fixture were misses.

The fixture also compared the preferred-first and alternate-first publication wrappers, giving 4,836 publication-result comparisons. Both implementations preserve the nominated-first behavior and the fallback to the other recid when the nominated one misses. Four simulated missing-u1/Q allocation checks retained rejection. Those checks do not claim that every possible OpenSSL allocator failure was forced; the optional precompute call adds no acceptance branch.

The fixture retains both recids even though the ordinary nominated-first path often returns after one successful verification. No recid is removed from the algorithm, and there is no key-space restriction, approximate field helper or skipped cryptographic round. Production still derives and verifies each nominee from the actual problem's digest and constants.

The GPU and host SHA ownership mechanisms are unchanged. The candidate does not alter accepted owner snapshots, job descriptors, pending-work accounting, retained planes, DMA completion or slot reuse. The independent CPU co-grinder remains enabled by its ordinary production default, and its backend selection and controller are unchanged. The precompute insertion is in the main thread's verification-group initialization; it does not add a new worker, a queue or a worker scheduling policy.

## Integration results and reproduction

The protected benchmark and verifier source files are unchanged. Public production reproduction commands are:

```sh
./setup.sh pinning
./benchmark.sh pinning
```

The normal production source uses the ordinary CPU co-grinder and host SHA policy. The local qualification used the documented command adapter `cmd:python3 harness/gpu_wrap.py --src candidates/pinning/pinning.cu --no-build` because the configured bridge executable was unavailable on the local machine. It used a 75-second diagnostic interval, N=24 and fresh problem seed 11323965. There were no finite-work instrumentation hooks in the production source and no CPU co-grinder disable flag. The source was fixed before that fresh seed was selected.

The GPU/CPU execution emitted 8,986 nominees. The outer local controller expired at 360 seconds while a two-worker canonical verification was still running, after the grinder had ended. The exact captured artifact was retained, its orphan verification workers were stopped, and the unchanged canonical verifier finished validation of that same artifact with six workers. All 8,986 distinct nominees passed, with no verification failures. No GPU experiment was repeated or selected. Because the interrupted outer harness did not persist its scored clock, this note records no local qualification score and does not reconstruct one from the grinder's self-reported rate. The captured artifact SHA-256 is `69c08e4d81c0eac431713e6f0a013fa6e4ea3636b609f6ac22a22e25eb3cb48d`.

A separate integration fixture used the complete production GPU geometry with the ordinary natural host SHA controller and the CPU co-grinder disabled, to compare the promoted control against this host change. Both native images remained exactly `2ae6c12c4693eec356a8d09e35aa3f9459e3dbfcf51d52720ba359b94d3d54d6`. Each row performed 128 warm sequences followed by 256 timed sequences, with K=4, full 131,072 subtiles and the unchanged owner/slot protocol. The expected warm/timed completed work was 19,913,601,024 and 39,827,202,048 candidates; expected subtile counts were 151,936 and 303,872. Every row required the exact respective nominee sets, completed counters and drained owners, not a peak progress rate.

One fixed eight-row sequence was used: A1, B1, B2, A2, B3, A3, A4, B4. All eight rows passed exact nominees, native activation, root startup, completed-work, drain and hardware identity checks. All eight warm and timed endpoints had the same host-policy admission and backend. The optional group precompute reported result=1 and active=1 in all four candidate rows; the null result is not explained by an inactive cache.

| Comparison | Candidate/control timed throughput |
| --- | --- |
| First four rows, ABBA | 0.9998447750 |
| Last four rows, BAAB | 0.9997506474 |
| All eight rows | 0.9997977078 |
| Complete process work / wall time | 0.9989271198 |

The overall timed delta was approximately -0.0202%, and complete-process work/wall time was approximately -0.1073%. This fixture demonstrated correctness but no whole-process gain. All rows were retained; no thermal, clock or controller-state subset was selected. The eight-row result is inconclusive by its prospectively fixed threshold. CPU service latency and CPU-disabled pipeline throughput are different measurements. The normal CPU-enabled ranked run is the context this exploratory submission asks to evaluate; no gain in that context is established here.

## Group ownership, allocation and optional-cache fallback

The production `gate_grp` and `gate_ctx` are local to the main thread. They are initialized once before CPU worker startup, and only the serial main publication loops pass that group into the exact gate. CPU co-grinder workers create their own groups; their headers and initialization are unchanged. This proposal adds one main-group cache, not a cache per worker, and does not introduce concurrent mutation of a shared group.

A separate single-CPU OpenSSL 3.0.2 audit measured one precompute call at 0.665 ms. The observed extra retained heap allocation was 81,136 bytes, with zero additional mmap allocation. Process RSS rose by 2,076,672 bytes, including first-touched library and allocator pages; that RSS increase is not the size of the cache objects. These are one audit's observations, not a universal memory bound or peak transient allocation guarantee.

A separate optional-return-zero simulation omitted the precompute API call on an initialized group, leaving active precompute at zero. For all 2,418 canonical nominees and both recids, the exact joint gate matched both the no-precompute joint control and the actually precomputed group on all 4,836 full public-key/hash/digest/acceptance comparisons per arm. This tests the no-cache fallback path. It does not claim to exercise every internal OpenSSL allocation failure or partially failed precomputation state. The production caller never changes its acceptance predicate based on the optional precompute result.

## Lineage and attribution

The promoted frontier is the source lineage for the device pipeline and ordinary host/CPU mechanisms. Joint point multiplication is prior art, including public pinning work by jtaroreh and earlier implementations; this note does not claim a new elliptic-curve identity. The proposed source edits were derived from our own isolated JOINT_ONLY candidate and the existing OpenSSL API. No unrelated device, table, geometry, host SHA or CPU scheduler change is included in this combination.

The new result is the source-bound observation that explicit generator precomputation substantially changes the current joint gate's CPU service cost on the tested library. Its practical value for the ordinary CPU-enabled ranked run is unresolved. The integration data above showed no measured CPU-disabled pipeline gain. This submission records that limitation and requests one complete evaluation of the exact production source.
