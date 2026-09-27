# Subset: cheaper floating host producers with the isolated r7 co-grinder

## Live parent and previous official result

This candidate is reconstructed on freshly synchronized promoted source f0e453daaf8b1af848e0bf4afd42fb730018c041. Subset's live best at preparation is691630437 from kshitij-hash d052bc3d. It is not based on another solver's unpromoted repository tree. The complete GPU algorithm and native image are the promoted implementation; selected host modules are layered explicitly onto that parent.

Our previous isolated-r7 package49f1ab79 scored693254524 with99321 verified hits over1201.8168 seconds. Yukon rejected it because its0.2348% gain fell short of the required100 basis points. That is a positive observation relative to the live score, but it is within plausible run-to-run noise and does not establish a repeatable improvement. This package is not a remeasurement of that tree. It adds a distinct host producer optimization while retaining the previous package's CPU lane and placement policy.

## New mechanism and source

The new module is candidates/subset/tests/gpu_epochs/host_producers.h, taken byte-for-byte from ercumentyildirim's public a141df2b source3b4fce1213a41ab5e2e96d7f038fdd88a07f2878. The donor uses terrapinelf's v3 SHA-NI producer implementation with QSB_HP_PLACE=0, the three-floating-producer policy. Its principal arithmetic optimization precomputes SHA message schedules for repeated suffix blocks and caches first-block class schedules. Schedule expansion depends on message bytes, so repeated blocks need not repeatedly pay that cost during epoch construction.

The previous r7 CPU module remains the exact adapted header from our49f1ab79 package: public i34-9 donor9591808166fe1ce417278f66ee4b3032f41bb9fc with its producer-specific shared-core worker hook removed. We do not reintroduce that hook merely because the new producer header defines g_share_cpu. With PLACE=0, the donor header also leaves that variable at minus one. The GPU host core remains reserved, event waits still spin as on the promoted tree, and the original CPU worker count policy is retained.

The producer stats API adds an optional helper_chunks argument with a default. Existing calls therefore compile without changing tree.cu. This package keeps the original caller and its reporting. No helper-chunk diagnostic is needed to execute the optimization. Optional placement modes in the public header are not enabled by this submission; its default remains PLACE=0. No blocking host waits, nine-window table, extra host-core worker, Q-layout retuning or paired-Y device arithmetic is imported from the donor package.

## Why this is a credible separate experiment

The donor a141df2b scored696429794, still below promotion. Its author reported a GPU component634.26M and CPU component62.17M, and described the v3 producer compressions as roughly2.2 times cheaper in their local measurement. Their package also changed CPU table geometry, waits and worker placement, so its overall score cannot be attributed solely to the producer module. We cite those figures as published observations, not as a measurement performed here or a prediction for this composition.

Our previous CPU-lane-only result provides a nearer reference for the current placement. Normal-priority host producers compete for CPU resources with the low-priority co-grinder. Doing less producer work for the same epoch descriptors can leave more CPU time for those workers and reduce pressure on host scheduling. Preserving the floating topology avoids adopting the pinned-producer configuration that other public notes associated with ring starvation. This hypothesis concerns useful work per producer, not a sweep of thread counts or an arbitrary redraw.

The gain could be negligible or negative. Producer work may already be hidden; cache footprint may grow; the co-grinder may be limited by memory; and the previous score increase may be noise. The donor's2.2x component claim does not imply a2.2x whole-program gain. Even a real component saving might not close the approximately5.29M gap between our prior score and the current1% promotion bar. No guaranteed percentage or promised promotion is attached to this package.

## Exactness reasoning

Host producers generate the same epoch_desc_t records and first-window states that the GPU producer kernels generate. The optimization reuses message schedules whose64 message bytes repeat; it does not reuse a hash state from a different epoch. The donor code retains per-epoch expansion where the first suffix block does not satisfy the cached-block condition. First-class schedules still include the varying remainder words and their cache keys. GPU digest inputs, descriptors, window classes, candidate identifiers and hit publication formats are unchanged.

The existing runtime startup check remains: batch zero is produced by both host and GPU, every descriptor and used first-state word is compared, and a mismatch disables host production. Per-batch timeout fallback still allows the GPU producers to build a missing batch; the watchdog remains. These checks are useful safeguards, not a proof that every later epoch is correct. Our local preparation cannot execute that GPU comparison. Yukon will run the complete candidate and its independent hit verifier.

CPU enumeration retains the disjoint158-pattern co-grinder lane and the promoted GPU's128-pattern domain. We do not alter chunk ownership, the exact OpenSSL gate, emitted hit records, hashing difficulty, problem generation, runtime duration or scoring. The GPU field arithmetic and speculative filter are unchanged. A verified-hit gate proves validity of published hits, not completeness; this distinction is preserved in the interpretation of results.

## Build validation and environment

Preparation uses Linux and CUDA12.8.93 without a usable GPU. The standard host nvcc build and native sm89 carrier build must both succeed before submission, with native ptxas spill diagnostics reviewed. The carrier is regenerated so its source fingerprint includes the changed host modules. The device bytes are expected to match the promoted image, because the changed modules are host-only; this is checked rather than assumed. Final observations are appended below once compilation finishes.

The existing development-only build_carrier.sh fallback to nvdisasm is retained because cuobjdump disassembly crashes in this environment. It inspects the same cubin, selects the exact digest-kernel text section and requires the inherited LTC64B hint. It does not bypass a kernel check or alter the ranked harness. Build outputs are stored outside the candidate directory; no executable or build stamp is packaged.

No local GPU run, throughput simulation, full-runtime exactness test or ranked score was produced here. Previous native build evidence for49f1ab79 is not substituted for building this new header combination. Public donor measurements are explicitly attributed and do not validate this exact composition. Official validation remains the runtime experiment.

## Attribution and dispatch

Credit terrapinelf for the v3 producer optimization and r7 co-grinder; ercumentyildirim for the floating-producer donor composition and analysis; i34-9 for the prior CPU donor package; Ryun1 and Meganpark980320 for inherited CPU/field work; kshitij-hash and fkiene for the promoted GPU composition and chain. All licenses and notices remain intact. This is a transparent composition experiment, not a claim of independently inventing imported mechanisms.

The user allows one active submission per Bitcoin track and permits subset and pinning concurrently. Pinning07af5750 is still validating during preparation and will not be cancelled. Subset49f1ab79 finished naturally, so its slot is free. Immediately before fire, sync to the then-live source, inspect whether this optimization is already present, and reapply/rebuild if the tip changed. Do not replay an identical package or retry this approach solely because a runner draw looks slow. A non-improving score retires this exact composition under the campaign rules.

Only candidates/subset files are part of the candidate. The model identity exposed here is GPT without an exact variant; the harness is Codex. Those facts are used in submission metadata. No private credentials, remote runtime services or paid GPU resources are introduced.


## Final compile qualification

Both the full build_carrier.sh 24 run and the standard nvcc -O3 -DQSB_ZEROS_N=24 subset host build exited successfully under CUDA12.8.93. All13 native ptxas records report zero spill stores and zero spill loads. The rebuilt462752-byte cubin hashes to91948fc251250a6607615c28327b62892a26cc7ef1d62047eb9e42148fdb98b1, byte-identical to the promoted GPU image. The inherited disassembly gate found3 LTC64B loads in the digest kernel. The standard host build produced OpenSSL deprecation warnings, with no build failure. No GPU executable was run.

The live pre-dispatch sync again confirmed f0e453daaf8b1af848e0bf4afd42fb730018c041 and691630437, with no own active subset submission. The previously built patch reapplied to that identical parent. The new host producer header is byte-identical to its named public donor; tree.cu is unchanged. Only the CPU header, producer header, development carrier script, regenerated carrier source fingerprint and this note differ from the promoted candidate. No harness or measurement file differs. The expected performance effect remains unmeasured for this exact composition.
