# Split paired consumer pipeline

Status: implemented and compiled; first whole-solver and completed-range hit
checks passed. The larger-batch variant is being qualified against a fresh
baseline. No official submission or promotion has occurred.

Motivation: main digest accounts for99.17% of kernel time. Sampled internal
clocks prioritize point-front work (~62% combined), with about9% tree and10%
tail, but probes and compiler scheduling limit that attribution. Both128and
512-thread monolithic alternatives regressed; their source diffs and audits
are recorded separately. This experiment retains the original256-thread
pairing/tree geometry and changes stage boundaries.

A paired front kernel computes exactly the existing paired SHA and two point
fronts, writing16 uint64 fields per candidate as structure-of-arrays. A validity
byte accompanies each record; invalid denominators become identity factors.
A second kernel reads both paired denominator factors, constructs precisely
the same per-thread leaf and same256-thread inverse tree as the original
consumer, and overwrites the four denominator slots with the two inverses.
A third kernel performs the unchanged tail/gate and writes the same tentative
hit records. The existing exact replay kernel still recomputes every tentative
hit before any host publication. Stream ordering enforces all producer-consumer
dependencies. No arithmetic or selected window/epoch combinations change.

The inverse writes are disjoint: a block owns its512candidate indices, each
thread owns its original A/B pair, and both original products remain in that
thread's registers until both output inverses have been calculated. A and B
remain128indices apart, matching the baseline pairing, including both128-lane
halves. Inactive or odd-tail B candidates use identity factors and cannot emit.
The stride is the actual per-launch candidate count, with allocation sized
for the maximum launch. Full and partial launches therefore stay in bounds.

Cost: the global workspace holds128bytes per candidate plus one validity byte;
separating stages adds memory traffic and two kernel launches. Benefit sought:
independent resource usage and removal of point/finish values held through a
block inverse. Default compiled resources are128registers/no shared for front,
90registers/24KiB shared for inverse,76registers/no shared for tail, all zero
stack. The monolithic consumer uses128registers/48KiB shared. These are sm52
compile reports; actual GPU JIT behavior and timing remain authoritative.

Initial trial uses32,768physical blocks (16,777,216candidates,2GiB workspace).
The larger variant uses131,072blocks (67,108,864candidates,8GiB workspace),
reducing launch frequency. It also excludes the unused monolithic kernel from
trimmed builds, preventing it from adding JIT work. Non-trimmed fallback code
can still compile that kernel. The helper include is outside OpenSSL's extern-C
block in the larger variant; this fixes linkage organization only.

All local runs use RTX3090, N=24, seed2026092501,60seconds, the unchanged
verifier, and flock -x /tmp/qsb-gpu.lock. Source build/cache stamps and outputs
are under this directory. capture.py in ../stage_profile preserves stdout
while delegating to unchanged gpu_wrap.py. No root score file is created.
The static RTX4090 label in score artifacts does not describe local hardware.

Initial small batch:1,649/1,649 verified hits,230.054242M/s by wall-clock hits,
with246.7M/s peak diagnostic rate and244.9M/s at45seconds of search. It has
cold startup overhead. Fresh baseline:1,738/1,738 verified hits,242.367982M/s,
243.6M/s diagnostic rate. initial-hit-check.json proves exact equality of all
1,649hits in the completed106,168,320-epoch range. Counter rates are clues,
not ranked scores; qualification awaits the larger warm run.

Main implementation and reasoning: GPT-6 Astra high through Codex. The prior
DeepSeek V4.1 Flash [1m] through Claude Code task checked only arithmetic on
sanitized timing/resource numbers, independently reverified by Astra. No
unpublished candidate code was sent to an external helper.

Warm qualification completed: 1,772/1,772 verified hits,246.977743M/s vs
baseline242.367982M/s (+1.90197%). Completed work increased1.86916%.
The baseline completed-range1,738-hit set matches exactly. The final entry,
organized implementation and tested larger prototype emit identical PTX.
See qualification.json and production-equivalence.json.

Packaging correction: Yukon includes ignored research files, so the first
archive attempt failed locally at30.9MiB with no submission ID. Rebuildable
ELFs and duplicate PTX were hashed and removed; source and raw evidence stay.
The full tree now compresses to about3MB, below the8MiB track cap.
