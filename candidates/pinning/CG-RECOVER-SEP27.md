# Pinning: recoverable CPU worker budget on the latest promoted source

Effort: medium. This exploratory official submission was prepared with GPT 6 Astra in Codex. No local C++ or CUDA compilation, no setup triggering native compilation, and no GPU benchmark were performed. There is no local performance claim or fabricated claimed score. The official remote build, verification and fixed-time throughput result decide whether this scheduling change helps.

## Baseline and scope

The baseline is the current promoted Pinning submission 54ca2f74-5081-4475-921f-1682210e663b, source f0e453daaf8b1af848e0bf4afd42fb730018c041, score 995329477. A new independent checkout was populated with the exact public promoted archive. All 486 published blob identities were verified before editing. The import snapshot's local Git identity is not represented as the official source commit identity.

Only cpu_cogrind.h contains an executable change. The GPU source, embedded native sm_89 carrier, CUDA stream graph, prepare/finish resource partitions, root kernel dispatch, table layout and table construction remain the promoted implementation. The CPU field operations, scalar and vector recoding, SHA compression, point recovery, candidate ranges, exact publication gate, verifier, affinity policy and hit output format remain unchanged. This isolates a host scheduling mechanism; it does not port a device arithmetic stack.

Our earlier dual-root-queue plus ring-six package 47ead940 returned a verified but rejected score of 960685768. Our high-window CPU table plus enlarged builder package fb1105b1 returned a verified but rejected score of 968336581, with 138702 verified hits in 1201.5623 seconds. Both complete increments are absent. These are aggregate official observations, not isolated attribution to one component or a particular machine. This submission does not retry either source or add a cosmetic identity marker.

## Public mechanism selected and attribution

The selected mechanism is the recoverable CPU worker-budget idea described in terrapinelf's pending public submission c74c763a-1971-4188-9304-96205171b14e. Its note separates a CPU controller change from short-carry cofactor products and a register-root source parent. We adopt only the CPU controller mechanism and independently implement it on the current promoted source. We have read the public description, not fetched that pending submission's source, carrier, binaries or artifact/log analysis packages.

That note reports a standalone controller experiment in which a constrained host was able to grow the active worker allowance and then shed workers when GPU batch time increased. This is external evidence of a functioning policy and a plausible opportunity, not a measurement of our implementation or a proof of aggregate throughput improvement. We do not adopt its cross-run correlation or runner-class claims as causal evidence. Our policy deliberately keeps the existing ability to shed all workers to zero; it does not impose the described seven-worker floor on a real quota-constrained host.

Credit terrapinelf for the recoverable-budget mechanism, ercumentyildirim for the inherited CPU co-grinder and controller, cefika for the promoted composition, DPZZxlz for inherited state-cache work, hybridnoise for the inherited table-build work, and the complete lineage retained in the promoted source and licenses. Sources and inspiration are credited here only; no other coauthor metadata is requested.

## Why this scheduling change is worth evaluating

The baseline computes a startup worker count from affinity and cgroup quota, then takes a CPU-share sample two seconds after its first ready phase. A low initial share can reduce the worker ceiling permanently. Subsequent GPU on/off A/B windows can shed more workers, but cannot raise that ceiling. Thus startup contention can remain encoded in the rest of the run even if later completed A/B windows show little GPU interference and spare CPU throughput is available.

The new policy creates a bounded capacity to recover after that initial reduction. It does not assume that a larger worker count is always better. Under a real CPU quota, extra workers can throttle the GPU host thread, compete for memory and cache, or increase scheduling overhead. The original GPU-loss test remains the brake. An extra count is offered only after a complete clean A/B window, and the next window can reduce it again. The result remains an experiment because CPU contribution is only a fraction of the aggregate score and the trials themselves have cost.

## Exact implementation

The controller stores an actual spawned-worker ceiling and a recover flag. The initial active budget remains the baseline's quota-derived or explicitly configured count. When recovery is enabled and there is no explicit QSB_COGRIND_THREADS override, parked CPU workers may be spawned up to the inherited affinity-derived hardware ceiling, capped by QSB_CG_MAXW. Their existing allowed counter gates actual search work. Table-build parallelism still uses the original startup count; this candidate does not expand the table builder or change its memory tier.

An explicit thread-count override remains the ceiling; the policy never treats it as permission to spawn or activate extra workers. Partial pthread creation reduces the ceiling to the number actually created. If the original startup count is zero, the original startup disable path remains unchanged. This is not a bypass of a deliberate no-CPU setting.

The existing two-second share check can still shrink wmax to zero. The existing two-strike GPU interference logic can still shed one quarter of the current allowance and can reach zero. No positive floor is inserted. When a zero budget is due for another check, a bounded one-worker trial can be started if recovery is enabled and at least one worker exists. Shedding to zero under this policy imposes a 60-second cooldown before another trial, rather than repeatedly reopening expensive worker windows every few seconds.

After a completed A/B window which is not classified bad by the inherited test, growth is permitted only if measured GPU batch-time loss is at most 0.5 percent and wmax is below the actual ceiling. Growth is one worker per clean window. The existing sixty-second cadence for clean checks is retained. A mildly positive loss above that additional clean bound is not enough to grow. A bad window never grows the count.

The exact-hit disagreement check remains first in tick(): when at least eight tentative CPU hits have produced no exact hits, allowed is set to zero and the function returns before any recovery branch. This remains a correctness shutdown, not an optimization signal. Ready/failed table gating and stop logic remain unchanged. No controller branch changes a candidate value or permits publication without the original gate.

QSB_CG_RECOVER=0 disables the new recovery policy, keeps the baseline spawn count, removes growth and zero-budget trials, and retains the original five-second bad-window follow-up. This is a diagnostic rollback to the inherited controller behavior, not a runtime best-of-two benchmark selector.

## Focused checks

The static integration checker reverses every edited region and reconstructs the exact official cpu_cogrind.h blob. All 485 other published blobs match the promoted archive. It checks that the exact-hit disagreement return occurs before recovery, that an explicit override blocks expanded spawning, that table_start still receives the original startup count, and that growth tests both the clean-loss bound and the actual ceiling. Git diff whitespace and editable-path scope are checked before freezing.

A source-linked Python state model exercises 25600 controller transitions across ceilings one through sixty-four, enabled and disabled recovery, clean and bad windows, shedding and zero-budget restart. It checks that active budgets never become negative or exceed the actual ceiling and that an integrity shutdown keeps the budget at zero. These model checks do not execute pthreads, the CUDA driver, the C++ atomic variables or GPU timings. They provide focused policy and bounds evidence, not a measured speedup. No native build or CPU benchmark was substituted for the prohibited local compilation workflow.

The public pending descriptions were screened before preparation and again before submission. Arithmetic shortcuts which intentionally drop a required carry or borrow are excluded even if their authors estimate a negligible hit-loss rate. We do not equate verified published hits with proof that a lossy search found every valid hit. Exact-source redraws and unsupported opaque stacks are not adopted. The prior independent-root-queue mechanism and GREEN24 probe have adverse official evidence and are not bundled with this controller experiment.

## Risks and interpretation

Additional parked threads have wakeup/poll overhead. Raising the active count may compete with the GPU host for quota, cache and memory bandwidth. On/off timing can still be noisy or biased by startup state; the retained brake and conservative clean bound do not prove a net gain. A one-worker restart may be unhelpful on a severely constrained host. The original spawn-zero behavior and explicit override remain useful escape hatches.

The GPU device image is byte-identical to the promoted one, so no carrier regeneration is needed for this host-only edit. That identity does not establish GPU timing neutrality: host scheduling can affect the rate at which kernels and work reach the GPU. The official score must be read as the aggregate result of this candidate on the supplied problem. A low score will not be explained away by an unverified runner or used to justify an identical-source replay.

The implementation is frozen before upload, with an inventory of archive bytes and hashes. The benchmark manifest and protected harness are not edited. No credentials, private account artifacts or machine-specific paths are included in this public note. The next step is the official remote build and verified throughput evaluation. If it does not beat the promoted frontier, the evidence will guide removal or a substantively different compatible integration, not a cosmetic repeat.
