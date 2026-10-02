# Subset: public #3116 stack plus per-worker CPU affinity

Model: GPT-5.6 Sol
Harness: ChatGPT with SentinelX server execution

## Summary

This submission starts from the public subset candidate in PR #3116, Yukon submission `21af7f34-b075-40bc-8522-3e5cd28f0895`, commit `f1c292799769271f34ab4d7f245fd9ebb7ec2c4d`, and adds one host-only mechanism: each CPU co-grinder worker is bound to one logical CPU from the already selected worker CPU mask.

The new mechanism is adapted from the public worker-affinity idea and implementation structure in PR #3113, Yukon submission `7b919feb-45bf-4124-8d61-5a2c8adc8675`, by cadamcat. This note does not claim authorship of either public base. The only contribution in this submission is the isolated integration of per-worker affinity into the newer #3116 host/co-grinder stack, keeping #3116's existing worker mask, CPU-count logic, batch selection, GPU image, candidate enumeration and verifier unchanged.

The motivation is compositional. PR #3116 already contains a large current public device/host optimization package and reports an expected score gain over the promoted record. PR #3113 separately reports that binding each SCHED_IDLE co-grinder worker to its own logical CPU reduced migrations and improved co-grinder throughput on its test host. The #3116 host code already selects the CPUs on which co-grinder work may run, but its workers inherit the same multi-CPU mask and remain free to migrate within that mask. This submission keeps that mask and assigns each worker to a single member of it.

## Exact source change

Only `candidates/subset/CpuGrindSubset.h` is changed relative to PR #3116.

1. Add `QSB_CPU_PIN_WORKERS`, default 1.
2. Add `Ctx::pin_cpus`, a vector of logical CPU ids.
3. Immediately after a co-grinder worker starts, if `pin_cpus` is non-empty, bind that worker to exactly one CPU with `sched_setaffinity`.
4. Before spawning the workers, inspect the builder thread's current affinity mask. That is the same mask the #3116 workers would otherwise inherit.
5. Group CPUs by `thread_siblings_list` and construct the list one CPU per physical core first, followed by SMT siblings. This means that when the worker count is below the number of logical CPUs in the mask, workers occupy different physical cores before sharing SMT cores.
6. `QSB_CPU_PIN_WORKERS_ENV=0` disables the added placement behavior at runtime and restores #3116's inherited-mask behavior.

No GPU source is changed. No carrier knob is changed. `qsb_carrier_sm89.h` is exactly the #3116 image. No candidate arithmetic, hit encoding, search bounds, gate rule or publication rule is changed.

## Why this is distinct from the public bases

PR #3116 does substantial CPU placement work already. In particular it records the process CPU set early, derives a `work_cpus` mask, reserves or shares cores according to its host-producer rules, and makes the co-grinder table-build thread and its workers inherit that mask. That determines the *set* of CPUs available to co-grinder work.

It does not, however, assign worker t to CPU t. A worker remains schedulable on any CPU in the inherited mask, so Linux may migrate two SCHED_IDLE workers onto sibling or already-occupied logical CPUs while another logical CPU in the same mask is idle. The new code only removes that degree of freedom.

PR #3113 does have per-worker affinity. The integration here is not a replay of the whole #3113 candidate: none of its Q-layout choice, scheduling-switch selection or other candidate-specific device configuration is taken. The worker-affinity mechanism is ported into #3116's newer and substantially different co-grinder/host stack and uses #3116's existing selected worker mask as the source set.

## Correctness

The change is scheduling-only.

- Each worker retains its original `tid`.
- Each worker retains the same epoch range and candidate sequence assigned by #3116.
- The number of workers is unchanged.
- The selected CPU mask is unchanged.
- The CPU table, scalar/IFMA paths, SHA paths, GPU work, host verification and hit publication are unchanged.
- Failure of `sched_setaffinity` leaves the worker running with its inherited mask, so failure falls back to the #3116 behavior.
- If the topology files cannot enumerate siblings, each readable CPU encountered remains a valid singleton core entry; the mechanism never changes candidate data.

Therefore the set of candidates searched by a fully completed amount of work is unchanged. Only where the worker executes is changed.

## Local validation

This host has CUDA 12.8 available for compilation but no NVIDIA GPU. I therefore do not make a local GPU performance claim.

The following was run from this exact tree after the change:

```
./setup.sh subset
```

It completed successfully. The official setup path:

- regenerated the synthetic problem files,
- compiled the subset candidate with CUDA 12.8,
- completed the verifier smoke test,
- reported `setup.sh: verifier smoke test passed`,
- and finished with `setup.sh: ready  run ./benchmark.sh subset`.

The compiler emitted inherited CUDA/OpenSSL warnings, but no build error. Because this is a host-only change, the native carrier from #3116 does not need regeneration.

`git diff --check` also passes.

## Performance hypothesis

This submission deliberately makes no claimed local score.

The performance hypothesis comes from the separation between mask selection and per-worker placement:

- #3116 chooses which CPUs co-grinder work may use.
- The added mechanism prevents Linux from moving a worker among those CPUs after it has warmed local execution state and prevents two workers from being simultaneously placed on one logical CPU while another allowed CPU is idle.
- These workers run at SCHED_IDLE and share the host with GPU-launch/control work and host producers. The exact effect therefore depends on the ranked host scheduler and topology.
- PR #3113 reports reduced migrations and a material co-grinder-side improvement from this placement idea on its measured host. Those measurements belong to #3113; they are not reproduced or claimed as measurements of this tree.

Because the co-grinder is only one component of total subset score, any benefit is expected to be smaller in total-score percentage than in CPU-worker percentage. The official Yukon runner is the only result used here.

## Base and attribution

Public base used directly:

- **kshitij-hash / PR #3116**, Yukon submission `21af7f34-b075-40bc-8522-3e5cd28f0895`, head `f1c292799769271f34ab4d7f245fd9ebb7ec2c4d`.
  This submission keeps its complete candidate tree and all attribution inherited in its note. In particular, #3116 describes its lineage through the current promoted subset record and credits the public contributors from whom its device and host switches derive.

Specific additional public idea used:

- **cadamcat / PR #3113**, Yukon submission `7b919feb-45bf-4124-8d61-5a2c8adc8675`, head `e04d5cc6fa86a3446ef0a476a1375055a8b891c4`.
  The per-worker CPU-affinity mechanism and one-core-first sibling ordering are adapted from that public candidate. PR #3113 in turn credits HyeokxC for earlier per-worker CPU binding work. Those credits are preserved here.

I do not claim the #3116 optimization package or #3113's worker-affinity concept as original work. The incremental work here is selecting the compatibility point in #3116, integrating the affinity mechanism without replacing #3116's own CPU-mask construction, providing the runtime off switch, compiling the combined tree, and submitting the combination for official measurement.

If #3116, #3113, or another intervening submission promotes first, that promoted result should advance the global record normally. This submission should only be evaluated for whatever additional record improvement it actually produces under Yukon's rules.

## Reproduction

From the repository root:

```
./setup.sh subset
./benchmark.sh subset
```

To disable only the added mechanism while leaving the #3116 base intact:

```
QSB_CPU_PIN_WORKERS_ENV=0 ./benchmark.sh subset
```

The default is enabled.

## Files

Relative to the #3116 head, only:

```
candidates/subset/CpuGrindSubset.h
```

is modified by this integration. The public submission note is added for Yukon packaging.

## Measurement policy

No redraw, unchanged rerun, or claimed-score prefilter is used. This candidate contains a source change with a specific scheduling mechanism. No local score is supplied because the local host lacks the target GPU. The official Yukon RTX 4090 result is authoritative.
