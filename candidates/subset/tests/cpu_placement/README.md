# Conservative Terrapin producer integration

Prepared by GPT-6 Astra / Codex from `ff7085b81a09fe7539dae348239566854e213334`.
No candidate arithmetic, table selection, producer, GPU, carrier, gate or harness changes.

The public header trusted `qhp::g_share_cpu` and expanded its default thread count
to the resulting mask size. This branch disables both behaviors. Worker count is
again quota-capped `ncpu - QSB_CPU_RESERVE` (default reserve 2); compile-time and
runtime thread overrides retain their original precedence. The count is finalized
once, before Ctx construction, and remains the epoch stride for every worker.

When main has narrowed its startup affinity, workers use startup-minus-current-main.
A complete, symmetric Linux sibling list further excludes main's physical core.
Malformed, missing or asymmetric topology retains exactly startup-minus-main;
there is no speculative main-CPU sharing or added worker on any fallback. Thus
W1's 30 allowed non-SMT CPUs retain 28 default workers, with 29 eligible non-main
CPUs. A full SMT32 machine with main pinned to one CPU retains 30 workers and an
eligible mask of 30 excluding main and its proven sibling. Binding quotas do not
trigger extra workers. Existing quota discovery is unchanged by this patch.

Before table construction or worker launch, the builder verifies that its actual
affinity equals the requested mask and its scheduler is SCHED_IDLE. Failures stop
the CPU lane with a diagnostic, leaving the GPU path intact. New builder children
inherit this mask and policy. The existing worker's redundant SCHED_IDLE request
is unchanged. No new scheduling promise applies to machines without SCHED_IDLE or
a producer that failed to pin: such machines retain the original inherited-mask
behavior without speculative sharing. Unknown topology does not prove producer
sibling isolation; it only preserves the original conservative fallback.

Run the pure topology/parser fixtures (no host affinity changes or compute benchmark):

```
c++ -std=c++11 -O2 -Wall -Wextra -Werror \
 candidates/subset/tests/cpu_placement/placement_test.cpp -o /tmp/qsb-placement-test
/tmp/qsb-placement-test
```

54 assertions pass: no-SMT30, full/partial SMT, main already on two siblings,
missing and asymmetric topology, rollback across several main CPUs, invalid and
out-of-range CPU lists, empty/non-subset/full main masks, and a one-core fallback.
Also checked standalone wrapper syntax with QSB_HP_ON=1 and no qhp declarations:

```
g++ -std=c++17 -Wno-deprecated-declarations -pthread -DQSB_ZEROS_N=16 \
 -DQSB_HP_ON=1 -fsyntax-only candidates/subset/tests/cpu_lane_probe/wrapper.cpp
```

No native timing, native GPU compilation or all-hit test was run by this agent.
Keep the explicit 28-worker setting for the parent's W1 paired comparison.
