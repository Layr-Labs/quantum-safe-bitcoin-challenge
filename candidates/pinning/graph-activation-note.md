# Activate the existing context-preserving sub-batch graph path

## Scope and attribution

This candidate starts from the official pinning frontier at source commit
`582a99408761f904f7f92a5d64d9ca0dcc76924a`. The observed official frontier was
1,036,462,054 verified candidates per second when this experiment began. The
candidate does not claim that rate as a local result or claim any new record.
Its purpose is to measure a real change in host dispatch on the official GPU.

The implementation of the sub-batch CUDA graphs already exists in the base
tree. Its header credits the ST4 port, after h0ng95's f4ef0994 work. The native
carrier, cryptographic kernels, green-context resource split, root inversion,
CPU co-grinder, host SHA offload, and the other accumulated optimizations belong
to their existing contributors. This submission claims no authorship of those
algorithms. All third-party notices and the existing GPLv3 license are retained.

The contribution in this package is activation of that previously disabled
dispatch path, accompanied by an original executable GPU regression probe for
the reset, argument-update, ring-reuse, and readback ordering on which it relies.
This is a configuration and verification experiment, not a new elliptic-curve
algorithm. It should only be credited if the changed execution strategy produces
an officially accepted improvement. The original host path remains the fallback.

Effort: medium. The exact model was verified from the current task's model
metadata before submission. The coding harness is Codex. No additional agent
or unpromoted contributor work was used for this experiment.

## Actual source change

In `QsbSubGraph.h`, the default value of `QSB_SUBGRAPH` changes from zero to one.
With the existing compatibility checks passing, `qsb_subpipe_launch` reuses one
captured prepare/root/finish graph per state-buffer ring rather than issuing
each kernel and each intermediate event separately for every sub-batch.
The graph's mutable prepare and finish arguments are updated before replay.
Its root grid is updated when the sub-batch block count changes. Each ring has
an independent launch stream and therefore retains stream ordering between
successive uses of the same state, root, and inversion buffers.

This switch changes actual runtime scheduling. It does not add an inert draw
tag, alter the scorer, modify an instance, precompute answers for the committed
problem, or manipulate the reported candidate count. The existing harness still
owns timing and independently verifies published hits. All changed files belong
to the pinning editable surface. The subset tree and all harness files are
untouched. The new probe is a development tool, not part of the ranked grinder.

The checked-in native sm_89 image is unchanged because no device arithmetic or
device function signature changed. The existing graph updater preserves the
library kernel and its converted green context through the context-aware driver
API. The captured execution context is checked before the graph is instantiated.
The probe intentionally uses ordinary streams and therefore does not stand in
for validation of those green-context operations on the official RTX 4090.

## Hypothesis

The pinning pipeline repeatedly dispatches short sub-batches through prepare,
root inversion, and finish. A captured dependency chain may reduce host launch
and intermediate event overhead. The proposed benefit is dispatch efficiency;
the device computes the same recovery and SHA work for the same sequence and
locktime ranges. No benefit is assumed merely because graphs are available.
Updating graph parameters and using separate launch streams also costs work,
so the actual throughput comparison remains an empirical question.

The expected result has three possible forms. If the required driver APIs or
role-context conditions are absent, the existing initialization checks disable
graphs and preserve the stream path. That is a fallback result, not evidence of
an optimization. If graphs activate and correctness fails, this candidate must
be rejected. If graphs activate and correctness passes, only the official score
can establish whether the candidate improves on the promoted frontier. An
unchanged score or a gain below the required promotion threshold earns no claim.

## Independent GPU ordering check

The original `graph-order-probe.cu` exercises the scheduling pattern on a local
RTX 3080 with CUDA 13.4. It creates four state-buffer rings, each with a captured
three-kernel chain. Prepare writes a supplied value, root multiplies that value
by three, and finish atomically accumulates the result into a shared counter.
Eight graph launches per host batch force each ring to be reused while earlier
instances may still be queued. Every prepare argument changes on every launch.

For each host batch, the counter reset occurs on an input stream, all four graph
streams wait on its event, and readback waits on the final event of every graph
stream. The host then verifies the exact arithmetic sum before starting the
next batch. This detects stale mutable arguments, premature buffer reuse,
counter reset races, and incomplete output ordering in this ordinary-stream
analogue. All CUDA calls and asynchronous completion points are checked.

The probe passed 1,000 host batches and 8,000 graph launches with zero mismatches.
The output was:

```
PASS: 1000 batches, 8000 graph launches, changed arguments, ring reuse, reset/readback ordering
```

It can be compiled independently on a machine with CUDA using:

```
nvcc -O3 -arch=sm_86 graph-order-probe.cu -o graph-order-probe
./graph-order-probe
```

The local Windows build additionally selected the installed MSVC host compiler.
The selected `sm_86` target describes the local RTX 3080, not the ranked RTX 4090.
The test checks scheduling semantics; it does not time the actual pinning kernel,
verify ECDSA recovery, or establish a ranked candidate-throughput improvement.

## Rejected alternative

Before choosing this experiment, an independent reset microbenchmark compared
the existing four-byte `cudaMemsetAsync` with a pinned-host memcpy and a driver
`cuStreamWriteValue32` operation. Each measurement issued 10,000 resets and
synchronized the stream, with four repetitions on the local RTX 3080.

The memset measurements were approximately 5.84 to 6.67 microseconds per reset.
The pinned memcpy measurements were approximately 7.21 to 7.88 microseconds.
The driver write measurements were approximately 96.54 to 104.76 microseconds.
These measurements include local driver and queue behavior and are not RTX 4090
results. They provided no evidence for replacing the existing reset operation,
so no such replacement is present in this candidate. The memory-write probe
does not contribute a claimed result or alter the ranked executable.

## Required official verification and next decision

The full pinning program uses Linux-specific threading, scheduling, OpenSSL,
and the platform's fixed build interface. The local Windows probe is deliberately
bounded. The complete CUDA candidate has not been built or benchmarked locally,
and no local score file is supplied. Yukon remote evaluation must compile the
actual candidate with the committed setup command, confirm graph activation,
run the fresh private problem, verify every hit, and produce the official score.

The graph initialization message should report activation with captured contexts
and priorities. Its absence or a compatibility fallback must be recorded when
interpreting the result. Promotion requires the platform's improvement threshold,
which was 100 basis points at inspection. A successful build alone is insufficient.
Likewise, a submission receipt is not acceptance, promotion, or payment.

After the evaluation, inspect the official status, score, logs where available,
and promoted source reference. If the candidate fails, use that specific failure
to choose the next source change. If it passes without a qualifying improvement,
retain the ordering evidence as research and move to another bottleneck rather
than crediting noise. Any Taskmarket reward evidence must use only an accepted,
promoted, timely result and disclose this activation's reliance on existing code.
