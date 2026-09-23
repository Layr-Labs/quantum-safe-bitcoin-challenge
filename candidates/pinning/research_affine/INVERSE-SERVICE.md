# Research: asynchronous root-inversion service

This header is a component experiment, not an enabled production candidate. It
changes no search ranges or benchmark rules. GPU execution and throughput remain
unmeasured. The root task owns the affine worker implementation separately.

## Architectural distinction

A worker CTA keeps its candidate affine points in registers, retains its local
product tree in shared memory, and submits one denominator product for inversion.
A permanently assigned service lane processes that CTA's mailbox. Thirty-two
service lanes therefore invert up to thirty-two independent roots together.
Workers wait only for their own roots; there is no grid-wide round barrier and no
intermediate global store of every candidate's coordinates.

This is distinct from the existing single-active-lane block inverse and from the
historical W5 wavefront. Public submission c2900b57 used a 512-thread on-chip affine
wavefront with one root inverse per wave and scored 138,466,473/s. The cached
7cb2974d notes report a -9.2% persistent cooperative prepare/finish fusion, using
block-level inverse waves. Neither note describes this independent-root service.
A bounded search of 669 cached notes, recent PR bodies through #1152 and 71 cached
runtime blobs found no equivalent service. This is not universal novelty proof.

## Mailbox contract and happens-before proof

`Slot` is 128 bytes, aligned to 64 bytes: two `uint4` records for the root, two for
the inverse, request and response epochs, and reserved padding. All slots start
zeroed before the launch. Each slot has one producer (its worker's lane zero) and
one consumer (its permanently assigned service lane).

Epochs start at one, increase by one, and never wrap. Zero means no request;
0xffffffff is STOP, so the largest request epoch is 0xfffffffe. The worker calls:

1. `worker_publish(slot, epoch, root)` after consuming the preceding inverse;
2. `worker_wait(slot, epoch, output)`, then consumes/copies the output;
3. after its last completed epoch, `worker_stop(slot)`.

Root payload stores precede a device-scope release store of `request`. The service
acquires that request before reading the root, so all four root limbs are visible.
The inverse payload stores precede a release store of `response`; the worker
acquires that exact epoch before reading the inverse. The next request is published
only after the worker has copied the preceding inverse, so the service cannot
replace that inverse prematurely. The next root is not stored until the preceding
response has arrived, so the service has finished reading the old root. STOP is
published only after the last response is consumed. These two handshakes prevent
both torn payloads and stale-generation reuse.

The implementation uses `cuda::atomic_ref<uint32_t, cuda::thread_scope_device>`
with explicit acquire/release order; ordinary volatile accesses are not substituted
for synchronization. In debug builds assertions check strictly consecutive epochs
and one outstanding request. They do not repair an invalid caller.

## Warp and residency requirements

Every physical lane in every service warp enters `service`, including lanes whose
index exceeds the worker count. Inactive lanes behave as already stopped. A warp
votes uniformly whether all lanes have stopped and whether any request is pending.
When any lane has work, every lane calls the finite inversion callback; idle lanes
receive the field identity. This prevents independent per-lane polling from
stranding lanes needed by a warp collective. The callback may not use a CTA-wide
barrier. Each service warp returns only when all its assigned workers have stopped.

The kernel must launch cooperatively with the exact compiled kernel's occupancy
capacity checked using the CUDA runtime, and must refuse unsupported cooperative
launches. Worker CTAs plus service CTAs must fit that capacity. A speculative
ordinary launch with spinning workers is unsafe: workers might occupy every slot
while a service CTA remains unscheduled. Other streams must not hold resources
needed by the launch. The host must check the actual launch and synchronization
results. Polling assumes forward progress for resident warps; finite worker loops
and a terminating inversion callback then provide an acyclic dependency graph.

With 128 threads per service CTA, `service_blocks*128 >= worker_blocks`. At most
one request is pending per worker, and each service lane visits its single mailbox
on every outer iteration. No dynamic work queue, arbitrary claim order, host
interaction, or cross-worker acknowledgment is involved. Workers with no real
candidates still call STOP; partial candidate blocks contribute identity leaves
to their local denominator trees and retain independent validity masks.

## Cost model, not a speed claim

A standard 128-leaf inverse tree costs 127 upward and 254 downward multiplies,
plus one root inverse. One ordinary affine addition then costs two multiplies
and one square after its inverse: slope and ordinate. Thus the ideal ledger is
(5-3/128)M + 1S + I/128 per addition, excluding barriers, field additions,
normalization, loads, idle service lanes and polling.

At 800M candidates/s and fourteen inverse rounds, the service demand would be
87.5M roots/s. A hypothetical 4-microsecond inverse warp serving 32 active lanes
would process 8M roots/s; that latency and full utilization are unmeasured. Actual
runtime divergence, polling, mailbox access and delayed workers can reduce it.
The earlier estimate of six resident worker CTAs is not established: the inverse
path shares the kernel's register allocation and can reduce occupancy.

## Necessary latency budget

The service architecture can hide one worker's stall behind other resident
workers, but it cannot remove that worker's dependency on its own root inverse.
Its candidate population is bounded by residency. Let B be the **measured**
resident CTAs per SM for the entire compiled kernel, assume 128 SMs, 128 candidate
lanes per worker, and fourteen sequential inverse rounds per candidate chain.
If C=128B, the maximum workers W satisfy W+ceil(W/128)<=C; S=ceil(W/128) dedicated
128-lane service CTAs occupy the remaining slots. With all candidate lanes useful,
a throughput target T requires an average complete candidate-chain time below
128W/T. Thus the average round, including worker arithmetic, tree barriers,
mailbox wait, inversion and polling, must fit below 128W/(14T). This is a necessary
condition, not a sufficient performance prediction.

At T=814,000,000 candidates/s:

| Resident CTAs/SM B | Total CTA capacity | Service CTAs S | Worker CTAs W | Candidates in flight | Complete chain budget | Average round budget | Round budget / 272 Fermat operations |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 384 | 3 | 381 | 48,768 | 59.912 us | 4.279 us | 15.73 ns |
| 4 | 512 | 4 | 508 | 65,024 | 79.882 us | 5.706 us | 20.98 ns |
| 6 | 768 | 6 | 762 | 97,536 | 119.823 us | 8.559 us | 31.47 ns |

These budgets already assign the entire execution time to those fourteen rounds.
Any additional scalar generation, initialization, hash/tail work, launch overhead,
partial tiles or underfilled service calls makes the available inverse budget
smaller. The actual measured `_ModInv` register allocation in the full kernel
must establish B; setting a launch bound alone does not establish residency.

For the Fermat resource probe, 257 squares plus fifteen products give 272 generic
256-bit multiplication operations per inverse. The final column is only the
entire round time divided by that operation count, not a measured operation
latency or an assertion that all those operations form one dependency chain. Even
at six resident CTAs/SM the inverse would need to complete within 8.559 us before
accounting for any worker work; at four CTAs/SM it must fit below 5.706 us. Its lower
register allocation therefore does not establish feasibility. The long addition
chain may consume more time than it saves through occupancy.

A further queueing cost is possible inside a service warp: a request arriving
while that warp already executes an inverse waits for the running invocation
before joining the next one. Static mailbox assignment avoids starvation but does
not bound this delay to zero. Required GPU measurements are the completed
whole-chain throughput and the distribution of request-to-response latency under
the actual resident workers. Isolated callback throughput cannot substitute for
those measurements.

## Checks completed

`test_inverse_service.cpp` compiles and executes the actual header using C++20
atomics, CPU threads and 32-lane barrier-backed ballots. Eight configurations cover
0, 1, 3, 31, 32, 33 and 63 workers, mixed finite epoch counts, zero epochs, and an
entirely unused service warp. A deterministic four-limb transform checks every
payload bit and generation. It is not a field-inversion or GPU scheduler test.
The test passed with UndefinedBehaviorSanitizer and fail-on-error enabled.

`test_inverse_service_model.py` exhaustively visits 770 abstract interleaving
states for one/two workers and zero to three epochs. No stale/torn response or
nonterminal deadlock is reachable. Publishing the request before its payload or
the response before its payload is detected by two negative controls. This is a
finite ownership model, not a model of every CUDA memory-model behavior.

CUDA 12.8.93 compile probes with `-DNDEBUG -O3` produced:

| Probe | Target / launch bound | Registers | Stack | Spill stores / loads |
|---|---|---:|---:|---:|
| Mailbox, identity callback | sm52 / 128,4 | 16 | 0 | 0 / 0 |
| Mailbox, identity callback | sm89 / 128,4 | 18 | 0 | 0 / 0 |
| Mailbox + actual `_ModInv` | sm89 / 128,4 | 118 | 120 B | 0 / 0 |
| Mailbox + actual `_ModInv` | sm89 / 128,6 | 80 | 192 B | caller 32 / 36 B; callee 80 / 60 B |
| Mailbox + exact Fermat chain | sm89 / 128,4 | 80 | 0 | 0 / 0 |
| Mailbox + exact Fermat chain | sm89 / 128,6 | 76 | 0 | 0 / 0 |

The arithmetic probe canonicalizes before and after `_ModInv`, using a five-limb
work buffer. It does not run affine workers; whole-kernel resource results remain
a separate check. Six-CTA register limits demonstrably cause spills here.

A separate Fermat resource experiment uses a checked addition chain for
p-2 = 2^256-2^32-979: 257 squares and fifteen products. Its squares deliberately
use the literal exact general multiplier copied from `pinning.cu`; therefore
this experiment is not optimized square arithmetic. Its source operations are parsed by `test_inverse_service_fermat.py`, which
verifies the exact symbolic exponent and compares the chain with Python modular
exponentiation on boundary and deterministic random inputs. This is not CUDA
arithmetic execution. The compile probe fits the six-CTA register limit without
spills, but increased arithmetic and code size can outweigh that resource benefit.
No GPU correctness result or timing is claimed.

## Full persistent probe resources

The streamed-tail `persistent_affine_probe.cu`, including actual affine workers,
was also compiled for sm89 with CUDA 12.8.93, `-O3 -std=c++17 -DNDEBUG`. These
measurements supersede minimal-callback probes when reasoning about this full
kernel. `persistent_resources.json` records source hashes, exact commands, compiler
logs, ELF function boundaries, and static SASS instruction counts.

| Requested minimum CTAs/SM | Registers | Shared bytes | Stack bytes | Spill stores / loads | Worker LDL / STL instructions |
|---|---:|---:|---:|---:|---:|
| 3 | 148 | 12,288 | 120 | 0 / 0 | 0 / 0 |
| 4 | 128 | 12,288 | 120 | 0 / 0 | 0 / 0 |
| 6 | 80 | 12,288 | 352 | 356 / 248 B | 51 / 74 |

For bounds three and four, ELF symbol ranges show that every local-memory
instruction belongs to the service helper or its `_ModInv` callee; the worker
entry/dispatch contains no LDL or STL. PTX identifies a 40-byte address-taken
service buffer and an 80-byte `_ModInv` local depot, explaining the 120-byte stack.
The service path still accesses local memory even though compiler spill counters
are zero. Static instruction counts are not dynamic traffic measurements.

At bound four, 128 registers times 128 threads consumes 16,384 registers per CTA;
four CTAs require 65,536 registers and 49,152 shared bytes. This is compatible
with four CTAs under that SM register budget, without spills. The runtime must
still query occupancy for the actual device/kernel before its cooperative launch;
the arithmetic resource calculation does not replace that query. Requesting six
CTAs introduces spills in workers as well as the inverse path.

`compile_probe.py` rebuilds bounds 3/4/6, writes the JSON report, and by default
builds an executable at bound four under `_build/`. It runs only `--describe`,
which succeeds without accessing a GPU. Pass `--ptx` to capture local-depot
analysis, and `--keep-intermediates` to retain cubin/SASS/PTX files. Without that
flag, only this script's intermediates are removed; logs and the executable stay.
No GPU work or speed measurement is performed by the build script.

Example CPU reproduction, from the benchmark root:

```sh
g++ -std=c++20 -O2 -pthread -fsanitize=undefined -fno-sanitize-recover=undefined \
  candidates/pinning/research_affine/test_inverse_service.cpp -o /tmp/inverse-service-test
/tmp/inverse-service-test
python3 candidates/pinning/research_affine/test_inverse_service_model.py
nvcc -O3 -arch=sm_89 -DNDEBUG -c -Xptxas=-v \
  candidates/pinning/research_affine/inverse_service_math_compile.cu -o /tmp/inverse-service.o
```

The decisive future experiment must include the actual persistent affine kernel,
cooperative residency checks, exact point and hit comparisons, and equal completed
work on a GPU. Compilation and these CPU checks do not establish a speedup.
