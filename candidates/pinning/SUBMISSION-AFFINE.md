# Pinning: resident affine workers with an asynchronous inverse service

Effort: ultra. This research and implementation use GPT 6 Astra through Codex.
The model and effort were checked against this task's active session metadata;
they were not copied from the previous submission's attribution. The authoring
machine has no usable NVIDIA GPU runtime. No local GPU speedup or claimed ranked
score accompanies this experiment.

## Starting point and purpose

The checkout starts at promoted commit
[`9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/9f239c386c7e99f8815103d9c6cc4465d7c5a9ba).
The latest read-only Yukon check reports 813,651,852 verified candidates/s as the
current best. A one-percent promotion requires at least 821,788,371/s after
integer rounding. These are the promoted control's results, not measurements of
this proposal. Our separate completion/transfer submission `b67219a6` has now
completed validation and was rejected at 797,802,186 verified candidates/s.
Its scheduling changes are a different experiment and are not evidence for
the present architecture.

The fresh clone followed Yukon's printed benchmark work directory. The schema-v2
manifest and candidate directory were inspected before changes. The required
`yukon setup --track pinning` passed, including the independent CPU verifier smoke
test. The unmodified `yukon run --track pinning` was attempted, but this machine
lacks the organizer's GPU bridge and a usable NVIDIA device. Consequently the
baseline attempt did not produce a ranked local score. CUDA 12.8.93 compilation
is available independently of execution on the ranked GPU.

The experiment attacks the dependency structure of scalar multiplication and
recovery. The promoted implementation is already heavily optimized in its field
operations, deferred-ordinate XYZZ walk, product trees and parity calculation.
Several additional arithmetic rearrangements reduced an intermediate count or a
register count without improving the actual compiled loop. The architectural
question is whether fewer ordinary field products can compensate for more
inversions when those inversions run efficiently in dedicated service warps.

## Scope and inherited work

Only `candidates/pinning/` changes. The benchmark manifest, harness, verifier,
problem generator, scoring, sibling track and organizer build interface remain
outside this submission. The existing GPLv3 notices and `COPYING` are preserved.
The promoted field arithmetic, hashing, table recoder and output contract remain
the starting point, with their existing credits. No external solver's unpromoted
implementation is imported into this work.

Affine point addition, Montgomery batch inversion, shifted fixed-base tables and
SIMD processing of independent inverses are established ideas. This note does
not claim their invention or universal novelty. A bounded audit examined 669
cached public notes, later PR descriptions and 71 cached recent source files.
It found earlier affine wavefront and persistent-kernel experiments, including
the W5 affine-wavefront submission `c2900b57` and block-level inversion work
reported in `7cb2974d`. Those negative results matter: merely keeping affine
coordinates on chip is not by itself a new or demonstrated improvement.

The specific proposal combines a 15-window affine walk, a last table containing
both recovery arms, and permanently assigned inverse mailboxes in one resident
cooperative grid. Each service lane handles an independent worker root; workers
need no global barrier between scalar windows. The bounded prior-art audit did
not identify this exact composition. That finding is a limited search result,
not a claim that no other implementation exists.

## Scalar walk and paired final table

Let `A = -r_inverse * G` on the original curve. The signed odd-digit recoder has
fifteen windows: eighteen bits first, then seventeen. Its final positive table
entry is `L_j = (2j+1) * 2^238 * A`. After the first fourteen windows, let the
affine prefix be `P`. Store two final-table points:

```
T_0(j) = L_j + R
T_1(j) = L_j - R
```

The two recovered points can then be computed directly as `P + T_0(j)` and
`P + T_1(j)`. There is no separate final scalar addition followed by a projective
normalization and recovery pair. Associativity justifies the transformation;
it does not change the underlying candidate or recovery identifier.

A negative final digit must both negate the stored point and swap arms:
`T_recid(-L) = -T_(recid XOR 1)(L)`. Negating without swapping would exchange the
recovery identifiers. The signed loader implements this mapping explicitly.
Replacing the old final 4 MiB table with two 4 MiB arms increases the actively
accessed fixed-base tables from 64 MiB to 68 MiB. This first integration retains
the original builder's complete 64 MiB allocation and allocates the shifted
8 MiB table separately: allocated table storage is therefore 72 MiB, with the
old final 4 MiB chunk unused during search. These points are built on the original curve;
the promoted recovery isomorphism and its weighted inverses do not apply here.

For each arm, write the difference of x coordinates as `d_r` and the difference
of y coordinates as `n_r`. Form one leaf denominator `D=d_0*d_1`, obtain its
inverse from the local batch tree, and use
`lambda_r = n_r*d_(r XOR 1)*inverse(D)`. This recovers both affine slopes from one
root-inverse request per candidate group. Each output x coordinate follows the
ordinary affine addition formula. The compressed key needs only y parity, so a
specialized implementation can use the existing parity-product window instead
of materializing a full final ordinate.

Zero denominators, cancellation, doubling, infinity and partial candidate tiles
must not contaminate a batch product or silently discard valid search work.
Their product-tree leaves use the identity while their validity is tracked
separately. A complete ranked integration needs a correct exception path and
must retain each original candidate's sequence, locktime and recovery mapping.
The isolated research primitive explicitly declines these exceptional cases;
its CPU tests do not by themselves establish the complete solver contract.

## Resident inverse service and ownership

A worker CTA has 128 candidate lanes. Its shared product tree uses 8 KiB, and
its inverse array uses 4 KiB. Each ordinary affine-addition wave keeps coordinates
in registers, constructs one nonzero root denominator and requests its inverse.
The complete local tree costs `3N-3` field multiplications for N leaves, plus one
external inverse. The original leaf-validity mask survives the identity
substitution.

A service CTA contains 128 lanes, each permanently assigned to one worker CTA.
A service warp can therefore process thirty-two independent roots in parallel,
instead of running one useful root inverse with the other lanes idle. This does
not eliminate inversion work: each worker still depends on its own response.
Other resident workers can make progress while it waits, subject to scheduling
and arithmetic resource limits.

Each 128-byte mailbox contains root and inverse payloads and request/response
epochs. One worker lane publishes a root, then release-stores the new request
epoch. Its service lane acquire-loads that epoch before reading the payload.
The inverse payload is published before a release-store of the response epoch;
the worker acquire-loads the matching response before reading it. Only one
request may be outstanding, and the next request is not published before the
previous inverse is consumed. Epochs never wrap; a distinct terminal value stops
the service after finite worker work completes.

Every physical lane of a service warp participates in the service votes and
inverse invocation. Lanes without pending work invert identity. Per-lane spin
loops around a warp collective are avoided. Service code has no worker CTA
barrier. Worker barriers preserve each product tree until all readers finish,
including the boundary before the next wave overwrites its leaves.

The host must launch one cooperative grid and query occupancy for the actual
compiled kernel and actual device. Every worker and service CTA must fit its
resident capacity, with enough service lanes for all workers. A normal
oversubscribed grid with spinning workers could deadlock before the service CTAs
are scheduled. Launch bounds alone do not prove residency. Direct allocation,
launch and synchronization errors must propagate rather than look like completed
search work.

## Cost model and important limits

The algebraic ledger below separates multiplications M, specialized squares S,
parity windows W and inversions I. It excludes field additions, normalization,
loads, synchronization, SHA256, table construction and exceptional cases. The
baseline includes its actual TOP16 collective schedule at batch size 2^23; the
new architecture uses 128-leaf local inversions in fourteen sequential waves.

| Per candidate | M | S | W | I |
| --- | ---: | ---: | ---: | ---: |
| Promoted active path | 106.14843726 | 28 | 2 | 1 / 8388608 |
| Specialized affine target | 72.671875 | 15 | 2 | 14 / 128 |
| Initial exact research probe | 89.671875 | 0 | 0 | 14 / 128 |

The initial probe and this first integrated GPU-test candidate intentionally
implement every square and final parity product with a complete general
multiplication. Their active arithmetic ledger is 89.671875 M + 14/128 I. The
cheaper specialized target is not enabled and must not be advertised as this
submission's implemented operation count. The inverse
service also increases inversion demand enormously relative to the promoted
global hierarchy. A smaller multiplication ledger is not a throughput result.

On a hypothetical fully utilized 128-SM device with four resident 128-thread
CTAs per SM, reserve four service CTAs and retain 508 worker CTAs: 65,024 candidate
lanes are in flight. To reach 814M candidates/s, the average complete chain would
need to finish within 79.882 microseconds, or 5.706 microseconds per inverse wave
before allowing for other work. At three or six resident CTAs per SM, the
corresponding complete-wave budgets are 4.279 or 8.559 microseconds. These are
necessary feasibility constraints, not measured latencies or predictions.

A separate Fermat-inverse resource screen uses a verified p-2 addition chain:
257 squares and fifteen products. It compiles with fewer registers, but its long
arithmetic path may exceed the latency budget. That alternative is retained as
research evidence; it is not treated as a proven service-speed optimization.

## Completed component evidence

The following checks were completed on the research components before ranked
integration. They have distinct scopes and are not a substitute for executing
the final candidate on the remote GPU:

- The actual inverse-tree header passed 112 CPU-scheduled blocks, 13,440 leaf
  inverse comparisons and 39,984 counted tree multiplications. The fixtures
  included 2,034 masked leaves, multiple widths, aliases and allocation sentinels.
- The actual mailbox header passed multithreaded C++20 transport tests with
  UndefinedBehaviorSanitizer, mixed epoch counts, partial service warps and unused
  warps. The exhaustive finite protocol model visited 770 states without a torn
  payload or nonterminal deadlock. Two premature-publication mutations failed.
- The shifted-table audit executed the C++ builder, signed loader and finish
  bodies with exact field shims. Independent Python curve arithmetic checked
  2,368 table points, 552 signed loads, 543 recovered-key pairs, 1,086 hashes,
  225 recoder cases, 2,172 recid-priority checks, singular/infinity declines and
  a table-builder batch-boundary crossing, with zero mismatches.
- The canonical subtraction PTX audit checked 34,641 operand pairs, 69,282
  output-alias cases and 34,641 negations, with zero mismatches. Two negative
  controls were detected. This is a PTX semantic model, not native GPU execution.
- The literal Fermat source chain was parsed and its exponent verified as p-2;
  136 modular reference cases passed. This checks that research chain's algebra.

The standalone streamed-tail probe compiled for sm89 with these resources:

| Minimum CTAs/SM | Registers | Shared bytes | Stack bytes | Spill stores / loads |
| --- | ---: | ---: | ---: | ---: |
| 3 | 148 | 12,288 | 120 | 0 / 0 |
| 4 | 128 | 12,288 | 120 | 0 / 0 |
| 6 | 80 | 12,288 | 352 | 356 / 248 B |

ELF function boundaries and SASS show zero worker LDL/STL instructions at bounds
three and four. Their 120-byte stack belongs to the service buffer and inverse
path, which still perform local-memory accesses despite zero spill counters.
At bound six the worker itself has spills. These are the isolated probe's
resource results; final integrated kernel resources are reported separately.

## Active ranked integration

`QSB_AFFINE_SEARCH=1` is the submitted default in `pinning.cu`. It sets the table
configuration to original-curve coordinates (`QSB_ISO_XR=0`, `QSB_YOFF=0`), includes
`affine_search.cuh` and `affine_driver.cuh`, and routes the real search to the new
driver after the original problem checks and GPU fixed-base table construction.
`QSB_AFFINE_SEARCH=0` retains the promoted search for a diagnostic control build.
The new header requires carry-complete shared inverse-tree products. Its point
arithmetic uses complete general multiplications followed by canonicalization;
squares and parity products deliberately use the same complete multiplication
for this first remote experiment. It does not enable the lower-operation-count
specialized-square/parity target or the experimental Fermat inverse.

The actual SHA path consumes the sequence-dependent precomputed state and actual
locktime, applies the existing optimized double-SHA256 message transforms, and
passes the resulting scalar through the existing signed-recode setup. It does
not infer locktime from blockIdx because some blocks are service CTAs. The
fifteen recoded table indices are stored in 7.5 KiB of per-CTA shared memory,
separate from the 12 KiB inverse tree. Each useful lane then executes thirteen
ordinary affine additions followed by the paired shifted-table finish and the
compressed-public-key SHA256 gate for each recovery arm.

The host builds 65,536 pairs of shifted final-table points from the actual
problem's recovery constants, uploads them, and queries cooperative support and
occupancy independently for the real-search and startup-audit specializations.
It requests the shared-memory carveout needed by the full kernel. The runtime
capacity determines service and worker counts, with at least one worker and one
service CTA. Allocation, table-generation, cooperative-launch and synchronization
failures terminate with an error. No occupancy value from the isolated probe is
assumed for the integrated kernel.

A worker traverses its own finite sequence of 128-lane tiles, with a stride equal
to all worker lanes. Each candidate index in the current batch belongs to exactly
one worker/lane/tile. Partial final tiles retain every physical lane in the
collectives while masking inactive leaves and output writes. Workers with no
useful tile still stop their service mailbox. All mailboxes and output masks are
zeroed before each independent cooperative launch; request epochs are private to
that finite launch and cannot wrap within the permitted batch bounds.

Hit reporting uses a two-bit-per-candidate mask, one bit for each recovery arm,
plus a one-bit-per-candidate exception mask. These fixed-size masks cover every
candidate in the batch; there is no small hit-list capacity that can overflow.
Every mask update is atomic. After a completed launch, the host scans the copied
masks once per candidate. For any nominated or exceptional candidate it invokes
the existing exact OpenSSL acceptance check on **both** recovery identifiers,
then appends each genuinely passing `sequence=... locktime=... recid=...` record.
The GPU nomination bits are not trusted as proof of a hit. Exceptional zero
denominators or infinity therefore reach a complete CPU recovery path instead
of being silently discarded.

The default batch remains 8,388,608 candidates. Sequence partitioning uses the
selected GPU and total-GPU arguments, advancing without sequence repetition.
The locktime interval remains `[500000000, 1744600000)`, with a shortened final
batch. The host waits for the entire finite batch, transfers masks, performs the
exact acceptance work and closes any result file before advancing. This first
architecture test does not claim the host-overlap optimization of that separate
submission. Its progress counter increases only by
completed batch sizes. The local progress timer starts before CUDA setup and
table construction; the organizer's clock and verified-hit score remain decisive.

## GPU startup audit, not a local GPU result

Before ranked search, the new driver runs three cooperative audit batches of
257, 129 and 255 candidates, spanning three sequences and distinct locktime
ranges, including byte-carry boundaries and partial CTAs. After checking the
actual audit kernel's occupancy capacity, these launches use one service CTA and
two worker CTAs so the 257-candidate case exercises worker tile reuse as well as
partial lanes and unequal worker completion. For all 641 candidate
pairs, an independent OpenSSL reference reconstructs the scalar from the actual
message suffix and recovery constants without reading generated tables or GPU
intermediates. The device's two x coordinates, two y parities and nomination bits
must agree. Any ordinary-fixture exception, mismatch or API failure aborts before
the search begins.

This startup gate will execute on the remote GPU. It has **not** executed on the
authoring machine. Its existence is not reported as 641 locally passing GPU
cases. It samples the active arithmetic and launch protocol on the actual
problem, complementing the much wider CPU component audits without proving every
GPU schedule or every possible input. Startup work and OpenSSL checks contribute
to the process time seen by the external harness.

## Final integration validation

The fixed organizer-style command compiled and linked successfully with CUDA
12.8.93, without an architecture override or a claimed local score. A direct
sm89 build also compiled into a CUDA cubin. The final default executable was rebuilt
after adding explicit CUDA-initialization error checks. Invoking it on the
authoring machine immediately reports that the CUDA driver is insufficient,
then exits with an error instead of continuing with an unusable device.

The required `yukon run --track pinning` was attempted again on the integrated
candidate. It exits unsuccessfully because the organizer's bridge/metrics path
is absent here. This is not a candidate verification pass and does not produce
a local ranked score. The configured remote run is 1,200 seconds (twenty minutes),
with actual process timing and fresh-problem hit verification owned by the
unchanged harness. No shorter diagnostic time is substituted for the ranked run.

`integration_resources.json` records the exact source hashes and compiler
resource logs. The complete production search and audit kernels use 19,968 bytes
of shared memory, including both the tree and recoded digit storage:

| Compilation path | Search registers | Search stack | Search spill stores / loads | Audit registers | Audit stack / spills |
| --- | ---: | ---: | ---: | ---: | --- |
| Default sm52 target | 128 | 144 B | 32 / 28 B | 126 | 120 B / zero |
| Direct native sm89 | 128 | 120 B | 0 / 0 | 126 | 120 B / zero |
| Default compute52 PTX assembled offline for sm89 | 128 | 120 B | 0 / 0 | 126 | 120 B / zero |

The third row is an explicit compatibility/resource screen using ptxas on the
PTX produced by the unchanged default build target. It is not execution by the
remote CUDA driver JIT, not a measurement of driver compilation latency and not
a promise of identical device runtime behavior. Final runtime occupancy is
still queried by the submitted host code. The direct-native build preceded the
last host-only initialization-error check; the default build and offline PTX
screen were rebuilt after that change.

Three integrated CPU audits passed in addition to the earlier component checks:

1. `search_primitives_result.json` records execution of the actual scalar SHA,
   digit-extraction and compressed-key filter helpers: 2,048 real SHA256d cases,
   6,155 signed recodings, 2,048 independently reconstructed recovery pairs and
   6,666 public-key first-word checks. There were zero mismatches. The test-only
   difficulty N=4 produced seven cases where both recids pass, as well as cases
   where only one passes. Ranked difficulty is not changed by these fixtures.
2. `search_cpu_result.json` executes the actual search-kernel body with CPU
   thread/barrier shims and independent exact field arithmetic. It checks 513
   active candidates and 1,026 affine arms against OpenSSL, with seventy root
   request/response round trips, two worker CTAs, partial tiles, 127 inactive lane
   iterations, early worker STOP and output-mask canaries. All results agree;
   none are declined. At test N=4 there are 63 matching candidates and 65 matching
   recids, including two candidates with both arms passing. This exercises the
   real kernel's control flow, arithmetic equations and candidate association;
   it does not execute its PTX instructions or the GPU scheduler.
3. `affine_publish_result.json` executes the actual host mask-drain and text
   writer with a controlled acceptance predicate. Ninety fixtures cover 10,020
   candidate slots and exactly 8,088 expected output lines, with no duplicates
   or mismatches. They include 1,859 false GPU nominations, 2,040 double-hit
   candidates and 391 exception-only matches, plus partial-word boundaries and
   ignored bits beyond the batch end. The controlled predicate isolates output
   selection; cryptographic acceptance is checked by the separate OpenSSL audits.

Together, these checks establish the tested algebra, indexing, data ownership,
mask decoding and build compatibility. They do not establish native GPU
correctness, deadlock freedom under every GPU schedule, hit recall on all
possible inputs, or a performance advantage. The remote startup audit and the
independent ranked verifier remain essential parts of the experiment.

Shared components, research source and result files live in `research_affine/`, including
`inverse_service.cuh`, `inverse_tree.cuh`, `shifted_tail.cuh`, the standalone
probe, protocol/tree/tail audits, `count_operations.py`, resource reports and
`compile_probe.py`. The production entry headers are `affine_search.cuh` and
`affine_driver.cuh`. Generated executables and intermediate disassembly are not
needed to reproduce the source change. The source archive is checked against the track manifest's 8,388,608-byte
submission limit before upload; compiler logs and source-bound JSON records
preserve the local evidence without requiring generated binaries.

## Reproduction and remote decision

From the benchmark work directory, the component audits can be reproduced with:

```sh
python3 candidates/pinning/research_affine/test_inverse_service_model.py
python3 candidates/pinning/research_affine/test_inverse_service_fermat.py
python3 candidates/pinning/research_affine/audit_shifted_tail.py
python3 candidates/pinning/research_affine/audit_affine_sub.py
python3 candidates/pinning/research_affine/count_operations.py
python3 candidates/pinning/research_affine/audit_search_primitives.py
python3 candidates/pinning/research_affine/audit_search_cpu.py
python3 candidates/pinning/research_affine/audit_affine_publish.py
python3 candidates/pinning/research_affine/compile_probe.py --ptx
```

The C++ CPU tests additionally need a C++20 compiler and pthread support; the
exact big-integer audits need Boost and OpenSSL. The compiler probe needs nvcc,
cuobjdump, readelf and OpenSSL, and accepts explicit tool paths. It builds an
executable and runs only `--describe`; it deliberately does not execute GPU work.
The organizer interface remains the required nvcc build of `pinning.cu`:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning candidates/pinning/pinning.cu -lcrypto -lm
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -o pinning-sm89 candidates/pinning/pinning.cu -lcrypto -lm
yukon run --track pinning
```

The first command uses the configured ranked difficulty. Local CPU fixtures
that use N=4 do so only inside their extracted test translations.

The remote fresh-problem run is the decisive test: verified hits and the
harness-owned elapsed time determine the score. Service latency, divergent
inverse iterations, register pressure, tree barriers, cache behavior and larger
tables can all erase the field-operation reduction. A valid but slower outcome
is informative and will be reported as such. No percentage speedup, promotion,
or huge advance is asserted before that measurement. Any follow-up GPU A/B
comparison should use equal completed search ranges, sorted verified hits,
final drains and the same device, with alternating run order.
