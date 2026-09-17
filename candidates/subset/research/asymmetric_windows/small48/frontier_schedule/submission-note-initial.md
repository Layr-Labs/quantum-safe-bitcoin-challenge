# Subset: guarded fourteen-window chain with a 48 MiB small-table prefix

Effort: xhigh for the main GPT 6 Astra implementation and high for GPT 6 Astra
reviewers, using Codex. No local GPU throughput or claimed score is supplied.
This is a substantial table-geometry and point-order experiment combined with
the promoted frontend selection and inherited external inversion. Its expected
benefit remains a hypothesis until official CUDA execution and scoring.

## Source, frontier and attribution

The selected production/audit include-closure fingerprint is
`140133ddb7a0408750a5396ce20658a1b48591422b5974315c8b76110de3117c`.
The exact source is installed in the production path; the prior production
source is preserved separately. The frozen source is under
`candidates/subset/research/asymmetric_windows/small48/frontier_schedule/candidate`;
its `prepared-source.json` records the exact donor hashes. All paths here are
relative to the benchmark root. The benchmark has schema version 2 and this
submission targets only the `subset` editable path. Runtime tables are derived
from the supplied problem base; search scalars still come from the actual
SHA-derived candidate data. No answers or hits are cached across problems.

The public frontier used for this integration is
[PR77](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/77),
head `e3a95b0f07e426443d5801b71588e469b3670bb3`, promotion
`d2772418e0f372767b4c59f7382d71f9142585fe`, at **488210159 verified
candidates/s**. We import its exact 256-window selection and lane grouping,
with 54 first-key and 56 second-key classes. We do not import its entire grinder
architecture or attribute its score to this port. The schedule/hash bodies
following the selection were already identical in the compared sources.
The selected implementation retains our external checkpoint/root inversion
pipeline and checked earlier field primitives.

The inherited lineage includes our promoted
[PR60](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/60),
submission `65fb673d-5014-4ee8-866d-97d972e7b1f0`, promotion
`8e5cd89b50dafa0dbef41ce92cab1f7974e531c2`, at 451135044 verified
candidates/s, and welttowelt's promoted
[PR62](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/62),
submission `80a850f0-3bd5-484b-b890-70faec8a39d0`, promotion
`106a6826ecf8998e684169f46e8de9dce1cf3f0f`, at 477182283. These whole-program
results are historical baselines, not measurements of isolated mechanisms.

The retained, inactive 16 GiB geometry code and the bounded table builder descend through
our [PR74](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/74),
validation commit `154b2bd97a9c918ac3d992c31a6cf7901aeab1a0`, and the previous
adaptive subset candidate. That work substantially used alvaroborras's
unpromoted [PR64](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/64),
head `a7b21d0f62e6d73b66fe820e228f8db50d504716`, and batched ladder construction
informed by MakiRH4's [PR46](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/46)
and jacklightChen's [PR53](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/53),
head `9274883051636def6db5add0d3ba0e02314813f0`. These substantial unpromoted
contributions remain credited as coauthors.

ercumentyildirim's unpromoted pinning submission
`e2fd8093-2ba5-4d40-8f25-dabb0a4807c5` reported an RTX 4090 maximum persisting
L2 set-aside of 50 MiB and a policy-only improvement of 1.203% on that author's
64 MiB case. This motivated the 48 MiB prefix and optional cache policy here;
ercumentyildirim is also credited as a coauthor. Those hardware observations
and timing belong to the author. We have not independently reproduced them,
and they do not predict this geometry's performance. This submission changes
subset only and performs no new pinning experiment.

The promoted parent already attributes jacklightChen, DPZZxlz, nullforest8200,
alvaroborras and Meganpark980320, along with earlier epoch contributors.
Streamed signed recoding, deferred-Y XYZZ, ranked SHA reuse, direct recovery,
and the inverse hierarchy remain inherited work. Original GPL notices and
`COPYING` are retained. Prior review by Gemini 3.8 Flash High and Grok 4.6
covered earlier adaptive components; it is not validation of this final source.

## Mechanism and testable performance hypothesis

The active compact branch now has fourteen windows: twelve 17-bit windows
followed by two 26-bit windows, covering all 256 scalar bits. The small tables
occupy exactly 50331648 bytes (48 MiB), and the two cold tables occupy
4294967296 bytes (4 GiB). Total permanent table storage is 4345298944 bytes.
Affine X/Y coordinates remain interleaved at 64 bytes per entry.

The point chain loads cold windows 12 and 13 first as an independent seed pair,
then adds windows 0 through 11. Its ordinary path costs **88M + 26S**, versus
95M + 28S for the earlier compact fifteen-window branch. Logical table payload
falls from 960 to 896 bytes per candidate. These counts exclude hashing,
inversion, recovery, checkpoints, field additions and startup. They cannot be
converted directly into a throughput prediction. Two large random accesses,
table construction, guarded exceptional handling, and external pipeline traffic
can outweigh the arithmetic saving.

The earlier asymmetric control used a 52 MiB small prefix and 3 GiB cold tables.
Shrinking its small prefix to 48 MiB costs another GiB of cold storage. The
motivation is that all twelve small tables can fit within the externally
reported persisting-cache capacity. Actual usable capacity is queried at runtime;
fitting a capacity does not establish cache residency or a hit rate.

The independently implemented optional policy applies only to the 48 MiB prefix
on the same default stream as search. It checks the selected device's capacity
and maximum access-policy window, sets the persisting limit, reads its effective
value, and installs a hit-ratio-1 access window. Unsupported or insufficient
capacity leaves ordinary caching active. Where installation cannot complete
after changing the limit, the code restores the previous limit. Expected optional
errors are cleared with checks for unrelated errors; unexpected failures remain
visible. No policy applies to the 4 GiB cold region. This is a cache-priority hint,
not a promise that all table lines stay resident.

This experiment forces the fourteen-window table: `d_wide` is explicitly null,
so the older 16 GiB option and runtime tuner are inactive. The 48 MiB plus 4 GiB
table is mandatory; allocation failure is fatal rather than selecting a smaller
geometry. Only the optional L2 policy can fall back to ordinary caching. Retained
wide-branch source and its audit coverage are ancestry and dead-path evidence,
not a second runtime algorithm in this candidate. Startup and JIT costs remain
part of total-process cost and can cause a regression even if steady search is
faster.

The active table builder retains bounded GPU construction with runtime-dependent host
ladders and tiled normalization. This avoids materializing the full large table
in host memory. CPU builder projections and sampled runtime checks are useful
evidence, but they do not certify every entry of a resident GPU table.

## Correctness correction: exceptional final additions

Simply reordering incomplete mixed additions is unsafe. The original ascending
asymmetric chain and the unguarded cold-first chain both passed sampled checks,
yet independently constructed scalar witnesses produce equal-point final
additions. The old formula returns zero denominators instead of doubling.
The preserved cold-seed controls document separate failing witnesses for those
orders; this is a concrete failure of the unguarded designs, not speculation
based on a score.

The selected final helper tests its already computed X and slope differences.
When the X difference is zero, it constructs twice the affine input if the slope
difference is also zero; otherwise it represents infinity. Ordinary-path field
operation counts stay unchanged, but the predicate, branch, and exceptional code
have real compilation and possible runtime costs.

For the selected geometry, let the odd positive recoding representative be M,
with 1 <= M <= n. The two cold terms combine to
`C = ((M >> 204) | 1) * 2^204`. After small windows through c, their sum is
`L = M - ((M >> S) | 1) * 2^S`, where `S = 17 * (c + 1)`.
For c through 10, the source-specific bounds place both accumulator-plus-input
and accumulator-minus-input coefficients strictly between zero and n. The seed
pair also cannot be equal or opposite: its normalized sum/difference is nonzero,
odd, and has magnitude below n. Thus only the final addition requires the guard
for this ordering, assuming the stated recoder identity and a valid nonzero
order-n runtime base. This argument is not a proof of the CUDA arithmetic,
memory loads, other retained geometries, or the complete program.

## Startup correction and a failed official control

The earlier adaptive subset [PR86](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/86)
failed after approximately 5.6 seconds with zero hits. The available artifact
did not expose candidate stderr or exit status. The cause remains unknown;
this note does not diagnose it as allocation exhaustion or connect it to the
rare curve witnesses.

The selected source nevertheless retains an independently identified startup
correction: request a checked 4 KiB stack for ranked mode before builder launches,
retain 32 KiB for generic mode, check mandatory search-buffer allocations, and check 22 previously unchecked
allocation results. The predecessor also moved optional wide selection after
mandatory buffers; that optional selection is disabled in this experiment.
Five existing pipeline allocation checks are preserved. A prior free-memory
query could otherwise precede both the larger stack request and mandatory
buffers. CPU mocks establish allocation ordering and error-handling behavior,
not the actual device's allocation size, automatic stack growth, or PR86's cause.

## Evidence and its boundaries

The current `frontier_schedule/frontend-results.json` is bound to the selected
fingerprint above. It passes 3840 inverse values, 6144 complete SHA256d cases,
10262 signed recodings, 8086 unranking cases and all 256 frontend window
selection cases. The recoding portion covers the inherited wide branch;
the active fourteen-window compact chain has separate checks. This is CPU
reference execution with synchronization emulation and OpenSSL operations.

The following existing component reports are useful precursor evidence. Their
original source identities are explicit, so they must not be confused with an
unqualified whole-source pass after the frontend port:

| Component | Existing result | Source identity / scope |
|---|---|---|
| Guarded 48 MiB chain | 12769 recodings, 628 chains, 1242 recovered keys, 532 actual vector loads, 7536 prefetch addresses | `df16d1bd5cd279bb881f29049cdd797e2f0da6be1f065e530dae926c472cd117`; CPU OpenSSL/sparse-table oracle |
| Final helper and domain | 159 equal, 159 opposite, 159 ordinary cases; all 159 old-helper doubling controls fail; 214 extra scalar cases and 11 prefix bounds | Same geometry source; actual extracted helper with independent affine oracle |
| Host ladder / builder formulas | 101494 decodes, 4482 entries, three runtime bases | Same geometry source; source-derived CPU formulas |
| Builder pipeline | 13107 entries, three runtime bases including the earlier failed-run base | Same geometry source; actual kernel bodies projected to CPU threads and OpenSSL, sparse storage and canaries |
| Optional L2 policy | 180 host scenarios and 30 injected failures | `76c07307fc835215e0974b0fe73118da9c2375ca0c9cf757661b73fb53678d5f`; mocked CUDA API |
| Startup budget | 279 budget scenarios, 29 mandatory faults, five optional-policy cases; unchecked-allocation mutation caught | Preserved startup component report; mocked API and modeled budgets |

The geometry, field, builder, exception helper and policy source files are
unchanged by the frontend port; `prepared-source.json` lists hashes for review.
Fresh final-source exceptional-chain checks now pass the same 159 equal, 159
opposite and 159 ordinary helper cases, 628 complete chains, 1242 recovered keys,
12769 recodings, 532 actual loads and 7536 prefetch-address checks, bound to
`140133ddb7a0408750a5396ce20658a1b48591422b5974315c8b76110de3117c`.

Fresh production native compilation for that exact source passes CUDA 12.8.93
sm89 and default flags. Compact prepare uses 124 registers, 24 KiB shared memory,
zero stack/spills and 9743 static non-NOP instructions; finish uses 80 registers,
24 KiB shared memory, zero stack/spills and 4685 non-NOP instructions. The older
policy/control comparison also has identical per-kernel instruction listings,
but this does not establish driver-JIT equivalence or GPU behavior. The audit translation unit also passes fresh sm89 and default-target native
compilation. Exact-source CPU pipeline checks pass 1282 leaf inverses, 2000
recoveries, 768 controlled candidates, 237 generated hit records and three
singular lanes. Actual host field tests pass 60540 multiplies and 40360 squares;
the PTX semantic model checks 2180 cases each, including 23 overflow cases and
a carry-mutation negative control. CPU PTX modeling is not device execution.
`yukon setup --track subset` also passes its verifier smoke checks on the local
Mac. A subsequent `yukon run --track subset` stopped before GPU execution
when the configured privileged benchmark bridge could not launch on this host.
That host has no nvcc or NVIDIA GPU; the native CUDA compilation above ran
in a separate ARM Linux compiler VM. No GPU audit, sanitizer execution, cache
measurement or throughput measurement has been performed locally.

The checked prior `GPUMath.h` and `square32.cuh` are retained rather than importing
the whole frontier's arithmetic changes. Their near-p carry witnesses and stale
condition-code negative controls remain part of the research record. OpenSSL
substitution in chain tests alone cannot validate PTX arithmetic.

## Reproduction

Run from the benchmark root. The frozen candidate generators deliberately refuse
to overwrite existing destinations. Use the source paths below to check the
selected artifact without regenerating it:

```sh
SRC=candidates/subset/research/asymmetric_windows/small48/frontier_schedule/candidate
python3 -B candidates/subset/check_candidate.py --selection grouped --source "$SRC" --report /tmp/qsb-small48-frontend.json
python3 -B candidates/subset/research/asymmetric_windows/cold_seed/check_exceptions.py --source "$SRC" --first-width 17 --output /tmp/qsb-small48-exceptions
PYTHONPATH=candidates/subset:candidates/subset/research python3 -B candidates/subset/research/wide_windows/check_builder.py --source "$SRC" --compact --widths 17,17,17,17,17,17,17,17,17,17,17,17,26,26 --report /tmp/qsb-small48-builder.json
python3 -B candidates/subset/research/asymmetric_windows/small48/check_policy.py
python3 -B candidates/subset/research/startup_budget/check_startup.py
```

The last two commands intentionally reproduce their frozen precursor reports;
they are not source-root overrides for the final tree. The exception checker
also runs the actual full chain with independently expected widths, cold-first
order, and constructed scalars. Legacy regular16 source assertions in older
deferred-Y checks do not describe this fourteen-window chain.

On a CUDA build host, compile a fresh copy with
`nvcc -O3 -DQSB_ZEROS_N=24 subset.cu -lcrypto -lm`; add
`-arch=sm_89 -Xptxas=-v,--warn-on-spills` for comparable static resource reports.
Compile `tests/gpu_epochs/tree_audit.cu` separately. After installation in the
production path, `python3 -B candidates/subset/preflight.py --cuda` fingerprints
the closure and compiles production and audit in a fresh temporary tree.
The trusted wrapper caches using only `subset.cu`'s timestamp: invalidate its
generated binary and `.subset.build` stamp after header changes, then use
`yukon setup --track subset` and `yukon run --track subset` on a GPU host.
A fresh temporary compile does not refresh that wrapper cache or execute CUDA.

## Interpretation of the next official result

The intended experiment is whether the smaller reusable prefix, one fewer
addition, independent cold seed loads and promoted frontend selection together
justify this larger table and inherited pipeline. The 488210159 frontier result
does not establish a gain from the transplanted frontend, and individual
historical gains must not be added together. Check the actual cache-policy and
startup logs and compare verified throughput against both the
inherited bases and the frontier at evaluation time. A loss would reject this
particular integration under those conditions, not all asymmetric geometries
or every external-inversion schedule. Keep the exact source, validation logs
and failure artifacts so the next experiment can distinguish startup, memory,
arithmetic and scheduling effects.
