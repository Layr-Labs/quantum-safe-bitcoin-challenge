Effort: max

# Pinning: 128-leaf trees, seven-block finish, symmetric recovery, hot L2, and sparse prepare SHA

## Summary

This candidate starts from the live promoted source `372a325` / submission
`e2fd809`, whose official RTX 4090 score is 653,505,529 verified candidates/s.
It composes four largely independent improvements found in the public Yukon
submission history, then adds integration-specific audits and an actual local
`sm_89` CUDA compile. There is no NVIDIA GPU on the local host, so this note
makes no local throughput claim. The ranked 1200-second run remains the only
performance verdict.

The production configuration contains:

1. The 128-leaf candidate product/inverse trees and lean mixed-add field
   helpers from public submission `1a22808` (`5c85ae0`). That source scored
   659,859,051 officially when evaluated without the promoted persisting-L2
   policy.
2. The promoted persisting-L2 policy, refined to skip the double-sized and
   half-as-hot first table chunk. On an RTX 4090-sized 50 MiB policy window,
   the old prefix covers 11.5 of 15 expected random table reads while the new
   `[8 MiB, 58 MiB)` window covers 12.5 of 15. This refinement comes from the
   public `21d45a5` work; its reported local A/B was positive, but its official
   result was still pending when this archive was prepared.
3. The three-field symmetric recovery from public submission `0c6f4c8`
   (`3bccced`), adapted to the 128-leaf source. That source scored 657,885,190
   officially on the pre-persist tree. It reduces checkpoint state from eight
   to six `ulonglong2` planes and recovery from 10M+4S to 10M+2S.
4. Only the prepare-stage `Tail11` and `Digest32` sparse SHA-256 transforms
   from public submission `0a5de48` (`9bca47d`). Its author reported one
   million device-side bit-exact comparisons and a positive 4090 local run.
   The previously losing finish-stage Pk33, read-only-load, and asynchronous
   host-drain bundle is deliberately not imported.
5. A new seven-block launch bound for the 128-thread finish kernel. This is
   not the published losing 256x4 experiment: it raises finish residency from
   6x128 to 7x128 threads while capping registers at 72 rather than 64. Under
   CUDA 12.6 it adds only one local load and one local store to the fast finish
   image while exposing four additional resident warps per SM. It is compiler
   evidence and an occupancy hypothesis, not a local throughput claim.

This is not a claim of independent invention of those public components.
Attribution and the exact integration boundary are recorded below.

## Why this composition

The public history already closes most isolated tuning lanes. The 64 MiB,
15-term table beat the smaller 32 MiB geometry because eliminating one mixed
addition outweighed cache savings. A radix-373 GLV proposal was also tested in
the public history and scored 499,336,895, so the cache-capacity intuition is
not enough to justify a much more complex recoder. Prefetch, `__ldg`, streaming
cache hints, tree offload, host-loop pipelining, larger batches, the heavily
spilling 256x4 finish variant, and several field-helper rewrites were neutral
or negative.

The remaining positive results touch different bottlenecks:

- 128-leaf trees shorten barrier tails while retaining 16 resident warps in
  prepare; the seven-block finish bound raises finish residency from 24 to 28
  warps with a much smaller spill delta than the rejected 256x4 geometry.
- Lean field helpers shorten the fixed-base dependency chain.
- Persisting L2 protects the random-read GTable from the multi-gigabyte
  checkpoint stream; offsetting the window spends those bytes on hotter table
  regions.
- Three-field recovery removes 32 bytes of global state traffic in each
  direction and two field squarings from finish.
- Sparse SHA removes known-zero work from the prepare kernel, which dominates
  total batch time.

Because these mechanisms are not the same knob, combining them has higher
expected value than another isolated experiment. There can still be overlap:
register allocation, cache pressure, and instruction scheduling are nonlinear.
The CUDA resource report below is therefore part of the submission evidence.

## 128-leaf tree and lean mixed addition

`QSB_TREE_N` is 128 for the production candidate trees. The prepare launch is
four 128-thread CTAs per SM and finish is seven 128-thread CTAs per SM. The
candidate checkpoint stride follows `N`; the two root-hierarchy levels remain
256-wide. The super-root inversion indexes roots globally so a 16,777,216
candidate batch remains covered.

The finish launch bound uses 896 threads per SM. PTXAS chooses 72 registers,
so seven blocks consume 64,512 of AD102's 65,536 registers; their 12 KiB each
consume 84 KiB of shared memory. The exact CUDA 12.6 fast-finish image grows
from 3,328 to 3,336 instructions relative to six blocks. Spill traffic moves
from three local loads/stores and 20 bytes reported each way to four local
loads/stores and 28 bytes each way. Eight blocks was rejected before submission:
it forced 64 registers, 80 bytes of fast-path spills each way, and a 3,432
instruction image. Seven is the only new production geometry.

`GPUMath.h` imports the public `QSB_LAZY=1` implementation:

- `_ModSub256` folds a borrow by subtracting `2^32+977` under the mask.
- `_ModAddLazy` folds the `2^256` carry for the slope input.
- `_ModX3Fused` combines `R^2 + PPP - 2V` in one carry chain.

The source retains the public experiment switches, all disabled by default.
No table-prefetch, streaming-state, dense-tree-offload, probe-mask, or
shared-memory parking experiment is enabled in the ranked configuration.

## Three-field symmetric recovery

Let the fixed recovery point be `R=(a,b)` and the accumulator be represented
as `P=(X/U,Y/V)` with `U=ZZ`, `V=ZZZ`, and `V^2=U^3`. Preparation computes:

```text
d = a*U-X
W = U^2*d
```

Only `Y`, `V`, and `W` cross the kernel boundary. They occupy six aligned
128-bit structure-of-arrays planes: two planes per field. The 16,777,216
candidate state allocation is exactly 1.5 GiB instead of 2 GiB. The product
tree receives exactly the same denominator `W` as before.

The host computes the fixed constant `K=3*a^2 mod p` once with OpenSSL and
uploads four little-endian limbs to constant memory. Given the collective
inverse `I=1/W`, finish evaluates:

```text
h = V*I
t = V*h
u = b*t
v = Y*h
F = 2*u^2-K*t+a
H = 2*u*v
x_plus  = F-H
x_minus = F+H
y_plus  = (u-v)*(a-x_plus)-b
y_minus = b-(u+v)*(a-x_minus)
```

The output order and recovery identifiers remain `P+R`, then `P-R`. Both x
coordinates and both parity-bearing y values are normalized before compressed
key construction. Inactive and zero-denominator lanes still contribute the
identity to the collective and return only after the required synchronization.

The inherited square helper can lose a final carry on a tiny near-prime input
window. The symmetric formula reaches such inputs on constructible curve
points, so the imported recovery-local guard normalizes `u` and squares `p-u`
when the top bit is set. The original `u` remains live for `H` and the y
formulas. The existing general field-helper caveat elsewhere is inherited; this
candidate does not pretend to repair all of GPUMath.

## Access-frequency-aware L2 window

The GTable is 64 MiB. Chunk 0 has `2^17` records (8 MiB); each of the other
fourteen chunks has `2^16` records (4 MiB). Every candidate reads one record
from every chunk. A byte in chunk 0 is consequently half as likely to be read
as a byte in any later chunk.

The promoted policy starts a limited persisting window at table byte zero.
This candidate computes `cold = gt_entries(0)*64` and, when the table is larger
than `cold + want`, starts the same-size window at `d_gt+cold`. The available
length is bounded before applying the device's maximum access-policy window.
On hardware where the remaining table fits, or where persisting L2 is not
available, the guarded policy falls back safely or remains advisory exactly as
before. It cannot alter arithmetic results.

## Sparse prepare SHA

The ranked fast-tail path hashes two blocks with fixed sparsity:

- Tail11 has only `W[0..2]` variable, `W[3..14]=0`, and
  `W[15]=9995*8=79960`.
- Digest32 has `W[0..7]` variable, fixed `W[8]=0x80000000`,
  `W[9..14]=0`, and `W[15]=256`.

The specialized transforms fold those constants into rounds 0–15 and into the
first circular `WMIX`. Rounds 16–63 use the unchanged GPUHash macros. The
caller no longer initializes thirteen dead Tail11 words or eight fixed
Digest32 words. No Pk33 specialization is used in finish.

## Independent integration validation

All checks below ran on the final combined source, not on the component
archives in isolation:

```text
PASS: full 15-point deferred-Y chain; 20000 arbitrary-field, 1000 curve,
      and 1000 mixed-window accumulations
PASS: 210 128-leaf split trees; 4018 on-curve baseline/affine/compressed
      recovery comparisons; 7 singular skips; 10M+2S recovery
PASS: fast-tail single-hash contract; 20 host-selection cases
PASS: field final-carry audit; 200576 targeted/random cases
PASS: 64 MiB table; 50 MiB offset policy covers 12.5 vs 11.5 lookups
PASS: 211 shared product-tree cases including zeros/inactive identities
PASS: sparse Tail11 and Digest32 schedules match generic WMIX in 200000 cases
PASS: streamed recode; 51404 scalars and every one of 1048576 table slots
PASS: two-level root representation boundary cases through 65536 roots
PASS: two-level root inversion; 137492 roots across 12 boundary sizes
PASS: six-plane vector state is bijective/aligned and exactly 1.5 GiB
PASS: 2320 SHA256d fast-tail comparisons across synthetic prefixes
PASS: git diff --check
PASS: Yukon setup and CPU verifier smoke test
```

The SHA audit independently evaluates the generic circular schedule and the
two sparse first-mix schedules across randomized and boundary words. The
external-pipeline audit uses actual secp256k1 points and projective scales,
compares both recovered coordinates to the old recovery and an independent
affine addition, checks compressed encodings, and exercises singular points.
The vector audit proves every logical limb has one aligned address in the six
plane layout.

The local machine has no NVIDIA GPU, but an existing ARM64 CUDA 12.6 build
container allowed a real compile and link outside the candidate tree:

```text
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 -Xptxas -v

fast prepare: 126 registers, 0 spill stores, 0 spill loads,
              0-byte stack, 8192-byte shared memory
generic prepare: 126 registers, 0 spills, 192-byte stack,
                 8192-byte shared memory
finish: 72 registers, 28-byte spill stores, 28-byte spill loads,
        32-byte stack, 12288-byte shared memory; 7x128 resident threads
root prepare/finish: 44 registers, 0 spills
super-root inverse: 114 registers, 0 spills
```

Adding the sparse SHA transforms did not change any of those resource counts
relative to the combined build without them. The final executable linked
successfully against OpenSSL. Compilation produced only inherited unused,
unreachable-path, and OpenSSL deprecation warnings. Build products were kept
outside the repository, so the archived binary/stamp are not falsely presented
as a build of this source.

`yukon run --track pinning` was attempted before edits as requested. The local
configured runner bridge required unavailable privileged runner access, and
there is no local GPU, so it produced no baseline metric. The setup verifier,
CPU mathematical audits, and CUDA compilation are validation evidence, not a
substitute for the ranked hardware result.

## Provenance and credit

- Promoted base and persisting-L2 policy: `e2fd809` / `372a325`, solver
  `ercumentyildirim`, plus the repository's earlier credited development line.
- 128-leaf trees and lean arithmetic: `1a22808` / `5c85ae0`, solver
  `0xCramJam`, public note attributed to Claude Fable 5.1 in Claude Code.
- Three-field recovery: `0c6f4c8` / `3bccced`, solver `xlib`, public note
  attributed to GPT 5.6 Sol in Codex.
- Offset persisting window: public `21d45a5` work by `ercumentyildirim`.
- Tail11/Digest32 transforms: `0a5de48` / `9bca47d`, solver `Gajesh2007`,
  public note attributed to Grok 4.6 in Grok Build.

The current integration, adaptation, audits, CUDA compile, and submission note
were produced by GPT 5.6 Sol in Codex at max reasoning effort. The user also
provided a prior LLM's GLV hypothesis as a lead; it was not imported after the
public ranked evidence showed the tested radix-373 design was substantially
slower.

## Reproduction

From the benchmark checkout:

```sh
yukon setup --track pinning
for f in candidates/pinning/audit_*.py candidates/pinning/check_*.py; do
  PYTHONDONTWRITEBYTECODE=1 python3 "$f"
done
git diff --check
```

On a CUDA host, the official setup command builds the kernel at ranked `N`.
Run the unchanged benchmark harness for the authoritative verified score. No
claimed score is supplied because this host cannot execute the GPU workload.

## Caveats

- Component deltas are not guaranteed to add; the official 1200-second run is
  required to measure cache/occupancy/scheduling interactions.
- The seven-block finish kernel has 28 bytes of spill load/store traffic under
  CUDA 12.6, eight bytes more each way than the six-block control. The extra
  four resident warps are expected to hide more inverse-tree and SHA latency;
  this is an expectation, not a measurement.
- Rare field carry behavior inherited from the public base remains outside the
  recovery-local signed-square guard.
- The persisting window assumes only the documented per-chunk access frequency;
  actual cache associativity and replacement are hardware-controlled.

All edits are confined to `candidates/pinning/`. No harness, problem, subset,
or benchmark configuration file is changed.
