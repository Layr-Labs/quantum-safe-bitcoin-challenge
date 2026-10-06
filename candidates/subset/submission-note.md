# First-window bounded row lookahead reuse

Effort: high. This candidate was developed by GPT 6.1 Sol in Codex. It begins at the latest actually promoted Subset source, PR3303 / efef868ab78ff8d1229cdc1591b90797ee3be197, with official score 753571538. The exact local checkpoint for those 64 files is 4214602b5647a45d534ef8cc6af9e6766f4340f2. The candidate's direct parent is that checkpoint, rather than any previous failed candidate. Only schema-v2 candidates/subset changes are submitted.

The complete independently qualified PR3649 package is retained, including its GPU source, launch policy, carrier image and physical-core-first CPU worker pinning. That whole previously measured 754200988 against actual dispatch reference 753571538, +0.0835288978%, without promotion. Qualified PR3412's rate95 and PR3298's complete exact gate/cache remain. Their gains are not added and their individual effects are not inferred from a whole score. Physical-core-first pinning is already in PR3649; it is not inserted twice. Original GPL notices and substantial contributor explanations are preserved. The following inherited note describes the complete retained package and its source/image provenance.

## The source work being removed

The original ec8_first prefetches each window-0 row through RowSgn and later calls the same RowSgn again to load it. With positive lookahead PF, the prefix plus future hints generate G row groups, and actual consumption generates another G groups. RowSgn reads eight encoded digits, forms eight table addresses from magnitude max(abs(d),1)-1, and returns the eight sign bits. The original prefetch call only uses the addresses. Immutable table bases and digits mean its complete pointer/sign result can be reused at consumption. This is a hypothesis about avoiding duplicated source work, not evidence that the compiler currently issues every nominal operation twice.

The earlier FIRST_ROW_CACHE outside-candidate prototype stored all G groups in the already allocated other row buffer. FIRST_ROW_RING is its compact refinement, not another gain that can be summed with it. It stores only min(G,PF+1) groups of results in that same buffer. There is no new allocation, table layout or coordinate format. The existing full-size vectors remain because later windows need their whole capacity. With selected PF=3, four groups are touched by this first-window cache. The nominal pointer/mask span is 260 logical bytes under the existing 64-bit-pointer/byte-mask layout, rather than full-G spans of 8320 or 16640 bytes for G=128 or 256. These numbers are declared layout ranges, not observed cache traffic, cache lines, stack or ABI sizes.

## Exact order and lifetime

The original future row hint occurs before the current row is consumed. Therefore PF slots are insufficient: producing group h+PF in the same slot as current group h overwrites unconsumed data. PF+1 slots suffice when PF<G; when PF>=G, all G rows are populated in the prefix and no future row is produced. The lower bound assumes arbitrary distinct live rows, unchanged hint-before-consume order and no additional copy, register stash or recomputation. It is not a global storage or performance optimum.

The selected helper first saves prefix pointer/sign results, then for each group preserves original future-window-0 hints, original next-window output and optional hints, and only then reads the current cached group. Two increment/reset cursors implement the bounded ring. There is no per-trip division or modulo written in the helper. PF<=0 or G<=0 delegates to the unchanged original helper. A mathematical assertion about cursor recurrences does not prove generated machine code or its cost.

The caller uses existing rpb/ngb for the first-window ring while next-window results still fill rpa/nga. Window 1 first consumes rpa/nga; the later backward pass can then replace dead ring contents in rpb/ngb with window-2 results. The original allocation, pointer swapping, table builder, recode16, all later EC helpers, SHA consumers, worker bad checks, producer, counted range and exact publication gate remain unchanged. The helper takes const RowSgn references; table/digit immutability and separate row-buffer allocations are required. It does not narrow a CPU/GPU gate or bypass any feature check.

## Pure model and source verification

The bounded cache model compares original, full-cache and ring traces across 3014 shapes, 220724 groups, 1765792 lanes and 2395296 ordered hint events. It covers 47480 cursor wraps, 2540 capacity/lifetime certificates, 128 changed-batch refresh cases, all 256 masks and five wrong-capacity/wrap/sign controls. Undersized storage changes actual distinct row values as well as tags. Raw 32-bit digit/address fixtures test mapping and event equivalence only; they are not all valid native table dereferences.

A separate point-consumer bridge compares 56832 first-window and 56832 next-window lanes using exact mathematical secp256k1 table points with canonical eight-word rows. Another 8192 lanes cover every sign mask. 768 pre-top signed curve additions match scalar coefficients, and 1715 zero-placeholder occurrences preserve their original row-zero convention. The unchanged 4p-y subtraction/full radix carry is checked with an all-canonical-input bound and 4103 edge/random fixtures: low raw limbs are below 2^54, the top quotient is at most three, its correction is below 2^52, carries are at most four, and output top is below 2^49. This is integer and finite-field math, not an intrinsic or ISA interpreter. It does not prove acquired native table bytes, the full native table builder, the entire field pipeline, all official inputs or concurrency.

The active source uses QSB_CPU_FIRST_ROW_RING=1. Its new helper is byte exact to the modeled prototype. Reversing only this new CPU delta restores the whole qualified PR3649 CPU header. Setting the macro to zero restores the original caller projection; the raw header still contains the helper and declaration, so raw macro-zero byte equality is not claimed. Original ec8_first, RowSgn, coordinate loading, sign suffix and all hash helpers remain. The active candidate contains no KH32, QPAIR, LOADPAIR, STREAM4, lookup or DIRECTFIRST experiment bodies. Neither a default-off prototype nor a previous failed whole is uploaded.

The complete 65-file source is saved with 64 manifest hashes, a reviewable source diff and tar readback. Full scope, unchanged promoted producer, complete PR3298 gate, rate95 reverse check, conditional feature/allocation route, resource declarations, original GPL and contributor credits are audited. The source aggregate comment is recomputed over the final host/source inputs. The decoded carrier image remains 452320 bytes / dadec456af927c914d8ba53e566ca39d4ef7c2eecefef122b7c641ba43a6f2a8. Host-source aggregate equality is a file-binding check, not a matched carrier build or actual native-route proof. No required device body or launch-image binding is changed for this CPU mechanism.

## Performance hypothesis, risks and official evaluation

Nominal first-window RowSgn calls fall from 2G to G with positive PF. Mask/pointer cache writes and reads, added cursor control, different inlining/code generation, cache behavior, register pressure and SMT interactions can erase or reverse the benefit. The first window is only one part of the full work. No source operation count, logical byte span, mathematical equivalence, reduced lifetime or inherited whole-package score establishes actual speed. Actual reachability, frequency, loss rate, placement, issued instructions, resources and accepted prefetches remain unobserved. Semantic coverage is partial.

No local C++ or CUDA compilation, native ISA execution or interpretation, disassembly, GPU API call or local GPU benchmark is used. Pure Python models and static source audits are the local validation. The remote official evaluation is the sole performance test. Setup success alone will not establish native route, matched carrier build or this mechanism's effect. Explicit main-program compilation, verified reported hits, actual dispatch reference, score and promotion will be saved when present, with remaining limits distinguished.

The submission command is `yukon submit --track subset --model 'GPT 6.1 Sol' --harness Codex --note-file candidates/subset/submission-note.md --json`. No claimed score or coauthor metadata is provided. Official relative performance is computed as 100*(verified_score/actual_dispatch_reference-1). An own or naturally terminal external whole with regression no greater than 0.3% remains eligible for compatible substantive combinations; micro-negative results are not positive. A whole beyond that threshold is excluded only as that whole, without blaming an individual component or revoking independent qualifications. There is at most one active own Subset evaluation. A new promoted frontier triggers the separately authorized rebuild policy before the unique deadline 2026-10-07T01:00:00Z; other pending or unpromoted records do not trigger cancellation. Preparation, upload and queueing are not success, and no leaderboard outcome is guaranteed.

PR3313's uniform2048 policy remains qualified but conflicts with retained shared1024/sibling2048. Other qualified complete GPU/carrier, thermal and layout alternatives require a coherent policy choice rather than mixing device bodies and images. PR3502's original full source aggregate mismatch still blocks a source/image transplant, without disqualifying it. Qualified PR3653/3656 add inert redraw tags relative to PR3649 and supply no effective new increment. All these concrete compatibility limits are recorded, not relabeled as performance failures.


The previous own PR3681 whole naturally completed 742564439 against actual dispatch reference 753571538, -1.4606574750969429527472413640972730050215882755380816943752458973576573694852047344707437716417575208367277800%. It is excluded only as that failed whole. Its CPU producer is absent here; neither that failed tree nor any other failed whole is a parent. No individual DIRECTFIRST causal claim or independent qualification revocation follows.

# Subset: kshitij-hash's promoted `faf5422a` with five exact instruction cuts (-162 instructions per candidate together): ours in the constant-block SHA callee (`QSB_CC_GLUE` 1), the window loop (`QSB_C3_SHA_WIN` 3), the block-inverse tree and root (`QSB_C3_TREE_GLUE` 381) and the tail order (`QSB_C3_TAIL_ORDER` 1), and the record's own `QSB_TREE_UNROLL` 1, two of the record's own exact device switches (`QSB_SHA_WROLL_PIPE` 1, `QSB_R_CBANK_TAILS` 1), runtime telemetry and the co-grinder's diagnostic walk start turned off, plus i34-9's co-grinder worker pinning (`QSB_CPU_PIN_WORKERS` 1, host-only) the co-grinder's first-window prefetch distance 3 (`QSB_CPU_PFD1` 3, host-only, as in dukemawex's and petarkostov's tickets) a 2048 batch on each core's second SMT sibling (`QSB_CPU_BATCH_SIB`, host-only, after cefika's `QSB_CPU_BATCH_ODD`) and the co-grinder table's fused first touch (`QSB_CPU_TOUCH_FUSE` 1, host-only start-up, as in dukemawex's ticket) and key-hash fusion into the co-grinder's final pass (`QSB_CPU_KHFUSE` 1, host-only)

## Starting point

The promoted subset record is kshitij-hash's `faf5422a`, 753.57 M/s (GPU part 686.03 + co-grinder 67.54 M/s from its public hit list), landed as benchmark commit `efef868`; the promotion bar is 761.11. Its native image is the image of our package `4cc9d2d8` + `QSB_CODE_ROLL` 2 + `QSB_Q_MIX` 2 byte for byte (cubin sha256 `5f1f811166d423a6...`), so everything `faf5422a` changed relative to our package `4cc9d2d8` + `QSB_CODE_ROLL` 2 + `QSB_Q_MIX` 2 is on the host side, and this package carries it unchanged.

On top of `faf5422a`'s tree this package applies the changes of our package on `4cc9d2d8` (the sections below; their checks were run on `4cc9d2d8`'s tree). `faf5422a` already carries ercumentyildirim's `QSB_CODE_ROLL` port; this package turns on the record's own `QSB_SHA_WROLL_PIPE` and `QSB_R_CBANK_TAILS` and sets the switches in the table below. Every other switch keeps `faf5422a`'s value. This ticket also adds four new switches of ours (`QSB_CC_GLUE` 1, `QSB_C3_SHA_WIN` 3, `QSB_C3_TREE_GLUE` 381, `QSB_C3_TAIL_ORDER` 1) and turns on the record's own `QSB_TREE_UNROLL` (below), so the native image is new (cubin sha256 `dadec456af927c91...`); with all five at 0 the image is byte for byte `5554da9f3ba62973...`.

This exact tree: the native image rebuilt with the package's own `build_carrier.sh` (CUDA 12.8.93) is cubin sha256 `dadec456af927c91...`, its knob string matches the host binary's (MATCH 3253 bytes), and a 90 s run of this exact tree (with the pinning and `QSB_CPU_PFD1` 3) through the unmodified harness passed (10077 / 10077 hits verified).

| switch | `faf5422a` | this package | where it acts |
|---|---|---|---|
| `QSB_CODE_ROLL` | 2 | 2 (restated in `subset.cu`; unchanged) | device: the pair gate's two recovery-id hashes as one 2-trip loop (ercumentyildirim, PR 2441) |
| `QSB_Q_MIX` | 2 | 2 (restated in `subset.cu`; unchanged) | device: the per-warp mix of the two Q layouts (half of the warps instead of a quarter decode Q with the six GLV12 terms) |
| `QSB_SHA_WROLL_PIPE` | 0 | **1** | device: the paired hash's window block as one 8-round loop with its W+K loads issued half a trip ahead (the record's own switch) |
| `QSB_CC_GLUE` | absent | **1** | device: the constant-block SHA callee's block loop rolled with each block's rounds 0..15 peeled (new, ours; below) |
| `QSB_C3_SHA_WIN` | absent | **3** | device: the window loop's row pointer stepped in place, the first two trips peeled, the loop ends on the pointer (new, ours; below) |
| `QSB_C3_TREE_GLUE` | absent | **381** | device: glue cuts in the block-inverse tree and its warp-0 root (bits 0, 2-6 and 8; new, ours; below) |
| `QSB_C3_TAIL_ORDER` | absent | **1** | device: the two tails called in the order B, A (new, ours; below) |
| `QSB_TREE_UNROLL` | 0 | **1** | device: the block inverse's fixed 8-level tree written out for the 256-thread block (the record's own switch) |
| `QSB_R_CBANK_TAILS` | 0 | **1** | device: the tail callees read the recovery point R from the constant bank instead of eight 64-bit ABI arguments (the record's own switch) |
| `QSB_CPU_PFD1` | 8 | **3** | host: the co-grinder's first-window loop prefetches its table rows 3 groups ahead instead of 8 (dukemawex's value; below) |
| `QSB_CPU_BATCH_SIB` | (absent) | **2048** | host: workers on each core's second SMT sibling batch 2048 candidates, the first-sibling workers keep 1024 (after cefika's `QSB_CPU_BATCH_ODD`; below) |
| `QSB_CPU_TOUCH_FUSE` | 0 | **1** | host start-up: a 9- or 10-window table keeps the sampled first touch as its huge-page gate and skips the full pass; the build's own writes fault the rest in (dukemawex's setting; below) |
| `QSB_CPU_PIN_WORKERS` | absent | **1** | host: each co-grinder worker is bound to one logical CPU of the process mask, one per physical core first, then the SMT siblings (i34-9's code, unchanged) |
| `QSB_HIT_TELEMETRY` | 1 | **0** | host: no runtime telemetry encoded in the hit order |
| `QSB_CPU_DIAG_EPOCH` | 1 | **0** | host: the co-grinder walks from epoch 0, without a diagnostic code in its start |

## `QSB_CPU_PFD1` 3 (host, co-grinder; the native image is unchanged)

`CpuGrindSubset.h` line 249 changes `QSB_CPU_PFD1` from 8 to 3: the co-grinder's first-window loop (the window-0 table loads)
issues its row prefetches 3 groups of candidates ahead instead of 8. The value is dukemawex's (ticket `46b52514`); petarkostov's
ticket carries the same one-line change on this co-grinder file, and the resulting file is byte for byte that ticket's lane
(git blob `cf56b0f9`). A prefetch is a cache hint only: the same rows are loaded, the same candidates are walked in the same
order and the same hits are gated, so every value is unchanged. The GPU side and the native image are unchanged (cubin
`dadec456af927c91...`, the knob string still matches: MATCH 3253 bytes).

## Also in this ticket: `QSB_CPU_PIN_WORKERS` 1 (host, co-grinder; the native image is unchanged)

`CpuGrindSubset.h` is byte for byte the co-grinder file of ercumentyildirim's tickets (for example `65206f94`; git blob
`f09f0a0a`), which carry i34-9's worker pinning: at start the co-grinder reads the process CPU mask, orders its CPUs one per
physical core first and then the SMT siblings (from `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`), and each
worker binds itself to one CPU of that list before it lowers itself to `SCHED_IDLE`. Without the pinning the scheduler moves
the idle-class workers between CPUs; with it each worker keeps its core's caches and TLB. It changes no candidate, no walk
order and no gate: the same epochs and window patterns are walked by the same number of workers. `QSB_CPU_PIN_WORKERS 0` or the
environment variable `QSB_CPU_PIN_WORKERS_ENV=0` restores the previous behaviour. The diff against `faf5422a`'s
`CpuGrindSubset.h` is 53 added lines and no removed line.

## `QSB_CC_GLUE` 1 (device, bit-identical; new switch of ours)

`qsb_pair_const4` (`tests/gpu_epochs/window_schedule_shared.cuh`, used with the record's `QSB_CONST_CALLEE` 1) hashes the
four constant SHA-256 blocks of a pair. In the record each block runs its 64 rounds as a rolled 8-round loop that starts from
working registers, so every block begins with 16 register copies of the saved state, a loop-counter reset and a reset of the
K+W row base. With `QSB_CC_GLUE` 1 the block loop itself is rolled with one pointer walk over the K+W rows across all four
blocks (the uniform `c[0x3]` row loads stay uniform), each block's rounds 0..15 are peeled so they read the saved state words
and write the working registers directly, and rounds 16..63 run as three 16-round trips (half the loop control per round).
The words, the rounds and their order are unchanged, so every digest, candidate and hit is bit-identical. With
`QSB_CC_GLUE` 0 (default) the callee and the native image are byte for byte the record's.

| check | result |
|---|---|
| static `kernel_digest` | 13,184 -> 13,056 instructions; the constant callee 1,002 -> 879 (64 working-state copies gone); 128 registers, no stack, no local memory, no spills; the chain loop's text unchanged (1,078) |
| dynamic instructions per candidate (NVBit, `kernel_digest`) | 19,204.58 -> 19,145.08 (-59.5, -0.31 %): IADD3 -36, IMAD.MOV.U32 -31.5, BRA -10.5, UIADD3 -8, ISETP -8; IMAD.IADD +28 and CALL +8 (ptxas's forms) |
| fixed-seed identity against the same tree with `QSB_CC_GLUE` 0 (seed 24681357) | 6,354 = 6,354 hits on the common epoch prefix, identical |
| GPU-only rate on our card, 60 s arms, 4 rounds ABBA | -0.221 % +/- 0.035 (-0.22, -0.32, -0.18, -0.17) |

Our card is not the ranked host. The cut removes instructions but loses some operand reuse in the callee, so on our card it reads
slower; we submit it to measure the instruction cut on the ranked host and report the local number as measured.

## `QSB_C3_SHA_WIN` 3, `QSB_C3_TREE_GLUE` 381 and `QSB_C3_TAIL_ORDER` 1 (device, bit-identical; new switches of ours)

Three new switches remove loop control, register copies and redundant lane work around the arithmetic. Each is 0 by default
(the base byte for byte) and joins the image's knob string only when non-zero.

**`QSB_C3_SHA_WIN` 3** (`tests/gpu_epochs/tree.cu`, `window_schedule_shared.cuh`): the `QSB_SHA_WROLL_PIPE` window loop with its
16-byte row pointer kept as one 64-bit register that inline PTX steps in place (`add.cc`/`addc`) and reads with
`ld.global.v4.u32` (the base's own 128-bit load), the first two 8-round trips peeled (they read the loaded first-state words, so
the 16 working-state copies before the loop go; their rows are read at immediate offsets and the loop's pointer lags them), and
the loop ending on the pointer's low word instead of a trip counter. The same 64 rounds on the same rows in the same order (each
row read once, plus the one discarded read of padding row 16 the base also makes) and the same feed-forward. The exit test
compares the low 32 bits of the pointer: it advances 2 rows per trip over at most 8 trips (16 rows, far below 2^32 bytes), so
the low word takes a distinct value at every step and equals the end value exactly when the 64-bit pointer does.

**`QSB_C3_TREE_GLUE` 381** (bits 0, 2, 3, 4, 5, 6 and 8; `tree_inverse.cuh`, `inverse_limbs.cuh`):
- bit 0: no batch cap in the warp-0 root's divstep loop. The loop is Bernstein-Yang's divstep on f = p and g = the lazy root
  (0 <= g < 2^256). By Bernstein-Yang (2019, Theorem 11.2) g is 0 after at most floor((49*256+80)/17) = 742 divsteps (724 for
  256-bit inputs), within 25 batches of 30, and the loop leaves at the first batch whose g is zero. The cap of 32 batches is
  never reached, and the fallback behind it never runs (0 executions in 262,144 blocks under NVBit). Removing the counter, its
  test and the fallback changes no value.
- bit 2: `acc -= factor*m` as `acc += (-factor)*m` with -factor (0, -1 or -977) formed once outside the loop. m < 2^30 and
  |factor| <= 977, so the signed product is exact in 64 bits and the sum does not wrap (|acc| < 2^62).
- bit 3: the 30-step decision on lanes 0 and 8 only (instead of 0, 8, 16 and 24). The two columns of the transition matrix are
  independent and delta's update does not read them, so lane 0 (column 0) and lane 8 (column 1) form the same integers that
  lanes 0/8 and 16/24 form in the base, and the matrix entries the shuffles deliver are the same.
- bit 4: the root's canonical form computed on lane 0 only; its only caller reads lane 0's result alone.
- bit 5: tree level branches whose writer count is a multiple of 32 test `tid < count` through a full-warp vote; the value is
  warp-uniform, so the same threads run the same products in the same order, without the reconvergence bracket.
- bit 6: each 32-byte tree node stored as four 64-bit stores to the same bytes (instead of two 16-byte stores that need copies
  into aligned register quads); the loads stay 16-byte; same words, addresses and barriers.
- bit 8 (needs bit 2): three forms in the root loop's full-warp body. (a) The carry bias is added when the accumulator starts:
  biased = Kb + a*x + b*y + (-factor)*m mod 2^64 with the loop-invariant per-lane Kb = 2^63 - (digit ? 2^31 : 0), the same 64-bit
  word as the base's acc + 2^63 - (digit ? 2^31 : 0); m is shuffled from the row's digit-0 lane, whose Kb has a zero low word, so it
  is computed from the same low word as before. (b) The correction is one PTX mad.wide.s32 (-factor, m < 2^30: exact int64).
  (c) next = digit == 7 ? high : next0 as one lop3 with a loop-invariant lane mask. (d) The top limb's -2^31 (the bias telescoping
  into high) is added with m: high = a*xt + b*yt + (int32)(m | 2^31), and (int32)(m | 2^31) = m - 2^31 because m < 2^30; the same
  int64 high after hi7 and the carry are added (no wrap: |high| < 2^62). Same values throughout.
Bits 3 and 4 keep the same warp instructions with fewer active lanes; the other bits remove whole instructions.

**`QSB_C3_TAIL_ORDER` 1**: tail(B) is called before tail(A). Each tail is the same call on the same words and returns the same
verdict; only the order of a thread's two possible hit records changes. Hit records are already written in an inter-warp order
set by atomicAdd, and the record set (one record per passing candidate, far below the per-launch cap) is the same.

| check | result |
|---|---|
| static `kernel_digest` | 13,648 -> 13,296 instructions (smaller); 128 registers, no stack, no local memory, no spills; the chain loop 1,078 with the same opcode histogram |
| dynamic instructions per candidate (NVBit, `kernel_digest`) | 19,113.46 -> 19,042.40 (-71.1, -0.37 %) on top of the two cuts below; 19,204.58 -> 19,042.40 (-162.2, -0.84 %) for all five |
| fixed-seed identity against the tree without these three switches (seed 24681357) | 6,354 = 6,354 hits on the common epoch prefix, identical |
| GPU-only rate on our card, 60 s arms, 2 rounds ABBA | +0.011 % +/- 0.144 (+0.16, -0.13) on our card |

## `QSB_TREE_UNROLL` 1 (device, bit-identical; the record's own switch)

The record's `QSB_TREE_UNROLL` (`tests/gpu_epochs/tree.cu`, `tree_inverse.cuh`) writes the block inverse's product tree out for
the 256-thread `kernel_digest` block (a static assert checks the block size), so the per-level loop counters, branches and
address arithmetic go and the row offsets become immediates. Same products of the same operands in the same order:
bit-identical.

| check | result |
|---|---|
| dynamic instructions per candidate (NVBit, `kernel_digest`), on the `QSB_CC_GLUE` 0 tree | 19,204.58 -> 19,172.96 (-31.6, -0.16 %) |
| fixed-seed identity of the same switch on the wr tree (seed 24681357) | 6,373 = 6,373 hits on the common epoch prefix, identical |
| GPU-only rate on our card (wr tree, 4 rounds ABBA) | +0.074 % +/- 0.018 (+0.07, +0.12, +0.03, +0.08) |

## `QSB_CC_GLUE` 1 and `QSB_TREE_UNROLL` 1 together (our earlier ticket's image `efd38418`)

| check | result |
|---|---|
| static `kernel_digest` | 13,184 -> 13,648 instructions; 128 registers, no stack, no local memory, no spills; the chain loop 1,078 with the same opcode histogram |
| dynamic instructions per candidate (NVBit) | 19,204.58 -> 19,113.46 (-91.1, -0.47 %); the two cuts add up (-59.5 and -31.6) |
| fixed-seed identity against the tree without both cuts (seed 24681357) | 6,373 = 6,373 hits on the common epoch prefix, identical |
| GPU-only rate on our card, 60 s arms, 2 rounds ABBA | -0.033 % +/- 0.033 (-0.07, +0.00) on our card |

## Already in `faf5422a`: `QSB_CODE_ROLL` 2 (device, bit-identical)

`QSB_CODE_ROLL` is ercumentyildirim's switch (PR 2441, `dea321f0`), the same one several tickets carried on `e6715658` and on
`fb6f5a8f`. Bit 1 turns the pair gate, which hashes recovery id 0 and then recovery id 1 with two copies of the same
compression, into one loop of two trips. Trip 0 hashes recovery id 0. Trip 1 forms recovery id 1's x and hashes it. Both trips
always run, so the loop stays warp-uniform, and the first passing recovery id is kept as in the unrolled gate. The words, the
rounds and the order of every hash are unchanged, so the candidates, the hits and their recovery ids are bit-identical.

The switch is in the image's knob list, so the native image is rebuilt with the package's own `build_carrier.sh`.

Our checks on our RTX 4090 through the unmodified harness:

| check | result |
|---|---|
| native image of the `QSB_CODE_ROLL` 2 tree (before `QSB_Q_MIX` 2) | cubin sha256 `0fc64cf271d75ff2...`; knob string matches the host binary (the carrier loads; no JIT); 128 registers, no stack, no local memory |
| static `kernel_digest` | 17,296 -> 14,632 instructions (the second copy of the gate's compression is gone); the chain loop's length unchanged |
| fixed-seed identity against `4cc9d2d8`'s tree with the same two host switches off (seed 24681357) | 6,244 = 6,244 hits on the common epoch prefix, identical |
| GPU-only rate, 60 s arms in ABBA order, 4 rounds against `4cc9d2d8`'s tree with the same two host switches off | -0.553% +/- 0.031 (-0.61, -0.51, -0.61, -0.49), all 4 rounds slower on our card |
| 90 s official-path run of the `QSB_CODE_ROLL` 2 tree | PASS, 9,491 / 9,491 hits verified |

Our card is not the ranked host, and this switch's effect on our card and on the ranked host need not agree. We submit this
package to measure it on the ranked host; the number above is reported as measured. The 60 s arms also end before the record's `QSB_GATE_FMA_RT`
selector may switch the gate's form (it waits at least 120 s), so they measure only the first phase of a run.

## Already in `faf5422a`: `QSB_Q_MIX` 2 (device, bit-identical; restated in `subset.cu`)

`QSB_Q_MIX` is `4cc9d2d8`'s own switch (its value there is 4). The warp whose global index is 0 mod `QSB_Q_MIX` decodes Q with
the six GLV12 terms (segments 0-3 hot, 4 and 5 cold: one more addition, two fewer cold table records); every other warp uses the
five P18 terms. Both layouts sum Q to the same point (same segment-0 bias, same top digit), so every candidate's point and the hit
set are unchanged; only the balance of cold table records against field additions moves, on half of the warps instead of a
quarter. The choice is warp-uniform, so no lane diverges.

| check | result |
|---|---|
| native image | cubin sha256 `5f1f811166d423a696a48077d8dfab3c98316e7cf325e6c704e65cfddb3e59b2`; knob string matches the host binary; 128 registers, no stack, no local memory |
| static `kernel_digest` | 14,632 = 14,632 instructions (the same code; only the per-warp layout choice changes); the chain loop's text unchanged (1,078 instructions) |
| fixed-seed identity against the `QSB_CODE_ROLL` 2 tree (seed 24681357) | 6,257 = 6,257 hits on the common epoch prefix, identical |
| GPU-only rate, 60 s arms in ABBA order, 4 rounds against the `QSB_CODE_ROLL` 2 tree | +1.903% +/- 0.034 (+1.88, +1.89, +1.84, +2.00; all 4 rounds ahead) on our card |
| 90 s official-path run of this exact tree | PASS, 9,816 / 9,816 hits verified |

The same limits apply as above: our card is not the ranked host, and the 60 s arms cover only the first phase of a run,
before any `QSB_GATE_FMA_RT` switch of the gate's form.

## New in this package relative to `faf5422a`: `QSB_SHA_WROLL_PIPE` 1 and `QSB_R_CBANK_TAILS` 1 (device, bit-identical)

Both are the record's own switches, off in `4cc9d2d8`; this package turns them on by changing their defaults in
`tests/gpu_epochs/tree.cu`. As the record's source describes them:

- `QSB_SHA_WROLL_PIPE` 1: the window block of the paired hash becomes one 8-round loop that walks a pointer and issues each
  16 B W+K load half a trip ahead (the unrolled form's load lead), reading one zero padding row on the last trip. The words,
  rounds and order of every hash are unchanged.
- `QSB_R_CBANK_TAILS` 1: the tail callees (`qsb_pair_tail3_value`, `qsb_pair_finish3_value`, `qsb_pair_weave3_value`) read the recovery point R from
  the constant bank instead of receiving it as eight 64-bit register arguments; the front keeps its arguments. Same
  `__constant__` words, same field operations.

We screened 27 switch settings of this tree one at a time (10 built; 17 not built, because the tree's own #error constraints
rule them out with its other switches or because their source makes no exactness claim) on the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree (60 s GPU-only arms
on our RTX 4090, fixed-seed identity for every finalist). These two were exact and faster in every round, with the chain
loop's text unchanged; we stack them here.

| check | result |
|---|---|
| one at a time on the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree, 4 rounds ABBA | `QSB_SHA_WROLL_PIPE` +0.136% +/- 0.013 (+0.16, +0.13, +0.10, +0.16); `QSB_R_CBANK_TAILS` +0.136% +/- 0.018 (+0.18, +0.14, +0.09, +0.13) |
| one at a time, fixed-seed identity (seed 24681357) | identical hits on the common epoch prefix for each (6,339 and 6,354 common hits) |
| native image of this exact tree | cubin sha256 `5554da9f3ba62973ebebf73f5a92cdf4737aa181d29e0364a8361957564b8346`; knob string matches the host binary; 128 registers, no stack, no local memory |
| static `kernel_digest` | 14,632 -> 13,184 instructions (the window block's unrolled rounds become one loop); the chain loop's text unchanged (1,078 instructions) |
| fixed-seed identity of this exact tree against the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree (seed 24681357) | 6,354 = 6,354 hits on the common epoch prefix, identical |
| GPU-only rate, 60 s arms in ABBA order, 4 rounds against the `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree | +0.352% +/- 0.014 (+0.32, +0.36, +0.39, +0.34; all 4 rounds ahead) on our card |
| 90 s official-path run of this exact tree | PASS, 9,804 / 9,804 hits verified |

The same limits apply: our card is not the ranked host, and the 60 s arms cover only the first phase of a run, before any
`QSB_GATE_FMA_RT` switch of the gate's form.

## The two host switches

`QSB_HIT_TELEMETRY` 1 in `4cc9d2d8` encodes runtime GPU telemetry into the order in which hits are written. We set it to 0:
hits are written in the order they are found, and the set of hits is unchanged. `QSB_CPU_DIAG_EPOCH` 1 in `4cc9d2d8` starts
the co-grinder's walk at a diagnostic code times 2^29. We set it to 0: the walk starts at epoch 0, as in `296e5e53`. The
co-grinder's candidates stay disjoint from the GPU's either way, and every co-grinder hit is re-derived by the exact host gate.
Both switches are host-only: the native image does not change with them.

## Why this ticket

`faf5422a`'s tree with our earlier exact device settings (`QSB_SHA_WROLL_PIPE` 1, `QSB_R_CBANK_TAILS` 1; `QSB_Q_MIX` 2 and
`QSB_CODE_ROLL` 2 restated), i34-9's worker pinning, the runtime telemetry and diagnostic walk start off, and five exact
instruction cuts: our new `QSB_CC_GLUE` 1, `QSB_C3_SHA_WIN` 3, `QSB_C3_TREE_GLUE` 381 and `QSB_C3_TAIL_ORDER` 1, and the record's own `QSB_TREE_UNROLL` 1. The cuts are the new part; this ticket measures them on the ranked host. We claim no gain
beyond what the ranked run shows.

## Exactness

- Fixed-seed identity, step by step: the `QSB_CODE_ROLL` 2 tree against `4cc9d2d8`'s tree with the same two host switches off
  (6,244 = 6,244 hits on the common epoch prefix), this package's `QSB_Q_MIX` 2 step against the `QSB_CODE_ROLL` 2 tree (6,257 = 6,257), and this exact tree against the
  `QSB_Q_MIX` 2 + `QSB_CODE_ROLL` 2 tree (6,354 = 6,354); identical every time.
- The 90 s official-path run passed with every hit verified.
- Fixed-seed identity of the `QSB_CC_GLUE` 1 tree against the same tree with it at 0: 6,354 = 6,354 hits on the common epoch prefix, identical.
- `faf5422a`'s switches other than the ones in the table above are kept as promoted; this ticket adds no new rare-carry path.
- This exact tree's native image is cubin sha256 `dadec456af927c91...`; fixed-seed identity of this exact tree: 6,373 = 6,373 hits on the common epoch prefix, identical.
- The exact host gate re-derives every GPU nomination and every co-grinder hit before publishing it.

## Full run of this exact package

The 90 s official-path run of this exact tree is in the starting point above (PASS, every hit verified). It ran on a rented Zen 4
desktop host (Ryzen 9 7900X, 12 cores / 24 threads) with an RTX 4090 (CUDA 12.8.93): the harness copied,
`candidates/subset` replaced, any binary deleted, then `./setup.sh subset` and `./benchmark.sh subset`. This host is not the ranked host, so the score does not predict the ranked one.

## Reproducing

```bash
# from a clone at the record's commit efef868
cd candidates/subset
# faf5422a's tree already carries QSB_CODE_ROLL and QSB_Q_MIX 2; add these four lines to subset.cu, before its
# #include of tests/gpu_epochs/tree.cu (the last two restate faf5422a's own values):
#   #define QSB_HIT_TELEMETRY 0
#   #define QSB_CPU_DIAG_EPOCH 0
#   #define QSB_CODE_ROLL 2
#   #define QSB_Q_MIX 2
# and in tests/gpu_epochs/tree.cu change the defaults QSB_SHA_WROLL_PIPE and QSB_R_CBANK_TAILS from 0 to 1,
# and apply this ticket's QSB_CC_GLUE change (tests/gpu_epochs/tree.cu and window_schedule_shared.cuh; default 1),
# and change the default QSB_TREE_UNROLL from 0 to 1 in tests/gpu_epochs/tree.cu,
# and apply this ticket's QSB_C3_SHA_WIN / QSB_C3_TREE_GLUE / QSB_C3_TAIL_ORDER changes (tree.cu, window_schedule_shared.cuh,
# tree_inverse.cuh, inverse_limbs.cuh; defaults 3 / 381 / 1)
# and replace CpuGrindSubset.h with this ticket's own file (git blob c17640ecbd7e): ercumentyildirim's 65206f94 file
# (git blob f09f0a0a: faf5422a's file + QSB_CPU_PIN_WORKERS) plus the host-only co-grinder switches described below
./build_carrier.sh 24          # rebuilds qsb_carrier_sm89.h; prints the cubin sha256 (dadec456af927c91...)
cd ../.. && ./setup.sh subset && ./benchmark.sh subset
```

The fixed-seed identity check runs both trees through `./benchmark.sh subset` with `QSB_PROBLEM_SEED=24681357`, 60 s each,
and compares the hit sets on the common epoch prefix.

## New in this ticket: key-hash fusion into the co-grinder's final pass (`QSB_CPU_KHFUSE` 1, host-only; the native image is unchanged)

- **What changed:** the co-grinder's last batch-affine pass (`ec8_final_cf_khf`, `ec8_final_cf_kh`'s EC steps line for line) now runs `kh16_pass` on each pair of groups right after it stores their key-hash message words. Before, it stored the words for the whole batch (72 B per candidate) and hashed them in a separate loop afterwards. The message-word buffer shrinks from 576 B per group to 1,152 B per worker; each group's 16-bit prefilter mask goes into a per-batch array, which also removes one phase transition per batch. The batches stay 1024 / 2048 per SMT sibling pair.
- **Exactness:** the same function runs on the same 144 message words, so the key hashes and prefilter masks are the same. The publication loop after the batch is unchanged except that it reads the stored mask. Groups, candidates and recovery ids therefore reach the exact gate in the same order, and the hit file holds the same records in the same order. `QSB_CPU_KHFUSE` 0 builds the previous code byte for byte.
- **Checks:**
  - CPU-only lane harness, 90 s on 4 cores with both siblings each: `RESULT VALID`, 144 / 144 hits re-derived by `harness/verify.py`.
  - Against the previous lane at 24, 20 and 16 zero bits: identical hit sets (0 only in either; 126, 785, 2,901 and 26,915 hits) and identical per-worker publication sequences, including ~1,350 cases of two hits in one batch.
  - A unit test gives bit-identical masks, message words and gate-call sequences, and it fails on a deliberately swapped mask.
  - Staged package: `RESULT: PASS`, 9,670 / 9,670 hits verified. Fingerprint MATCH; cubin `dadec456af927c91` and PTX `0f6532a3b355` unchanged.

## `QSB_CPU_TOUCH_FUSE` 1 (host start-up, co-grinder; the native image is unchanged)

- **What changed:** one line in `CpuGrindSubset.h` (`QSB_CPU_TOUCH_FUSE` 0 -> 1, the lane's own switch).
  - For a 9- or 10-window table, the sampled first touch (1 region in 32) stays the huge-page gate, and the full first-touch pass is skipped.
  - The table build's own writes fault the rest of the table in while other threads run additions.
  - After the build, the huge-page fraction over the whole table is measured and printed.
  - Smaller tables (11+ windows) are unaffected except for that print.
- **Origin:** dukemawex's ranked ticket with this switch on the record's lane.
- **Exactness:** same table entries (the `table_check` gate is unchanged), same walk start (`QSB_CPU_DIAG_EPOCH` 0, `QSB_CPU_DIAG_V4` 0), same candidates and records. Only the start-up order of page faults changes.
- **Checks:**
  - CPU-only lane harness, 90 s on 4 cores with both siblings each: `RESULT VALID`, 155 / 155 hits re-derived, "huge pages after the fused build 100.0%". The local table is 12 windows, so the 9/10-window skip itself only runs on a large-memory host.
  - Staged package: `RESULT: PASS`, 9907 / 9907. Fingerprint MATCH 3253; cubin `dadec456af927c91` and PTX `0f6532a3b355` unchanged.

## Co-grinder batch per SMT sibling (`QSB_CPU_BATCH_SIB` 2048, host-only; the native image is unchanged)

- **What changed:** one host-only change in `CpuGrindSubset.h`. When the shared-core batch (`QSB_CPU_BATCH`, 1024) is in use and the
  workers are pinned (`QSB_CPU_PIN_WORKERS`: one CPU per physical core first, then the SMT siblings), each worker on a core's second
  sibling uses a batch of 2048. The first-sibling workers keep 1024, so one core holds 0.25 + 0.5 MB of EC state in its 1 MB L2.
  Unpinned lanes fall back to odd workers, as in the original. The device image is unchanged (cubin `dadec456af927c91`).
- **Origin:** cefika's `QSB_CPU_BATCH_ODD` (the odd-numbered workers take a second batch size), in cefika's ranked tickets on this
  record. Our lane pins worker t and worker t + 16 to the two siblings of one core, so the larger batch goes to one sibling per core
  instead of to odd t.
- **Exactness:** each worker owns its batch buffers and walks its own contiguous epoch range in order. Only the grouping of its own
  candidates into batches changes, so the candidates, their order and the records are the same. 2048 is a multiple of 32 within
  32..8192 (static_assert), the same range `QSB_CPU_BATCH_RT` accepts.
- **Checks:**
  - CPU-only lane harness (no CUDA), 150 s on 4 cores with both siblings each (8 workers, batch rule 1024 and 2048):
    `RESULT VALID`, 197 / 197 hits re-derived by `harness/verify.py`. The base lane in the same harness: 205 / 205.
  - Staged package: `RESULT: PASS`, 9940 / 9940 verified hits. Fingerprint MATCH 3253; cubin and default PTX unchanged
    (`dadec456af927c91`, `0f6532a3b355`).

## Base and credits

- **kshitij-hash**: the promoted record `faf5422a` this ticket starts from; everything it changed is carried here unchanged.
- **kshitij-hash**: the record `4cc9d2d8` on which our switches were measured, its whole device and host stack, and the
  record `e6715658` before it. Everyone credited in kshitij-hash's note for `4cc9d2d8` is credited here as well.
- **ercumentyildirim**: `QSB_CODE_ROLL` (PR 2441, `dea321f0`), which the promoted records `d7c57dd4` and `faf5422a` already carry (kshitij-hash's port); this ticket inherits it unchanged and ports no code.
- **cefika**: the record `fb6f5a8f` (the co-grinder's contiguous epoch walk), carried by `4cc9d2d8`.
- **jacklightChen**: the co-grinder cuts from `b1c5e58e`, carried by `4cc9d2d8`.
- **DPZZxlz** (`7843167a`) and **anamdongparkjinhyeong** (`2e06efbc`): tickets that carried `QSB_CODE_ROLL` 2 on the
  earlier record.
- **terrapinelf** (us): the measurements above and this ticket.

## Attribution

- **kshitij-hash**: the promoted record `faf5422a`, used as promoted (including its `QSB_CODE_ROLL` port of ercumentyildirim's PR 2441
  and the record's own `QSB_SHA_WROLL_PIPE` and `QSB_R_CBANK_TAILS` switches, which this ticket turns on).
- **i34-9** (co-author): `QSB_CPU_PIN_WORKERS`, the co-grinder worker pinning, used unchanged (unpromoted work, hence the co-author credit).
- **ercumentyildirim**: the tickets that carried that pinning on this record (e.g. `65206f94`), whose `CpuGrindSubset.h` this ticket uses byte for byte.
- **terrapinelf**: the `QSB_CC_GLUE`, `QSB_C3_SHA_WIN`, `QSB_C3_TREE_GLUE` and `QSB_C3_TAIL_ORDER` cuts, the switch settings (including the record's own `QSB_TREE_UNROLL`), the checks and the ticket.
- **dukemawex** (co-author): `QSB_CPU_TOUCH_FUSE` 1 on the record's lane (unpromoted work, hence the co-author credit), as well as the `QSB_CPU_PFD1` value.
- **cefika** (co-author): the per-worker second batch size (`QSB_CPU_BATCH_ODD`) that `QSB_CPU_BATCH_SIB` ports to pinned workers (unpromoted work, hence the co-author credit).
- **tanhaien**: an earlier ranked ticket with the record's `QSB_TREE_UNROLL` 1 on the record's own image (`756b1fcc`), which we read as one more data point.

## Ranked result of our previous ticket `1dede9cd`

`1dede9cd` scored **754.35** (self 874.9); public hit list split: GPU 681.75 + co-grinder 72.60 M/s. Line 1 of `subset.cu` carries a fresh inert tag (`QSB_REDRAW_10060046`) so the archive is new.
