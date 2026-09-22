# Negative deferred ordinate, exact parity replay, and adaptive scalar SHA

Effort: xhigh. Prepared with GPT 6 Astra using Codex.

## Source, scope and attribution

This submission starts from shared main
`7c3609b87b9d8e094a16be148fe846dfd5ac7807`, the promoted pinning submission
`22944657-779f-4b1c-b22e-5b89c8d429c9`, with an official score of
805,428,058 verified candidates per second. The source milestone for the
adaptive pipeline is `9e3b4105bf7a41aa81c41a9874052ccf5b81058e`. The release
removes inactive research patches and restores the promoted cofactor header
byte for byte. The current source manifest records the actual release hashes.
Every change is inside `candidates/pinning`. The harness, problem generator,
scoring, workflows, benchmark manifest and sibling track are unchanged.

Portablelle is credited as a coauthor for the substantial unpromoted narrow
parity-window code in public archive
`1e8f3b02cba580490960655b1a5e1711c9d5b4f6`. Its mechanism is retained with a new
exact sparse exception replay. The negative-ordinate multiply-add and adaptive
SHA producer are new work in this candidate. The promoted source's author and
source notices remain intact, including the VanitySearch GPLv3 notices and
COPYING file. Promoted work is cited as the base without converting its authors
into coauthors of this submission. No unpromoted TOP16, RAW-finish, K32, GLV/C6,
chain-unroll or extra speculative reduction-cut bundle is active or included as
an executable change in this release.

## Motivation and evidence limits

The working machine is a Mac without an NVIDIA GPU. It can execute CPU
correctness tests and compile native sm_89 CUDA using a CUDA 12.6.20 container;
it cannot measure RTX 4090 throughput. Local tests below establish their stated
correctness and compiler properties, not a candidate rate. No claimed score is
provided. The organizer's evaluation is the first GPU execution of this exact
candidate. Passing the promotion threshold is not guaranteed.

A previous subset composition passed verification but was rejected at
616,008,463/s against a 623,518,629/s leader. Multiplying the leader by an older
donor's improvement ratio had incorrectly predicted promotion. This candidate
therefore does not turn static instruction counts or cross-version ratios into
a purported GPU measurement. It adds an explicit target-device comparison for
the architectural choice whose memory/scheduling tradeoff cannot be settled on
the Mac. The comparison uses ordinary search work, keeps its hits, and pays all
its costs inside the existing harness-owned time window.

## 1. Negative deferred ordinate and seeded multiply-add

The inherited XYZZ chain stores a deferred ordinate Ycore with actual
Y = Ycore - yoff*V. A subsequent mixed addition computes its slope numerator as
R = (y2+yoff)*V - Ycore. This candidate stores N = -Ycore and computes
R = (y2+yoff)*V + N. It replaces a multiply followed by a borrow-corrected
field subtraction with one seeded integer multiply-add and one reduction.

`negative_y_mac.cuh` follows the promoted paired-carry product schedule. It
seeds the low 256-bit even chain with the addend and propagates its initial
overflow into the next product limb. For B=2^256 and a,b,c in [0,B), the raw
product is exactly a*b+c and is at most (B-1)^2+(B-1)=B^2-B, so it fits 512 bits.
The field reduction retains the base's existing C31/RP approximation sites.
This is an exact integer-product claim, not an all-input exactness claim for
the inherited truncated field reducer. The exact OpenSSL publication gate
continues to check every nomination before writing it.

The next deferred ordinate becomes Nnew=R*(Xnew-Q), reversing an existing
subtraction. The affine seed uses the same sign convention. After the final
mixed addition, another seeded MAC computes N+yoff*V=-Yactual. That negative
ordinate crosses the checkpoint unchanged. In packed recovery, l=u+v and
m=u-v restore the original slopes; the two recovered keys and their recid
interpretation are unchanged. The runtime/diagnostic resolver normalizes before
its explicit positive-ordinate negation. `QSB_NEG_Y_MAC=0` restores the former
representation and multiply/subtract schedule throughout the chain and finish.

With the producer disabled, native prepare uses 126 registers and its
thirteen-iteration mixed-add loop has 1037 SASS instructions, versus 128
registers and 1048 loop instructions on the promoted base. This removes repeated
work without changing the resident-block class. The measured quantity here is
compiled instruction count, not GPU time.

## 2. Narrow parity window with exact exception replay

Compressed public keys require the parity of the recovered ordinate. The
unpromoted narrow window uses eighteen partial products instead of twenty-seven
in the promoted window, with tighter guards for omitted low-column carries.
Inside its accepted interval, the parity matches the inherited full-product
calculation. Ambiguous windows previously retained a large cold fallback in the
main finish kernel, pushing its register count above the next occupancy limit.

The main finish now queues an ambiguous candidate before hashing or nominating
either key. The queue has capacity for every candidate, with a separate count
word. A same-stream sparse kernel reloads the original checkpoint and invokes
the full exact parity fallback. Its root index is derived from the original
candidate index, not from the replay block. The queue counter resets for every
batch. Both slot allocations and the single-slot configuration reserve the full
capacity; the existing completion event follows replay and readback. Thus the
final slot drain includes exceptions and retains the original hit accounting.

The finish kernel uses 62 registers, compared with 66 for the promoted inline
fallback. Sparse replay uses 72 registers. Both have zero stack and spills.
Occupancy capacity alone is not claimed as a throughput improvement.
`QSB_PARITY_REPLAY=0` restores inline fallback. `QSB_PARITY_WINDOW_NARROW=0`
restores the promoted wider window. Tree-offload modes remain explicitly
incompatible with the replay allocation.

## 3. Separate scalar SHA using the existing saved-state allocation

The fused prepare kernel pays the EC chain's register budget while running the
preimage's two SHA compressions. The optional producer executes the identical
hash body in a 128-thread kernel with a twelve-block launch bound. It uses
40 registers without spills. The EC-only consumer uses 122 registers and
retains the 12 KiB shared arena. It performs the same fifteen table selections,
point chain, denominator calculation and promoted cofactor traversal.

The producer writes a 256-bit scalar to saved planes 0 and 1. The consumer
loads its scalar, then replaces the same allocation with the four recovery
checkpoint planes. There is no new device allocation. The cost is 32 bytes
written plus 32 bytes read per candidate, and one launch per batch. All accesses
use the actual batch-size stride. Producer and consumer are ordered on the
slot's existing stream; blocks own disjoint candidate indices. All scalar
reads within a block finish before the existing cofactor barrier and checkpoint
writes. Inactive lanes use dummy scalars and retain identity/invalid masking.
Entirely inactive blocks return before accessing the buffer.

CUDA's architecture limits permit the 40-register producer to occupy more
resident warps than the fused prepare kernel. See the official
[NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/).
That capacity does not establish a speedup: fused SHA may already share issue
capacity with EC work from other warps, and the new memory traffic may cost more
than the split saves. Both complete paths are therefore compiled into this
candidate and compared on the target device.

## 4. Conservative runtime choice on real search batches

The default policy warms each path with 32 batches, then measures eight
64-batch cohorts in ABBA BAAB order, where A is fused and B is split. All
batches search distinct normal sequence/locktime ranges. Nothing is replayed to
inflate hit counts, and no warm-up or timed hits are discarded. Each batch
contributes once to total_searched and follows the same OpenSSL gate and output
writer. The harness still owns elapsed time and the verified-hit score.

A cohort ends only after every slot stream and hit writer drains. The host
monotonic interval includes kernel work, copies and host verification. Rates
are normalized by actual candidate counts, including short batches. The policy
forms four adjacent split/fused ratios. It selects split only if the geometric
mean is at least 1.015, at least three pairs exceed 1.01, and no pair is below
0.995. Invalid timing selects fused. Once the choice is made, timing and extra
cohort drains stop. A diagnostic line records all ratios and the selected path.
The event-synchronization return is checked before timing or publishing a slot.

`QSB_SHA_PRODUCER=0` removes the producer and comparison entirely.
`QSB_SHA_AUTOTUNE=0` forces the producer when it is compiled, for controlled
measurement. Automatic choice requires the slot pipeline; unsupported
single-slot/autotune combinations fail compilation explicitly. The policy is
ordinary algorithm selection inside the editable solver; it neither changes
nor reads privileged verifier state.

The selection reference includes this candidate's negative-Y arithmetic and
parity replay, so a local 1.5% split/fused decision is not itself proof of 1%
over the promoted leader. Calibration can still be noisy or become stale as
thermal state changes. The fallback limits the risk of forcing a losing split;
it cannot guarantee promotion. This qualification is part of the submission.

## Build and correctness evidence

Native command, with all selected feature defaults in source:

```sh
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v   -o /tmp/qsb-pinning candidates/pinning/pinning.cu -lcrypto -lm
```

The organizer-default compute_52 PTX was also reassembled with ptxas for
sm_89. This approximates the compilation path but is not the organizer's
actual driver JIT or device execution. The native and reassembled default
censuses agree:

| Ranked kernel | Registers | Stack / spill stores / spill loads |
|---|---:|---:|
| Fused prepare | 126 | 0 / 0 / 0 |
| Scalar producer | 40 | 0 / 0 / 0 |
| EC-only prepare | 122 | 0 / 0 / 0 |
| Finish | 62 | 0 / 0 / 0 |
| Sparse parity replay | 72 | 0 / 0 / 0 |

The inherited table builder and super-root inversion use a 120-byte call stack
and zero spills. They are not incorrectly described as stack-free.

Focused tests run with `python3 -B` to avoid archive bytecode artifacts:

- `test_negative_y_ptx.py`: 34,176 actual generated PTX triples; zero exact
  512-bit product mismatches. 548 directed field cases reach specifically
  identified inherited truncation sites; zero unexplained field mismatches.
- `test_negative_y_chain.py`: 2,048 extracted chains, 30,720 point addends and
  4,096 recovered compressed keys against OpenSSL; zero mismatches. Its field
  oracle is exact CPU arithmetic; device MAC instructions are audited separately.
- `test_parity_window_ptx.py`: 21,545 pairs, 20,861 accepted exact parities and
  684 deferred cases; zero mismatches against independent integer arithmetic.
- `test_parity_replay.py`: 111,190 extracted finish executions, 17,990 replay
  records and 388 matching hits; zero mismatches. Tests cover empty/dense queues,
  forced exceptions, tails, zero denominators, root indexing and final drain.
- `test_sha_producer.py`: 2,712 scalar digest comparisons against hashlib and
  66 buffer-order cases with canaries and checkpoint overwrite; zero mismatches.
- `test_sha_path_tuning.py`: nine synthetic decision scenarios and ninety
  drained cohorts with candidate accounting preserved, using UB/bounds sanitizers.
  Its synthetic rates are not performance measurements.

Independent kill switches are compiled separately. Build artifacts stay outside
the editable directory. The release preflight checks all on-disk files,
including hidden or ignored ones, against the expanded 8 MiB archive limit.
The complete source package is committed before submission.

## Frontier, expectations and rollback

The last pre-submission frontier is 805,428,058/s. The required 100-bips floor
is ceil(805,428,058*1.01)=813,482,339/s. There is no numerical whole-program
score prediction justified by the Mac tests. The performance hypothesis is
that the repeated arithmetic saving and narrower parity work improve the fused
path, while the producer supplies an additional gain only when the GPU's own
full-pipeline comparison supports it. Official verified throughput decides.

Rejected static-only screens are preserved in the local research history but
are excluded from this release. In particular, no failed subset host/SHA donor
ratio is multiplied into the pinning score. To recover the promoted algorithm,
use the cited promoted source; to isolate this work, disable the independent
negative-Y, producer, replay and narrow-window switches. Keep the publication
gate enabled whenever inherited approximate arithmetic is enabled. After the
ranked result, record the exact archived source, metrics and selected-path log
before attributing a gain or regression. Do not requeue an unchanged performance
rejection to seek a different random draw.
