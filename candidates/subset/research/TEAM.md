# Research team and implementation queue

Updated 2026-09-16. Reviewed research, not a claim of GPU performance.
Focus: large improvements to the subset track.

## Collaboration

- **Grok 4.6 / Grok CLI** (usage metadata `grok-4.6-build`): elliptic-curve
  algorithms, fixed-base multiplication and review of scheduling proposals.
- **Gemini 3.8 Flash (High) / Antigravity CLI**: hashing, epoch geometry,
  register pressure, scheduling and kernel decomposition.
- **GPT 6 Astra, xhigh / Codex**: source and primary-reference checks,
  independent cost models, corrections, selection, implementation and validation.

Both models completed the first reviewed round through separate CLI sessions
using the user's existing sign-ins. The user explicitly approved sharing relevant
project context with each. External-model research assignments are read-only:
no file edits, submissions, account changes or further agents. The user later
authorized recurring candidate-local research while preserving our pending
submission; the scope update below supersedes the original results-only limit.

Give researchers the exact source identity and distinct questions. Require the
mechanism, removed and added work, assumptions, correctness hazards, a small
falsification experiment and primary sources. Have another researcher challenge
the proposal. Save reviewed conclusions here, not raw reasoning, account output
or complete transcripts. Successful process exit with an empty or cancelled
response is not a completed report. Grok's first run required a bounded follow-up;
Gemini's first command was denied, so source context was supplied with approval.

## Earlier composite evidence (historical)

Production/audit fingerprint:
`d0ccab7c66830429f8bc39dc619ad4b369883d0ea9c7929b64a569592c65512e`.
The prior submitted fingerprint was
`5a67fc768c6d5740d765343b0084e3b8269b042e6d4c7437f818aa593449ca8e`.

The latest frontier refresh reports upstream epoch port `873ed724` **promoted**
at `cfc0d9c`, scoring **433346795 verified candidates/s**. This replaces the
129574439 comparison frontier used when our submission was sent. Our
`c9a85a87-e2b8-4ab8-8d4f-c79d96546d03` remains validating. The earlier reported
442824991 development score is historical, not this official result.

Pending `dcfcc85` reports 128 registers/thread, 24576 shared bytes and zero
spills for its sm89 parent and different inverse variant. These are its reported
compiler results, not our exact four-lane build. Its fused tree retains 16 KiB
and reduces 18 barriers to 16; our first submission uses 4 KiB and 14 barriers.
Correction: the prior advice not to combine them was too broad. PR36 composes
the four-lane reduction with fusion of different, higher shared levels, giving
4 KiB and 12 barriers. This combined schedule is now independently checked.
Pending `7bf15b2` specializes ranked flags on an older base; the selected PR40
specialization instead applies directly to the current epoch architecture.

This Mac cannot compile or run CUDA. CPU source checks and symbolic models
cannot establish PTX correctness, register allocation or GPU speed.

## Reviewed queue

| Priority / state | Idea | Evidence and next decision |
| --- | --- | --- |
| 1 / hypothesis | Separate SHA generation from EC recovery in bounded tiles | Both models recommend testing independent register budgets. A digest write and read adds 64 bytes/candidate, or 27.7 GB/s at the new frontier before other traffic. This proves neither a gain nor a bandwidth failure. Compare total pipeline runtime with an unchanged monolithic control. |
| Supporting diagnostic | Compiler resource and occupancy sensitivity | Compile unchanged base and our exact source first. Inspect registers, stack, spills, allocated shared memory and stalls. A register-budget sweep is a diagnostic, not itself a large-improvement submission. |
| Deferred research | Cooperative EC arithmetic across 2 or 4 lanes | No demonstrated gain. Account for fewer independent candidates, communication and registers per candidate. Grok's blanket 4x requirement ignores possible occupancy changes; do not treat it as an impossibility proof. |
| Deprioritized | Simple epoch/window geometry changes | The bounded scan below finds under 1% SHA-only equal-cost upside among sustainable choices. No large architectural gain demonstrated. |
| Deprioritized | Schedule-table packing and producer overlap | No evidence of a large runtime fraction. Divergent constant-memory lookups serialize. Overlap is not guaranteed by adding streams. |
| Rejected as stated | “GLV automatically halves fixed-base work” | Current 16 windows already remove doublings. Two 128-bit multiplications must count both sets of windows and their combination. Require a lower operation count and feasible table size. |

For experiment 1, keep 256-lane epoch and inverse semantics and carry an
unambiguous global candidate/epoch index between stages. Candidate tile sizes:
32768, 65536 and 262144 (1, 2 and 8 MiB digest buffers). Preserve both recovery
flags, runtime inputs, inactive-lane identity factors and hit mapping. Memory
capacity fitting a cache does not guarantee residency.

Use matched compiler options, GPU, problem seed, difficulty and duration. Measure
the complete producer/SHA/EC/hit pipeline. Verify hits independently and run the
GPU arithmetic/synchronization audit. Reject if verified end-to-end throughput
does not show a substantial repeatable gain against the current architecture.
If the compiled EC-only kernel retains the same constraints and profiling shows
no relevant improvement, stop before broad tuning. Neither model has established
that this split will help: maximum live state may already be dominated by EC.

No production prototype was installed by the first research round. The queue does
not authorize GPU rental, automatic submissions or trusted harness edits.

## Independent geometry check

Run from the benchmark repository:

```sh
python3 candidates/subset/research/epoch_geometry.py
```

The model counts compression calls using distinct symbolic row bytes. It
reproduces the current 84 first-state classes and 26 second-message classes.
Including the epoch producer, SHA256d and two pubkey hashes gives
`(21 + 84)/256 + 1 + 4 + 1 + 2 = 8.41015625` transforms/candidate.
Equal block-two schedules do not imply equal incoming SHA states.

For windows 4..32, omissions 1..9 and the first at most 256 lexicographic choices,
requiring unique space for an assumed 500M/s over 1200 seconds, the best modeled
count is 8.339393939. The SHA-only equal-cost speed ratio is 1.0084853. Field
arithmetic, inactive-lane padding and hardware costs are not modeled. This is a
narrow screening result, not a proof against other orderings, cross-epoch reuse
or a GPU speedup bound under different hardware behavior.

## Source-checked EC review

The source implements one affine-pair XYZZ addition (4M+2S), fourteen mixed
additions (8M+2S each), and a 3M homogeneous conversion: 119M+30S before recovery.
The conversion plus prepare and finish total 13M+3S, excluding block inversion.
Grok correctly identified the already-present 8x32 multiply schedule, specialized
squaring and amortized block inverse. These are not new opportunities.

Grok found prior experiments in `TREE_INVERSE.md`, independently checked by
Codex: streaming recoding reported 428061650 and direct XYZZ recovery 434122561,
against that author's historical 437975843 local base. These are upstream reports,
not our measurements or universal negative results. Obtain the actual previous
variant before repeating either experiment. Grok's illustrative three-factor
XYZZ finish is not a proof that all direct formulations need three factors.

## Corrections retained from review

Gemini retracted unsupported cycle estimates, the supposed proof of zero inverse
speedup, guaranteed stream utilization, and its blanket rejection of kernel
splitting because an entire batch exceeds cache capacity.

- `__launch_bounds__(256,2)` supplies a desired **minimum** resident-block target
  for compiler register budgeting, not a maximum. Check the actual compiler
  version and resource report when combining register controls.
  [NVIDIA language extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#launch-bounds).
- Ada has 64K 32-bit registers and at most 48 resident warps per SM. If the exact
  build uses 128 registers/thread at 256 threads/block, register capacity allows
  only two such blocks. This conditional calculation measures neither runtime
  nor the value of fewer barriers.
  [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#occupancy).
- Higher occupancy need not improve throughput; divergent constant-memory
  accesses serialize. Lexical scopes do not guarantee shorter compiler live
  ranges. Resource targets are not promised performance gains.
  [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy).
- Use `S = 1 / (1 - f + f/s)` with a measured runtime fraction `f` and justified
  stage acceleration `s`. Compression counts with cached versus dynamic
  schedules are not interchangeable cycle measurements.
- Only 26 second-schedule columns are populated; unused columns are zero,
  not copied schedules. Cache misses and DRAM traffic remain unmeasured.

Primary research leads: [Bitcoin Core fixed-base comb implementation](https://github.com/bitcoin-core/secp256k1/blob/master/src/ecmult_gen_impl.h)
and [NVIDIA CGBN cooperative big-integer library](https://github.com/NVlabs/CGBN).
These established techniques are leads to assess against this source, not
evidence of novelty or speedup on this benchmark.

## Final composition review, 2026-09-16

The user subsequently authorized submission based on expected superiority when
GPU testing is unavailable, using this Mac only. Selected production now
combines PR36 (`472b536`) and PR40 (`08a4b7b`): streamed deferred-Y arithmetic,
composed inverse levels, and ranked kernel templates. Public provenance and
the exact reproduction commands are in `../submission-deferred-ranked.md`.
Source review found the two patches operate on separate parts of the pipeline;
PR40 applied to PR36 with zero fuzz. This is the strongest expected combination
we identified, not a verified ranking above either parent.

Grok and Gemini completed final read-only reviews of the combined source. Both
found no blocking semantic defect. Grok checked the affine anchor invariant,
final resolved ordinate, disjoint shared-tree ownership and both recids, and
conditioned submission on the CPU audits passing. Those audits passed.
Gemini also recommended the composition, but its claims of successful CUDA
compilation, guaranteed ptxas stripping, exactly allocated registers/stack,
and a 77% EC runtime fraction were unsupported and are explicitly rejected.
Dead `_FixedBaseSignedAffine` removal was inherited, not a new improvement.
An old non-template overload would not itself necessarily be a compilation
error, contrary to one Grok checklist item; the actual source has no such call.

CPU evidence for the selected fingerprint: 3840 inverse outputs, 6144 complete
SHA256d hashes, 10262 recodings and 8086 unrankings through extracted source;
2000 arbitrary-field and 128 on-curve chains through extracted point helpers,
with 31920 intermediate-state comparisons; plus the independent PR36 model's
19200 inverse outputs, 20000 arbitrary chains, 1000 curve chains, 10262 recodes
and 28 folded multiplies. No CUDA compiler or GPU was used.

The isolated split generator supports the prior submission, PR36 and this
template-specialized combination. Across the prior and PR36 bases it passed
6144 complete hash/candidate-identity checks and 2112 additional scalar
transport checks. The final composite base adds 3072 and 1056 respectively.
This checks the generated hash/transport source, not its full EC kernel or GPU
execution. Splitting remains deferred: it adds tiled launches and digest traffic
without removing EC arithmetic, and an EC register bottleneck may remain.

Latest additional frontier notes read: PR39 streams recoding already included;
PR40 supplies the selected specialization; PR42 has an older two-pass epoch
design and author-reported 99-104M/s on RTX 4080. No new stronger architecture
was established by those reports. At that point the monitor was results-only.

## Continuing research and monitoring authorization

The user subsequently chose to preserve `c9a85a87` and explicitly requested
continued improvement research, inspection of incoming submissions and an alert
if later submissions finish first. The existing follow-up now checks every
20 minutes, preserves both original pending entries, and permits bounded local
subset research. It does not submit or cancel. `monitor_state.json` retains
observed statuses and notification history. There is no visible FIFO guarantee;
later scored results establish completion order, while early failures or
unrelated CodeQL checks do not establish GPU execution order.

New pinning notes PR44, PR45 and PR46 were read for transferable mechanisms.
PR44/45 port the promoted subset field arithmetic back to pinning; that source
is already present here. PR46 adds batch inversion and faster host table
generation to an older pinning design; subset already has the stronger shared
inverse tree and GPU table construction. Their reported gains do not apply
incrementally to the prepared subset control. No speculative port was selected.

PR47 was also screened: grouping launches and restoring earlier dead-code
removals on old pinning source supplies no demonstrated large increment for
our 8M-batch ranked subset kernel. The useful remaining arithmetic research is
in `ARITHMETIC.md`: a verified 48-product limb model, and a concrete inherited
carry-drop counterexample with an isolated corrected multiply/square variant.
The original source stays frozen for comparisons; future submissions must
address and disclose the primitive finding rather than rely solely on the
OpenSSL-substituted point tests. No new speedup has been measured.

## Current pipeline experiment, superseding the earlier pending state

PR27 returned rejected at 418504460 versus 433346795. The current source ports
the pinning checkpoint hierarchy into the strongest prepared subset composite,
retaining 16-window planar tables and adopting direct XYZZ recovery. Pinning's
related 6ce23203 was promoted at 644546620 during this work; this is supporting
architecture evidence, not a subset prediction.

Grok 4.6 and Gemini 3.8 Flash(High) reviewed actual relevant source through signed-in
CLIs. Initial missing-signature, table-layout and dropped-carry claims used stale
PR24 context and were rejected after inspecting the exact port. Whole-file C++
projection type checking also passes; it is not a CUDA compile.

Grok's remaining proposed carry counterexample (low=B-1 and carry=1 after the
second fold) is unreachable. Let B=2^256,C=2^32+977. The first fold's high limb
is at most C, so the second fold is at most B-1+C^2. If carry=1, its low limb is
less than C^2; adding C is below 2^96. If carry=0, the third addition is zero.
Thus no carry can escape z2 in the shortened third fold. This is an algebraic
bound, not a claim that random testing proves all inputs. Gemini's correction
quoted a stale non-.cc high addition; actual production uses addc.cc and a
mutation test catches the stale-flag alternative. Neither suggestion was
blindly applied.

Remaining risks are real but unmeasured: roughly 320bytes/candidate checkpoint
traffic, extra launch overhead, larger prepare shared state and finish register
pressure under three-CTA launch bounds. No invented cycle estimate, cache
eviction guarantee or spill result is accepted. All public performance claims
remain separately labeled by origin. See submission-ranked-pipeline.md for the
final reproducible experiment and validation boundaries.

## Final pending scan

The last two arrivals were inspected before upload. PR57
(`3b1bba4b-5095-4dc5-a072-1589c5589107`) uses direct XYZZ recovery at 10M+3S
and retains the per-CTA inverse. It saves one square relative to our direct
checkpoint recovery but uses a different saved denominator/state relationship.
Our selected four-field transport avoids adding another saved field or
reconstructing W for the finish collective; we retain that already checked
port rather than assume the monolithic formula's saving transfers unchanged.
PR59 (`5605ad84-dd72-4aed-9e6d-60a6bb64a61b`), validation head
`da066cac70e48ca97ce70942f1f60ea17539ec0c`, combines two-stream epoch overlap,
constant recovery coordinates and paired compressed-key SHA. Its source still
uses the homogeneous recovery/per-CTA inverse. The author reports matched
full-duration relative gains of 1.67% and 2.03% on two power-limited RTX 4090 hosts.
Those are author results, not ours or an official new frontier.
Paired SHA holds two schedules live; its reported monolithic-kernel benefit
is not established under our separate three-CTA finish register budget.
Multi-stream overlap would also change buffer/event lifetimes. Both remain
recorded follow-up options pending exact-pipeline GPU evidence, rather than
being treated as automatically composable wins. Neither new entry supersedes
the external-inversion architecture selected for this submission.
