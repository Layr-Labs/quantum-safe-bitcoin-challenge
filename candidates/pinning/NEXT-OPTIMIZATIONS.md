# Pinning: next-optimization backlog after the 778 M frontier

Snapshot: 2026-09-20. This is a campaign plan, not a performance claim. It
is scoped to `candidates/pinning/` and assumes the currently promoted source
`7b0a15b` at 778,624,395 verified candidates/s.

## Current queue and ready work

- Our account `f034a9c4` **rejected at 750,065,705** (−3.67%). SHA ST flags
  stay off. Treat as LeaderGPU-class host draw; do not retry SHA.
- ercumentyildirim `960da801` / PR #743 **rejected at 786,386,945** (+0.997%).
  Missed the 786,410,639 floor by 23,694/s. Multiply-tail is real. Compose it
  with carry62 (this is commit `432cab3`).
- Commit `432cab3` is ready locally. It composes PR #743 with `QSB_CARRY62`,
  removes ten serial-chain instructions per point-add round, has an explicit
  below-2^-62 new error budget, and is predicted at roughly +1.216% over the
  778 M source. It cannot be queued until our SHA submission finishes.
- `eb6d9871-3655-4284-b627-da65949f0e89` is a new exact four-wave top-16
  cofactor candidate. It fixes the old unsafe reassociation with a complete
  multiplier but performs 63 products instead of the frontier's 43. Its
  official result is useful; do not copy it before that result exists.
- `6efddf37-b57c-485f-98a7-c557c035be40` is primarily a frontier
  remeasurement with a small carried mechanism and no measured gain. Treat it
  as noise calibration, not as an optimization source.

The immediate rule is simple: do not cancel `f034a9c4`, and do not submit a
new bundle without refreshing the promoted score and all four results above.

## Proofreading corrections to WINNING-STRATEGY.md

The shared strategy file is valuable, but these claims need correction before
they drive implementation:

| Thesis claim | Corrected reading | Action |
| --- | --- | --- |
| A 32 MiB joint GLV comb is the large untried bet | Grouped/joint GLV has already been tried repeatedly: `334b1839` scored 674.7 M, `5ffaef34` scored 722.8 M, `0c846cf0` scored 714.3 M and `4c77f9bd` scored 523.5 M. The arithmetic-floor submission `808bddbf` explains why the ordinary 32 MiB window family dominates once random-load cost is included. | Retire GLV unless a new cache/layout proof invalidates those measurements. Do not build radix-373 GLV. |
| `d = a-x` may still save two additions | `QSB_PARITY_SUM=1` already replaces those differences with the identity based on `x-a = sum*(l or m-c)` and computes the needed parities without the old `a-x` subtractions. | Closed. Verify source before porting any PR #686 hunk. |
| Enable `QSB_TAIL_TAB` and `QSB_SHA_SMEM_W1` one at a time, then together | The call site has a combined ST body and an SMEM-only S body. `TAIL_TAB=1` while `SHA_SMEM_W1=0` does not select a distinct table-consuming transform; it mainly adds setup work. | The meaningful experiments are SMEM-only and both-on. Our in-flight run is both-on. |
| Top-16 is simply unsafe | The old short-carry reassociation is unsafe. The new `eb6d9871` uses a complete multiplier and boundary normalization, so it can be correct, but it trades 20 extra products for two fewer dependent waves. | Wait for its official score and inspect registers/SASS before composing. |
| Joint GLV or filter/exact is next if SHA loses | Only filter/exact survives the public evidence. The cleanest first version may be an exact host publication gate using the existing hit-index queue, not a second GPU kernel. | Prototype the gate first; use it to unlock deeper per-candidate approximations. |
| Another agent currently has an unroll tree dirty | This checkout is now clean at `QSB_UNROLL=1`; unroll 2/3/4 was reverted after static analysis and prior official losses. | Treat the coordination header as stale. |

Two further wording fixes matter:

1. Approximate arithmetic before hashing can create both false positives and
   false negatives. It becomes safe for publication only when an exact gate
   removes false positives; the remaining false-negative rate must still be
   included in the score budget.
2. A favorable runner assignment is not a kernel optimization. Host spread is
   useful uncertainty information, but a plan should not depend on repeated
   cancellation or on identifying a particular runner.

## Result-driven submission decision

Refresh the ledger immediately before the next submit.

### If `f034a9c4` rejects or scores at/below the 778 M source

Keep both SHA flags off. Then inspect PR #743:

- If PR #743 also rejects, submit `432cab3` unchanged. The composition was
  designed to clear the old 786,410,639 floor, though its modeled margin is
  only about 0.21%.
- If PR #743 promotes, do **not** submit carry62 alone: +0.585% is below the
  next 1% gate. Move directly to the exact-publication-gate bundle described
  below, or combine carry62 with another independently positive mechanism.

### If `f034a9c4` promotes

The new frontier includes the SHA flags but not PR #743. Sync the promoted
source, then rebuild a three-piece composition:

1. promoted SHA flags;
2. PR #743's measured multiply-tail truncation;
3. `QSB_CARRY62`.

Compile the rollback and composed modes again. The arithmetic mechanisms are
independent of the SHA path, but the new promotion score raises the absolute
bar, so the old 788.1 M center is no longer sufficient evidence by itself.

### If `eb6d9871` promotes

Treat its complete top-16 schedule as proven only on its exact submitted
source. Rebase rather than hand-merging old PR #700 code. Compose it with
carry62 only after checking:

- stage-0 registers and spills;
- chain and tree SASS size;
- that its dedicated complete multiplier did not replace PR #743's hot
  `_ModMultCore` paths;
- that its manifest/runtime files really match the newly promoted base.

## Priority 1: exact publication gate plus a faster approximate chain

This is the strongest genuinely new direction left by the public record.
Subset already proves the architecture: a fast approximate path may nominate
hits, but only exact results may be published. Pinning currently writes the
stage-2 tentative indices directly to its 1024-entry hit buffer and trusts
them. Add an exact gate between that buffer and the output files.

### Prefer a host gate as the first experiment

Pinning already links OpenSSL and already has the complete problem constants,
sequence, locktime and recid on the host. The expected true-hit rate is about

```text
778,000,000 candidates/s * 2 recids / 2^24 ~= 93 tentative hits/s.
```

Exact CPU recovery and hashing for roughly 93 records/s should be small next
to the GPU grind, especially if drained in batches on the two existing slots.
This must be measured rather than assumed, but it avoids adding a large exact
CUDA specialization and its JIT/register footprint.

Required behavior:

1. Stage 2 writes `(candidate index, recid, hash mode)` to the existing queue.
2. On slot drain, the host reconstructs the exact candidate preimage, computes
   exact ECDSA recovery and the pubkey hash, and discards any mismatch.
3. Only host-verified records are written in the benchmark's hit format.
4. The verifier must preserve first-hit/recid semantics expected by the
   harness; a valid later recid cannot be suppressed by a false earlier one.

The last point is easy to miss. The current kernel returns after the first
tentative hit. With approximate arithmetic, a false recid-0 nomination could
hide a real recid-1 hit. Either queue both passing recids or, when the exact
gate rejects the first, exact-check the other recid before declaring the
candidate empty.

### Approximations unlocked by the gate

Start with one switch at a time and keep the approximations per-candidate.
Do not first alter fixed-table construction, shared super-root inversion or a
value that fans out to thousands of candidates.

Promising switches:

- **C31 second-fold tail.** Stop before propagating into `z3`, rather than the
  carry62 version that consumes z3 and stops before z4. Temporary CUDA 12.6
  analysis reduced the point loop from 1,077 to about 1,068 instructions.
  Its per-operation divergence is around the 2^-31 class, unacceptable for
  direct publication but reasonable behind an exact gate.
- **Low-64 split-3p correction.** Subtract `3K` through two limbs instead of
  carry62's three. Its divergence condition is `low64 < 3K`, about 2^-30.4.
- **One-limb K corrections.** Revisit the second limb retained by
  `_ModSub256`/`_ModAddLazy`. These sites were kept to achieve about 2^-95
  direct-publication bounds. Behind an exact gate, a 2^-31-class local
  divergence may be acceptable if it stays within the per-candidate chain.

The score loss from false negatives is approximately the fraction of
candidates whose point result is corrupted, not the fraction of all corrupted
results that happen to pass the gate. Therefore calculate a union bound over
all invocations per candidate. A nominal 2^-31 site used 100 times gives a
roughly 4.7e-8 candidate corruption fraction before correlations: tiny, but it
must still be measured with a matched exact hit set.

### Kill tests for the publication gate

Do not stack aggressive truncations until the gate itself passes all four:

1. **Injected-false-hit test:** force a known wrong tentative index and prove
   that no output record is written.
2. **Two-recid priority test:** reject a fake first recid while preserving a
   valid second recid for the same candidate.
3. **Baseline overhead ABBA:** exact gate with exact GPU arithmetic must cost
   less than 0.2% on fixed work, or move verification to a tiny GPU kernel.
4. **Long matched prefix:** every baseline true hit must survive; extra
   tentative hits may appear but must be removed before output.

If CPU verification is too expensive, the fallback is a compact GPU replay
kernel launched only when the tentative count is nonzero. Keep that kernel
separate so its exact field code does not increase the hot stage's register
allocation. Its PTX/JIT cost still counts and must be included in the ABBA.

Expected band after the gate is validated: **0.8-2.5%**, depending on how many
serial carry instructions can be removed before false-negative loss or code
shape dominates. This is a research estimate, not measured throughput.

## Priority 2: evidence-gated composition of current live entries

The cheapest source of another percent is to combine mechanisms only after
each has produced evidence:

| Mechanism | Evidence required | Composition rule |
| --- | --- | --- |
| PR #743 multiply tail | Its current official score, plus its existing mirrored +0.627% local result | Keep if it is not a large official regression. It changes multiplies only. |
| carry62 | Current static proof and official score of its first run | Keep behind a rollback flag. Do not infer +0.585% solely from instruction count after a contrary ranked result. |
| SHA ST flags (`f034a9c4`) | Official result | Compose only if positive enough to survive noise. Stop pinning SHA work if neutral/negative. |
| exact top-16 (`eb6d9871`) | Official result and static register/SASS audit | Compose only from its exact source, never from unsafe PR #700 reassociation. |

Avoid a four-way speculative bundle. The 1% gate makes composition necessary,
but an unknown -1% piece can erase two real +0.6% pieces.

## Priority 3: systematic exact carry-chain census

The last exact gains are likely small PTX changes, not a new curve formula.
Turn the ad-hoc search into a reproducible census:

1. Extract every `addc`/`subc` tail in `_ModMultCore`, `_ModSqr`,
   `_ModSqrAddSub2`, `_ModSub256`, `_ModAddLazy` and the packed finish.
2. For each tail, record the proven range of its incoming carry/borrow, the
   number of propagation limbs, dynamic calls per candidate and whether the
   result fans out to a group.
3. Classify each removable suffix as exact, below 2^-90, below 2^-60,
   2^-31-class, or unsafe/unbounded.
4. Generate a rollback macro and a host word-operation model for one site at a
   time.
5. Compile sm_89 both ways and retain only changes that reduce the serial
   backedge without increasing registers, spills or loop footprint elsewhere.

The direct-publication candidate should stay at or below roughly the 2^-60
class. The exact-gated filter may test 2^-31-class sites. Never infer
independence when a carry value and the following limb share operands; use an
executable predicate audit and a conservative bound.

## Priority 4: exact top-tree scheduling, only after the live result

The four-wave complete top-16 proposal is the only current exact tree idea
that is not already refuted. Its potential comes from reducing dependent warp
waves from six to four, but it adds 20 scalar multiplications. The official
result will be more informative than another source-level argument.

If it is close but negative, possible follow-ups are narrowly scoped:

- use the dedicated complete multiplier only for intermediates that can be
  raw, while retaining the frontier multiplier at canonical boundaries;
- test whether one of the four waves can preserve the original association
  without the 16 extra exclusion products;
- shorten register lifetimes around the 17 final normalizations.

Kill immediately if it raises stage-0 registers across an occupancy boundary,
spills, or loses more than 0.5% officially. Do not spend another submission on
short-carry reassociation; the counterexample is already public.

## Priority 5: low-ceiling operational/code-shape experiments

These cannot clear the gate alone and should be used only as measured bundle
fillers:

- **CUDA graph per slot:** capture the stable prepare/root/finish launch chain
  and update only pointers/scalars. This preserves phase separation unlike the
  failed fused kernel. Expected ceiling is launch overhead, probably below
  0.3%.
- **Further ranked-only PTX trimming:** inspect the emitted function list and
  remove only specializations that cannot be launched in ranked mode. The big
  unreachable-half cut has already landed, so the remaining ceiling is small.
- **Host drain batching:** exact-gate work may make this relevant. Drain and
  verify one slot while the other runs; never introduce a device-wide sync.
- **SMEM-only SHA path:** if both-on SHA is ambiguous, compare the existing S
  body against the current Q body. Do not test TAIL_TAB alone because it does
  not select a meaningful distinct transform.

Each needs fixed-work timing and identical hits. None deserves a standalone
submission without at least about +0.5% local evidence to combine with a
larger proven mechanism.

## Retired directions

Do not spend more implementation time on these without qualitatively new
evidence:

- classic, shifted, grouped or joint GLV; radix-373 GLV; wider or 14-term
  tables;
- chain unroll 2/3/4;
- PR #700 short-carry top-16 reassociation;
- L1 prefetch, L1 bypass, larger batch, extra slot, five-block occupancy;
- Karatsuba, 5x52, 10x26, Montgomery, Edwards, Pippenger or tensor-core field
  arithmetic;
- more pinning SHA work if `f034a9c4` is neutral or negative;
- root/barrier and checkpoint variants already screened against PR #706;
- cosmetic remeasurements as evidence of a mechanism.

## Concrete next sequence

1. Wait for `f034a9c4`; refresh all statuses and the promoted score.
2. Apply the result-driven decision above to commit `432cab3`.
3. In a separate branch, implement the exact host publication gate with no
   arithmetic changes and measure/prove its overhead first.
4. Add C31 only, then low-64 split-3p only, each behind its own switch and with
   matched exact hit sets.
5. Consider the exact top-16 source only if `eb6d9871` supplies positive
   official evidence.
6. Continue the carry-chain census for one more direct-publication-safe
   below-2^-60 edit.

That sequence learns from every paid 1,200-second run, preserves rollback
isolation, and avoids the two largest traps in the shared thesis: retrying GLV
and porting recovery work already absorbed by `QSB_PARITY_SUM`.
