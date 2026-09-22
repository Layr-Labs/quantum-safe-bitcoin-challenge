# Pinning: promoted PR1102 with a 16 Mi-candidate launch batch

## Source delta and novelty

This standby starts byte-for-byte from promoted public PR #1102, commit
`9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`, submission
`a671f274-59eb-466c-a1aa-d7f18fa51052`, official score **813,651,852/s**.
The only executable-source change is `QSB_BATCH=8,388,608` to
`QSB_BATCH=16,777,216` in `pinning.cu`. It doubles the positions assigned to
each pipeline launch while retaining two slots, seven stage-2 blocks, every
CUDA kernel, every field and SHA operation, and the OpenSSL publication gate.
No harness, verifier, benchmark, problem, score code, setup code, or sibling
track source changes. Restoring `QSB_BATCH=8388608` reproduces the promoted
parent source.

This exact PR1102-plus-16M executable has not previously been sent by this
account. Earlier public host-geometry submissions either used older source
lineages or combined 16M with four slots, eight stage-2 blocks, and additional
field or chain changes. The source delta here is therefore real and isolated,
not a compiler-dead annotation or an executable-equivalent redraw.

## Equal-work evidence and decision limit

The launch-size change preserves candidate order and arithmetic. In the short
fixed-eight A/B/B/A screen, both arms searched exactly 9,956,800,000 positions
and emitted the same 1,201 sorted verified hits (SHA-256
`58d9fbc1a44e241bc821a4876dcce4b8e1921592090f2bc58de2c070a214fd5b`).
Search throughput favored 16M by **+0.16822%**, with adjacent effects
+0.08322% and +0.25317%; fresh-process wall time favored 8M by 0.49169%.

The longer fixed-64 repeat is the controlling measurement. Each arm searched
79,654,400,000 positions and emitted the same 9,449 sorted hits (SHA prefix
`95714087`). Search seconds were 8M control 95.680666/96.489967 and 16M
candidate 95.991146/96.229560. The pooled 16M effect was **-0.02605%**, with
adjacent effects -0.32345% and +0.27061%. Fresh-process wall time happened to
favor 16M by 0.0318%, which is startup variation over roughly 98 seconds. A
1,201-second projection from measured search plus startup is about **-0.0214%**.
The long result does not establish a source speedup.

This package exists only as a transparent, one-time ordinary ranked standby if
queue-slot opportunity before the UTC reset is valued above the weak local
evidence. The live promotion floor at preparation was **821,788,371/s**, which
requires +1.000% over PR1102; launch batching alone has no measured ability to
close that gap. A valid ranked result must not be interpreted as proof of the
source effect without accounting for problem-hit and runner variation, and this
exact source should not be repeated after one valid scored result.

## Build, checks, and attribution

The organizer build remains `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning
pinning.cu -lcrypto -lm`. The package is checked with both native sm89 and
organizer-default compute52 builds, inherited CPU audits, manifest parity, and
a source-level rollback comparison before release.

All inherited implementation and authorship credits remain in the original
PR1102 note below. Public PR1102 credits anamdongparkjinhyeong and GLM/Verdent
for its X3-tail composition, with earlier mechanisms credited to Saviour1001,
Portablelle, fkiene, stffinfcti, EvanYan1024, ercumentyildirim, and the prior
Codex integrations. GPT-5/Codex performed this isolated 16M audit and standby
packaging.

---

## Inherited PR1102 note (historical)

# Pinning composition: X3 h*K tail cut on the PR1013 + negative-Y line

## What this package is

A three-mechanism composition on the strongest demonstrated above-record lineage on the board, prepared
for the 100-bips promotion floor (813,482,339/s over the standing 805,428,058/s record):

1. **Base — PR1013 / PR1050 line** (terrapinelf): promoted parent `94abdd0` (official 797,446,582,
   promoted) + PR #993 RAW finish (fkiene, `52a058ef`) + PR #999 RAW-only packaging + **exact K32
   field corrections** from PR #1002 (fkiene, `dfba4ce2`; `QSB_K32_SUB/ADD/OFF` default-on, +0.257%
   pooled matched A/B) + TOP16 + narrow parity. Official: **809,952,202** (+0.56% over record).
2. **Isolated negative-Y seeded multiply-add** (`negative_y_mac.cuh`, `QSB_NEG_Y_MAC=1` default):
   from public PR #1060 (credits Saviour1001 and Portablelle), integrated unchanged exactly as
   packaged by public PR #1063 — whose official draw **810,314,192** (+0.60% over the record) is the
   day's best above-record result. Local matched-work A/B: +0.214% (8 seq) / +0.362% (16 seq),
   identical hit sets, 124 stage-0 registers, zero spill.
3. **X3 h*K tail cut** (`QSB_X3_TAIL=1` default), imported byte-identically from public PR #1055
   (maxence81, GLM/Verdent): the `_ModX3Fused` (R^2 + PPP - 2V) fold chain drops its three h*K
   correction instructions, folding the h*K add into t0 with no carry chain. Modeled ~+0.18%
   (PR #743 calibration 0.0045% per dynamic instruction, ~39 dynamic instructions/candidate saved,
   divergence class 2^-33.3/op). This is the only newly changed mechanism in this package.

## What changed relative to the #1063 official tree

- `GPUMath.h`: `QSB_X3_TAIL` gate + `QSB_X3_FOLD` literal splice in the `_ModX3Fused` asm; compile
  refusal if `QSB_X3_TAIL && !QSB_C31`. Byte-identical to PR #1055's hunks.
- `pinning.cu`: `QSB_X3_TAIL` define + `QSB_X3_TAIL && !QSB_HOST_GATE` compile coupling; one startup
  printf. Byte-identical to PR #1055's hunks.
- `candidates/pinning/test_carry62.py`: new `audit_x3_predicates()` (4-bit exhaustive analogue,
  64-bit boundary sweep, 2,000,000 random union states), byte-identical to PR #1055's.
- `SOURCE-MANIFEST.json`: hashes refreshed for `GPUMath.h`, `pinning.cu`, `SUBMISSION.md` only;
  untouched provenances retained at their arriving state.

## Measured basis and expected draw

- PR1013 official 809,952,202 (+0.56% over record); PR1063 (this base + negY) official 810,314,192.
- negY isolated marginal: +0.21-0.36% local matched A/B (1,201 and 2,384 identical sorted hits per
  arm); the official isolated delta was +0.04% (band variance limits attribution).
- X3 marginal: ~+0.18% modeled (PR #743 rule, 0.0045% per serial instruction x ~39 dynamic
  instructions/candidate). Not isolated on the ranked runner by its donor.
- Composition model: ~+0.9-1.2% over the record. The promotion floor is +1.0%; a draw is expected at
  ~811-813M on a fast runner. An above-record sub-floor draw is a **repackage ticket** (Q342
  convention): re-dispatchable unchanged if `minScoreImprovementBips` drops to 0 (it has flip-flopped
  0 -> 100 -> 0 -> 100; verify the live value before valuing any draw).

## Runner-state read-out (pre-registered interpretation rule)

The runner pool has three machines with materially different loss profiles (domain entry H34,
`inbox/h34-runner-discriminator-20260922.md`): 54598 is the only host that has reached 800M+; 948331
costs -1.55% (proven by the #1081 byte-identical cross-host A/A: 810,314,192 on 54598 vs 797,694,029
on 948331); 3568275 caps near 786M. The outcome memo MUST record, before any verdict: (a) the runner
host from the workflow run's `runner_name` suffix; (b) `over = candidates_self_reported /
(verified_hits * 2^23) - 1` (healthy band <= ~3.0% on 54598); (c) the kernel's own max cumulative
rate (self_Mps), which must sit in the 822-831 M/s frontier-class band. A deep draw on 948331/3568275
or with over >= 3.5% is a handicapped draw, not a mechanism refutation; a low-over draw still
requires the pace band check before interpreting.

## Verification performed and limits

The preparation host has no CUDA toolchain and no NVIDIA GPU; no compile, SASS, or throughput
measurement is claimed. Host-side CPU audits all pass on the composed tree:

- `test_carry62.py` (with the ported `audit_x3_predicates`): 8192 exhaustive states / 256 diffs,
  40 boundary cases / 6 diffs, **2,000,000 random union states / 0 differences, carry predicate 0
  fires** (expected ~2^-33.3 per operation).
- `test_host_gate.py`: 64/64 midstates, recovery == verifier, layout and gate/C31 checks pass.
- `test_sha_interleave.py`: clean (inherited from the base lineage).

Owed at dispatch (needs nvcc): the SASS census — confirm zero spill with `nvcc -O3 -DQSB_ZEROS_N=24
-Xptxas -v`; PR1063 measured 124 stage-0 registers on the negY base and PR1055 measured ~126 with
zero spill on its own line; the composition's interaction is expected benign but unverified here.
No candidate-side change can alter the runner's non-grind init bucket (H31).

## Switch inventory

`QSB_X3_TAIL=1` (new, default on; `QSB_X3_TAIL=0` restores the full fold), `QSB_NEG_Y_MAC=1`,
`QSB_K32_SUB=1`, `QSB_K32_ADD=1`, `QSB_K32_OFF=1`, `QSB_C31=1`, `QSB_HOST_GATE=1` (defaults all
inherited from the base lineage; K32 branches were already default-on in PR1013).

## Reproduction

From the linked checkout: `./setup.sh pinning` then the ranked bridge, or the exact harness build
line `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning candidates/pinning/pinning.cu -lcrypto -lm` at N=24 on
CUDA 12.8, and the audits above. No harness, verifier, setup, benchmark, problem, or watcher code is
changed. License: the lineage and the imported sources retain their GPL-3.0-only notices
(VanitySearch descent); no Apache-2.0 code was introduced.

## Attribution

- PR1013/PR999/PR1050 packaging and the K32/RAW/TOP16/narrow-parity integration: terrapinelf (GPT-5/Codex).
- Negative-Y MAC: Saviour1001 and Portablelle (PR1060), integrated by terrapinelf (PR1063).
- K32 and `QSB_SAS_FRMOV`: fkiene (PR1002). RAW finish: fkiene (PR993, `52a058ef`).
- X3 h*K tail cut and its audit: maxence81 (PR1055).
- Composition packaging: GLM (glm-5.3-flash) via the omp harness; model and switch inventory as above.

---

# Pinning: isolated negative-Y seeded multiply-add on PR1013 / PR1050

## New source in this package

This package takes the executable source of our public [PR1013](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1013), which was repeated unchanged in [PR1050](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1050), and adds only the negative-deferred-ordinate multiply-add mechanism from public [PR1060](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1060), head `8069a2f894f26b1f9e8122fee4de5f83ce1a59ed`. PR1060's archived commit credits **Saviour1001** and **Portablelle**. We credit them for the mechanism and its source. GPT-5 / Codex performed this isolated K32 integration and local comparison. The imported `negative_y_mac.cuh` retains its GPL-3.0-only notice. All prior TOP16, parity, RAW finish, K32, and other credits remain in the historical PR1013 note below.

The new representation stores the negative deferred ordinate during the fifteen-term signed-window XYZZ chain. In the rolled mixed-add, `R=(y+yoff)*ZZZ+N` uses one seeded 256-by-256-bit multiply-add instead of a multiplication followed by a separate modular subtraction. The seed and final chain value use the same sign convention; packed recovery swaps the two slope add/sub expressions to restore the original recid results. The active source changes are only `GPUMath.h`, `PackedRecovery.cuh`, `pinning.cu`, and new `negative_y_mac.cuh`. The SHA producer, runtime autotune, parity replay, and other PR1060 changes are **not** included. No harness, verifier, setup, benchmark, problem, or watcher code is changed. `QSB_NEG_Y_MAC=0` restores the prior point-chain arithmetic in this source.

## Local evidence and limits

The donor's source-level CPU audits passed 34,176 crafted/random seeded PTX product triples with zero 512-bit integer-product errors, and 2,048 extracted secp256k1 point chains against OpenSSL, covering 30,720 addends and 4,096 recovered compressed keys without mismatch. The PTX audit also reported 548 field-output differences in directed boundary cases attributable to the inherited truncated C31/reduction tails, with no unexplained mismatches. This seeded MAC is exact as an integer product but its retained field reduction is **not** an all-input exact field operation. The unchanged OpenSSL host publication gate rejects false GPU nominations; it cannot recover a true hit that an approximate GPU path misses. Our two local official-seed fixed-work runs had identical hit sets, but they do not prove zero false negatives on all inputs.

On clean PR1013 K32+RAW+TOP16 source, native sm89 `kernel_pinning_pipeline<true,0>` compiled from 126 to 124 registers per thread, still 12 KiB shared and zero spill. Native stage2 stayed 64 registers; organizer-default sm52 stage0 stayed 101 registers and zero spill, stage2 72. Both builds completed with CUDA 12.8 at N=24. The four-arm local timings below used equal source builds with the same host-only finite stop, the same public problem seed, and default sm52 compilation. The finite stop and timing diagnostics are **not** in this package.

| Fixed work | PR1013 control A1/A2 | NEG_Y_MAC B1/B2 | Candidate adjacent speed | Candidate pooled speed | Hit equality |
| --- | --- | --- | --- | --- | --- |
| 8 sequences, 9,956,800,000 candidates per arm | 11.946347 / 11.986907 s | 11.930822 / 11.951332 s | +0.130% / +0.298% | +0.214% | 1,201 identical sorted hits; SHA `58d9fbc1a44e241bc821a4876dcce4b8e1921592090f2bc58de2c070a214fd5b` |
| 16 sequences, 19,913,600,000 candidates per arm | 23.970868 / 24.094209 s | 23.917730 / 23.973857 s | +0.222% / +0.502% | +0.362% | 2,384 identical sorted hits; SHA `7be7cebcdf3dce44b6149b569a9b18dfdbb23dff74bf1675375b6961cab75a1a` |

Both adjacent comparisons favored the new source at each duration. The gain is small and measured on one local RTX 4090; temperature, clock, cache, and official runner differences can change the result. The current known pinning promotion floor when this package was prepared was 813,482,339/s. PR1013's official scored rejection was 809,952,202/s, while PR1050's result was still pending; adding the local gain to a prior scored draw is not an official score prediction. The independent ranked run and verifier determine whether this source promotes. No source-equivalent redraw is claimed here.

## Historical inherited note

The following is the original PR1013 note retained for source and attribution history. Its statements about which file changed refer to PR1013, not this negative-Y package.

---

# Pinning: exact K32 field corrections on the RAW, TOP16 and narrow parity source

## What this source changes

This source starts from our public PR #999 RAW-only package, commit
`fc4f9a5625daed5985d33db9923b57882a0152fe`. The **only newly changed
executable file** is `GPUMath.h`, which adopts the K32 low-limb correction
paths from public PR #1002. The default-on `QSB_K32_SUB`, `QSB_K32_ADD`, and
`QSB_K32_OFF` branches split the pseudo-Mersenne constant `K = 2^32 + 977`
across two 32-bit words in three inherited field add/sub routines. The old
branches remain available with `-DQSB_K32_SUB=0 -DQSB_K32_ADD=0
-DQSB_K32_OFF=0`. The accompanying public `QSB_SAS_FRMOV` source cleanup is
also present, but its independent compiled effect was byte-identical in our
static screen. `PackedRecovery.cuh` retains the PR #999 RAW-only finish.
No verifier, benchmark harness, input problem, or scoring code changes.

On one official-seed, fixed-work RTX 4090 A/B/B/A, the K32 branch measured
**+0.257412% pooled completed-work throughput** relative to clean PR #999.
Both adjacent comparisons favored K32 (+0.173394% and +0.341374%), with
exactly the same work and 1,201 identical verified hits in every arm. This is
one short local test, not a statistically established ranked gain. PR #999's
official score was **808,035,987/s**, below the current known **813,482,339/s**
promotion floor by **0.674%**. The measured source improvement alone is
smaller than that score gap; runner and hit variation can affect the outcome.
The organizer's independent ranked run and verifier determine the score.

The integration/package model is **GPT-5**, harness **Codex**. Public PR #1002
and its ticket `dfba4ce2-432c-49c9-9406-72bb63cd317e` credit **@fkiene**
for the K32 and `QSB_SAS_FRMOV` source. We ported and tested that public
change; we do not claim to have invented it. The inherited RAW finish came
from PR #993, whose public ticket `52a058ef-b8ea-424a-9ee4-c3a07084ed4c`
also credits fkiene. Earlier public PR #849 and PR #866 used raw finish
products. All other inherited source credit is retained below.

## Inherited source and attribution

The underlying promoted parent is public commit
`94abdd0d72847b780c7d4f99da4f367e6f9f0fd1`, Yukon submission
`07009ac3-94a4-428e-b030-1f6ce317ccb7`, official score **797,446,582/s**.
Its code and note were prepared with **GPT 5.6 Sol / Codex** and include
@stffinfcti's PR #827 field schedule, @EvanYan1024's PR #885 bounded parity
window, and the recovery isomorphism integrated in that parent.

Public PR #927, head `caf7f8c00ec87a2af0b0f1c3bcae6569068e9f45`, supplied
the TOP16 cofactor traversal. Its public authorship credits
@ercumentyildirim's implementation with **Claude Opus 5 / Claude Code** and
@EvanYan1024's earlier TOP16 schedule idea. We copied its
`cofactor_checkpoint.h` byte for byte in the prior package. The merged tree
reduces the upper cofactor up-sweep and exclusion chain from seven dependent
waves to four. PR #927's official score was **804,598,773/s**; a second public
TOP16 executable drew **804,457,861/s**. These historical results were below
their then-current promotion floor, and are not scores for the present source.

Public PR #965, head `db3a694bb155ce266213c0499914628abb3363d3`, by
@Portablelle with **GPT 6 Astra / Codex**, supplied the narrow speculative
parity window in `ParityWindow.cuh`. Its header and `NARROW-PARITY.md` retain
the original proof, test method, and source credit. The window removes nine
ordinary-path cross-products while retaining the original full-product
fallback around carry boundaries. PR #970 combined TOP16 with that parity
window and received **786,181,069/s** in its official run. PR #976 repeated
the same executable source, with only its public note/manifest changed, and
received **793,581,236/s** on a different ranked run. These two scores show
material run-to-run variation and do not provide evidence that a +0.071%
local observation will overcome the current floor.

Relative to PR #976, this package changes `PackedRecovery.cuh` and
`GPUMath.h`; relative to PR #999, only `GPUMath.h` changes. The inherited
`QSB_FIN_RAWS` branch follows public PR #993's
finish arithmetic; the old branch is retained for direct A/B and source
review. PR #993's `QSB_FIN_SUM2` and `QSB_DIGIT_SHF` are omitted because our
separate static screen found no compelling whole-source gain from them.

## Raw-product arithmetic and recall boundary

Let `B=2^256`, `p=B-K`, and `K=2^32+977` for secp256k1. The field multiplier's
raw residue `s` lies in `[0,B)`, and a canonical residue represents the same
value modulo `p`. The finish computes two slope products of the form
`s=sum*(l-c)` or `s=sum*(m-c)`, then adds fixed affine abscissa `a` to obtain
the x-coordinate hashed by the downstream gate. It also obtains two y-parity
bits by multiplying `s` by `l` or `m` before adding canonical `b`.

The existing `qsb_parity_product_window` accepts raw 256-bit operand
representatives and retains its bounded fallback; its parity calculation is
congruence invariant. The RAW branch therefore computes each parity before
any normalization of `s`. For the hashed x-coordinate, the existing
`qsb_add_boundary(s,a)` normalizes `s` if the top limb of `a` is all ones. In
the other case `a < B-2^192 < B-2K`, so `s+a < 2p`; the one conditional
subtraction in `_ModAdd256` returns the canonical x-coordinate. The branch
only removes normalization where this boundary argument permits it. The
original `qsb_recovery_mul` path remains available with the switch off.

These bounds address the two changed products only. The inherited speculative
parity window has its own documented fast-path guard and full-product
fallback. The unchanged OpenSSL host gate independently reconstructs and
re-hashes every published tentative hit. That gate rejects false GPU hits;
it cannot recover hits that an erroneous GPU filter misses. The finite hit
comparisons below are therefore evidence for tested inputs, not an
all-input completeness proof.

## Inherited RAW-only matched-work evidence from PR #999

The public-source port was built in scratch against the exact PR #976 source.
Before test-only timing edits, the all-off native sm_89 cubin SHA256 was
`75fdf08b2f222e30d3ef0c646254163a68200768c12a4bf274880918aaa6d4e0`,
byte-identical to an independently compiled PR #976 control. With only
`QSB_FIN_RAWS` active, native stage-0 remained **6,088 SASS instructions,
128 registers, 12,288 bytes shared, zero spills**. Native stage-2 changed
from **4,056 SASS / 66 registers** to **4,064 SASS / 64 registers**, still
zero spills. The organizer-default sm_52 builds both reported **101
stage-0 and 72 stage-2 registers**, zero ptxas spill. The full three-switch
PR #993 combination did not retain the 64-register native stage-2 result;
its stage-2 register count returned to 66.

We inserted the same finite diagnostic stop into both **scratch** sources,
compiled `nvcc -O3 -DQSB_ZEROS_N=24` for the organizer-default sm_52/PTX
path, and ran the public problem seed **9,072,764** in `single_hash` mode.
Each arm completed exactly **9,956,800,000 candidates**. Every arm emitted
the same **1,201 sorted, OpenSSL-gated hit lines**, SHA256
`58d9fbc1a44e241bc821a4876dcce4b8e1921592090f2bc58de2c070a214fd5b`.

| Run order | Source | Fixed-work seconds |
|---|---|---:|
| A1 | PR #976 control | 11.971310 |
| B1 | PR #976 + RAW only | 11.987853 |
| B2 | PR #976 + RAW only | 11.987452 |
| A2 | PR #976 control | 12.021120 |

The control mean was **11.996215 s**, RAW mean **11.9876525 s**. The
arithmetic pooled effect is **+0.07143% completed-work throughput**. A1/B1
favored the control; B2/A2 favored RAW. Clock and order drift at this scale
is unresolved. The result is too small to establish a durable gain. An
earlier related raw-finish source family was also near noise in local
experiments; PR #849's full combination was negative on its different
stage-0 base. These historical observations add caution, not a direct
measurement of this executable on the ranked runner.

The scratch report, source-port script, static cubins, four output logs, hit
hashes, and machine summary are retained outside this submission tree at
`/tmp/qsb-pin-pr993-screen/`. They are not submitted as source, binaries,
result files, or problem inputs. The production tree uses the unmodified
contest loop and has no finite stop or diagnostic timing code.

## PR #1002 K32 exactness and incremental evidence

Public PR #1002 by @fkiene supplies the four `GPUMath.h` switches in this
package. For the secp256k1 pseudo-Mersenne reduction, `K=2^32+977` has a
low 32-bit half of 977 and a high half of 1. The K32 branches split the
existing selected 64-bit correction over those halves. They preserve the
same carry and borrow policy and the same stop limb as the inherited paths;
the change does not add a new approximation. `QSB_K32_ADD` is mostly inactive
under default `QSB_YOFF=1`; SUB and OFF are active in the mixed-add path.
`QSB_SAS_FRMOV` removes a source-level unpack/repack of identical words in
the fused square path; in the preceding isolated static screen, enabling it
alone left the complete cubin byte-identical. The measured difference below
isolates K32 because both A and B enabled `QSB_SAS_FRMOV`.

An independent Python differential model compared the old and K32 raw
256-bit result limbs for **256 directed edge operand pairs plus 100,000
deterministic random full-width pairs per operation**, separately for SUB,
ADD, and OFF: **300,768 comparisons, all equal**. This directly checks the
three changed arithmetic outputs, though it does not prove every possible
upstream operand distribution. On the N24 native sm_89 compile, stage-0 SASS
fell from **6,035 to 6,026** instructions and stage-2 from **4,050 to
4,034**. Stage-0 register use fell **128 to 126**, still in the same
four-resident-CTA class with 12 KiB shared; sm_52 stage-0 remained 101
registers. Both configurations had **zero spills**. Clean PR #999's sm_89
and sm_52 complete cubins matched our independent baseline byte-for-byte
before applying K32.

The same finite stop was inserted only in two scratch sources; both used the
organizer-default sm_52 compile, N24 and public seed **9,072,764**. Every
arm completed exactly **9,956,800,000 candidates** and emitted the same
**1,201 sorted, OpenSSL-gated hits**, SHA256
`58d9fbc1a44e241bc821a4876dcce4b8e1921592090f2bc58de2c070a214fd5b`.

| Run order | Source | Fixed-work seconds |
|---|---|---:|
| A1 | clean PR #999 | 11.969222 |
| B1 | PR #999 + K32 | 11.948504 |
| B2 | PR #999 + K32 | 11.956393 |
| A2 | clean PR #999 | 11.997209 |

The arithmetic pooled completed-work throughput change was **+0.257412%**;
both adjacent comparisons favored K32, **+0.173394%** and **+0.341374%**.
Only one short balanced test on one GPU and seed was run. Clock drift,
runner variation, and hit statistics still limit any official-score
prediction; the signal is smaller than the current ranked floor gap.
Reproducible scratch source, arithmetic checks, static outputs, four run
logs, and a report are kept at `/tmp/qsb-pin-pr999-k32-screen/` outside this
package. The submitted source retains the original unbounded contest loop.

## Reproduction and interpretation

The organizer-default N24 build is
`nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm`
from the pinning candidate directory. `QSB_FIN_RAWS=1` is the inherited
source default; `-DQSB_FIN_RAWS=0` restores the PR #976 finish branch.
`QSB_K32_SUB=QSB_K32_ADD=QSB_K32_OFF=1` and `QSB_SAS_FRMOV=1` are the new
defaults; the three K32 flags set to zero restore PR #999 field arithmetic.
`SOURCE-MANIFEST.json` records SHA256 and byte counts for every included
pinning source, license, and note file. Its manifest hash list is recalculated
after note edits and before committing.

The ranked score is based on independently verified hits over elapsed time,
so fixed-work throughput and official score are different measurements. The
runner's timeout `candidate_count` can extrapolate a peak observed progress
rate over elapsed time; it is not a completed-work count. We base the local
+0.07143% inherited RAW and +0.257412% incremental K32 figures solely on
their respective matched fixed candidate counts above and do not infer an
official score from either number. Public prior source credit and
the prior package's runtime files remain preserved in the Git history and
headers. The score and promotion decision require the contest's own full
run, current floor, and exact verifier.
