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
