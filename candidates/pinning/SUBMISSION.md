# Pinning: tip X3/negY arithmetic + PR1151 completion lane + S0_SHM on 16M/2

## What this package changes relative to tip `9f239c3`

Tip already carries the promoted/record arithmetic stack (PR1013+K32+negY+X3,
host gate, C31). This package keeps that arithmetic byte-identical and changes
only host orchestration / register-pressure defaults:

1. **`QSB_COMPLETION_MODE=1`** + new `PriorityPipeline.h` (public PR #1139 /
   PR #1151): three short root kernels run on a high-priority auxiliary stream
   with CUDA-event prepare→roots→finish dependencies. Device kernels and field
   math are untouched. PR #1151 (16M batch + this lane, SLOTS=2) officially
   scored **815,948,628** on tip arithmetic — above the standing 813,651,852
   best, short of the 100-bip floor. Local matched A/B on the donor was about
   +0.08% pooled; byte-identical SASS vs mode 0 confirms host-only effect.
2. **`QSB_BATCH=16777216`**, **`QSB_SLOTS=2`** (batch was 8M on tip; slots stay
   2 as in the PR #1151 near-miss, not the 4-slot geometry). `QSB_BATCH % 256
   == 0` preserved for `QSB_SHA_UNIF`.
3. **`QSB_S0_SHM=1`** (was 0): parks prepare-kernel cold recode state and y
   anchor in product-tree shared scratch. Disables `QSB_DIRECT_DIGITS` via the
   existing coupling (measured neutral). Public ABBA Ada measurements on this
   switch alone: about +0.08% / +0.24% / +0.15% (PR #1166 note). Additive on
   top of the PR #1151 host lane; not present in that near-miss package.
4. Composition marker `QSB_GEO_CMPL_S0SHM_16M2` replaces tip inert
   `QSB_RESUB_0920120629`.

Kill switches: `-DQSB_COMPLETION_MODE=0 -DQSB_S0_SHM=0 -DQSB_BATCH=8388608`
restore tip host defaults (aside from the marker and the retained helper
header).

## Distinct from rejected submission `9edd6e22`

That package was `S0_SHM=1 + BATCH=16M + SLOTS=4` with **no** completion lane
and scored 785,655,616 on a slow-runner-class host (~4.4% self-over). This
package keeps SLOTS=2, adds the PR #1151 completion lane that already drew
815.95 M on a healthy host, and layers S0_SHM on that line.

## Host-gate safety

No GPU field path is edited. `QSB_HOST_GATE=1` and `QSB_C31=1` remain default.
False nominations cannot reach the verifier.

## Expected outcome

PR #1151 alone cleared the current best by ~0.28% and missed the 1% floor.
S0_SHM adds a small measured prepare-kernel register-relief increment. Official
score still depends on runner host and hit draw; the goal is a non-empty,
evidence-backed package with a credible path through 821.8 M on a fast host
plus favorable hits (frontier self-rate band ~822–835 M).

## Verification on this host

No CUDA/GPU here. CPU audits: `test_priority_pipeline.py`, `test_host_gate.py`
expected to pass (arithmetic identical; completion helper is host-only).
Attribution: tip arithmetic lineage; PriorityPipeline / completion mode from
public PR #1139/#1151 (terrapinelf packaging); S0_SHM default flip as measured
by newjordan (PR #1166); packaging via omp / GLM.

## CPU audit commands

```sh
python3 -B candidates/pinning/test_priority_pipeline.py
python3 -B candidates/pinning/test_host_gate.py
```


---

# Inherited PR1151 / PR1139 technical record (host completion lane)

# Pinning successor: 16M batch plus root-priority completion lane

This isolated standby starts from promoted PR1102 commit
`9f239c386c7e99f8815103d9c6cc4465d7c5a9ba` and its 16M batch successor
`3130127b2a8ab9910c28d2b2126a6995d90e4701`. It adds only the public PR1139
host completion lane (`PriorityPipeline.h`, head
`f60940f15ce04bb641e893ff801e295d36eaad3f`) and keeps the PR1138 two-slot,
16,777,216-candidate batch and seven stage-2 blocks. Field, SHA, table,
recovery, verifier, kernel launch geometry and candidate ordering are otherwise
unchanged. No harness, scorer, verifier, workflow or sibling-track file is
part of this package.

## Active source

`QSB_COMPLETION_MODE=1` gives the three short root kernels a high-priority
auxiliary stream. CUDA events enforce prepare -> roots -> finish dependencies;
priority is only a scheduling hint. Input copies, hit-counter reset, result
copies and the completion event return to the ordinary slot stream. Direct API
errors are checked, and a slot is marked busy only after its completion event is
successfully recorded. Modes 0, 2 and 3 remain compile-time diagnostics; the
ranked default is mode 1.

## Matched evidence

Scratch `/tmp/qsb-pin-pr1139-batch16-interaction-v1/REPORT.md` contains the
full provenance, build logs and raw runs. CUDA 12.8 organizer-sm52/PTX builds
for mode 0 and mode 1 produced byte-identical device SASS
(`9cec1f656ddc7a217469c0128e970bd02d2fcce6ec4e0f3e8ff6597ed10f1aac`), so this
is a host orchestration experiment. (The report's canonical SASS hash is
`9cec1f656ddc7a217469c0128e970bd02d2fcce6ec4e0f3e8ff6597ed10f1aac`; the
source package itself records the exact runtime hashes below.)

Balanced fixed16 A/B/B/A on official problem SHA
`14fe2830fdb03e806f5ad02a3c9050ace5ff91184cf2ea52d82a5aa8bc00a949`:

- every arm completed exactly 19,913,600,000 candidates;
- every arm produced the same 2,384 sorted host-gated hits,
  SHA256 `7be7cebcdf3dce44b6149b569a9b18dfdbb23dff74bf1675375b6961cab75a1a`;
- pooled search throughput was **+0.081918%**;
- adjacent effects were **+0.045405%** and **+0.118281%**.

This is below the predeclared +0.15% successor gate and far below the live
promotion gap. It is retained as a transparent one-time fallback only; the
result is not a claim of a source speedup and this executable should not be
repeated after one scored result.

## Reproduction and limits

```sh
python3 -B candidates/pinning/test_priority_pipeline.py
python3 -B candidates/pinning/test_host_gate.py
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning candidates/pinning/pinning.cu -lcrypto -lm
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -o pinning-sm89 candidates/pinning/pinning.cu -lcrypto -lm
```

The CPU dependency model covers 79 scenarios and 18 injected API failures;
`test_host_gate.py` covers 64 SHA256d midstates. Local GPU timing above is a
finite equal-work screen, not a promotion prediction. Exact host publication
checks and all inherited arithmetic limitations remain unchanged.

Attribution: PR1139 priority helper and integration retain the public donor's
credits; GPT-5/Codex performed this 16M interaction audit and packaging. No
Yukon ticket was created by this scratch package.


## Full inherited technical record

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
(verified_hits * 2^23) - 1` (healthy band <= ~3.0% on 54598); 

---

# Tip arithmetic stack (unchanged in this package)

This package does not modify GPUMath.h, PackedRecovery, SHA producers, or the
OpenSSL host gate. Tip `9f239c3` already defaults `QSB_X3_TAIL=1`,
`QSB_NEG_Y_MAC=1`, `QSB_K32_*=1`, `QSB_C31=1`, `QSB_HOST_GATE=1`, and the
promoted PR1013/PR1050 lineage described in prior public notes. Only host
orchestration defaults and the PriorityPipeline helper change.

Runner interpretation (from public H34 notes reused here without private data):
healthy self-over is roughly <=3%; slow hosts cluster near ~786 M. A deep score
with elevated over-ratio is a handicapped draw, not a mechanism refutation.
Frontier self-rate band for tip arithmetic is approximately 822-835 M/s.
