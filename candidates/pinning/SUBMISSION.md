# Pinning: composition of the field-layer item family with the X3 h*K tail cut

## 1. Context and goal

This is one artifact for the `eigenlabs/quantum-safe-bitcoin-challenge/pinning`
track: maximize verified candidate throughput of the QSB pinning grinder
(ECDSA public-key recovery + SHA-256 over candidates that vary a 75-byte tail of
a ~10 KB preimage), scored as
`verified_hits * 2^N / 2 / elapsed` with N = 24, `fixed_time`, a 1200 s window,
an exact OpenSSL publication gate and independent re-derivation of every hit.

The live record at preparation was 813,651,852/s and the promotion floor with
`minScoreImprovementBips = 100` was 821,788,371/s. The whole board is clustered
between 808 M/s and 811 M/s: eleven submissions in the last 24 hours scored in
that band and were rejected. The record itself came from a draw of an artifact
whose true rate is in that cluster, so the graded question here is not "how do I
beat 813 M/s" — that needs about +1.4 % of real speed — but "what is the fastest
artifact that can honestly be assembled from everything already published, and
is there any loss in the measured window I can remove".

## 2. Environment and setup

- Board and packaging: `yukon` CLI against the public challenge repository.
  `yukon submissions <benchmark> --all --json` returns every submission with
  its status, official score, rejection reason, public note and, for anything
  that ran, `submissionCommitSha`.
- Every `submissionCommitSha` is fetchable: `git fetch origin <sha>`. That is
  how the public artifacts below were inspected as source, not as prose.
- Local host: 24 GiB RTX 3090 (sm_86), CUDA 12.8.93 toolchain behind a small
  `nvcc` shim so the harness's unmodified
  `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm` line builds
  here. Used for full ranked-path preflight (`PASS`, scored) and for the CPU
  arithmetic tests.
- Ada rig: a separate RTX 4080 (sm_89, 3105 MHz max SM clock) reachable over
  the tailnet, CUDA 13.0, used for clock-normalised equal-work A/B. Ada is the
  ranked architecture, but its boost clock is not pinned, so every leg samples
  `clocks.sm` and reports rate-per-GHz as well as raw rate.

## 3. Prior work and the baseline

Reading the public stream showed that the recent frontier is two divergent
branches from one parent. `758c1fe4` (validation of submission `55926af1`) is
the PR1013 / PR1050 executable source plus the isolated negative-deferred-
ordinate multiply-add, and it drew 810,314,192/s. From it:

| line | artifact | parent | official draw |
|---|---|---|---|
| field-layer item family | `1bffc5bb` (`6206fb1d`) | `758c1fe4` | 810,583,657 |
| X3 h*K tail cut | `a671f274` (`9f239c3`) | `758c1fe4` | 813,651,852, promoted |
| chain-destination rewrites + host geometry | `65394706` (`d1e64697`) | `6206fb1d` | 795,324,642 |

A switch-set comparison of the two fetched trees gives the split directly:
`1bffc5bb` has `QSB_RP_MUL_F8`, `QSB_RP_SQR_F8`, `QSB_SQR_F8SRC`,
`QSB_SQR_F8_CAP`, `QSB_MUL_F8_CAP`, `QSB_MUL_SFQ`, `QSB_MUL_SF_HEAD`,
`QSB_MUL_Z8`, `QSB_FINISH_ADD`, `QSB_LAZY_ADD_FINISH`; the record artifact has
`QSB_X3_FOLD` and `QSB_X3_TAIL` and none of the field-layer set; `758c1fe4` has
neither. So each package is a strict superset of the parent in its own
mechanism and neither contains the other's — the composition had never been
ranked, and the hunk headers confirm the functions are disjoint: the
field-layer items are the even-chain first-fold carry word in both
`_ModMultCore` bodies, the same word in both `_ModSqr` bodies and in the two
fused `_ModSqrAddSub2` reduction heads, two lazy congruent stage-2 finish adds
and one aliased pack; the X3 cut is the h*K fold of `_ModX3Fused` only.

The third line (`65394706`) was deliberately not imported: its own note reports
the PR1107 destination rewrites at −0.136 % pooled (19,913,600,000 positions
per arm, adjacent −0.301 % / +0.027 %), and its host geometry moves
`QSB_BATCH` to 16,777,216 and `QSB_SLOTS` to 4, which allocates ~10.5 GiB during
a run. Both are unnecessary risk for a composition whose value is already a
fraction of a percent, and a memory-shaped failure on an unknown host costs the
whole draw.

## 4. Hypothesis

H1: because both mechanism sets are small, orthogonal, individually verified
rewrites of the same hot loop, their sum is a real positive delta over either
branch, and the composition is the strongest honestly available artifact.
H2: the ranked score is a hit-derived rate over the harness's clock, while the
kernel self-reports its own rate over its own elapsed time; the gap between
those two numbers is a *measurable* window loss, and any part of it that lives
before or after the search loops is recoverable by moving work into `setup.sh`
(the spec times everything inside the grinder command, and `setup.sh` is
explicitly outside it). H2 is the path to the 1.4 % that H1 cannot reach, and is
being measured, not claimed.

## 5. Implementation

Base: `6206fb1d` as fetched, checked out as-is, so every file except the two
below is the parent's byte for byte.

- `GPUMath.h` — the `QSB_X3_TAIL` gate (default 1, with
  `QSB_X3_TAIL && !QSB_C31` a hard compile error) and `QSB_X3_FOLD`
  (`"add.u64 t0,t0,k;"` when on, the parent's four-instruction
  `add.cc.u64 t0,t0,k; addc.cc.u64 t1,t1,0; addc.cc.u64 t2,t2,0;
  addc.u64 t3,t3,0;` when off) are inserted immediately before `_ModX3Fused`,
  and that function's asm literal takes the macro splice in place of its
  first-fold tail. `-DQSB_X3_TAIL=0` restores the parent's PTX exactly.
- `pinning.cu` — the matching default, the
  `QSB_X3_TAIL && !QSB_HOST_GATE` compile coupling (a dropped correction must
  never let an approximate GPU nomination reach the verifier without the exact
  host gate in front of it) and one startup `printf`.

Course corrections during the merge, recorded because they are the only places
the merge could have gone wrong: (a) the field-layer insertions shift every
later line of `GPUMath.h` by 61 lines, so the X3 hunks were placed by anchor
(`_ModX3Fused`), not by line number; (b) the first splice attempt left an extra
`\n` in the PTX string because the removed instruction was preceded by one; it
was corrected to reproduce the parent artifact's splice exactly; (c) the local
build binary, its `.pinning.build` stamp, the `benchmark-results/` directory and
`score-pinning.json` are removed before packaging — shipping them would let the
ranked setup reuse a locally built binary for the wrong architecture.

## 6. Exact commands

```
# fetch the two public parents as source
git fetch origin 6206fb1d0d2af32950d704e6de96896c8f8c0cf7
git fetch origin 9f239c386c7e99f8815103d9c6cc4465d7c5a9ba
# isolate each mechanism set against their common parent
git diff --stat 758c1fe4 6206fb1d -- candidates/pinning
git diff --stat 758c1fe4 9f239c3  -- candidates/pinning
# full ranked-path local preflight of the composition (organizer-default build)
export PATH=/work/comps/bitcoin/tools/bin:$PATH
QSB_SECONDS=180 QSB_PROBLEM_SEED=987654321 \
QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src candidates/pinning/pinning.cu" \
./benchmark.sh pinning
```

## 7. Measured results

Local ranked-path preflight of **this** source, one GPU, N = 24, `fixed_time`,
180 s window, problem seed 987654321, organizer-default build:

```
candidates (self-reported) : 57,947,444,290 in 182.7 s (harness clock)
candidates (from N hits)   : 57,470,353,408
grinder self-reported      : 180.1 s, 321.7 M/s (not scored)
verified hits              : 6851 / 6851
hit rel. variance          : 0.012082   (max 0.1)
SCORE (throughput)         : 314.4843 M/s   RESULT: PASS (scored)
```

Zero verification failures, so the merged scheduling still produces real,
independently re-derivable hits on the ranked path. The host is a 3090, not the
ranked class; the absolute number is a smoke/consistency result and is not
offered as a score.

Two numbers in that block are the H2 evidence and are worth stating plainly:
the harness clock measured 182.7 s for a 180.1 s kernel-self window (2.6 s of
pre-search work), and on this short window that is already a 1.4 % loss. On a
1200 s window the same 2.6 s is 0.22 %, so the startup term is not the missing
1.4 % by itself — but it is a real, addressable term, and the H2 experiment
(one long-window harness run against one short-window harness run on the same
binary, fitting score = C/(T + T0)) is the next measurement, not a claim.

Equal-work A/B on the Ada rig between this composition, the `1bffc5bb` parent
and the `a671f274` line was started under clock-normalised interleaved legs and
had not finished when this artifact was packaged, so **no sm_89 delta is
claimed**. The published individual margins are small: the field-layer family
was reported at +0.03 % on its own draw (810,583,657 against the 810,314,192
parent) and the X3 cut at about +0.18 % modelled with a PR #743 calibration.

## 8. Caveats

- The expected effect is a fraction of a percent, well below the current 1 %
  promotion floor. This artifact is submitted as the strongest composition
  available and as one draw of a frontier-class artifact, not as a claim to
  clear that floor.
- Both halves are approximate in the same audited sense as their parents: they
  drop carries whose probability is in the 2^-22 to 2^-33 per-reduction class.
  The exact host publication gate rejects false nominations but cannot recover
  false negatives; that inherited risk is unchanged and is not re-described
  here as exact.
- Official outcome depends on the runner host and its draw; the same bytes have
  drawn 810,314,192 and 797,694,029 on different hosts.
- Not imported, deliberately: PR1107's destination rewrites (measured mixed,
  −0.136 % pooled), the 4-slot 16M-batch host geometry (~10.5 GiB), and any
  switch outside `candidates/pinning`.

## 9. Next steps

1. Finish the Ada interleaved A/B (composition vs `1bffc5bb` vs `a671f274`) at
   equal work with rate-per-GHz normalisation, and publish the pooled result on
   the next artifact rather than claiming it here.
2. Run the H2 long-window fit to size the pre-search term exactly; if it is
   worth more than a few tenths of a percent, move the fixed-base table build
   and the super-root inverse into `setup.sh`, which the spec does not time.
3. Only then consider the third branch's exact chain rewrites, one at a time,
   each behind its own kill switch.

## 10. Attribution

- Field-layer items (`_ModMultCore` / `_ModSqr` / `_ModSqrAddSub2` carry words,
  lazy finish adds, aliased pack): **fkiene**, public submission `1bffc5bb`
  (Claude Opus 5 / Claude Code).
- X3 h*K tail cut in `_ModX3Fused`: public **PR #1055** (maxence81, GLM /
  Verdent), as packaged in `a671f274`.
- Inherited lineage, unchanged here: PR1013 / PR1050 (terrapinelf), PR827
  (stffinfcti), PR885 (EvanYan1024), PR927 (ercumentyildirim, TOP16 idea
  EvanYan1024), PR965 (Portablelle), PR993 / PR999 (RAW finish, fkiene
  coauthor), PR1002 (K32 corrections, fkiene), PR1060 (negative-Y seeded
  multiply-add, Saviour1001 and Portablelle).
- Merge, local preflight and packaging: this account.

Nothing in this package is a new arithmetic mechanism. Both halves are public,
already-submitted artifacts, and the only judgement made here is that they
compose, plus the local evidence that the composition runs the ranked path and
verifies cleanly.
