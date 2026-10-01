# Session ledger — 2026-09-30, pinning track

## Environment facts (verified this session, not assumed)

- No NVIDIA GPU and no CUDA toolkit on this host. Confirmed by the task brief.
  I also have **no shell tool** in this session: read/write/edit/grep/glob only.
  I cannot run `git`, `nvcc`, `ptxas`, or any Python oracle. Every correctness
  argument below is therefore by **source reading**, not by execution. I must
  say so rather than imply I ran anything.
- Board is empty. `main` is the only participant. I am the only worker.

## FINDING 1 — the brief's "best official 882,096,418 (v20)" is STALE.

The source tree on disk is **not** the v20 package. Verified by reading
`candidates/pinning/pinning.cu`:

- `#define QSB_GLV11 1` (line 161) — eleven-term fixed-base geometry.
- `#define QSB_PMIX12 65536`, `QSB_PMIX12_WARP 0`, `QSB_PMIX12_N 1` (183/201/211).
- `#define QSB_BATCH 4194304`, `#define QSB_SLOTS 3` (240/436).
- `#define QSB_SLOTPIPE 1`, `QSB_COMPLETION_MODE 1` (410/429).
- `QSB_PAIR_ORD 1` (348), `QSB_CHAIN_PIPE 1` (345), `QSB_SYM_FINISH 1` (378).
- Table geometry guard (666): `(QSB_GLV11?22688113472ULL: ... :1465193024ULL)`
  i.e. GLV11 table = 22,688,113,472 B (21.1 GiB); GLV12 = 1,465,193,024 B.
- Directory contains `cpu_cogrind*.h`, `cg_ec_scalar.h`, `cg_fe4.h`, `cg_sha.h`,
  `cg_table.h`, `cg_v26asm.h`, `qsb_carrier_sm89.h`, `QsbCarrier.h`,
  `build_carrier.sh` — the **CPU v2 co-grinder** and **native sm_89 carrier**
  packages.

This is the **GLV12-GPU + CPU-v2-co-grind** tree described in
`SUBMISSION-GLV12-V2.md` / `SUBMISSION-GLV12-V2-REDRAW.md` (Sept 26), not v20
(`QSB_GLV10` era, 12-term BIGTBL, `QSB_SLOTS 2`, no `cg_*` headers).
`ITERATIONS.md` stops at v23 (Sept 24) and never describes it.
`SOURCE-MANIFEST.json` is itself stale: it still claims
`base_commit b59484345...`, the ~826 M GLV14 promotion.

**Consequence:** the 15.4% gap in the brief is measured from v20, which is two
generations of mechanism behind what is in the working tree. Treating 15.4% as
the real engineering gap would badly misstate the problem.

## FINDING 2 — the real gap is much smaller than the brief states.

Public promoted scores for this track, read out of the track's own late notes:

| score | source | where recorded |
|---|---|---|
| 881,273,403 | anamdongparkjinhyeong pr1259 (GLV12 dense table) | SUBMISSION-v27.md |
| 904,971,814 | "F" package | SUBMISSION-CODEX-20260925-G3.md:15 |
| 948,943,797 | fkiene `3b423554` / `4f0f50e` — **eleven-term geometry**, native sm_89 carrier, gather-pipelined pair-ordinate chain | SUBMISSION-GLV12-V2.md:7 |
| 957,888,461 | redraw `4dc24cf6`, rejected vs floor 958,433,235 | SUBMISSION-GLV12-V2.md:64 |
| 960,830,125 | fkiene `ff524fd9` (GLV12 GPU + W16 CPU), promoted; floor 970,438,427 | SUBMISSION-GLV12-V2-REDRAW.md:23 |
| 936,505,399 | our own GLV12+CPUv2 `02b0141a`, r3 runner, rejected | SUBMISSION-GLV12-V2-REDRAW.md:23 |
| **1,008,206,828** | **current record** (brief) | brief |

The tree in hand is the GLV12 GPU of `ff524fd9` plus ercumentyildirim's CPU v2
engine. The promoted sibling of the *same GPU source* scored 960,830,125.
So the true gap from my in-hand GPU source to the record is:

    1,008,206,828 / 960,830,125 = 1.0493   ->  +4.93%

not +15.4%. Against the promotion floor 1,018,288,896 that is +5.98% over the
in-hand promoted-equivalent GPU rate. Note the CPU co-grinder adds only
~2.6-5.0 M verified candidates/s — about 0.3-0.5% — and cannot be the lever.

**This does not mean the task is easy.** It means the honest target is
~+6% on the GPU hot loop, and it must be stated against the correct baseline.

## FINDING 3 — the lever the brief calls "still open" is already CLOSED.

The brief and `DEAD-ENDS.md` list as open: "the exact host publication gate +
C31 on top of PR #743 + carry62" and "the 16M batch plus root-priority
completion lane". Both are already in this tree and both were scored long ago:

- `QSB_HOST_GATE 1` (pinning.cu:41), `QSB_C31 1` (pinning.cu:53) — present.
- `QSB_COMPLETION_MODE 1` (pinning.cu:429), `PriorityPipeline.h` present —
  present, and this tree uses `QSB_BATCH 4194304`, not the 16M standby.
- The whole `NEXT-OPTIMIZATIONS.md` / `DEAD-ENDS.md` "still open" list is
  written against the 778 M frontier and is two months stale.

The one genuinely-unexplored prize named anywhere in this track's notes —
terrapinelf's "L2-resident hot set with the 11-add chain intact, ~+7.5%" —
**is the geometry this tree already runs**: `QSB_GLV11 1`, eleven-term,
21.1 GiB table, promoted at 948,943,797. It is not a lever; it is the
baseline. Anyone reading only `DEAD-ENDS.md` would propose it as new work.

## Ledger

| # | item | status |
|---|---|---|
| 1 | env/tooling constraints verified | DONE |
| 2 | stale-brief finding (tree != v20) | DONE |
| 3 | real-gap recomputation vs record | DONE |
| 4 | "still open" list is stale; big prize already in baseline | DONE |
| 5 | read stage-0 hot loop / geometry / table build | OPEN |
| 6 | decide: one structural idea, or honest negative | OPEN |
