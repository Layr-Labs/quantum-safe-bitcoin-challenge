# Pinning: depth-1 gather pipeline (CHAIN_PIPE) on the 809.95M source

## What this source changes

This source starts from public commit
`0ace23d47b97a5b7816fd6e251c31a08582ac877` (submission `3c124ec`, official
**809,952,202/s** — the strongest measured public source, above the promoted
frontier `7c3609b` at **805,428,058/s**). The newly changed executable files
are `pinning.cu`, `PackedRecovery.cuh`, and `cofactor_checkpoint.h`.
`GPUMath.h` gains a comment block only; `LeafRecovery.cuh` is byte-identical
to the base. No verifier, benchmark harness, input problem, or scoring code
changes.

New work on top of the base:

- **`QSB_CHAIN_PIPE`** (`pinning.cu`, default on): depth-1 software pipeline
  of the table gathers inside the signed-digit chain. A new
  `_PointAddXYZZT_pipe` issues the gather for chunk `c+1` the moment its
  target registers die inside the current addition — `X2` after
  `U2 = X2*ZZ1`, `Yoff` after `S2 = (Y2+Yoff)*ZZZ1`. The remaining ~5M+2S of
  the addition covers the L2 latency. The caller rotates three buffers
  `(x,y,o)`; since `S2 = Y2 + Yoff` is a sum the anchor/ordinate roles
  commute and `y`/`o` swap each call, so the loop steps `c += 2`. The final
  point uses the classic `_PointAddXYZZT` tail. A
  `_Static_assert(GT_CHUNKS & 1)` guards the required parity. Unlike the
  retired `QSB_EARLY_LOAD` (dedicated registers, spilled), the prefetch
  reuses dead registers: sm_89 N24 hot kernel is **124 regs / 0 spills**
  with the pipe vs **126 regs / 0 spills** without. `-DQSB_CHAIN_PIPE=0`
  restores the classic loop.
- **`QSB_DEC_REP`** (`pinning.cu`, default on): the decoded-digit sign mask
  is the sign bit replicated to both 32-bit halves; `mov.b64 {m,m}` does it
  in one instruction instead of SHF+OR. `-DQSB_DEC_REP=0` restores it.
- **`QSB_SUM_2U`** (`PackedRecovery.cuh`, default on): the slope sum is
  formed directly as `2u` using `l + m == 2u (mod p)` — congruent, not a
  new approximation.
- **`QSB_PREP_MASK`** (`PackedRecovery.cuh`, default on): the shared `hc`
  factor is masked once for unusable lanes, so both downstream products are
  zero without separately clearing both outputs.
- **`QSB_TREE_FLAT`** (`cofactor_checkpoint.h`, default on): under
  `QSB_TOP16` the tree traversal starts and ends at compile-time-known
  sizes; the dead `count > 2` / `count == 2` / `N == 2` branches are
  removed. Value flow and synchronization are unchanged.

## Correctness evidence

CPU bitwise oracle (`research/check_chain_pipe.py`) extracting the real
device functions verbatim into a host harness:

- 1,500 fake-table cases: pipelined chain **bitwise identical** to the
  classic chain (X, Y, ZZ, ZZZ limb-for-limb).
- 120 OpenSSL-anchored cases: both chains affine-equal to the independent
  OpenSSL reference `z*A`.
- `test_carry62.py`: 200,000 random samples, 0 diffs.
- `test_host_gate.py`: pass.
- `test_sha_interleave.py`: 11,522 vectors / 34,566 digest comparisons vs
  hashlib, plus baseline-schedule and in-place alias checks.

Native build (CUDA 12.8.93): `nvcc -O3 -DQSB_ZEROS_N=24` clean for sm_89 and
organizer-default flags; `-Xptxas=-v,--warn-on-spills` reports zero spills
on every kernel. `git diff --check` clean.

## A negative finding: RAW_DIFF is invalid for secp256k1

The pending public "RAW_DIFF" idea (raw subtraction mod 2^256 in place of
`_ModSub256`) was evaluated and **rejected**. With `B = 2^256`,
`p = B - K`, `K = 2^32 + 977`: a borrow wraps by `B`, and `B == K (mod p)`
— not 0 — so the result is off by exactly `K` per borrow.
Congruence-tolerant consumers cannot repair that; the OpenSSL-anchored
oracle fails every case under it. The pending public bundle shipping it
also dropped the `R*(V-X3)` multiply; its validation failed
(`b7b11e5`/`614c5398`). The `GPUMath.h` comment records this; no raw
subtraction remains in this source.

## Inherited source and attribution

The base `0ace23d4` is @terrapinelf's public K32 package (GPT-5 / Codex):
promoted parent `94abdd0d72847b780c7d4f99da4f367e6f9f0fd1` (submission
`07009ac3`, official **797,446,582/s**, GPT 5.6 Sol / Codex), which
integrates @stffinfcti's PR #827 field schedule, @EvanYan1024's PR #885
bounded parity window, and the recovery isomorphism; plus PR #927 TOP16
cofactor traversal (@ercumentyildirim, Claude Opus 5 / Claude Code; TOP16
idea @EvanYan1024), PR #965 narrow speculative parity window
(@Portablelle, GPT 6 Astra / Codex), PR #993 RAW-only finish
(@fkiene co-author, ticket `52a058ef`), and PR #1002 K32 exact low-limb
corrections with `QSB_SAS_FRMOV` (@fkiene, ticket `dfba4ce2`). Earlier
public PR #849 and PR #866 used raw finish products. `QSB_DEC_REP` follows
a mechanism described in fkiene's public pending-bundle note. All other
inherited source credit is retained in file headers and git history.

The CHAIN_PIPE design, its port onto the QSB_YOFF offset-ordinate chain,
the RAW_DIFF falsification, the bitwise oracle, and this package were
prepared with **SWE-2 Max / Devin CLI**. We do not claim to have invented
the inherited mechanisms.

## Expectations and caveats

No local GPU is available; our own deltas are supported by structural
evidence (oracle, register counts) plus the published official measurement
of the base (809,952,202/s) and the +0.8–1.4% this mechanism measured on
our earlier pre-isomorphism base (submission `d83b6aa0`). The promotion
floor at packaging time is **813,482,339/s** (805,428,058 × 1.01). The base
sits within ~0.55% of it; official runs carry ±2–4% hit-draw variance — a
same-family source measured 809.9M and later 775.8M — so no ranked-gain
claim is made. The organizer's independent ranked run and verifier
determine the score.

`SOURCE-MANIFEST.json` records SHA256 and byte counts for every included
pinning source, license, and note file.
