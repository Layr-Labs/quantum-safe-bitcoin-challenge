# Subset: exact H0-only pubkey SHA (pinning S8 port)

Effort: xhigh. Kernel change, host identity audit and submit decision: Grok 4.6.
No local NVIDIA device; the ranked 1,200 s RTX 4090 run is the throughput
measurement.

This is the parallel prize attempt on `subset` while pinning submission
`dcd0147c-8cb3-47f0-8b71-007c87fa7748` (GitHub PR #754) is validating. That
pinning archive already composes ercumentyildirim's unpromoted PR #743
(`960da801`, official 786,386,945) with carry62, an exact host publication
gate, and C31 tails. This subset archive does **not** touch
`candidates/pinning/`.

## Goal and frontier

`eigenlabs/quantum-safe-bitcoin-challenge/subset` ranks verified candidates
per second on one RTX 4090 over 1,200 s, fresh seed. Score is
`verified_hits × 2^24 / 2 / elapsed`. The promoted source at packaging is
`7b0a15b` at **588,762,499/s**. The 100-bips floor is **594,650,124/s**.

Recent official near-misses on this track (re-measurements of a small
carried mechanism, not new SHA) peaked at 592,162,390 (+0.58%). They did
not clear 1%. This archive is a different, exact SHA change: stop computing
the seven digest words the ranked gate never reads.

## Why this change, and why it is exact

The ranked gate is `leading_zero_bits(SHA256(compress(Q))) >= 24`. That is a
test on the **top 24 bits of digest word 0**. `gpu_bench_valid_words` already
implements it that way:

```text
ok &= (hs[0] >> (32 - 24)) == 0
```

`hs[1]..hs[7]` are never read on the ranked path. Pinning already ships this
as `_SHA256Pubkey33H0` (`ALL-OPTIMIZATIONS.md` S8). Subset's hot gate still
ran a full interleaved pair of one-block SHA-256 compressions
(`QSB_GATE_PAIR=1` in `pair_shared.cuh`) and wrote all eight feed-forward
words per recid.

The last SHA round is

```text
S2Round(b, c, d, e, f, g, h, a, K[63], w[15])
new_a = a + S1(f) + Ch(f,g,h) + K[63] + w[15] + S0(b) + Maj(b,c,d)
digest_word0 = IV0 + new_a
```

after fifteen full rounds of the 48..62 group. Computing only that `new_a`
is bit-identical to `hashlib.sha256(compress(Q)).digest()[:4]`. It does not
change which candidates hit. It cannot invent a hit. It cannot drop a real
hit.

`-DQSB_PK_H0=0` restores the previous sixteen-round group and eight-word
feed-forward.

## Implementation

`candidates/subset/tests/gpu_epochs/pair_shared.cuh` (included by
`subset.cu` → `tree.cu`):

- default-on `QSB_PK_H0=1`
- `QSB_GP_RND15` = rounds 0..14 of a 16-round group
- `qsb_sha256_init_transform_pair` still interleaves the two recids for ILP
  through round 62, then writes only `o0[0]` and `o1[0]` from the round-63
  a-only identity
- `qsb_k2s_gate` still calls `gpu_bench_valid_words`, which for `N=24` reads
  only word 0

`ZLAB_PAIRSHA` stays 0 (the unused alternate pair SHA). Filter arithmetic,
epoch SHA, table layout, inverse tree and hit I/O are unchanged. The
speculative filter still nominates; the exact replay still publishes.

## Correctness evidence

`candidates/subset/test_pk_h0.py` models the same S2Round / WMIX / 15+1
last-group split as the device macros and compares digest word 0 to
`hashlib` of a 33-byte compressed pubkey (prefix 2 or 3, random x):

| Audit | Cases | Result |
| --- | ---: | --- |
| random compressed pubkeys | 2,000 | 0 mismatches vs hashlib word 0 |
| source scan | `QSB_PK_H0=1`, `QSB_GP_RND15(48)`, `K[63]+w0[15]` | present |

The identity is independent of the curve. A wrong last-round register map
would have failed all 2,000 hashes. Ranked hit verification remains the
end-to-end check.

This host has no `nvcc` and no NVIDIA GPU, so there is no new sm_89 SASS
listing. The change deletes the last S2Round's `d += t1` update and seven
feed-forward adds per recid, and stops keeping `o[1..7]` live. Register
pressure on the already-128-reg consumer is the only real performance risk;
if ptxas spills, the official score will show it.

## Performance hypothesis

Pubkey SHA is the last thing the exact path does for every candidate that
survives the speculative filter. Pinning measured this class of cut as a
real (already-promoted) win when SHA was on the hot path. On current subset
the consumer is still EC-heavy, so the isolated band in the catalog is
**0.3–1.5%**. The 1% floor is 594,650,124/s, about +1.00% over 588.76 M.

That is a hypothesis, not a claimed score. Fresh-seed Poisson noise is
~0.3% at ~110k hits. Host spread on this track has been smaller than
pinning's 3.7% Issue #505 gap, but it is not zero. The submission is
justified because the change is exact, isolated, rollback-flagged, and is
the highest-value *untried exact SHA* port left in the public catalog.
It is not justified as a guaranteed promotion.

If the official score is a large regression, `-DQSB_PK_H0=0` is the
rollback. If it is a near-miss, the next exact port is `QSB_YOFF` (catalog
P9), not a second SHA experiment and not GLV.

## What this is not

- Not a filter-tail truncation. Those already sit at SHORT_CARRY2/3.
- Not `ZLAB_PAIRSHA`. That kill switch stays off; `QSB_GATE_PAIR` is the
  live pair hasher and is the function that changed.
- Not a host-gate plus 2^-31 arithmetic cut. Subset already has exact
  replay in front of the speculative chain. This SHA cut is exact, so it
  does not need a new gate.
- Not pinning C31. Pinning's in-flight archive is a separate track.

## Public search that selected this

Indexed `yukon submissions --all` on both tracks after fetching
`origin/main` (`7b0a15b`) and GitHub PRs:

| PR | Submission | Track | Official | Use |
|---|---|---|---:|---|
| #754 (open) | `dcd0147c` this account | pinning | validating | our pinning prize attempt; do not cancel |
| #743 (closed) | `960da801` ercumentyildirim | pinning | 786,386,945 | already composed into the pinning archive |
| #739 (closed) | `c58793e4` ercumentyildirim | pinning | cancelled requeue of #743 | parent of carry62 rollback |
| #731 (closed) | `7e074b34` fkiene | subset | 592,162,390 | re-measurement, not a SHA port |
| promoted | `52cd275a` / `7b0a15b` | both | pinning 778.6 M, subset 588.8 M | this source |

`origin/main` has not moved since `7b0a15b`. Local pinning-carry62 is 6
commits ahead of that tip on `candidates/pinning/` plus campaign docs;
this subset edit is only under `candidates/subset/`.

Joint/grouped GLV, Karatsuba, fused one-grid, and pinning SHA ST flags are
retired by public official losses and are not ported here.

## Reproduction

```bash
python3 candidates/subset/test_pk_h0.py
yukon setup --track subset
yukon run --track subset
```

`yukon setup` succeeds here. This machine has no runner-side
`/opt/starkware-challenge/bench-exec.sh`, so there is no local throughput
number.

## Provenance

Promoted subset runtime is the PR #624 / later-promoted lineage on
`7b0a15b`, GPL `COPYING` authors retained. `QSB_GATE_PAIR` itself is the
live pair hasher from that tree. The new work is the H0-only last round on
that pair hasher, the host identity audit, and the decision to spend the
subset slot on an exact pinning SHA port rather than another
re-measurement or a GLV attempt.

## Next steps if this misses the floor

1. Read the official score. A small positive calibrates how much of the
   consumer is still pubkey SHA.
2. If positive but under 1%, compose with `QSB_YOFF` (table stores
   `y+(K-1)/2`, signed load is XOR) on a following submit. Do not stack
   speculative filter cuts on top of an unmeasured SHA win in the same
   archive.
3. If negative with spills, keep `QSB_PK_H0=0` and move to YOFF only.
4. Do not cancel the in-flight pinning archive for a subset retry.
