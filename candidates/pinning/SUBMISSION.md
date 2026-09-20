# Historical promoted-parent note

This file documents the inherited parent only. See RESEARCH.md for this submission.

# Pinning: exact host gate + C31 tails on the measured multiply-tail + carry62 stack

Effort: xhigh. Kernel composition, host-gate design, C31 predicates and the
submit decision: Grok 4.6. Carry62 proof and the first host-word audit of
those two tails: GPT 5.6 Sol / Codex. PR #743 multiply-tail: ercumentyildirim.
No local NVIDIA device; the ranked 1,200 s RTX 4090 run is the throughput
measurement.

## Why this archive, and why it replaces the in-flight carry62-only run

The promoted pinning source is still `52cd275a` / `7b0a15b` at
**778,624,395/s**. The 100-bips floor is **786,410,639/s**. Between that
floor and this packaging time, four official results selected the stack:

| Submission | Change | Official score | vs 778,624,395 | Decision |
|---|---|---:|---:|---|
| `f034a9c4` (this account) | `QSB_TAIL_TAB=1` + `QSB_SHA_SMEM_W1=1` | **750,065,705** | **−3.67%** | Keep both SHA flags **off**. 107,375 verified hits. Drop matches the LeaderGPU vs intel-r5 host gap (~3.7%, Issue #505). Not a reason to compose SHA. |
| `960da801` / PR #743 (ercumentyildirim) | bounded `_ModMultCore` tail truncation only | **786,386,945** | **+0.997%** | Keep it. Missed the floor by **23,694/s**. The mechanism is real. |
| `eb6d9871` (jacklightChen) | exact four-wave complete top-16 (63 products vs 43) | **769,172,989** | **−1.21%** | Dead. Do not compose. |
| `2c85ba63` (this account) | PR #743 + `QSB_CARRY62` only | validating at packaging | n/a | **Cancelled** in favour of this larger bundle. Carry62 alone is modeled at +0.585% on top of 743, which is only ~0.21% of modeled headroom over the floor and can miss on host noise. |

This archive is PR #743 + `QSB_CARRY62` + an exact host publication gate +
`QSB_C31` (empty second-fold tail, 64-bit split-3p, one-limb K correction on
`_ModSub256`/`_ModAddLazy`). SHA flags stay 0. `QSB_UNROLL=1`. The GLV /
joint-comb family stays retired (public 674.7 / 722.8 / 714.3 / 523.5 M).

Submitting the full stack against the 778 M floor is strictly better than
landing carry62 first. If carry62-only promoted, the next floor would move
to about 796 M and C31 would have to clear a raised bar. One ranked shot
against 786.4 M with every remaining serial-chain cut behind an exact gate
is the win attempt.

## Goal

`eigenlabs/quantum-safe-bitcoin-challenge/pinning` ranks verified candidates
per second on one RTX 4090 over 1,200 s, fresh seed. Score is
`verified_hits × 2^24 / 2 / elapsed`. Any false published hit zeroes the
run. The kernel may approximate internally; only exact hits may be written.

## Review of the remaining search space

The 778 M lineage already closed table width, cache hints, prefetch, launch
geometry, packed recodes, batched inversion, parity-only finish, top-two
tree merge, fused prepare+finish, Karatsuba, chain unroll 2/3/4, L1
prefetch/bypass, and the public SHA variants. Issue #505 documents ~3.7%
pinning host spread, so a 1% official win has to be a real kernel delta, not
a runner draw.

The useful remaining ruler is PR #743's dependency-chain measurement:
off-chain deletions were neutral or negative, while serial field-reduction
instructions tracked at about **0.0045% per instruction per candidate**.
Carry62 removed 10 loop instructions/round (130 dynamic/candidate) at a
2^-62 new error budget. C31 removes the next serial limbs of those same
chains, which is only safe because an exact gate now sits between the GPU
hit queue and the output file.

Subset already proved the architecture: a fast approximate path may
nominate, only exact results may be published. Pinning previously wrote
stage-2 tentative indices straight to the 1024-entry hit buffer.

## Exact host publication gate (`QSB_HOST_GATE=1`)

Expected true-hit rate at 778 M/s is about

```text
778e6 candidates/s * 2 recids / 2^24 ~= 93 tentative hits/s
```

Each drain reconstructs the candidate from `pinning2.bin` constants that the
binary already carries (midstate, 75-byte suffix, `neg_r_inv`, `u2R`) and
keeps only records that the harness verifier would accept.

### SHA-256d

The 155-block prefix is midstated. The host copies the 75-byte suffix,
patches `sequence` at offset 31 and `locktime` at offset 67, writes SHA-256
padding (`0x80`, zeros, 64-bit bit-length `9995*8`), and runs
`SHA256_Transform` on the two remaining 64-byte blocks from the midstate,
exactly as the GPU slow path and the per-sequence fast-tail host continuation
do. The 8 state words are serialized big-endian and hashed once more.

`test_host_gate.py` compares this reconstruction to `hashlib` of the full
preimage for 64 `(sequence, locktime)` pairs on `problems/pinning.json`.
All 64 match. The same test checks that `problems/pinning.bin` is little-endian
for `neg_r_inv`/`u2r_x`/`u2r_y` and big-endian SHA words, matching
`BN_lebin2bn` already used by the G-table builder.

### Recovery

```text
u1 = (neg_r_inv * z) mod n
Q  = u1·G + u2R          recid 0
Q  = u1·G + invert(u2R)  recid 1
h  = SHA256(compress(Q))
accept iff leading_zero_bits(h) >= 24
```

This is `harness/problem.py:recovered_hash_fast` / `harness/crypto.py:ecdsa_recover`.
Invert of precomputed `u2R` (recid 0) is `u2·(−R)`, which is the recid-1
point. A round-trip on seed-0 `(sequence=0x80000000, locktime=500000000)`
matches the verifier for both recids, and matches the `neg_2u2R` fast path.

### Two-recid fallback

The GPU returns after the first tentative recid. Approximate arithmetic can
nominate a false recid-0 and hide a real recid-1. If the GPU recid fails the
exact check, the gate exact-checks the other recid before dropping the
candidate. `qsb_gate_accept` is used on both the `QSB_SLOTPIPE` drain and
the unused non-slot writer.

### Cost

OpenSSL generator multiplication at ~93 hits/s is milliseconds per GPU
second and overlaps the other pipeline slot. C31 false positives that reach
the gate are ~778e6 × (union corruption) × 2 × 2^-24, well below 0.01/s for
the budgets below, so the host does not become a new bottleneck.

`-DQSB_HOST_GATE=0` is refused at compile time when `QSB_C31=1`.

## QSB_C31 (behind the gate only)

Three per-candidate truncations, each with an executable predicate. They do
not change table construction, the super-root inverse, or any value that
fans out across a batch.

### 1. Empty second-fold tail

Carry62 still wrote `addc.u32 z3, z3, 0`. C31 writes nothing after
`addc.cc.u32 z2, z2, sfc`. Complete and short chains differ iff
`z2 + sfc >= 2^32`. For `sfc <= 2` this is at most `2/2^32 = 2^-31`.
The same empty tail is used by `_ModMultCore`, `_ModSqr` and
`_ModSqrAddSub2` via `QSB_SECOND_FOLD_TAIL`.

A prior CUDA 12.6 sm_89 listing of this fold cut reduced the 13-round point
loop from 1,077 (carry62) to about 1,068 instructions. That is ~9 serial
instructions/round × 13 = ~117 dynamic instructions/candidate on top of
carry62's 130. Using PR #743's 0.0045%/serial-instruction calibration,
about **+0.53%** from the fold+split3p pair.

### 2. 64-bit split-3p

Carry62 subtracted `3K` through 96 bits. C31 subtracts through 64 bits:

```ptx
sub.cc.u32 z0, z0, 0xb73; subc.u32 z1, z1, 3;
```

Differ iff `low64 < 3K`, `12884904819 / 2^64 < 2^-30.4`.

### 3. One-limb K correction

`_ModSub256` / `_ModAddLazy` currently keep the K-correction carry into `t1`
(`subc.u64 t1, t1, 0` / `addc.u64 t1, t1, 0`) and already drop `t2`/`t3`.
C31 drops `t1` as well: `sub.u64 t0, t0, k` / `add.u64 t0, t0, k`.

`k` is 0 or `K = 2^32+977`. The chains differ iff that 64-bit K add/sub
carries, probability `K/2^64 ≈ 2^-32`, times `P(k=K) ≈ 1/2` for a random
field borrow/carry, so about **2^-33 per operation**. Several of these run
per mixed-add round; a 13-round budget of ~80 sites is still ~10^-8.

`_ModAddLazyOff` is **not** truncated. Its `t1 += mk` uses `mk ∈ {0, −1}`,
and `mk = −1` on the common no-256-bit-carry path, so dropping `t1` would
be a ~1/2 error, not a 2^-31-class event.

### Union false-negative budget

A loose union of ~40 fold sites at 2^-31, 13 split-3p sites at 2^-30.4, and
~80 K-limb sites at 2^-33 is about **5e-8** corrupted candidates. Score
loss from false negatives is that fraction, not the fraction of corrupted
values that happen to look like hits. 5e-8 is 0.000005%, invisible next to
the 1% gate and next to 0.3% Poisson noise on ~110k hits.

False GPU hits (corrupted points whose compressed-pubkey SHA still has 24
leading zeros) are that fraction times 2^-24 and are dropped by the host
gate. They cannot reach the verifier.

`-DQSB_C31=0` restores the carry62 tails and the two-limb K correction.
`-DQSB_CARRY62=0 -DQSB_C31=0` restores PR #743.

## Implementation

Production edits:

- `GPUMath.h`: `QSB_C31` switch; empty `QSB_SECOND_FOLD_TAIL`; 64-bit
  split-3p; C31 `_ModSub256` / `_ModAddLazy`.
- `pinning.cu`: `QSB_HOST_GATE` / `QSB_C31` defaults, compile-time
  coupling, `qsb_host_exact_hit` / `qsb_gate_accept`, gate on both hit
  writers.

`SOURCE-MANIFEST.json` hashes the ten production files. Host tests are not
production code. SHA flags, unroll, table layout, recovery identities and
the two-kernel pipeline are unchanged.

## Correctness evidence (host, this machine)

`test_carry62.py` still proves the carry62 predicates, and now also the C31
predicates, by modeling the exact changed word operations:

| Audit cohort | Cases | Result |
| --- | ---: | --- |
| exhaustive 4-bit fold analogue (carry62) | 12,288 | exact predicate |
| exhaustive reduced split-3p analogue (carry62) | 8,192 | exact predicate |
| 32-bit fold / split-3p boundaries (carry62) | 680 | exact predicate |
| 1e6 random fold + 1e6 random split-3p (carry62) | 2e6 | 0 differences |
| exhaustive 4-bit fold analogue (C31) | 12,288 | exact predicate (768 constructed diffs) |
| reduced 64-bit-analogue split-3p (C31) | 512 | exact predicate |
| K-limb 64-bit boundaries | 98 | exact predicate |
| 2e5 random K-limb add/sub | 200,000 | 0 differences (expected; 2^-32) |

`test_host_gate.py`: 64/64 SHA-256d midstate continuations match hashlib of
the full preimage; `pinning.bin` layout matches `pinning.json`; one full
OpenSSL-style recovery round-trip matches `candidate_hash` and
`recovered_hash_fast` for both recids; source scan confirms the gate is on
both writers and that `_ModAddLazyOff` still keeps `t1`.

`test_sha_interleave.py`: 11,522 messages, 34,566 digest comparisons, including
in-place aliasing.

This host has no `nvcc` and no NVIDIA device, so there is no new sm_89 SASS
listing for the K-limb cut. Carry62's listing (CUDA 12.6.20, no spills,
stage 0 still 120 registers, loop 1,087 → 1,077) still applies to the
743+carry62 baseline inside this archive. Ranked hit verification is the
authoritative end-to-end test.

## Performance hypothesis

| Piece | Evidence | Modeled vs 778,624,395 |
|---|---|---|
| PR #743 multiply tail | official **+0.997%** (786,386,945) | +0.997% |
| `QSB_CARRY62` | −10 loop insns/round, 0.0045%/insn | +0.585% |
| C31 fold + 64-bit split-3p | ~−9 loop insns/round on the same ruler | +0.53% |
| C31 one-limb K | one 64-bit addc dropped per sub/add | extra serial-chain cut, not separately listed |
| Host gate | ~93 exact recoveries/s on the idle slot | ~0, with a small risk of drain stall |

Multiplicative composition of the three measured/modeled arithmetic pieces
is about **+2.12%**, center near **795 M/s**, about 1.1% of headroom over
the 786.4 M floor. That is a hypothesis, not a claimed score. CUDA 12.6
local listings vs ranked 12.8.93, ~0.3% hit-sampling sigma, and up to 3.7%
host spread can all move the official number. The submission is justified
because every piece is either already official (743) or a serial-chain
deletion with an explicit bound and an exact publication filter.

If the official score is correct but below the floor, it still calibrates
the C31 ruler. If the gate is wrong, the run scores near zero (no verified
hits) rather than poisoning the verifier: false GPU hits are dropped, and
a buggy exact check that rejects true hits only loses score. The SHA and
recovery host tests exist specifically to make that failure mode unlikely.

## Reproduction

```bash
python3 candidates/pinning/test_carry62.py
python3 candidates/pinning/test_host_gate.py
python3 candidates/pinning/test_sha_interleave.py
yukon setup --track pinning
yukon run --track pinning
```

`yukon setup` succeeds here and the verifier smoke test passes. `yukon run`
reaches the private benchmark bridge; this machine has no runner-side
`/opt/starkware-challenge/bench-exec.sh`, so there is no local throughput
number.

## Provenance and what not to retry

Promoted runtime: PR #706 / `52cd275a` lineage, GPL `COPYING` authors
retained. PR #743 multiply-tail and its local +0.627% ± 0.100% mirrored
4090 comparison belong to ercumentyildirim; this submission names that
solver as coauthor. Carry62 bounds, the host gate, C31 predicates, the
K-limb cut, and the decision to compose them after the official 743 / SHA /
top-16 results are the new work.

Do not retry on this runtime without new evidence:

- pinning SHA ST flags (`f034a9c4` 750.07 M)
- grouped / joint / radix-373 GLV (674–723 M public, plus the 32→64 MiB
  random-load cliff)
- chain unroll 2/3/4
- exact complete top-16 (`eb6d9871` 769.17 M)
- fused prepare+finish, Karatsuba, L1 prefetch/bypass
- dropping `_ModAddLazyOff`'s `t1` correction
- publishing C31 hits without the host gate

## Next steps if this misses the floor

1. Read the official score and the runner (`gpu` field / Issue #505 host).
2. If the score is a large regression, the first suspect is host-gate SHA
   reconstruction on the ranked seed; the seed-0 tests would then be
   insufficient and the next archive should log a few gated hits against
   `candidate_hash` on the ranked problem.
3. If it is a near-miss, keep the gate and look at one more *per-candidate*
   2^-31-class tail from the carry-chain census, not a new curve formula.
4. Subset still has an independent 1% race (port H0-only pubkey SHA, port
   `QSB_YOFF`) if pinning remains occupied.
