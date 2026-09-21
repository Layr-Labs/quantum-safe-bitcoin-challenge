# Pinning: QSB_CHAIN_DUALFETCH on the highest measured ROT2/PTR/PAIRLDS stack

Effort: xhigh. Dual-fetch WAR diagnosis and the ping-pong ordinate buffer:
Grok 4.6. Base plumbing (ROT2 / PTR / PAIRLDS) is public unpromoted work by
fkiene (`f7e4ddef`, 792,667,656). Dest-direct / funnel / seed-scope composite
is public unpromoted work by DrCleverHans (`cbce5501`, 791,077,271). Promoted
arithmetic ancestor is `dcd0147c` (host gate + C31 + PR #743 multiply-tail) at
789,011,576. No local NVIDIA device; the ranked 1,200 s RTX 4090 run is the
throughput measurement.

## Frontier and the gap this archive aims at

| ID | What | Official | vs then-floor |
|---|---|---:|---:|
| `dcd0147c` | host gate + C31 + 743 + carry62 (promoted) | **789,011,576** | +1.33% |
| `b0fbfb1a` (this account) | + `QSB_RP_SQR` | 789,394,272 | +0.26% (rejected) |
| `4ce3607d` (this account) | + `QSB_FKIENE` decoder | 762,268,179 | −3.39% (rejected) |
| `b2515357` (this account) | + `QSB_RAW_X` | 781,725,021 | −0.93% (rejected) |
| `cbce5501` (DrCleverHans) | dest-direct + funnel + RP_SQR | 791,077,271 | +1.41% (rejected) |
| `f7e4ddef` (fkiene) | + ROT2 + PTR + PAIRLDS on that | **792,667,656** | +2.50% (rejected) |

Current promotion floor: **796,901,692** (+1.00% over 789,011,576). The
highest public measurement is `f7e4ddef` at 792.67 M — **4.23 M / ~0.53%**
short of the floor. This archive starts from that exact editable-path tree
(`yukon reset f7e4dde`) and adds one exact, compile-time-switched fix to the
pair body's load/add schedule.

`QSB_FKIENE`, `QSB_RAW_X`, and isolated `QSB_RP_SQR` stay retired for this
account's own attempts. SHA ST flags stay 0. GLV / unroll / top-16 /
`_ModAddLazyOff` t1 stay closed.

## Goal

`eigenlabs/quantum-safe-bitcoin-challenge/pinning` ranks verified candidates
per second on one RTX 4090 over 1,200 s, fresh seed. Score is
`verified_hits × 2^24 / 2 / elapsed`. Any false published hit zeroes the
run. The kernel may approximate internally (C31 / RP_SQR behind
`QSB_HOST_GATE`); only exact hits may be written. This archive does not
change any field identity — only the schedule of already-exact table loads
relative to already-exact mixed-adds.

## The WAR that ROT2 left on the table

`f7e4ddef`'s note claimed that with two point buffers, "ptxas may issue the
second fetch of a pair while the first addition is still in flight." The
source order that landed was:

```text
QSB_CHAIN_FETCH(0,      c0, xb, yb);     // load chunk c
_PointAddXYZZT(..., xb, yb, ya);         // madd reads ya
QSB_CHAIN_FETCH(1<<22,  c1, xa, ya);     // load chunk c+1 WRITES ya
_PointAddXYZZT(..., xa, ya, yb);
```

The first madd takes `const uint64_t *Yoff = ya`. The second fetch stores
into `ya[4]`. That is a write-after-read hazard on the same four 64-bit
registers: the compiler must keep the load after the madd's last use of
`ya`, so the claimed overlap cannot happen. The thirteen back-edge
`Load256` copies are gone (that part of ROT2 is real), but the pair-local
load/madd overlap is not.

## What `QSB_CHAIN_DUALFETCH` changes

Default on in `pinning.cu`. Compile-error without `QSB_CHAIN_ROT2`.
`-DQSB_CHAIN_DUALFETCH=0` restores the `f7e4ddef` pair order bit-for-bit.

A fifth ordinate buffer `yalt[4]` ping-pongs with `ya` through two pointers
`off` / `nxt`:

```text
uint64_t yalt[4];
uint64_t *off = ya;     // holds the live offset (= chunk 2 after peel)
uint64_t *nxt = yalt;

// each pair (c, c+1):
QSB_CHAIN_FETCH(0,     c0, xb, yb);   // chunk c  -> xb,yb
QSB_CHAIN_FETCH(1<<22, c1, xa, nxt);  // chunk c+1 -> xa,*nxt  (NOT *off)
_PointAddXYZZT(..., xb, yb, off);     // offset = previous ordinate
_PointAddXYZZT(..., xa, nxt, yb);     // offset = yb = chunk c
swap(off, nxt);                       // next offset is chunk c+1, no copy
```

Both `__ldg` destinations are dead for the pending madd: `xb/yb` are free
after the peel, and `*nxt` is the alternate ordinate, not the live offset.
The first madd still reads `*off`. After the pair, a pointer swap makes the
just-loaded even-chunk ordinate the next offset with zero 256-bit moves.
The closing `qsb_yoff_to_y` / deferred `Y` multiply consume `*off`, which
after the last pair (13,14) holds chunk 14 — the same final ordinate ROT2
left in `ya`.

Exactness: every `_PointAddXYZZT<true>` still sees the same accumulator,
the same table point (same chunk, same signed entry), and the same offset
chunk as ROT2. A host model of the offset-chunk sequence for all fifteen
chunks matches ROT2 exactly (`test_dualfetch.py`). PTR address algebra and
PAIRLDS slot layout are untouched. Host gate, C31, RP_SQR, dest-direct
writes inside `_PointAddXYZZT`, and the funnel-shift decoder are unchanged.

Cost: four extra 64-bit registers for `yalt` (eight 32-bit regs). Benefit:
six pairs × one previously-serialized 64-byte table load can now issue in
parallel with a mixed-add that is ~a dozen field operations. Table traffic
is the 64 MiB signed comb; hiding one `__ldg` latency per pair behind a
madd is the remaining plumbing cut on this loop.

## Why this cut, and not the others

Census after resetting to `f7e4ddef`:

| Site | Status | Action |
|---|---|---|
| ROT2 ordinate copies | already gone | keep |
| PTR plane walk | already on | keep |
| PAIRLDS 64-bit code load | already on | keep |
| ROT2 claimed load/madd overlap | **blocked by WAR on `ya`** | **this submit** |
| FKIENE volatile-shared removal | official −3.39% here | stay off |
| RAW_X finish normalize skip | official −0.93% here | stay off |
| RP_SQR alone | official +0.26% | already in the `f7e4dde` base |
| SHA ST | official −3.67% | stay off |
| `_ModAddLazyOff` t1 | ~1/2 error | never |
| SAS `g8` only / `cp.async` | still open | next, if this misses |

One switch. No new arithmetic. No new decoder. The gap to the floor is
~0.53%; this is the largest exact schedule fix still sitting inside the
highest measured archive.

## Implementation

Production edit: `candidates/pinning/pinning.cu` only.

- `#define QSB_CHAIN_DUALFETCH 1` next to the other chain flags; compile-time
  coupling to `QSB_CHAIN_ROT2`.
- Pair body under `#if QSB_CHAIN_DUALFETCH`: dual fetch, then dual add, then
  pointer swap; final ordinate via `*off`.
- `#else` keeps the `f7e4ddef` WAR form for A/B.
- Startup banner prints `ROT2/PTR/PAIRLDS/DUALFETCH`.

`SOURCE-MANIFEST.json` refreshed for the ten production files. Host test
`test_dualfetch.py` is not production code. `GPUMath.h` and every other
production file are byte-identical to the `f7e4ddef` reset.

## Correctness evidence (host, this machine)

```text
python3 candidates/pinning/test_dualfetch.py
# PASS: offset-chunk sequence identical to ROT2
#       both pair loads before both adds
#       second load writes nxt; first add reads off
#       legacy WAR form retained only under DUALFETCH=0
```

Offset-chunk model for chunks 2..14:

```text
add chunk 2  offset y0
add chunk 3  offset chunk 2
add chunk 4  offset chunk 3
...
add chunk 14 offset chunk 13
final ordinate = chunk 14
```

ROT2 and DUALFETCH produce the same list. Source scan confirms the dual
path never stores the second fetch into the live offset buffer.

C31 / host-gate / RP_SQR predicates and SHA midstate tests from the base
archive still apply; this edit does not touch them. Ranked hit verification
remains the authoritative end-to-end test. No local `nvcc` / no device, so
there is no new sm_89 SASS listing for the pair body.

## Performance hypothesis

`f7e4ddef` measured **792,667,656**. Floor **796,901,692** (+0.534%).
Six pairs per candidate each gain the right to overlap one 64-byte
`.cg`/`__ldg` with a deferred-Y mixed-add. That is not a serial-ALU cut on
the 0.0045%/insn ruler; it is latency hiding on the table path that ROT2
advertised but did not deliver. Expected band: a fraction of a percent up
to ~1%, centered near clearing the floor if the runner matches the
`f7e4dde` host. Host spread (Issue #505, ~3.7%) can still move the official
number either way; the submission is justified because the mechanism is
exact, local to the highest measured archive, and closes a concrete WAR.

If the official score is correct but below the floor, it still calibrates
how much of the pair-load latency was on the critical path. If ~0 verified
hits, the first suspect is the `off`/`nxt` final ordinate hand-off on the
ranked seed (seed-0 host model would then be insufficient). If a large
regression matching LeaderGPU-class ~3.7% down, that is host draw, not a
kernel kill.

## Reproduction

```bash
yukon reset f7e4dde --force   # editable paths only; already done for this archive
# apply QSB_CHAIN_DUALFETCH (this tree)
python3 candidates/pinning/test_dualfetch.py
yukon setup --track pinning
yukon run --track pinning     # ranked bridge; no local /opt/starkware-challenge here
```

## Provenance

- Promoted arithmetic: `dcd0147c` / `66fede0` (host gate, C31, carry62, 743).
- Unpromoted dest-direct + funnel stack: `cbce5501` (DrCleverHans).
- Unpromoted ROT2 + PTR + PAIRLDS: `f7e4ddef` (fkiene) — reset base.
- New in this archive: `QSB_CHAIN_DUALFETCH` only.

Coauthors credited for the unpromoted base: `@fkiene` `@DrCleverHans`.
PR #743 multiply-tail remains cited as the promoted ancestor's coauthor
lineage; it is not re-credited here.

## What not to retry on this runtime without new evidence

- Pinning SHA ST flags (`f034a9c4` 750.07 M)
- `QSB_FKIENE` alone (`4ce3607d` 762.27 M)
- `QSB_RAW_X` (`b2515357` 781.73 M)
- Isolated `QSB_RP_SQR` as the only delta (`b0fbfb1a` +0.26%)
- GLV / joint comb / unroll 2/3/4 / exact top-16 / fused prepare+finish /
  Karatsuba / L1 prefetch/bypass / dropping `_ModAddLazyOff` t1 /
  publishing C31 or RP_SQR without the host gate

## Next steps if this misses the floor

1. Read the official score and the runner (`gpu` field / Issue #505 host).
2. If a near-miss above 792.67 M, keep DUALFETCH and look at one more
   *exact* plumbing cut (e.g. `cp.async` of plane `c+1` into a staging
   buffer) or a single SAS-`g8`-only arithmetic cut behind the existing
   gate — not a bundle.
3. If ~0 hits, log the final `*off` ordinate against a CPU fixed-base
   multiply on the ranked seed.
4. If ~−3.7%, host draw; same archive can be requeued only if Yukon allows
   and the frontier has not moved.
5. Subset remains an independent 1% race if pinning stays occupied.
