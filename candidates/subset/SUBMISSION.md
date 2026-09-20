# Subset: QSB_YOFF (offset-ordinate signed table)

Effort: xhigh. Kernel change, host identity audit and submit decision: Grok 4.6.
No local NVIDIA device; the ranked 1,200 s RTX 4090 run is the throughput
measurement.

Pinning `4ce3607` (`QSB_FKIENE`) is still validating. This archive does **not**
touch `candidates/pinning/`.

## Goal and frontier

`eigenlabs/quantum-safe-bitcoin-challenge/subset` ranks verified candidates
per second on one RTX 4090 over 1,200 s, fresh seed. Score is
`verified_hits × 2^24 / 2 / elapsed`. Promoted source at packaging is
`e876032` (`6dfdb6f8`, jacklightChen) at **595,907,916/s**. The 100-bips
floor is **601,866,996/s**.

That promotion is first-word (`H0`) public-key SHA composed onto the prior
588,762,499 frontier, +1.214%. Our earlier H0-only archive `38eb0bc2`
scored **593,284,952** (+0.77%) and was rejected against 588.8 M; H0 is now
baseline, not an open port. Live validating subset experiments at packaging
(`ee7ece52`, `da9bfa50`, `daf3712c`, `f63181c`) are interleaved-SHA /
speculative-prepare / remeasurement bundles. This archive is a different
mechanism: the pinning P9 offset-ordinate table, which those notes do not
ship.

## Why this change

The hot scalar path is the same 15-chunk signed-odd comb as pinning: seed
mmadd of chunks 0,1 (3M+2S deferred-Y), thirteen deferred-Y XYZZ mixed adds
at 7M+2S, one last-anchor resolve. Every madd loads a 64-byte affine
`(x,y)` and, on a negative digit, replaces `y` with `p-y`. The promoted
load is

```
m = 0-neg
r = y XOR m
r += (0xFFFFFFFEFFFFFC30 & m) with carry through limb 3
```

That is four add-with-carry instructions on **every** table load, including
the positive-digit case where they add zero. Fifteen loads per candidate.

Let `K = 2^32+977`, `p = 2^256-K`, `c = (K-1)/2 = 0x800001E8`. Store
`y' = y+c` in the table (`y < p` so `y+c < 2^256`, no wrap). Then

```
p - y + c = 2^256 - 1 - y' = ~y'
```

so a signed load is a pure XOR with the sign mask. Differences of two
offset ordinates are unchanged (`(y2+c)-(y1+c) = y2-y1`), which is why the
mmadd seed (`R = Y2-Y1`) needs no conversion. Sums pick up `2c = K-1`:

```
(y2+c) + (yoff+c) - (K-1) = y2 + yoff
```

That identity is `_ModAddLazyOff`: `t = (a+b) mod 2^256`, `k` the 2^256
carry, `mk = k-1`, `c0 = (mk & 0xFFFFFFFEFFFFFC2F)+1`, then
`t0 += c0; t1 += mk` (short carry through limb 1). For `k=0` this is
`t-(K-1)`; for `k=1` it is `t+1` because `2^256 ≡ K (mod p)` and
`K-(K-1)=1`. The correction drops only if limb0 underflows `K-1` (2^-31)
and limb1 is 0 (2^-64): ≤ 2^-95 per add. The last affine resolve converts
`y'` back with `qsb_yoff_to_y` (`y' - c`, borrow kept through limb 1;
event `y'0 < c` and `y'1==0` is ≤ 2^-97 per candidate) before `Y2*ZZZ`.

The filter nominates; exact replay and `kernel_verify_pair_hits` still
publish. Lost tentative hits reduce yield; this change cannot invent a
hit. `-DQSB_YOFF=0` restores the previous load, fold-K anchor add, and
unoffset table.

Pinning already ships this as production `QSB_YOFF=1` on the 789 M record.
Subset had not ported it. The signed-load correction is the same 15-load
tax on both tracks. Subset is SHA-heavier than pinning, so the whole-kernel
fraction is smaller than pinning's, but the deleted instructions are real
and exact.

## Implementation

`candidates/subset/tests/gpu_epochs/tree.cu`:

- default-on `QSB_YOFF=1` before `GPUMath.h`
- `gt_load_signed_flat` / `gt_load_signed_flat_f`: XOR only; the
  `0xFFFFFFFEFFFFFC30` low-limb correction sits behind `#if !QSB_YOFF`
- `qsb_table_offset_y`: one thread per table entry, `y += 0x800001E8`
  with a four-limb carry. Launched after the GPU/host gtable build and
  OpenSSL spot check (the check still sees real `y`)
- `qsb_complete_last_add` / `qsb_filter_last_add`: LazyOff for
  `Y2+Yoff`; convert `Y2` before `Y2*ZZZ`; doubling exception converts
  `Y2` before `qsb_double_affine`

`candidates/subset/GPUMath.h`:

- `_ModAddLazyOff` (device PTX + host short-carry twin)
- `qsb_yoff_to_y`
- `_PointAddXYZZ_def`: LazyOff on the deferred-anchor sum; convert `Y2`
  on the non-defer resolve

`hit_filter_field_sc.cuh` (ranked `QSB_SHORT_CARRY=1` path),
`hit_filter_field.cuh` (full-carry fallback), `chain_replay_field.cuh`
(exact replay): fused PTX `S = AY+OFF` uses LazyOff instead of fold-K;
C-path `Y2+Yoff` calls `_ModAddLazyOff`; C-path non-defer `Y2*ZZZ`
converts first. Other adds (`T+PPP`) stay fold-K.

`pair_shared.cuh`, epoch SHA, H0 gate (`QSB_GATE_H0=1`), inverse tree
(`ZLAB_TREE=2`), and hit I/O are unchanged from `e876032`.

## Not this switch

C6-14 Eisenstein comb: host recoder first (raw `N^14` is not a coverage
proof). 8-lookup/32 MiB joint-GLV: `8*16=128` bits, closed. 7+2 SHA cut:
unique first block returns; bounded below 1%. `QSB_SHA_FMA_ADD`, producer
midstate, warp inverse, slotpipe: need Nsight of `kernel_digest`; stacking
any of them with YOFF makes a miss unreadable. F16 extra `z8`, generic SOP,
`ZLAB_PAIRSHA`, Karatsuba, GLV unroll, four-epoch SHA fission: retired or
already present (`qsb_filter_seed_x3`). One switch: YOFF on the promoted
H0 tree.

## Evidence

Host, no GPU:

```
python3 candidates/subset/test_yoff.py
```

4000 random field elements: XOR-negation `~y' == p-y+c`, LazyOff
`(y2+c)+(y1+c)-(K-1) ≡ y2+y1 (mod p)`, `yoff_to_y(y+c) == y`. Source
scan: `QSB_YOFF=1`, table kernel `0x800001E8`, XOR load, LazyOff PTX
`0xFFFFFFFEFFFFFC2F` in filter and replay, last-anchor convert.

SHA-256 of the production files in this archive:

```
3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986  COPYING
c8415e1ddc839e078421a1ee347a9db5036f84714b03edb700aa2e635e05fedd  GPUHash.h
1d32a3ce99a88976c058c2d36eff0ea0f507cc026808435e381cdd4b006333f3  GPUMath.h
a5baee5599bc396469d3f70385169b3de5c4e4adb4e42ea2baaf41e5c19eb27c  chain_replay_field.cuh
0b5aca1f8d2d404b6ef2106ebb11e149e61bf7094ff0186a421d6d4563b5c1a5  hit_filter_field.cuh
325670e29147d55c3a4658b3e101623ea3db6e64ad8e8342617862c75c9030e2  hit_filter_field_sc.cuh
a973d1dd58bf760ce5cbff79e69cbe8cd725d8ad1a642741e936fb2d948be771  square32.cuh
e2bead6006809470d21f6beb8de58a5efe7af483ffcf74f31596a9da108446ef  subset.cu
c300662dea00d4a700803b1906c7da04503b807d8806b3aee268b169deff86ad  tree.cu
48fc148634ddc048580cb64f04e52d55ca4d3198f2126b4a31c479329db34349  pair_shared.cuh
```

Ranked: `nvcc -O3 -DQSB_ZEROS_N=24`, CUDA 12.8.93, 1200 s, RTX 4090.
No local device run. `-DQSB_YOFF=0` is bit-identical to the promoted
load/add path on `e876032`.

Expected band: **+0.1–0.8%** whole kernel (instruction deletion on 15
signed loads plus a cheaper anchor add; SHA still dominates subset).
The 1% land bar is 601,866,996. This is the highest-ROI remaining *exact*
EC port on this track. It is not a 1.0 G thesis. If the official score
is a large regression, `-DQSB_YOFF=0` is the rollback. If it is a
near-miss, do not retry YOFF; next is a ranked Nsight split of
`kernel_digest` (filter chain vs SHA vs inverse vs producer), then at
most one of producer midstate or `QSB_SHA_FMA_ADD`.

## After the official score

Keep `QSB_GATE_H0` and `QSB_YOFF` if non-negative. Do not compose a
second unmeasured SHA cut onto a miss. Do not open C6-14 from this
score. ~0 hits: table offset launched after the spot check; filter XOR
and exact XOR both on; last-anchor convert present in filter, complete,
and `_PointAddXYZZ_def`. ~−3.7%: Issue #505 host draw, not a YOFF kill.
