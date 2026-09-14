# Problem specification

Two benchmarks. Both share the same per-candidate core (ECDSA key recovery +
hash); they differ only in how each candidate's SHA-256 preimage is assembled.

## Common core — per candidate

1. Assemble a preimage (bench-specific, below).
2. `z = SHA256d(preimage)` interpreted as a big-endian 256-bit integer.
3. **ECDSA public-key recovery**, done for every candidate:
   ```
   u1 = (-z · r⁻¹) mod n
   Q  = u1·G + u2·R        with  u2 = s·r⁻¹,  R.x = r,  R.y parity = recid
   ```
   `(r, s)` are fixed for the whole problem, so the problem ships the
   precomputed constants `neg_r_inv = -r⁻¹ mod n` and the point `u2·R` (for
   `recid = 0`). What varies per candidate is `u1`, i.e. the scalar
   multiplication `u1·G` and the point addition that follows it.
4. `h = SHA256(compress(Q))` (33-byte compressed pubkey).
5. **Hit** iff `leading_zero_bits(h) ≥ N`.

Each candidate is tried at both `recid ∈ {0,1}`.

## Pinning benchmark

Per candidate the thread varies `(sequence, locktime)` — two 32-bit fields — and
patches them into a mostly-fixed preimage:

```
preimage = pin_prefix (fixed, 9920 B = 155 SHA-256 blocks)  ||  suffix (75 B)
suffix[seq_offset:+4]  = sequence  (little-endian)
suffix[lt_offset:+4]   = locktime  (little-endian)
```
Only the 75-byte tail differs between candidates; the 9920-byte prefix is
block-aligned and identical for all of them.

## Subset-selection benchmark

Per candidate the thread chooses `t = 9` **skip indices** out of `n = 150` (which
dummy-signature pushes to omit), and rebuilds a ~9.9 KB preimage:

```
preimage = fixed_prefix (8234 B)
        || concat( dummy_sigs[i]  for i in 0..n-1  if i not in skip )   # (n-t)·10 B
        || tail_section (218 B)
        || tx_suffix (44 B)
```
`skip` uses the **STORAGE index convention**: index `i` refers to `dummy_sigs[i]`
in the order stored in the problem/`.bin` (offset `i·10`). Hits report skip
indices in this convention. The variable section starts at byte 8234, so
everything from there on — ~1.7 KB, ~27 SHA-256 blocks — differs between
candidates, while the preceding 8192 bytes (128 blocks) are fixed.

## Problem files (`problems/<bench>.json` and `.bin`)

`gen_problem.py` emits both, from a fixed RNG seed (reproducible):

- **`.json`** — authoritative, human-readable, used by the CPU verifier. Fields:
  `r, s` (hex), `neg_r_inv, u2r_x, u2r_y` (hex), plus the full preimage pieces
  (`pin_prefix/suffix/seq_offset/lt_offset` or `fixed_prefix/dummy_sigs[150]/tail_section/tx_suffix`),
  and `total_preimage_len`.
- **`.bin`** — the same data in a flat binary layout: `uint32` little-endian, the
  SHA-256 state after the fixed block-aligned prefix as 8×`uint32` big-endian,
  and the 32-byte scalars/coords little-endian.

## I/O contract (how any kernel plugs in)

A conforming grinder produces a **run artifact** JSON (write it to `--out`):

```json
{
  "bench": "pinning" | "subset",
  "zeros_n": <int N>,
  "mode": "fixed_time" | "fixed_hits",
  "candidates": <int>,          // total candidates searched
  "elapsed_s": <float>,
  "throughput_Mps": <float>,    // candidates / elapsed / 1e6
  "hits": [ <hit>, ... ]
}
```
Each `hit`:
- pinning: `{"bench":"pinning","sequence":<int>,"locktime":<int>,"recid":0|1}`
- subset:  `{"bench":"subset","skip":[<t storage ints>],"recid":0|1}`

The verifier re-derives each hit from `(problem, hit)` and checks the gate. The
reference CPU grinder (`harness/cpu_grind.py`) emits this artifact directly; a GPU
kernel can emit it directly or via `harness/gpu_wrap.py`.
