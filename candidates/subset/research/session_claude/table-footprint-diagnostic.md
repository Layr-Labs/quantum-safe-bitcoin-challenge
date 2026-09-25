# Table-footprint diagnostic — 2026-09-26

Local RTX 3090 (6 MiB L2), CUDA 12.8.93, unmodified `harness/run_benchmark.py` +
`harness/gpu_wrap.py`, 60 s runs at `N=24`, `--max-rel-var none`, serialized with
`flock -x /tmp/qsb-gpu.lock`. Only the kernel's own periodic progress rate is read;
the diagnostic builds compute wrong points, so they publish nothing.

## What was varied

A one-line change in each of the two fixed-base record loaders
(`gt_load_signed_flat`, `gt_load_signed_flat_f`):

```c
size_t off = (((size_t)base + idx) * 64) & ((SIZE)-1);   /* SIZE = 4 MiB or 8 MiB */
```

Everything else — lookup count, recoding, arithmetic, SHA, gate, enumeration — is
untouched. The mask only changes which bytes of the table the lookups land on.

## Result

| Table footprint | Rate at 15 / 30 / 45 s | vs stock |
|---|---|---:|
| 64 MiB (stock; 10.7x the 6 MiB L2) | 253.2 / 253.6 / 252.4 M/s | — |
| 8 MiB (control; 1.33x the L2, still does not fit) | 264.3 / 265.1 / 264.5 M/s | +4.3% |
| 4 MiB (fits the L2 with room) | 299.0 / 296.9 / 297.2 M/s | **+17.4%** |

The 8 MiB control is the point: a footprint that still exceeds the cache buys only
the small DRAM-locality gain, while the large step appears exactly where the
footprint crosses the cache capacity. That is the signature of residency rather than
of the masking trick (masking alone would improve smoothly with size).

## Why this matters for the ranked device

Stock table traffic is 16 records × 64 B = ~1 KiB per candidate, i.e. ~253 GB/s at
253 M/s — only 27% of this device's 936 GB/s. The kernel is therefore **latency**
-bound on table access, not bandwidth-bound, which is the regime where a
cache-resident table pays. The RTX 4090 has 72 MiB of L2 (12x this device's) and
~1 TB/s of DRAM bandwidth with the same-order latency, so the same latency argument
applies there.

The stock kernel cannot keep its 64 MiB table resident because the batch's *own*
state is `ZLAB_LAUNCH_BLOCKS(262144) × PAIR_MUL(4) × FIRST_SLOTS(16) × 8 × 4 B =
512 MiB` of first states plus 64 MiB of epoch descriptors — seven times the whole L2.
`ZLAB_LAUNCH_BLOCKS=32768` drops that to 64 MiB + 8 MiB, which is what makes a 64 MiB
table resident-able in a 72 MiB L2.

## Limits

- The masking also makes many lanes share cache lines, so +17.4% is an upper bound
  on what residency buys, not a prediction; a resident table lives in L2, not L1.
- This device cannot show the effect of the *submitted* change in either direction
  (6 MiB cannot hold a 64 MiB table), which is why that change measures neutral
  (−0.08%) here. The diagnostic above is what replaces that missing measurement.
- The 3090 and the 4090 differ in SM count, clocks and DRAM type; the diagnostic
  establishes the *mechanism*, not a transfer factor.
