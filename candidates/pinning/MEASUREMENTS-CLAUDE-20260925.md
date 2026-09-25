# Pinning: RTX 4090 energy breakdown of the d59a969 frontier (no device-code change)

Measured 2026-09-25 on a rented stock RTX 4090 (450 W limit, driver 580.159,
CUDA 12.8 toolkit) with the frontier source at `d59a969`, built with the
ranked command. The embedded native sm_89 image rebuilt with CUDA 12.8 is
byte-identical to the committed one (sha256 f38efa14..., 282,336 bytes).
Every leg started from a 40 C GPU and used one fixed problem seed so hit sets
are directly comparable. This file changes nothing at build or run time.

| Measurement | Result |
|---|---|
| Throttle state, every run | SW power cap the whole time (~450 W, ~2,250 MHz of 3,105 max); no thermal slowdown |
| Startup before search | 0.90 s (carrier load 0.25 s, GPU table build 0.65 s) = 0.08% of the 1200 s window |
| Four random cold-bank reads per candidate | at most 5.8% (diagnostic: redirect cold reads into an L2-resident window) |
| Stage-0 tail + SHA256d | ~7% (diagnostic: replace with a cheap mix) |
| Two pubkey SHA-256 (stage 2) | ~8% (diagnostic: replace with a cheap mix) |
| Remainder | ~80%: the 11-addition XYZZ chain, recovery and batched inversion |
| Native image built by clang 18.1.3 + ptxas 12.8 instead of nvcc | -1.00% (898.2 vs 907.2 M/s, 4-round ABBA); identical hits |

Diagnostic builds compute wrong results on purpose and were never packaged.
Because the GPU is power-capped, throughput tracks work per joule: a change must
remove executed instructions or DRAM energy. Latency tricks that add loads or
work (prefetches, extra occupancy) cost energy. The DRAM cost is nonlinear:
four scattered 64-byte reads per candidate sit below the random-read bandwidth
knee, while the ledger's GLV10 (ten reads) scored -55% officially.
