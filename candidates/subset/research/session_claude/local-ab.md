# Local qualification — native sm_89 carrier, 2026-09-25

All numbers below are **local RTX 3090 diagnostics**, not ranked RTX 4090 results.
The harness artifact's static `RTX_4090` label is not hardware detection.

## Environment

- GPU RTX 3090, 82 SMs, 24 GiB, driver 595.95, CUDA 12.8.93, WSL2 Ubuntu 24.04.
- Source: the public Subset frontier tree at `7e95c40` (submission
  `7aef224a`, Akashneelesh), unmodified except for the carrier files listed in
  `SOURCE-MANIFEST.json`.
- Harness: the repository's own `harness/run_benchmark.py` with
  `harness/gpu_wrap.py`, `--bench subset --N 24 --mode fixed_time`, the committed
  public problem, `--max-rel-var none`. Every emitted hit is re-derived by the
  unchanged verifier before any number is quoted.
- All GPU work was serialized with `flock -x /tmp/qsb-gpu.lock`.
- `research/session_claude/abrun.py` is a local-only wrapper that selects the
  compile flags per arm and saves the kernel's stdout; it does not change the
  argv, the hit collection, the clock, or the verifier.

## Arm definition

| Arm | What runs |
|---|---|
| OFF | the stock ranked build: `nvcc -O3 -DQSB_ZEROS_N=24 subset.cu` → compute_52 PTX, driver-JIT'd for the device. `QSB_CARRIER_DISABLE=1` forces this path in the carrier binary. |
| ON | the same binary, but `kernel_digest` is launched from an embedded native image whose only device-code difference is `ld.global.nc.L2::64B.v2.u64` on the first 16 B slice of each 64 B fixed-base table record. All host symbol uploads write both images. |

Both arms come from **one binary**, so the archive, argv, problem, seed and
verifier are identical; only the launched code differs.

## Correctness

60 s arms, same problem, hits collected from `results/digest_hit_0.txt` and
sorted before hashing:

| Arm | Verified hits | SHA256 of the sorted hit file |
|---|---:|---|
| OFF | 1891 / 1891 | `d6381d27d35d21d949c2c60a6dd5f62e6ff5307eeba4631d4aa09174f5f93f46` |
| ON | 1891 / 1891 | `d6381d27d35d21d949c2c60a6dd5f62e6ff5307eeba4631d4aa09174f5f93f46` |

**Hit sets identical.** The carrier changes only the memory request made for a
table record, so the searched candidate set and every published hit are the same.
The unchanged OpenSSL host gate re-derived every one of them in both arms.

## Speed — shipped configuration (carrier OFF vs ON)

`300 s` arms, A/B/B/A, one binary:

| Order | Arm | Verified hits | Hits/s |
|---|---|---:|---:|
| 1 | OFF | 9131 | 30.44 |
| 2 | ON | 9233 | 30.78 |
| 3 | ON | 9233 | 30.78 |
| 4 | OFF | 9073 | 30.24 |

OFF mean 9102, ON mean 9233 → **+1.44%** on the same work. The two ON arms are
bit-reproducible in count; the OFF arms vary by 0.6%, so the true effect is
somewhere in the +1.0% to +1.5% band on this device.

## Speed — the load qualifier alone

To separate the qualifier from the native-vs-JIT codegen, the same experiment was
run with **both** arms built natively for sm_86 (`-arch=sm_86`), so the only
difference is the load form. `300 s`, A/B/B/A:

| Order | Arm | Verified hits |
|---|---|---:|
| 1 | plain `__ldg` | 8981 |
| 2 | `.L2::64B` first slice | 9183 |
| 3 | `.L2::64B` first slice | 9152 |
| 4 | plain `__ldg` | 8981 |

plain mean 8981, hinted mean 9167.5 → **+2.08%**. A control arm that used the same
inline-asm form *without* the prefetch qualifier compiled to SASS identical to the
plain arm, which is consistent with the qualifier, not the surrounding asm, being
responsible.

A host-side `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64)` arm was also
screened: 7335 (control), 7286, 7305 (limit set), i.e. neutral-to-negative. The
device-side qualifier is doing something the host limit is not.

## Why the mechanism should help a table larger than the cache

Each digest thread reads a 64-byte record (`X||Y`, four 16-byte slices) from a
64 MiB table that is re-read for every candidate. Reaching DRAM as four separate
16 B loads, that record costs **two independent 32-byte sector fetches**; the
`.L2::64B` prefetch size asks for the whole 64-byte record in one L2/DRAM
transaction and halves the DRAM activation count for exactly this traffic. The
local RTX 3090 has a 6 MiB L2, so almost every table access misses and the effect
is large there. The RTX 4090 has 72 MiB of L2 — but the kernel also streams a
larger first-state/epoch working set through the same cache, so the table cannot
be assumed resident and the same transaction reduction may still apply.

**This is a local-to-ranked transfer that cannot be measured here, and no ranked
improvement is claimed.** The submission's official RTX 4090 run and the
repository's own verifier are the only measurements that decide it.
