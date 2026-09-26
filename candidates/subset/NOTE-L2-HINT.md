# Subset: an L2::64B fetch-size hint on the table-record load, ported from pinning without its carrier image

## What this changes

One device function and one call site inside `candidates/subset/tests/gpu_epochs/tree.cu`.
The 64-byte gTable record is read as a whole point, so the first 16 bytes of each
record now load through `ld.global.nc.L2::64B` instead of a plain read-only load.

```cuda
__device__ __forceinline__ ulonglong2 qsb_ld_rec_head(const ulonglong2 *__restrict__ p) {
#if !QSB_NO_L2_HINT && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    ulonglong2 v;
    asm("{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.nc.L2::64B.v2.u64 {%0,%1}, [g]; }"
        : "=l"(v.x), "=l"(v.y) : "l"(p));
    return v;
#else
    return __ldg(p);
#endif
}
```

The mechanism is the one the promoted pinning source carries, and it is attributed
to that lineage. `QSB_NO_L2_HINT=1` restores the previous behaviour exactly.

## Why this is a port and not a copy

Pinning ships the same hint behind a much larger apparatus: `build_carrier.sh`
compiles a second image with `-arch=sm_89 -DQSB_CARRIER_BUILD=1`, base64s the cubin
into a 448 KB generated header, and `QsbCarrier.h` loads it at runtime through the
CUDA runtime library API, with a fallback to the normal kernels. Its comment gives
the reason: the fixed build line embeds compute_52 PTX, and PTX for `.target sm_52`
cannot use any sm_75+ instruction, so the table loads cannot carry the hint.

Measured on this toolchain, that premise does not hold. With CUDA 13.4.92 and the
harness's own fixed build line (`nvcc -O3 -DQSB_ZEROS_N=24`, no `-arch`):

| build | `.target` | `L2::64B` present |
|---|---|---|
| fixed line, no `-arch` | `sm_75` | yes, 8 loads |
| `-arch=sm_89` | `sm_89` | yes, 8 loads |
| `-DQSB_NO_L2_HINT=1` | `sm_75` | no, 8 plain loads |

`__CUDA_ARCH__ >= 750` therefore holds in the default image, and a standalone probe
of the hint compiled, ran and returned the correct value under the default flags.
Subset needs 23 lines, not a 448 KB embedded cubin and a runtime loader.

## What is verified

- The PTX swap is exact: 8 hinted loads replace 8 plain loads, and the total count of
  `ld.global.nc.v2.u64` is unchanged at 48.
- Loaded values are bit-identical either way. Only the cache fetch size changes.
- The kernel builds, runs and completes normally with the hint on.

## What is NOT verified, stated plainly

I could not measure the gain. On an RTX 4080 Laptop the effect is inside the noise
floor of the hardware. Three drift-cancelled ABBA rounds of 40 s per arm gave:

| round | hint off (M/s) | hint on (M/s) | delta |
|---|---:|---:|---:|
| 1 | 226.65 | 224.35 | -1.0% |
| 2 | 211.95 | 218.85 | +3.3% |
| 3 | 211.25 | 212.95 | +0.8% |

Mean +1.0%, spread 4.3 points. The same runs rank the arms by candidate count with a
mean of +0.4%. This machine also shows roughly 10% thermal drift over a ten-minute
session, which is an order of magnitude above the effect being sought, so I do not
treat any of these numbers as a measurement of the change.

**This submission therefore does not claim a validated speedup.** It asks the
official harness for the number. The reasoning for expecting a non-negative effect
is structural rather than measured: a 64-byte record is consumed as a whole, so
fetching it with a 64-byte hint cannot increase memory traffic, and it can avoid the
second 32-byte sector miss.

## Scope

Subset track only. `candidates/pinning` is untouched. No harness, scorer, verifier,
workflow, manifest or sibling-track file is part of this change. The editable path is
`candidates/subset`, and the diff is 23 added lines and 1 changed line.

## Attribution

The L2::64B table-load idea and the carrier design come from the promoted pinning
lineage, which credits Ryun1's public pinning work for the `ld.global.nc.L2::64B`
record load. This port keeps the mechanism and replaces the delivery: no embedded
cubin, no runtime module load, just the instruction, because the toolchain no longer
needs the workaround.
## Reproduction

```sh
# 1. Build with the hint on (default) and off, using the harness's own flags
nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/z-on  candidates/subset/subset.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_NO_L2_HINT=1 -o /tmp/z-off candidates/subset/subset.cu -lcrypto -lm

# 2. Confirm the PTX swap is the only device change
nvcc -O3 -DQSB_ZEROS_N=24 -ptx -o on.ptx  candidates/subset/subset.cu
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_NO_L2_HINT=1 -ptx -o off.ptx candidates/subset/subset.cu
grep -c 'L2::64B' on.ptx            # 8
grep -c 'L2::64B' off.ptx           # 0
grep -c 'ld.global.nc.v2.u64' on.ptx   # 40
grep -c 'ld.global.nc.v2.u64' off.ptx  # 48

# 3. Drift-cancelled ABBA timing (order is A B B A per round)
for r in 1 2 3; do
  for arm in z-off z-on z-on z-off; do
    mkdir -p /tmp/run-$arm-$r && cd /tmp/run-$arm-$r
    timeout 40 /tmp/$arm problems/subset.bin 0 3556510642 1047016331 1 0 single_hash
  done
done
```

## Raw measurement log

GPU: RTX 4080 Laptop, 12282 MiB, 360 MHz reported at start and 2415 MHz at end.
Fixed problem file `problems/subset.bin`, seed `0`, sequence `3556510642`,
locktime `1047016331`, `single_hash` mode, 40 s per arm.

```
GPU at start: 1250 MiB, 6 %, 60 C, 360 MHz

  -- round 1 --
    z-off  r1-a     rate=226.4    epoch=54525952
    z-on   r1-a     rate=224.6    epoch=53477376
    z-on   r1-b     rate=224.1    epoch=53477376
    z-off  r1-b     rate=226.9    epoch=54525952
  -- round 2 --
    z-off  r2-a     rate=207.8    epoch=49283072
    z-on   r2-a     rate=218.2    epoch=52428800
    z-on   r2-b     rate=219.5    epoch=52428800
    z-off  r2-b     rate=216.1    epoch=51380224
  -- round 3 --
    z-off  r3-a     rate=211.2    epoch=51380224
    z-on   r3-a     rate=214.8    epoch=51380224
    z-on   r3-b     rate=211.1    epoch=50331648
    z-off  r3-b     rate=211.3    epoch=50331648

GPU at end: 1578 MiB, 83 %, 78 C, 2415 MHz
```

### Paired reading of that log

| round | off mean | on mean | delta | off attempts | on attempts | delta |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 226.65 | 224.35 | -1.0% | 54,525,952 | 53,477,376 | -1.9% |
| 2 | 211.95 | 218.85 | +3.3% | 50,331,648 | 52,428,800 | +4.2% |
| 3 | 211.25 | 212.95 | +0.8% | 50,855,168 | 50,331,648 | -1.0% |
| **mean** | | | **+1.0%** | | | **+0.4%** |

Two independent metrics, the kernel-reported rate and the completed candidate count,
disagree in sign in round 3 and agree in rounds 1 and 2, which is what noise looks
like rather than what a mechanism looks like.

## Independent evidence that this host cannot resolve the effect

An earlier four-arm session on the same host, same problem file, 60 s per arm, one
pass each, produced a monotonic slowdown in the exact order the arms ran:

| arm | first pass | second pass |
|---|---:|---:|
| base | 241.6 | 238.1 |
| `ZLAB_LAUNCH_BLOCKS=32768` | 239.1 | 214.4 |
| `ZLAB_PAIRSHA=1` | 244.7 | 217.8 |
| `QSB_SE_WINDOWS=256` | 237.8 | 206.3 |

Every arm lost speed in its second pass. That is a thermal trend across the session,
not a property of any variant, and its magnitude is several times the effect under
test here. By contrast, `ZLAB_T14=1` was rejected on the same host at -41% in both
passes, which is what a real effect looks like: reproducible in sign and far outside
the drift.

## Why the change should not hurt even if the gain is unmeasurable here

- A 64-byte record is read and consumed as a whole. `L2::64B` states the true access
  size, so it cannot increase memory traffic.
- The instruction is only a fetch-size hint. It does not change the address, the
  data, the number of loads, or the ordering.
- The value path is unchanged: `gx` and `gy` come out of the same four XOR and five
  carry operations after the load.
- `QSB_NO_L2_HINT=1` reproduces the previous load path exactly, so the change is one
  flag away from being reverted.

## What I am asking the official run to settle

Whether a 64-byte fetch-size hint on the table-record load helps on the ranked
runner. My local hardware cannot answer that and I am not claiming it does. If the
official score is flat or lower, the flag should be turned off and this note stands
as the record of a measured negative.