# Compact checkpoints and warp synchronization

Latest official result: PR74 was rejected at **407376555** verified candidates/s
against **644546620** (-36.80%). Its valid run does not establish a cause for
the regression. Any earlier pending/expected-best wording below records the
pre-result experiment. No successor performance win is established.

The isolated `warp_compact_fused/` candidate combines pending pinning PR74 with
the checkpoint mechanism from [Meganpark980320 PR98](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/98),
exact head `2d5eba297f8d9823ca0f5a552d3d1788fbe3031c`. Muse round 9 also
recommended this transfer after our public-source review had identified it.
Credit `Meganpark980320` as a coauthor if this unpromoted contribution is submitted.
The pending production closure is unchanged.

Retain this as a promising successor component. It passes CPU, native compiler
and C++ race checks, but no GPU throughput has been measured. There is no claim
that it beats the current or pending candidates or independently meets the
large-improvement target. The full-checkpoint version remains a comparison arm.

## Mechanism and costs

Keep the four saved coordinate fields C, Y, W and B. Store only product-tree
nodes 448 through 509 in four 64-element planes. Finish reconstructs the 192
lower internal nodes using 64 owners. Each owner computes two disjoint pairs
and their parent, retaining intermediates in registers. This is separate from
the earlier three-field checkpoint experiments that added coordinate work.

The split prepare/finish tree now uses 8 block barriers and 9 warp barriers,
versus 16 block barriers. The combined variant also changes the existing fused
tree to 7 block and 9 warp barriers, retaining its 765 field multiplies and
one inverse. The raw-root publication still has a block barrier before expansion.
Cross-warp producer/consumer boundaries remain block-wide: upsweep count 128
and above, downsweep count 32 and above. All 256 lanes participate; inactive
and singular candidates supply identity factors before returning after the collective.
This matches [CUDA's explicit warp synchronization requirements](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#synchronization-functions).

| Source-level cost per search candidate | PR74 | Compact |
| --- | ---: | ---: |
| Coordinate-state write + read | 256 B | 256 B |
| Checkpoint-node write + read | 63.5 B | 15.5 B |
| Additional tree field multiplications | 0 | 0.75 |
| Search checkpoint allocation at 16M candidates | 512 MiB | 128 MiB |

The 48-byte saving is checkpoint traffic, not whole-kernel traffic. Table loads,
root hierarchy, caches, parameters and instruction costs also matter. The memory
admission guard remains conservatively sized for the old allocation. Root-group
checkpoints use the new stride consistently. Fused state, table geometry, field
primitives, SHA and the runtime path selector retain the pending implementation.
In-place reconstruction uses the existing field primitive, which loads all
operand limbs into its assembly inputs before writing output limbs.

## Independently checked results

CUDA 12.8.93 in the local ARM Linux VM compiled all three variants for sm_89 and
with the default benchmark architecture flags. Values below are the fast-tail
kernels. Static instruction counts are not dynamic work or measured speed.

| Variant | Prepare registers / instructions | Finish registers / instructions | Fused registers / instructions |
| --- | ---: | ---: | ---: |
| Exact PR74 control | 122 / 7817 | 78 / 4598 | 126 / 13607 |
| Warp barriers, full checkpoint | 122 / 7821 | 78 / 4601 | 126 / 13607 |
| Compact checkpoint and warp barriers | 122 / 7824 | 79 / 5032 | 126 / 13607 |
| Compact plus fused-tree warp barriers | 122 / 7824 | 79 / 5032 | 126 / 13614 |

All listed kernels have zero spill stores and loads. Prepare uses 16 KiB shared,
finish and fused use 24 KiB. The fused 120-byte stack frame is inherited and is
not a spill claim. The small root-group finish increases from 44 to 76 registers.
Complete native resources, including generic kernels, are in the native reports.

Each variant passed 2,048 individual checkpoint inverses and the actual pinning
pipeline bodies across 1,856 active candidates, including singular lanes and
partial blocks. A controlled XYZZ producer replaces fixed-base multiplication;
OpenSSL replaces field arithmetic. Actual CPU SHA and an independent affine/hash
oracle check hit values and recid packing. Buffer canaries pass. The combined
variant additionally passed 2,048 direct-tree inverses and exact barrier counts.

ThreadSanitizer passes the compact and combined correct helper bodies with
separate 32-thread warp and 256-thread block barriers. Both deliberately broken
cross-warp boundary variants produce data-race diagnostics. Counter operations
use relaxed atomics so instrumentation does not introduce cross-thread ordering.
macOS's default sanitizer abort behavior initially defeated an exact exit-code
assertion despite detecting the intended race; explicit `abort_on_error=0` and
`exitcode=66` made the negative controls portable to this local runtime.

These CPU and sanitizer checks do not execute PTX, establish CUDA memory ordering
on hardware or measure GPU throughput. Native compilation provides real resource
costs, with source hashes checked against every corresponding CPU report.

## Reproduction

From the benchmark repository, with pinning selected:

```sh
python3 -B candidates/pinning/research/warp_checkpoint/prepare.py
python3 -B candidates/pinning/research/warp_checkpoint/check.py --variant warp_compact_fused
python3 -B candidates/pinning/research/warp_checkpoint/check.py --variant warp_compact_fused --sanitizer
python3 -B candidates/pinning/research/compile_local.py --source candidates/pinning/research/warp_checkpoint/warp_compact_fused --default-build --report /tmp/warp-native.json
```

The last command requires the existing local CUDA VM to be started as documented
in `../LOCAL_CUDA.md`; stop it afterward. It was stopped after these builds.
`comparison.json` binds the exact base and three candidate closures to their
reports. `prepare.py` refuses to run if pending production files have changed.
