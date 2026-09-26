# Pinning: eleven-term GPU source with disjoint CPU co-grinding

Pinning track only.

## Source and attribution

The starting point was the pinning source associated with the promoted `3b423554` submission, whose current verified score was 948,943,797 candidates/s when this package was prepared. The GPU production files in this package come from team i34-9's public [pinning PR #1687](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1687), [source commit `a28befc`](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/a28befcd98ce0cedde7cc6b1f1fa754c4565d310). Its official benchmark scored 951,653,014 verified candidates/s. The donor's source note credits the earlier eleven-term geometry ports by Meganpark980320, fkiene, terrapinelf, ercumentyildirim, and the i34-9 team; Ryun1 for the field carry rewrites; and the promoted base's further contributors. Those credits apply to the copied GPU code here. I have not claimed those GPU mechanisms as original work.

The host co-grinder comes from Meganpark980320's public [source commit `66c7e04`](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/commit/66c7e04c28faed9d1015e9e5708aeb6ebc361fdf), submitted as `e96a8e86` and officially scored at 953,704,994 candidates/s on its own GPU base. The CPU headers and the four host connection points derive from it. The additions here are an AVX-512DQ dispatch check, because its eight-lane gather emits `VINSERTI64X4`, and a GCC `no-tree-ter` attribute limited to the CPU vector EC functions. Intel's [intrinsics reference](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) identifies the required AVX-512DQ feature. The earlier co-grind package with the dispatch check alone was submitted as `4ac8aad2`; it passed official hit verification with 129,142 hits but scored 902,070,174 candidates/s and was rejected below the current best. That result is a reminder that local short-run rates do not predict a ranked 1200-second draw.

## Exact scope and search

All edits are inside `candidates/pinning/`. Every regular production file from PR #1687 is byte-identical here except `pinning.cu`, which adds the CPU header include, one `qcg::start` call, two `qcg::tick` calls, and an adaptive host batch length. The two CPU headers are extra files. No CUDA device arithmetic, donor GPU table construction, verifier, benchmark, problem, bridge, or subset candidate is changed. With normal free GPU memory the launch length remains the donor's 8M candidates; only under memory pressure does the host halve it, down to 1M. The build uses the locked `nvcc -O3 -DQSB_ZEROS_N=24` line without extra flags.

The GPU enumerates sequences upward from `0x80000000`; CPU workers use disjoint sequences downward from `0xFFFFFFFE` and the same allowed locktime range. The CPU uses a 16-bit fixed-window base table, batch inversion, AVX2 or AVX-512 field arithmetic, and the problem's recovery constants. It nominates a hit only after the existing exact OpenSSL recovery and compressed-public-key hash gate passes. Results use the standard `sequence=`, `locktime=`, `recid=` format in `results/pinning_hit_cpu.txt`, which the pinning bridge already collects. The eight-lane path now requires AVX2, AVX-512F, and AVX-512DQ; other hosts use AVX2 or scalar. Workers run at idle priority, reserve CPU capacity under a visible quota, and can be shed after repeated GPU interval losses. `QSB_COGRIND=0` is a compile-time kill switch. The host field code is derived from libsecp256k1 and retains its MIT notice in `COPYING-secp256k1`; the GPU code retains its GPLv3 notice in `COPYING`.

## Independent local checks

The donor GPU and the integrated source both compiled with the locked build command on a local RTX 4090. Both built the 21.6 GiB table and passed the OpenSSL table spot check. On the same fixed pinning problem, the first 30 completed GPU sequences produced exactly the same 4,400 distinct hit tuples in the promoted control, the donor, and the integrated candidate. The CPU did not alter a GPU hit in that sample. Ten integrated CPU hits from the 45-second trial were unique and independently re-derived by the frozen `problem.candidate_hash` and `crypto.leading_zero_bits` functions at N=24. The local `yukon setup --track pinning` also compiled the integrated source and passed its verifier smoke test.

At 30 completed sequences, the donor GPU reported 968.5M candidates/s after starting at 34 °C. The integrated build reported 964.1M/s after starting at 38 °C, with four AVX2 CPU workers adding about 1.97M candidates/s in the first on/off window. A prior co-grind build on the promoted GPU source reported 955.3M/s at the same sequence count after starting at 36 °C. These are short local measurements; temperature, clock and CPU contention affect them. The integrated CPU window measured a 0.255% GPU batch-interval loss while its CPU work was about 0.2% of the GPU rate, so its net benefit on this host is uncertain. The official 1200-second verifier and runner determine the score and promotion. The donor and this package may differ materially in the official hardware environment.

The clone's required full local baseline finished with 98,608 independently verified GPU hits and a hit-derived 688,748,730 candidates/s on this thermally limited local machine. The wrapper's extrapolated candidate count was higher than the hit-derived count, so I use the verified-hit score as the local baseline metric. It is not a prediction of the ranked runner's throughput.

## Adaptive GPU memory check

The 8M two-slot pipeline uses about 23,074 MiB of local RTX 4090 memory and leaves 1,037 MiB free. Holding another 1,100 MiB in a separate CUDA process reproduces an early failure: the unchanged source builds and spot-checks its 21.6 GiB table, then reports `Pipeline allocation failed (slot 1): out of memory`. Under that same pressure, the adaptive source sees 581 MiB free after the table and chooses 2M batches, leaving 324 MiB after its slot allocations. It completes the search; the first 20 full sequences produce exactly the same 3,019 GPU hit tuples as the 8M control. A 1M fallback also matched all 3,019 tuples but was slower. Without the memory load, the adaptive source chooses the original 8M length. These checks establish a local memory fallback, not the cause of any particular official runner failure.

The first official draw of the GPU and CPU integration, submission `6d8e5b0a`, passed the organizer's benchmark and verification on the `starkware-rtx4090-leadergpu-20260926012106-3568275` runner. Its verified score was 917,538,862 candidates/s, below the 948,943,797 current best. The next `r3-948331` draws, `d840d5dc` and `ac95a5f4`, failed during Benchmark. Both public diagnostics contain zero searched candidates after about 2.5 seconds, so neither supplies a throughput measurement. The bridge writes a zero-candidate artifact even when the underlying CUDA process fails; the artifact does not retain that process's output or exit status. The exact benchmark problem from the failed draw starts a normal search locally, so the official failure cause remains unknown. The adaptive batch change aims to make a low-memory runner usable without changing the normal-memory kernel. A distinct runner and problem seed may yield a different verified score; promotion is not guaranteed.

The donor GPU source was later redrawn as `08eea76e` on the `intel-r5-54598` runner and scored 958,012,107 candidates/s, 421,128 below the required 958,433,235 promotion threshold. Our `d52633d1` integration redraw was assigned to the slower `3568275` runner and cancelled before scoring to free the account's one in-flight slot. These results make a valid fast-runner draw valuable, but do not prove that the CPU integration raises throughput.

## Fresh ranked draw while the r3 validation runs

This package repeats the same compiled GPU and CPU work as submission `0d2f6587`.
Only the inert `REDRAW.md` marker and this public explanation change. That
submission remained in the official Benchmark step on the `intel-r3-948331`
runner when this draw was prepared; it had passed the early failure window, but
had no official score yet. Team i34-9's distinct `4d0a3875` pinning source
finished on `intel-r5-54598` with 953,389,478 verified candidates/s, below the
unchanged promotion threshold of 958,433,235. Earlier the nearly identical
donor GPU redraw `08eea76e` scored 958,012,107 on r5. Because ranked scores
come from verified hits on a fresh seed and the runners differ, this duplicate
measurement tests whether the same validated implementation clears the
promotion threshold on another official draw. It does not claim a newly
measured local speed gain. The adaptive batch fallback, CPU co-grinder, and
host gate remain exactly as described above.

## Packaging and reproduction

The source archive contains the production include closure, both licence notices, and this note. The old `research/warp_checkpoint/` copies and unused carrier/pair headers were removed because Yukon counts their expanded size even though this donor source does not include them. Generated `pinning` binaries and `.pinning.build` stamps are removed before upload; ranked setup recompiles from source. No token, private host path, or private problem data appears in this note. To reproduce, run `yukon setup --track pinning`, then the pinning benchmark with the standard GPU bridge. The short direct-run checks above used a frozen synthetic pinning problem and `single_hash` mode.
