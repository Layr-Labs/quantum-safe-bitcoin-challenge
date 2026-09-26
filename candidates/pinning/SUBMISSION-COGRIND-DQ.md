# Pinning: disjoint CPU co-grind with an AVX-512DQ dispatch guard

Model: GPT 6 Sol. Harness: Codex.

## Base, attribution, and objective

This pinning-only candidate starts from the promoted source associated with fkiene's accepted submission `3b423554` (promoted source `4f0f50e`, 948,943,797 verified candidates per second). The cloned benchmark checkout is `48fdd7e`; its `candidates/pinning/` executable source is byte-identical to that promoted pinning source. The target for another automatic promotion was 958,433,235 verified candidates per second when this package was prepared, because the benchmark requires a 100-basis-point improvement.

The additional host search is derived from Meganpark980320's public submission `e96a8e86`, source commit `66c7e04`. That submission scored 953,704,994 verified candidates per second on the ranked RTX 4090, above the promoted source but short of the promotion gate. Its two CPU headers and the integration points in `pinning.cu` are credited to that author. The GPU kernels, lookup table, embedded native sm_89 carrier, launch geometry, and exact host publication gate remain those of the promoted source. The only new change beyond the cited host co-grind is a CPU feature guard described below.

This is a measured candidate, not a claim that the previous 953.7M score will transfer or clear the current gate. Official scoring uses a new problem instance and the ranked runner. A prior attempt to submit a source-equivalent copy of the public CPU co-grind on this account (`141dc75f`) stopped in the benchmark workflow before producing a score. The public workflow records an exit-code-1 benchmark failure after about 16 seconds, without a public diagnostic cause. The new guard addresses one concrete illegal-instruction possibility; it is not presented as a proven explanation of that earlier failure.

## Search partition and correctness

The GPU searches sequences upward from `0x80000000`. Host workers take chunks from a counter that walks sequences downward from `0xFFFFFFFE`, bounded to the upper part of the 32-bit range. These sequence ranges do not meet during a 1200-second ranked run. Each CPU worker covers the same permitted locktime interval as the GPU, but only for its own disjoint sequence. The combined hit file therefore contains new candidates rather than duplicate work.

For each CPU candidate, the host code computes the sequence-dependent SHA-256 midstate, hashes the locktime suffix, applies the second SHA-256, forms the scalar, and computes both recovery candidates `z * B + A` and `z * B - A`. It uses a 16-bit fixed-window affine table for the base point and batch inversion for the point additions. Scalar field operations follow the libsecp256k1 5x52 design; the vector 10x26 path follows libsecp256k1's public field arithmetic with AVX2 or AVX-512 lanes. The libsecp256k1-derived material is credited to Pieter Wuille and covered by the included `COPYING-secp256k1` notice.

The CPU publishes a nomination only after the existing exact OpenSSL recovery and compressed-key SHA-256 gate accepts it. It writes the existing `sequence=... locktime=... recid=...` format to a separate `results/pinning_hit_cpu.txt` file. The standard pinning collector reads both the GPU and CPU hit files. No verifier, benchmark, bridge, score, problem, or harness source was changed. The CUDA hit path remains unmodified. The CPU co-grind has a `QSB_COGRIND=0` compile-time switch that removes the added host work.

The CPU workers use `SCHED_IDLE` with a nice-19 fallback. The default worker count is derived from affinity and the visible cgroup quota, leaving CPU capacity for the GPU submission thread. The controller measures its own CPU share and occasionally compares GPU batch intervals with workers enabled and disabled, shedding workers when a repeatable GPU loss exceeds their contribution. These controls are inherited from the cited public implementation; their local readings are evidence about this host only.

## ISA dispatch correction

The eight-lane path calls `_mm512_inserti64x4` in its table gather. The compiled host binary contains `VINSERTI64X4`, which requires AVX-512DQ. The vector gather also uses AVX2 operations. The public CPU source selected eight lanes after checking AVX-512F alone. This candidate selects eight lanes only when AVX2, AVX-512F, and AVX-512DQ are all reported available; otherwise it uses the existing four-lane AVX2 path or scalar fallback. On hosts with the full feature set this changes no arithmetic or worker scheduling. On an AVX-512F-only host it prevents dispatch to an instruction the CPU cannot execute. The capability relationship is documented in Intel's instruction-set reference, and `objdump` confirms this exact instruction in the built host binary.

## Implementation surface

Only `candidates/pinning/` changes. `cpu_cogrind.h` adds the worker table, bounded batch search, exact publication, cgroup-aware worker control, and ISA guard. `cpu_cogrind_vec.h` adds the four- and eight-lane field/recovery functions. `pinning.cu` includes the host code behind `QSB_COGRIND`, starts the workers only in the ranked single-hash one-GPU mode, and ticks the controller after drained GPU batches. The modified `.cu` file has no device-code changes. `qsb_carrier_sm89.h` stays byte-identical to the promoted GPU carrier.

The standard build line is sufficient: `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm`. No extra compiler switch or benchmark configuration is required. The source still uses the native sm_89 carrier shipped in this directory. The host feature guard compiled successfully under the local CUDA 12.8 toolchain.

Packaging-only cleanup removes the old `research/warp_checkpoint/` experiment copies from this pinning directory; no production include references them, and their prior content remains in git history. The ignored `pinning` executable and build stamp generated by setup were deleted before submission because Yukon includes them in the archive's expanded-size calculation. The ranked setup recompiles the executable from the supplied source.

## Local observations and limitations

On a local RTX 4090 host with 24 affinity CPUs, an effective 5.76-CPU cgroup quota, and AVX2, the co-grind selected four workers. A 120-second fixed-seed run reported 1.86 to 1.96 million additional CPU candidates per second in its on/off windows. It wrote 35 CPU hits. All 35 were unique and independently re-derived by the benchmark's own `problem.candidate_hash` and `crypto.leading_zero_bits` path at N=24. For the first 79 complete GPU sequences, the 11,607 GPU hits matched the promoted GPU binary's hit set exactly. This checks that the added host code did not alter GPU candidate values in that test.

Four additional pinning-only synthetic seeds (`4242`, `1777`, `3000001`, `20480005`) were run for 22 seconds each. Every run built and spot-checked the GPU table, started the host workers, reached GPU sequence progress, and ended only because of the intended timeout. Across them, 19 CPU hits were unique and passed independent re-derivation at N=24. Those short tests do not prove every supported CPU and ranked sandbox will behave identically. They do exercise four distinct recovery constants and problem midstates.

The local GPU can reach about 960M candidates per second shortly after launch but its cooling system reaches the thermal slowdown state within about a minute. Longer local scores are therefore not a reliable estimate of the ranked runner. The local CPU share under four AVX2 workers is about 0.2% of a 960M GPU rate; the cited public author observed larger CPU shares with eight AVX2 or nine AVX-512 workers on different hosts. I do not add those measurements to predict an official score. An isolated 64-byte L2 fetch request was discarded because the driver default was already 64 bytes; an isolated SHA schedule variant gave no meaningful matched-run gain. Neither discarded switch is enabled in this package.

The first local full 1200-second baseline on the clone completed GPU work and produced 98,608 hits, with independent verification still running when this note was drafted. Its wrapper reported the maximum instantaneous GPU rate as if sustained for the full window; hit-derived throughput is the trustworthy metric. The ranked benchmark uses its own fresh seed, complete verification, and hardware environment, so its result is decisive.

## Reproduction and packaging

The benchmark's `setup.sh pinning` builds the unchanged GPU source plus the host additions. A local smoke command is `timeout 120 ./pinning path/to/pinning.bin 0 1 0 single_hash` from a writable work directory with `results/`; `QSB_COGRIND_VERBOSE=1` prints the selected ISA, worker share, and on/off batch timings. Generated problems were created through the frozen pinning generator function only; no subset candidate or subset source was examined or edited for this submission.

The package retains `COPYING` for the GPLv3 VanitySearch-derived GPU files and `COPYING-secp256k1` for the MIT-derived host field arithmetic. The executable source and its transitive local includes are supplied. All benchmark and verifier files remain frozen. The note contains no tokens, credentials, private host paths, or private problem data. The official verifier must accept every published hit; the local tests cannot replace that ranked result.
