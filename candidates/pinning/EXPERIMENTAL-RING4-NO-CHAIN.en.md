# Pinning: four state buffers on the restored PR #2169 device path

This candidate changes QSB_SUBRING from six to four on PR #2169. It removes the CHAIN_ALU increment from the more recent, slower PR #2215 instead of carrying that increment into the queued ring experiment. The new paired RTX 4060 Ti completed-work comparison is +0.1225%. Every measured output record passed the original CPU verifier, and paired hit sets were identical. No RTX 4090 improvement has yet been demonstrated for this exact package; this is an explicitly authorized experimental evaluation.

## Official regression and correction

PR #2169 (submission 8f37bc98-b581-4f28-ab0b-7b55f759f03a) scored 1,009,707,243 verified candidates/s, with 144,640 of 144,640 reported hits verified. It was rejected because the gain did not meet the required 100 basis points over the current best. Its source commit is 74a5f8065323819c0ea9b568a9ae18f1e6745aea.

PR #2215 (submission 2f1fc205-29ac-43e0-afd1-7d38ce0acf0e) scored 999,184,474 verified candidates/s, with 143,123 of 143,123 hits verified. It was rejected for not improving the current best. Its source commit is 68d2efaeb09b4ca2ada67c9a5547fe2ffc6a86c2. Relative to #2169, the official score fell 1.0421604%.

The ratio of self-reported candidate work to official elapsed time also fell 0.9546958%. This secondary observation is not substituted for the official hit-derived score. Both runs used newly provisioned leadergpu-intel-r5 RTX 4090 runners, with different seeds and sessions. These are not controlled same-host, same-input measurements.

The executable increment between the two packages was CHAIN_ALU=1 and its regenerated native carrier. That increment also lost 1.3463% in the earlier compact local paired test. The official prepare image grew from 7,336 to 7,360 SASS instructions: IMAD count fell from 2,871 to 2,841 while IADD3 rose from 1,790 to 1,844. Its inspected loop grew from 982 to 983 instructions. Reducing one instruction category did not establish a throughput gain.

The official prepare kernel remained at 128 registers with no stack frame or spills, so an official prepare spill regression is not supported. The compact local variant did introduce an 8-byte spill. Without authenticated detailed diagnostics or runtime counters, we cannot identify the precise issue-pressure, dependency or other stall mechanism. CHAIN_ALU is the strongest actionable suspect, not a proven hardware-level cause.

The correction restores the PR #2169 device path, including its fused first-tail SHA scheduling, and applies the four-entry ring experiment there. The previously queued ring-four-plus-CHAIN_ALU package is not submitted. This is a changed implementation, not an identical retry intended to obtain another seed or runner.

## Source scope and attribution

The promoted source base is 8d07d3ebad41a017dfaa5906b164f883a9b59348. Relative to the preserved PR #2169 package, the only executable source difference is one line in candidates/pinning/pinning.cu: QSB_SUBRING is 4 instead of 6. Its native carrier remains byte-identical. The package also adds the current experiment note and a content manifest. Earlier inherited reports and SOURCE-MANIFEST.json remain as historical provenance; CANDIDATE-MANIFEST.json describes this package.

The ring-depth hypothesis was motivated by fkiene's public PR #1911, which combined this parameter with register roots, L2 policy, SM placement and other changes. Several of those changes are already present here. Its reported combined percentage cannot be attributed to ring depth alone. The CHAIN_ALU implementation from the fkiene / ItlaStudent lineage and PR #2110 is not enabled in this candidate.

The retained settings are QSB_SLOTS=3, QSB_GREEN=20, QSB_GREEN_SHARED=8, two independent root queues, the register-root startup checks, the existing persisting-table policy, the fused first-tail SHA change and the V3 CPU co-grinder. No extra host slot, CPU IFMA8 implementation, safegcd adapter, SM partition change or additional arithmetic optimization is bundled here.

All inherited GPLv3 files and copyright notices are preserved. No changes are made to the benchmark harness, verifier, problems, specification, benchmark.json, setup.sh, benchmark.sh or GitHub Actions.

## Mechanism and scheduling risks

A ring entry owns state, roots, super roots and checkpoint storage until its finish kernel completes. The global sub-batch index selects an entry modulo QSB_SUBRING. Removing two entries lowers the main-state allocation from 48 MiB to 32 MiB at 131,072 candidates and 64 bytes per candidate. Including the associated root and checkpoint buffers, the allocation reduction is 16,974,080 bytes, approximately 16.19 MiB.

A smaller ring could reduce cache pressure or improve state reuse. It also reduces overlap capacity: prepare can reach a reused entry and wait for its preceding finish sooner. Finish discards consumed state lines, so allocated bytes do not equal live L2 occupancy. The net effect depends on stage balance and the table/cache workload. This is a hypothesis rather than measured proof of a cache bottleneck.

QSB_SUBRING is separate from the host-batch slot count QSB_SLOTS, which stays at three. All ring arrays, allocation loops, event arrays and ownership flags use the ring parameter. The global sub-batch index persists across host batches. Before reusing an entry, prepare waits for that entry's preceding finish event. Root work waits for its own prepare, and finish waits for its own root. Both finish streams wait for the host slot's hit-counter reset; readback joins both finish streams before consuming and reusing the hit buffers.

An even ring depth of four preserves the prepare/root lane parity of the prior six-entry ring. No event wait, event record, candidate boundary, hit index, denominator condition or error check was removed. The tests provide integration evidence, not a formal proof of every CUDA schedule.

## Build and native image audit

Both the official candidate and the separate compact local adaptation were rebuilt using CUDA 12.8.93 with the established GCC 13 compatibility environment. Host compilation used the unchanged harness compiler interface at N=24. The native sm_89 carrier was regenerated using build_carrier.sh 24 for each geometry.

Each generated carrier header is byte-identical to its corresponding PR #2169 parent. The official carrier header SHA256 is ccd4cd25a24fcda97135bff247868a6900af45542792c913a85db6c316d82f77, and its embedded native image is c0f1a91e031e8c40f299bb714fd0b16929a79134d622ceb016b5757084b836de. The official prepare kernel uses 128 registers and 14,336 bytes of shared memory, with zero stack frame and zero spills.

The official pinning.cu SHA256 is c4f1e35628ea78a3af2410b78f085087985f9f02bb3f8aa2630f5bcedcc9a3b6. There is no finite-work override in this submitted source. The host ring change leaves device instructions and launch dimensions unchanged from #2169, while concurrency and buffer reuse can still affect overall runtime.

## Fresh local paired validation

Hardware: Ryzen 5 7500F, RTX 4060 Ti 8 GB, Windows 11 and WSL2 Ubuntu. No separate RTX 4090 was available. The official roughly 21.1-GiB table cannot provide representative local timing on the 8-GB device.

Both local arms therefore use the pre-existing compact GLV12 adaptation, with identical 30-SM prepare / 6-SM finish / 2-shared geometry. These local adaptations are excluded from the official payload. Ring depth is the only difference between the paired local arms, and CHAIN_ALU is disabled in both.

CPU co-grinding is disabled equally for this GPU-isolated diagnostic. Each measured process completes 12 full sequences, or 14,935,200,000 candidates, at N=24. Timing includes process initialization. A warm-up precedes A/B/B/A order on two fresh synthetic inputs. Every reported hit is verified by the original CPU verifier, and canonical hit identities must match in each pair.

| Seed | Six-entry seconds | Four-entry seconds | Throughput change | Verified hits per arm |
|---|---:|---:|---:|---:|
| 1262985941 | 82.742275 | 82.695422 | +0.0567% | 1,748 |
| 1836557402 | 82.599705 | 82.444181 | +0.1886% | 1,803 |

Pooled completed-work throughput is 180.658294 M/s for six entries and 180.879689 M/s for four, a +0.1225% change. All 7,102 measured output records passed verification, with identical paired hit sets. The warm-up's 1,693 hits were also verified and excluded from the comparison.

These are finite-work diagnostic rates, not official hit-derived scores. A small local difference is not evidence of a reproducible 4090 improvement. We retain both paired outcomes rather than selecting the favorable order.

## Ordinary harness integration

The unchanged ordinary harness ran the compact candidate for 35 seconds at N=24 with default CPU co-grinding enabled, without the finite-work override. The random problem seed was 251942562, and elapsed time was 35.1231 seconds. All 797 of 797 reported hits passed verification: 749 in the GPU sequence range and 48 in the CPU sequence range. The harness marked the artifact valid.

The complete artifact SHA256 is 1eef60471001d5dd03b5fd907b9b422a3a988883803200ddb59f52d7b9b62246. This ordinary run is an integration test without a paired control; its hit-derived rate is not used to claim a speedup. The harness's static GPU label is RTX_4090, but the observed physical GPU for these tests was RTX 4060 Ti.

The protected-file hashes are checked against the recorded baseline before public submission. The independently generated official and local carrier matches are recorded in the build audit. Source differences, local raw results, verifier artifacts and the fixed-work protocol are preserved in the research workspace.

## Evaluation request and uncertainty

The owner requested investigation of the official regression, correction of the queued experiment and submission of the corrected candidate. This package removes the unsupported instruction-routing increment and evaluates only ring depth on the earlier device path.

The 4090 has a different SM count, table size and cache/workload balance from the local adaptation. The smaller ring may improve reuse, make no difference, or reduce overlap enough to lose performance. Restoring the previous device code does not guarantee its earlier official score will recur.

No claimed score is supplied. No guaranteed 1% or 3% gain is asserted, and no runner, seed, score or outcome selects a configuration. The official independently verified evaluation will determine whether this complete package clears the current frontier and threshold.

Sources: https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/2169, https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/2215, https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1911 and https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/2110.
