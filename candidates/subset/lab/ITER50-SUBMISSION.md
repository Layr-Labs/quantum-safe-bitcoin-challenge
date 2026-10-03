# Subset: descriptor-only host ring and dense first-state GPU prepass

Effort: xhigh

## Base and attribution

This is an incremental subset-track candidate on the retained qualified production tree. The promoted comparison is kshitij-hash's `faf5422a`, commit `efef868ab78ff8d1229cdc1591b90797ee3be197`, scored 753,571,538 verified candidates per second. The underlying project is GPL-3. Credit remains with the QSB authors and inherited public optimizations: Jean Luc Pons's VanitySearch arithmetic, Ryun1's field arithmetic, host co-grinder and native carrier design, kshitij-hash's paired subset pipeline, cefika's contiguous CPU walk, and jacklightChen's co-grinder partition and cuts. Existing source notices remain intact. This note does not claim those contributions as new.

The new change concerns who computes eight first-block SHA states per epoch and whether the host allocates, computes and uploads duplicate copies. It does not alter ECDSA recovery, candidate enumeration, the omission count, window patterns, the exact OpenSSL publication gate, or the independent producer self-check. It does not modify the benchmark verifier or score definition.

## Hypotheses and distinct initial screens

A previous default-off diagnostic placed timestamp boundaries inside the repeated digest. Paired SHA occupied about one quarter of its instrumented elapsed intervals. Those were perturbed intervals, not removable-cost estimates or RTX 4090 predictions. This experiment tests a distinct producer/consumer hypothesis instead of trying to infer savings from that profile.

Production materializes eight first states per epoch in host producer threads. A full batch has 2,097,152 descriptors and 16,777,216 first states. The first-state transfer is 512 MiB; four pinned ring slots are 640 MiB each, including descriptors. Three screens separate consumer effects, transfer effects and host-computation/allocation effects. All use N=24, synthetic seed 1789110211, the existing benchmark verifier and 120-second windows followed by complete draining.

* CTA-local first states: build sixteen states inside the paired digest CTA in 512 bytes of existing parking shared memory, with publication and reader-retirement barriers. Host computation and allocation remain. Exact verification passed 10,935 hits, but throughput fell 388.675150 to 365.742002 M/s (-5.900338%). This variant stays disabled; no rescue redraw.
* Dense prepass only: build the same global states once per batch after descriptor upload and omit the 512-MiB first-state upload. Digest consumption stays unchanged; host computation and allocation remain. Exact verification passed 11,211 hits, but throughput fell 387.229028 to 384.217482 M/s (about -0.7777%). This isolated variant is recorded as negative.
* Complete descriptor-only pipeline: additionally omit unused steady host first-state computation and pinned first arrays. Exact verification passed 10,779 hits. The initial screen was 371.196026 to 372.622976 M/s (+0.3844%). That single small pair was not treated as repeatable evidence; it triggered the actual-leader qualification below.

These screens used the fixed benchmark build's compute_52 PTX/driver-JIT route on an RTX 3090, not the ranked RTX 4090. They are paired observations, not cross-machine forecasts.

## Enabled source change

`QSB_FIRST_DEVICE=1` enqueues the existing flat first-state builder immediately after a host-produced descriptor upload, on the same CUDA stream. Its ordinary 256-thread producer blocks use dense lanes instead of inserting sparse SHA work and two barriers in the digest. GPU descriptor fallback already builds first states and continues to do so.

`QSB_FIRST_HOSTLESS=1` makes steady host slots descriptor-only. Both SHA-NI and software producer paths emit the complete descriptor before skipping first-state computation when its destination pointer is null. Full steady slots shrink from 640 to 128 MiB. Across four slots, this removes two GiB of pinned first-state storage. Device state buffers remain allocated and are initialized by the prepass; this is not a VRAM-saving claim.

The independent batch-zero checker is deliberately unchanged. It allocates complete CPU descriptor and first-state arrays, computes every state, and compares against independently GPU-built data. Host steady batches are enabled only after comparison succeeds. Logs report all 2,097,152 descriptors and 16,777,216 first states bit-identical. Checker failure disables host production and selects GPU descriptor/state production, never stale first states.

Every required descriptor copy precedes the pinned-slot copied event. That event protects host-source reuse; subsequent kernels consume device copies. The prepass and digest are ordered on the same stream, and separate device-slot completion fences protect descriptor/state buffer reuse. Descriptor-only mode never copies a null first-state pointer. Cleanup is null-safe. Compile guards reject incompatible layouts and producer versions. `QSB_FIRST_LOCAL` remains off.

## Compiler-route correction and actual-leader qualification

The retained actual promoted leader binary is native sm_86. An initial nine-run alternation accidentally compared it with PTX/JIT candidate and production arms. Those runs passed 48,336 exact hits and numerically looked positive, but were EXCLUDED from qualification after fatbin inspection revealed the discrepancy. The evidence remains retained. Correcting a mismatched comparator is not a noise rescue of either negative screen.

The corrective experiment uses matched native sm_86 candidate and production builds and the retained native leader. Its order is L/H/P, P/H/L, L/H/P: three 120-second runs per arm, same N and seed. The retained leader executable SHA-256 is `a673445b0e337e690fb4d664640259f6b8840a09d7a5eb9dd013217233c83088`. All three fatbins target sm_86. Scores below are verified M/s on the local RTX 3090:

| Pair | Promoted leader | Hostless candidate | Production | vs leader | vs production |
|---:|---:|---:|---:|---:|---:|
| 1 | 354.425057 | 375.709537 | 367.687354 | +6.0054% | +2.1818% |
| 2 | 358.484151 | 375.328030 | 364.904262 | +4.6986% | +2.8566% |
| 3 | 351.659176 | 375.001824 | 336.055682 | +6.6379% | +11.5892% |

Means: leader 354.856128 M/s, candidate 375.346464 M/s, production 356.215766 M/s. Candidate mean is +5.774266% versus the actual leader and +5.370536% versus production. Every leader comparison exceeds the preregistered +4% local gate; every production increment is positive. All 47,123 hits across these nine runs passed the independent benchmark verifier.

The third production control dropped noticeably. Its large increment and the mean production increment are NOT stable removable-cost estimates. The first two increments are the less exceptional +2.1818% and +2.8566% observations. Candidate scores remain 375.001824–375.709537 M/s. Finite local windows under local thermals and scheduling are not confidence bounds or a guarantee on ranked hardware.

## Commands, compiler evidence and verification

Benchmark entrypoint: `./benchmark.sh subset`, with `QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS=120 QSB_PROBLEM_SEED=1789110211 QSB_MAX_REL_VAR=none`. A lab adapter supplies each precompiled arm and replaces the external wall timeout with a 120-second SIGTERM followed by complete natural draining. It changes neither problem generation, hit verification nor scoring. CUDA work was serialized with the operator's GPU lock.

Matched local builds use `nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1 -arch=sm_86`, with ordinary OpenSSL and math dependencies. Carrier regeneration uses `bash candidates/subset/build_carrier.sh`. The fixed runner build remains the ordinary `nvcc -O3 -DQSB_ZEROS_N=24` route. The enabled final source is separately built through that fixed route and passed through the same benchmark verification before packaging.

Dense-prepass and hostless device images are byte-identical to each other: existing producer math and digest instructions are unchanged. Digest kernels remain at 128 registers, 24 KiB shared memory, zero stack and zero spill stores/loads on inspected native sm_86 and sm_89 images. Enabled candidate images match the measured images. Disabling the first-state switches reproduces production cubins byte-for-byte.

## Lossless carrier transport and package integrity

Native-image base64 embedding is now losslessly LZ4-HC packed. Decoded CUDA image size stays 473,504 bytes. A bounded host-only raw-block decoder reconstructs it before CUDA loads it; the runner needs no LZ4 runtime library. The development generator optionally uses liblz4 to pack, or emits legacy raw base64 if unavailable. Both formats are understood by the loader. The production decoder and generator were checked against byte-exact native images. This packaging is not cryptographic proof or a measured throughput benefit.

All inherited evidence remains losslessly reconstructible. New measurements retain source, compiler manifests, build logs, exact score/run records and provenance. Public notes preserve inherited attribution. Package size is checked against the eight-MiB editable-surface cap; a compression or integrity test is never substituted for cryptographic hit verification.

## Caveats, disposition and next steps

The local GPU is an RTX 3090. Benchmark JSON's RTX_4090 label is harness metadata, not the device used here. Ranked RTX 4090 performance is unknown at submission. Some runs warn that the self-reported candidate counter excludes co-grinder work; scoring uses independently verified hits and elapsed time, not that counter. No real coin material was used and no hits were removed to improve a variance band.

Only the complete descriptor-only dense-prepass path is enabled. The negative isolated variants remain recorded and disabled. The board was rechecked after matched qualification and still named faf5422a as promoted leader. This submission follows the operator's instruction to submit qualified local winners rather than hold them for a check-in. If ranked throughput does not improve, the next materially distinct experiment is gather latency versus field arithmetic within the recovery fronts, not a redraw of the negative CTA-local variant.


## Final enabled-source and retention check

After enabling the qualified switches and regenerating the carrier, the existing
benchmark entrypoint passed all 5,319 hits in a separate fixed-build/PTX-route
120-second run (368.253156 verified M/s). That last run is a correctness check,
not folded into the native-sm86 leader qualification. Enabled sm86 and sm89
images reproduce the measured candidate cubins byte-for-byte. The complete
screens, mixed-route qualification, matched qualification, and final run logs,
hit artifacts, image manifests and compiler analysis are retained with hashes.
A joint XZ archive shares dictionary matches with the prior evidence. Its prior
archive member is byte-exact; no old hit, negative result or timestamp sample
was removed. The original source snapshots and first-stage control carrier are
also reconstructible byte-exact. `lab/iter50-state-audit.py` checks reconstruction,
retained verified-hit counts, the gate and the editable-tree package cap. It is
an integrity audit, not a replacement for the existing cryptographic verifier.
The final package is below 8,388,608 bytes. Official score remains unknown.
