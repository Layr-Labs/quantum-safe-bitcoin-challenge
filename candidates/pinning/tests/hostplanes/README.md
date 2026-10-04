# Host plane decoder CPU proof

This suite tests the submitted host decoder against the literal b59 baseline
(`b59a947d5c4d0ac61d2b1136ffdd7b3362010434`) and an independent OpenSSL SHA-256
oracle. It includes the exact source-extracted A/C record functions. The only
fixture adaptations are relative include paths and a closing namespace brace;
`provenance-manifest.json` records every original and public file hash.

The production `cg_sha.h` and `pksha_host_h0.h` are used by relative includes.
The small baseline H0 header is included locally so the A comparison remains
reproducible. No CUDA lifecycle or GPU execution is part of this suite.

## Build and prove

Requirements: g++ with C++17, Python3, libcrypto with SHA256/OpenSSL_version,
and an AVX2-capable CPU. SHA-NI mode2 is checked when the CPU supports it.
From `candidates/pinning/tests/hostplanes`:

```sh
g++ -O3 -std=c++17 -Wall -Wextra -fPIC -shared -pthread \
  -o /tmp/hostplanes-n24.so cpu-runner/service_bench.cpp
g++ -O3 -std=c++17 -Wall -Wextra -fPIC -shared -pthread \
  -DQSB_ZEROS_N=8 -o /tmp/hostplanes-n8.so cpu-runner/service_bench.cpp
python3 cpu-runner/run_fixture.py --library /tmp/hostplanes-n24.so --verify
python3 cpu-runner/run_fixture.py --library /tmp/hostplanes-n8.so --verify --bits 8
```

Successful output contains `verification.status="PASS"`. The runner checks
all nine variable message words and H0 for both recids across four serialized
128-lane records (1,024 comparisons), complete ordered A/C hit tuples, and
dead/batch-tail lanes at lengths 1, 3, 7, 31, 64, 127, 128. On a CPU with SHA-NI this
gives 12 full-record and 84 tail-record mode checks. Poisoning inactive X words
checks output behavior; it is not a guard-page test of physical memory reads.
The N8 build additionally checks positive recid0-only, recid1-only and
both-hit/recid0-priority publication in every available mode. Those directed
X values test hashing/publication semantics, not elliptic-curve membership.

## Optional record-service measurement

Choose an allowed CPU number and pass it as `CPU_ID` below. This command is
optional; correctness does not require timing:

```sh
python3 cpu-runner/run_fixture.py --library /tmp/hostplanes-n24.so \
  --verify --bench --calls 2048 --warm-calls 64 --groups 16 --cpu CPU_ID
```

The benchmark uses the same initialized record for every call, 64 warm calls,
and 16 groups per available mode with alternating ABBA/BAAB orders. It reports
every row, CPU time and wall time. Setup is outside the measured loops.
Mode1 forces AVX2; mode2 is the unchanged SHA-NI control. Production selects
SHA-NI when available, so a forced AVX2 gain alone does not establish default
CPU or whole-program throughput. This suite makes no GPU or score claim.
