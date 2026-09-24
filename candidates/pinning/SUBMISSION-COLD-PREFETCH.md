# Pinning: early cold-record prefetch with per-device startup selection

Effort: xhigh. Track: pinning. Prepared with GPT 6 Astra in Codex. This is a new
memory-access experiment over our completed four-bank candidate. No NVIDIA GPU
is available locally. CPU checks and CUDA compiler evidence below establish
limited correctness and resource properties; they do not establish throughput.
The official RTX 4090 run is the first device execution of this new mechanism.

## Baseline, completed rejection and attribution

The promoted baseline is commit `1fe5a8e40008befcd917668ea9b1a23c6ee590c4`,
submission `2c7a195e-48d6-4497-8530-ecea28b042df`, at 881,273,403 verified
candidates/s. The current promotion floor, rounded up, is 890,086,138. The
inherited lineage includes odinfree's GLV12 table and anamdongparkjinhyeong's
promoted replay, plus the arithmetic and pipeline contributors credited in the
original files. All inherited licences and notices remain.

The immediate parent is our submission `3ecc74b2-b847-4b58-beb3-9670e14c5cdf`,
[PR 1382](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1382),
server commit `5ab5328d78f154d7faa5b5b88284b6ab33def959`. It completed at
**871,612,747 verified candidates/s and was rejected**. The run verified all
124,784 emitted hits over 1,200.9508 seconds. Its score was 1.096% below the
promoted record and would need a 2.119% increase to clear the current floor.
No unchanged retry is represented here as an optimization.

That parent's two table changes remain:

- 0xCramJam's four cached-bank geometry from `c13f3832`, commit
  `90f89008c599bd046906e61794218cc2a22edbcd`: four dense banks in the same
  48 MiB prefix, with two wider streaming banks. The table is 9,803,211,584
  bytes and each scalar still consumes twelve records and the existing point
  chain. 0xCramJam is a coauthor for this unpromoted contribution.
- terrapinelf's direct GPU sample readback from `86f500cc`, commit
  `8f5b6e124a47a6c07e23ea9fc10bb0be5dad474e`: retain the OpenSSL sample checks
  and fallback, while avoiding a successful-path full host mirror of the
  table. terrapinelf is a coauthor for this unpromoted contribution.

fkiene's public `871963fd` independently reproduced the parent production
files and reported a CUDA 12.8 compute_52-JIT RTX 4090 ABBA comparison of
+2.630% +/-0.165 against the promoted source, plus an N=20 verification check.
We compared its published commit `7e95c40c99e57bded233ce57c7f453fbde9fd21c`
against our submitted source: the five production files checked were byte
identical. This is third-party reported timing of the parent, not a measurement
of this prefetch candidate. The new prefetch implementation and calibration
helper are independent. fkiene is credited as a coauthor because the reported
reproduction materially informed retaining this parent for the new experiment.

Attribution clarification: a public reproduction note assigns a previous
+2.70% / +2.74% comparison to our account. Those were 0xCramJam's reported
measurements, cited in our parent note. We have not performed a local RTX 4090
benchmark. No runner-normalized or donor-ratio projection is our official score.

The top five numerical public scores were studied separately from promoted
status. At the latest refresh they were 885,300,646; 883,205,562; 882,510,673;
882,096,418; and 881,322,891, all below the promotion floor. Several are repeats
of already published sources. Their values do not demonstrate a new mechanism.
The earlier shared-anchor / 104-register overlap candidate on this account
scored 840,542,283 and was rejected. Its anchor parking, register cap and
additional coefficient/seed rewrites are absent from this successor.

## Mechanism and hypothesis

The four-bank layout puts terms 0-3 of each GLV component in the dense prefix.
Terms 4 and 5 are streaming records; across the two components their code-plane
indices are 4, 5, 10 and 11. The baseline discovers those record addresses when
the rolled point chain reaches them. A cache miss can then expose memory latency
on that chain's dependency path.

The new `qsb_prefetch_glv_cold` helper reads those four existing code words after
scalar recoding and before the seed point addition. It masks away the Y sign,
forms the record's 64-bit address, converts the generic pointer to the global
address space, and issues `prefetch.global.L2`. The default issues hints at
record offsets 0 and 32. The regular table loads still supply every operand.
No prefetched data is substituted for an arithmetic result, and a dropped hint
cannot change the mathematical computation.

[NVIDIA's PTX prefetch specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu)
describes cache-line prefetching and supports the basic form from sm_20. The
selected form therefore compiles through the organizer's compute_52 path; it
does not rely on the newer sm_80 eviction-priority qualifiers. The specification
does not prove which sectors this workload fetches or that the data survives
until use. The offset-32 hint could be redundant on the target. Actual cache
traffic, hit rates and speed remain unmeasured locally. NVIDIA documents and
source references used for this work were downloaded and studied from a local
cache outside the submitted directory.

The address calculation uses `size_t` before multiplication by 64, so records
above 4 GiB retain their upper address bits. The highest masked record index
is below 153,175,181 and the record's last byte remains inside the allocation.
All four target code planes are populated by both component decoders, including
zero-component inputs. The whole-zero scalar exits before any prefetch. For a
zero Q component with nonzero P, the two Q hints are unnecessary but point to
valid table records. No lane reads another lane's code word. The existing
cofactor handoff barrier remains unchanged.

## Startup selection on the assigned device

A fixed prefetch choice without target hardware would be a guess. With default
`QSB_COLD_PREFETCH_TUNE=1`, the first ranked pipeline invocation measures the
same preparation batch with hints off and on. `ColdPrefetchTune.h` uses CUDA
events on the preparation stream, with the mirrored order:

```
off, on, on, off, on, off, off, on
```

Each arm performs one untimed warmup followed by 32 timed preparation launches.
There are 264 calibration launches in total: 8 warmups and 256 timed launches.
The helper waits for the first slot's input uploads before starting. It waits
for each measured arm to complete before changing the device constant. The
constant remains fixed after selection for the rest of the search.

Enablement requires an improvement in each four-arm block of more than 0.5%,
plus an aggregate preparation-time reduction of more than 1%. Flat timings,
a regression, or inconsistent blocks select off. Nonpositive/nonfinite timing
or a CUDA error aborts before normal hit production. These are selection
thresholds for preparation time, not a promised official-score margin.

This calibration has an explicit cost: **all calibration wall time counts
against the official run**. It repeats the first preparation input, does not
run root inversion or finishing, and does not increment `total_searched` or
mark any slot complete. The normal first preparation is launched again after
selection, overwriting the sampled state and roots. Ordinary root processing,
finishing, readback, host verification and publication then proceed once.
Consequently the warmup cannot manufacture verified hits or candidate credit.

The launch lambda captures the original stage-0 argument list and returns its
launch status. The first call precedes `begin_roots` and all slot-busy updates;
there is no already-running finishing slot during calibration. The ranked host
launches only the true/fast-tail specialization. The compile-time non-slot
fallback does not calibrate and uses the enabled default. The helper prints
timings and selection to stderr; the stock wrapper need not retain that text
in its published run JSON. It prints no throughput-like `M/s` field.

Important limitations: preparation-only timing omits full-pipeline contention;
repeated inputs differ from the long fresh-input stream; short startup timings
can differ from later clocks and thermal state. Runtime off retains the flag
check and compiler layout of the tuned kernel. It is not identical to compiling
prefetch out. The selector reduces the risk of a fixed bad choice but does not
prove an end-to-end gain or guarantee promotion.

## Compiler observations

Toolchain: CUDA 12.8.93 in a linux/arm64 container without a CUDA device. The
organizer-default compute_52 PTX was assembled for sm_89, and the native sm_89
and default sm_52 executables were linked. The driver may generate different
code when it JITs the submitted PTX.

| Arm, default PTX assembled for sm_89 | Prepare registers | Spill store/load bytes | Static prepare instructions |
| --- | ---: | ---: | ---: |
| Parent / compile-time prefetch off | 122 | 0 / 0 | 6,688 |
| Fixed early prefetch, tuning off | 122 | 0 / 0 | 6,720 |
| Submitted early prefetch with runtime selection | 122 | 0 / 0 | 6,728 |

Preparation uses 12,288 bytes of shared memory in every arm. Finishing uses
64 registers, zero spills and 4,040 static instructions. In the tuned build,
adding the device flag relocates `pin_one_mul` by four bytes in constant memory;
finishing instructions match the parent after that single constant-address
relocation. The constant's value remains one. All other functions except
preparation match the parent. The compile-time off arm matches every parsed
parent function exactly, without address normalization.

The native sm_89 result has the same preparation/finishing resource counts.
The default sm_52 executable uses 97 preparation and 72 finishing registers,
with zero spills. Every emitted kernel and called device function is spill-free
in these builds. Eight static CCTL prefetch instructions are emitted in the
selected preparation path. More instructions can hide latency, or can slow the
kernel; static counts alone decide neither outcome.

## Correctness checks and their scope

- `test_cold_prefetch.py` compiles and executes the actual helper and call sites
  on the CPU, replacing only the PTX hints with address captures. Ten build
  configurations cover enabled/disabled prefetch, one/two addresses, early and
  staggered scheduling, both lead distances, runtime on/off and non-four-bank
  controls. Each executes 5,120 lane/component paths. Bounds, signed-code
  masking, high addresses and exact hint sequences pass UBSan.
- `test_cold_prefetch_tune.py` includes the actual production timing header with
  an ordered fake CUDA runtime. Six timing cases cover consistent gain, equal
  timing, regression, exact threshold and inconsistent blocks. It checks all
  310 API/launch failure positions and 32 invalid-timing cases, propagation of
  errors, event cleanup, upload-before-launch ordering, mode-change ordering,
  264 total launches and exactly 256 timed launches. It does not simulate GPU
  caches, scheduling or device arithmetic.
- The retained four-bank recoder test checks 402,572 signed components and
  2,415,432 digits per layout, independent integer reconstruction, exact
  rational bounds, address limits and GLV relations. It was rerun on this tree.
- The actual sparse OpenSSL table-check body with CUDA-copy stubs checks 216
  samples per geometry and rejects six injected faults per geometry. The exact
  host publication-gate check was also rerun.
- Source review confirms the calibration launch is stage 0, whose unchanged
  body writes preparation buffers then returns before hit detection. It cannot
  publish hits. The scorer, verifier, problem generation and accounting logic
  are unchanged.

These CPU and compiler checks do not constitute an end-to-end GPU hit-set test.
The exact host publication gate remains enabled for every ordinary candidate
hit. No new approximate field operation or lossy coefficient shortcut is added.

## Build and controls

```bash
python3 -B candidates/pinning/test_cold_prefetch.py
python3 -B candidates/pinning/test_cold_prefetch_tune.py
python3 -B candidates/pinning/test_fourhot.py
python3 -B candidates/pinning/test_sparse_check.py
python3 -B candidates/pinning/test_host_gate.py

git diff --check
python3 -B candidates/pinning/submission_preflight.py

nvcc -O3 -DQSB_ZEROS_N=24 candidates/pinning/pinning.cu \
  -o /tmp/pinning-cold -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -ptx candidates/pinning/pinning.cu \
  -o /tmp/pinning-cold.ptx
ptxas -arch=sm_89 -v /tmp/pinning-cold.ptx -o /tmp/pinning-cold.cubin
cuobjdump -sass /tmp/pinning-cold.cubin
nvcc -O3 -DQSB_ZEROS_N=24 -arch=sm_89 candidates/pinning/pinning.cu \
  -o /tmp/pinning-cold-native -lcrypto -lm
```

`QSB_COLD_PREFETCH=0` disables hints and calibration. `=1` hints offset zero
only; `=2` hints zero and 32. `QSB_COLD_PREFETCH_TUNE=0` fixes prefetch on.
`QSB_COLD_PREFETCH_EARLY=0` moves hints into the rolled loop, with
`QSB_COLD_PREFETCH_LEAD=1` or `=2`; these are controls, not multiple submissions.
`QSB_FOUR_HOT=0` restores the promoted three-bank geometry and disables this
four-bank prefetch helper. `QSB_GT_SPARSE_CHECK=0` restores the full host mirror.

## Evaluation decision and packaging

The mechanism advances existing cold-record reads without extending point-value
register lifetimes. It preserves spill-free preparation and adds a conservative
on-device comparison before committing to the hints. This is the reason for
one new official experiment, distinct from repeating the rejected parent.
There is **no established performance margin** for this candidate yet. The
1,200-second verified-hits result is authoritative; a rejection will require
new evidence and diagnosis, not an unchanged re-upload.

Only `candidates/pinning` is packaged. The current note is this file; older
submission notes remain historical records. `SOURCE-MANIFEST.json` lists all
packaged source/test/document hashes. Preflight audits every on-disk file,
including ignored and hidden files, checks track scope and hashes, rejects
build artifacts and enforces the archive limit. Binaries, PTX, cubins, SASS,
logs and downloaded references remain outside the editable path.
