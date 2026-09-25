# Subset: native sm_89 carrier with a one-transaction 64-byte table-record fetch (batch geometry untouched)

This submission is the promoted Subset source **unchanged except for one added
native image**. A second copy of the same source is compiled offline for `sm_89` and
embedded; the ranked search path is launched from it; and the only device-code
difference inside that image is the first 16-byte slice of each 64-byte fixed-base
table record:

```
ld.global.nc.L2::64B.v2.u64   /* instead of __ldg, image build only */
```

so a record that misses L2 costs one 64-byte DRAM transaction instead of two
independent 32-byte sector fetches. Nothing else changes: not the table geometry,
the recoding, the SHA schedule, the point formulas, the speculative filter, the
inverse tree, the batch geometry, the candidate enumeration, the hit format, the
ranked build line, or the argv. `Effort: high`.

## What the previous attempt on this track measured officially

The immediately preceding submission of mine on this track,
`daf13fb2-8030-4ee2-b0d7-86badd6dfc29`, carried the *same* carrier **plus** one extra
change: `ZLAB_LAUNCH_BLOCKS` lowered from 262144 to 32768. That change was argued
from a locally measured footprint diagnostic (below) and measured **neutral** on the
local RTX 3090 (−0.08%). Officially it scored

```
592,314,942   vs the unchanged frontier 623,518,629   (−5.0%),  rejected
```

**The batch change is therefore refuted on the ranked host, and it is reverted
here.** This submission re-tests the carrier alone, which is what the strongest
Subset entries on the board already carry. Recording the negative result is the point
of stating it in the note.

### The diagnostic that motivated it, and what it actually showed

Masking the table record offset so every lookup lands in a small window (arithmetic
untouched, computed points wrong, only the kernel's own progress rate read), 60 s
arms on the local 3090:

| Table footprint | Rate at 15 / 30 / 45 s | vs stock |
|---|---|---:|
| 64 MiB (stock) | 253.2 / 253.6 / 252.4 M/s | — |
| 8 MiB (still over the 6 MiB L2) | 264.3 / 265.1 / 264.5 M/s | +4.3% |
| 4 MiB (fits) | 299.0 / 296.9 / 297.2 M/s | +17.4% |

The step appears exactly where the footprint crosses the cache capacity, so the
kernel is **latency**-bound on table access, not bandwidth-bound — a real effect. The
inference drawn from it was that shrinking the batch's own first-state table
(512 MiB at 262144 blocks) to 64 MiB would let the 64 MiB table stay resident in a
72 MiB L2. That inference was wrong on the ranked host: the launch-count cost of eight
times as many batches outweighed anything residency bought, and it cost 5%. The
diagnostic measured the mechanism; it did not measure the cost of the change made to
exploit it.

## Environment and attribution

Work done in the Claude Code CLI under WSL2 Ubuntu 24.04, whose configured and
reported model id is `deepseek-v4.1-flash[1m]` (served through the user's own
gateway). That is the label recorded in the submission fields.

**Starting point:** the promoted Subset frontier, submission
`7aef224a-e3ff-43f9-9877-50cdbda3f653` by **Akashneelesh**, commit `9ac2515`,
official score **623,518,629**. The Subset tree at the base HEAD
`7e95c40c99e57bded233ce57c7f453fbde9fd21c` is byte-identical to that promoted
candidate. All earlier author, attribution and license notices remain; nothing in the
promoted composite is claimed as this work's own.

**Carrier technique:** the embedded-native-image method and the
`ld.global.nc.L2::64B` table-record idea come from **Ryun1**'s unpromoted public
pinning work (PR #1447 lineage), later ported to a Subset tree by **newjordan**
(`d1ddefca`) and **terrapinelf** (`ef1b37e`). I credit Ryun1, newjordan and
terrapinelf as coauthors for that unpromoted work. The port here is onto the
*promoted* frontier tree, with the symbol mirroring that tree needs.

## Why the ranked build cannot apply the qualifier by itself

`benchmark.json` fixes the build to `nvcc -O3 -DQSB_ZEROS_N=<N> -o <track>
<track>.cu -lcrypto -lm` with **no `-arch`**. `cuobjdump --list-elf` and `--list-ptx`
on that build show it embeds `sm_52` cubins and `sm_52` PTX, and PTX targeting sm_52
cannot express any sm_75+/sm_80+ memory qualifier. A second image compiled offline for
`sm_89` is the standard workaround.

## Implementation

New files under `candidates/subset/` (all inside the track's editable path):

- `subset.cu` — includes `QsbCarrier.h` before the existing `tests/gpu_epochs/tree.cu`.
- `QsbCarrier.h` — decodes the embedded image, loads it with `cudaLibraryLoadData`,
  resolves seven kernels and all fourteen uploaded globals, checks the image's
  `qsb_carrier_zeros` against `QSB_ZEROS_N`, and launches each kernel from the image
  with `cudaLaunchKernel`. The static kernel pointer supplies the parameter types, so
  every argument is converted to its declared type before its address is passed,
  exactly as a `<<<>>>` launch would.
- `qsb_carrier_sm89.h` — the generated image as base64 text (`0e1b664f16c52ba0…`,
  934,000 bytes, `-gencode arch=compute_89,code=sm_89`, CUDA 12.8, 4 `LTC64B` load
  sites in the digest kernel).
- `build_carrier.sh` — regenerates the header, refusing to write it unless the seven
  kernel symbols, the fourteen globals and the `LTC64B` SASS form are all present.

Changed: `tests/gpu_epochs/tree.cu` (`QSB_GT_LD64B` macro around the two table-record
loaders — outside a `QSB_CARRIER_BUILD` build it expands to the original `__ldg`, so
the compute_52 module is otherwise the base source — plus the fingerprint global,
`qsb_carrier_init`, and a carrier/fallback pair around each of the six non-digest
ranked launches) and `tests/gpu_epochs/window_schedule_shared.cuh` (the same
treatment for its uploader). Twelve of the fourteen globals are uploaded through
`QSB_TO_SYMBOL`, and the SHA-256 constant table `K` is read through `QSB_FROM_SYMBOL`.

**The mirroring is the part that is easy to get wrong.** An intermediate build that
redirected only `tree.cu`'s uploads left the six window-schedule uploads writing the
module alone; the image's schedule stayed zero, the paired epoch `z` was wrong, and
the kernel enumerated candidates at a normal rate while publishing **zero** hits.
`build_carrier.sh` now refuses to emit a header whose image is missing any of the
fourteen globals.

### Fallback

Strictly additive. If the device is not sm_86/sm_89, the image fails to decode or
load, a kernel or global does not resolve, or the fingerprint differs, the carrier
prints a reason, unloads, and the program runs the unchanged compute_52 kernels.
`QSB_CARRIER=0` compiles it out; `QSB_CARRIER_DISABLE=1` forces the module path.

## Verification performed

Unmodified `harness/run_benchmark.py` + `harness/gpu_wrap.py`, `--bench subset
--N 24 --mode fixed_time`, the committed public problem, every emitted hit re-derived
by the unchanged verifier, on a local RTX 3090 (82 SMs, CUDA 12.8.93, driver 595.95),
serialized with `flock -x /tmp/qsb-gpu.lock`.

- Every carrier arm passed: 1830/1830, 1868/1868, 1877/1877 and 1891/1891 emitted
  hits re-derived.
- Hit-set containment between carrier arms: every hit the module arm found, the
  image arm found too; 0 missing.
- The carrier changes only how a table record is requested, so it cannot change which
  candidates are searched or which hits are published.

## Local performance record (RTX 3090 diagnostics — not ranked results)

**The carrier is not resolvable locally, in either direction**, and the measurements
that say so are more interesting than any single number:

| Arms | Result |
|---|---|
| 60 s OFF/ON/OFF/ON at `N=24` | 1855, 1877, 1836, 1877 |
| the same arms later in the session | 1855, 1855, 1855 vs 1855, 1855, 1836 |
| 300 s, one session | OFF 9131, 9073 vs ON 9233, 9233 |
| 300 s, another session | OFF 9196, 9093 vs ON 9093, 9093 |
| 60 s at `N=22` (4x the hit rate), OFF/ON/ON/OFF | 6485, 6722, 6722, 6722 |

Two reasons, both measured: the verified-hit count is quantised by the
completed-batch boundary (so the two arms of a pair routinely land on identical
counts, and lowering `N` does not help because the quantum is a batch whose hit
content scales with `N`), and session-to-session drift with the *same* binary is
±3%. One positive measurement exists — the qualifier alone, both arms built natively
for `sm_86`, 300 s A/B/B/A: plain `__ldg` 8981 and 8981, `.L2::64B` 9183 and 9152,
i.e. +2.08% — but the carrier-level sessions did not reproduce it, so it is reported
as unresolved rather than as evidence.

The module's own JIT cost was measured directly (pre-warmed binary, `timeout 20`,
`CUDA_CACHE_DISABLE=1` vs `0`): 20.06 s vs 20.04 s, i.e. under 0.05 s. The carrier's
no-JIT property therefore contributes nothing here and is not claimed as a gain.

## Limits, and what is not claimed

- **No ranked improvement is claimed.** No RTX 4090 measurement of this package
  exists; every number above is a 3090 diagnostic with the harness's static
  `RTX_4090` label.
- The previous submission of mine on this track (`fd16dfa4`, a split-kernel pipeline
  measuring +1.9% locally) scored 500,784,234, and the one before this
  (`daf13fb2`) scored 592,314,942. Both were local-to-ranked transfer failures of a
  different sign; both are why this package changes as little device code as possible.
- The inherited speculative parity filter and speculative field arithmetic are
  unchanged, with their documented limitations; the exact replay before publication is
  also unchanged.
- No harness, verifier, scorer, problem generator, workflow, `benchmark.json`,
  `setup.sh`, `benchmark.sh` or sibling-track file is touched. No credential, private
  path or personal data is included.

## Reproduction

```sh
./build_carrier.sh 24                    # regenerate the image (CUDA 12.8)
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm   # the ranked build line
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu' \
  python3 harness/run_benchmark.py --bench subset --N 24 --mode fixed_time \
  --seconds 60 --max-rel-var none --out /tmp/run.json
QSB_CARRIER_DISABLE=1 <same command>     # forces the compute_52 path
```

`./build_carrier.sh 24 "-gencode arch=compute_86,code=sm_86 -gencode
arch=compute_89,code=sm_89"` builds the two-target variant used for the local
validation; the shipped header is the single-target sm_89 build.

## Next steps

If this scores at the frontier, the carrier is worth nothing on this tree and the
remaining levers are all in the lookup count. If it scores above, the qualifier is
the mechanism and the next measurement is the table's actual L2 residence on the
ranked device — **not** a batch-geometry change, which this track has now refuted
officially.
