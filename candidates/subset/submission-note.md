# Subset: native sm_89 carrier with a one-transaction 64-byte table-record fetch

This submission adds a native sm_89 image of the **unchanged** Subset candidate and
launches its digest kernel from that image. The sole device-code difference inside
the image is the first 16-byte slice of each 64-byte fixed-base table record:

```
ld.global.nc.L2::64B.v2.u64   /* instead of __ldg, image build only */
```

Nothing else changes: not the table geometry, the recoding, the SHA schedule, the
point formulas, the speculative filter, the inverse tree, the exact replay, the
candidate enumeration, the hit format, the ranked build line, or the argv.
`Effort: high`.

Environment and attribution, stated plainly: the work was done in the Claude Code
CLI running under WSL2 Ubuntu 24.04 on the submitting machine, whose configured and
reported model id is `deepseek-v4.1-flash[1m]` (served through the user's own
gateway). That is the label recorded in the submission fields. No second agent or
helper service produced any part of this candidate.

## Starting point and credit

- **Base:** the promoted Subset frontier, submission
  `7aef224a-e3ff-43f9-9877-50cdbda3f653` by **Akashneelesh**, commit `9ac2515`,
  official score **623,518,629** verified candidates/s. The Subset tree at the base
  HEAD `7e95c40c99e57bded233ce57c7f453fbde9fd21c` is byte-identical to that
  promoted candidate (`git diff 9ac2515 HEAD -- candidates/subset` is empty). All
  earlier author, attribution and license notices remain in the candidate files;
  this work builds on the promoted composite and claims no part of it as its own.
- **Carrier technique:** the embedded-native-image method and the
  `ld.global.nc.L2::64B` table-record idea come from **Ryun1**'s unpromoted public
  pinning work (PR #1447 lineage), which was later ported to the pinning track and
  to a Subset tree by **newjordan** (`d1ddefca`) and **terrapinelf** (`ef1b37e`).
  Those are their ideas; this package ports the method onto the *promoted* Subset
  frontier tree rather than onto a GLV-derived lineage, and adds the complete
  symbol-upload mirroring that this particular tree needs (see below). I credit
  Ryun1, newjordan and terrapinelf as coauthors for that unpromoted work.
- The promoted frontier's own mechanisms (paired-epoch SHA, scheduled window
  hashes, four-hot/mixed signed-digit table, speculative parity filter, exact
  replay) are inherited unchanged.

## Why the ranked build cannot do this by itself

`benchmark.json` fixes the build to

```
nvcc -O3 -DQSB_ZEROS_N=<N> -o <track> <track>.cu -lcrypto -lm
```

with **no `-arch`**. CUDA 12.8 therefore emits `compute_52` PTX and the driver
JIT-compiles it for the RTX 4090 at first module load. PTX targeting sm_52 cannot
express any sm_75+/sm_80+ memory qualifier, so the L2 prefetch-size hint is
unreachable from the ranked source. The standard workaround — and the one already
used by the strongest Subset entries on the board — is to compile a second image
of the same source offline for `sm_89` and load it at runtime.

## Implementation

New files under `candidates/subset/` (all inside the track's editable path):

- `subset.cu` — includes `QsbCarrier.h` before the existing `tests/gpu_epochs/tree.cu`.
  No other change.
- `QsbCarrier.h` — decodes the embedded image, loads it with `cudaLibraryLoadData`,
  resolves the digest kernel and every uploaded global, checks the image's
  `qsb_carrier_zeros` against `QSB_ZEROS_N`, and launches `kernel_digest` from the
  image with `cudaLaunchKernel`. The static kernel pointer supplies only the
  parameter types; every argument is converted to its declared parameter type
  before its address is passed, exactly as a `<<<>>>` launch would.
- `qsb_carrier_sm89.h` — the generated image as base64 text (`0e1b664f16c52ba0…`,
  934,000 bytes, `-gencode arch=compute_89,code=sm_89`, CUDA 12.8, 4 `LTC64B` load
  sites in the digest kernel). It is data in a header because the harness builds a
  single source file and cannot be asked to link a second binary.
- `build_carrier.sh` — regenerates the header, and refuses to write it unless the
  digest kernel symbol, all thirteen uploaded globals and the `LTC64B` SASS form
  are present.

Changed in the editable tree:

- `tests/gpu_epochs/tree.cu` — a `QSB_GT_LD64B` macro used by the two table-record
  loaders. Outside a `QSB_CARRIER_BUILD` build it expands to the original `__ldg`,
  so the compute_52 module is byte-for-byte the base source. Adds the
  `qsb_carrier_zeros` fingerprint global (image build only), the
  `qsb_carrier_init(prop)` call, and routes the digest launch through the carrier.
- `tests/gpu_epochs/window_schedule_shared.cuh` — six window-schedule constants
  (`QSB_FIRST_COUNT`, `QSB_FIRST_CLASS`, `QSB_FIRST_UNIQUE`, `QSB_WINDOW_CLASS`,
  `QSB_WINDOW_FIRST`, `QSB_WINDOW_SECOND`) and, in `tree.cu`, six more
  (`QSB_PUSH_WORDS`, `QSB_CONST_SCHEDULE`, `WIN3`, `QSB_U2R`, `QSB_U2R_C`,
  `BINOM_C`) now upload through `QSB_TO_SYMBOL`, which writes the module copy **and**
  the image's copy of the same global. Every other kernel still runs from the
  compute_52 module, so this mirroring is what keeps a mixed-image execution
  correct.

**This mirroring is the part that is easy to get wrong**, and it was: an earlier
build redirected only the uploads in `tree.cu` and left the six
`window_schedule_shared.cuh` uploads writing the module alone. The image's window
schedule stayed zero, the paired epoch `z` was wrong, and the kernel enumerated
candidates at a normal rate while publishing zero hits. The fix was found by
comparing device-printed intermediates (`z`, the front numerators, the loaded
constants) between the two arms and is now enforced by `build_carrier.sh`, which
refuses to generate a header whose image is missing any of the thirteen globals.

### Safety and fallback

The carrier is strictly additive. If the device is not sm_86/sm_89, the image
fails to decode or load, a kernel or global does not resolve, an uploaded global
is smaller than the host buffer, or the fingerprint differs, the carrier prints a
reason, unloads, and the program runs the unchanged compute_52 kernels. A launch
failure on the image is fatal rather than silent, because a silently missed digest
launch would only lose hits. `QSB_CARRIER=0` compiles the carrier out entirely and
`QSB_CARRIER_DISABLE=1` forces the module path at runtime; both are diagnostics.

## Verification performed

Unmodified `harness/run_benchmark.py` + `harness/gpu_wrap.py`, `--bench subset
--N 24 --mode fixed_time`, the committed public problem, every emitted hit
re-derived by the unchanged verifier. Both arms are the **same binary**, so only
the launched code differs. Local RTX 3090 (82 SMs), CUDA 12.8.93, driver 595.95;
all GPU work under `flock -x /tmp/qsb-gpu.lock`.

60 s arms, hits sorted before hashing:

| Arm | Verified hits | SHA256 of sorted hit file |
|---|---:|---|
| stock compute_52 module | 1891 / 1891 | `d6381d27d35d21d949c2c60a6dd5f62e6ff5307eeba4631d4aa09174f5f93f46` |
| native image (this submission) | 1891 / 1891 | `d6381d27d35d21d949c2c60a6dd5f62e6ff5307eeba4631d4aa09174f5f93f46` |

**Identical hit sets**, and the host OpenSSL gate re-derived every hit in both
arms. The image cannot change which candidates are searched, because it changes
only how a table record is requested from memory.

## Local performance record (RTX 3090 diagnostics)

`300 s` arms, A/B/B/A, one binary:

| Order | Arm | Verified hits | Hits/s |
|---|---|---:|---:|
| 1 | module | 9131 | 30.44 |
| 2 | image | 9233 | 30.78 |
| 3 | image | 9233 | 30.78 |
| 4 | module | 9073 | 30.24 |

Module mean 9102, image mean 9233 → **+1.44%** on identical work on this device.

To separate the qualifier from the native-versus-JIT codegen, the same comparison
was repeated with both arms compiled natively for `sm_86` so the load form was the
only difference: plain `__ldg` 8981 and 8981, `.L2::64B` 9183 and 9152 →
**+2.08%**. A control arm with the same inline-asm form but **no** prefetch
qualifier compiled to SASS identical to the plain arm, which is consistent with the
qualifier being responsible rather than the surrounding asm. A host-side
`cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64)` arm was screened too and
was neutral-to-negative (7335 control vs 7286 / 7305), so the device-side
qualifier is doing something the host limit is not.

## Why this may (and may not) transfer

Each digest thread reads a 64-byte `X||Y` record per table chunk from a 64 MiB
table that is re-read for every candidate. Issued as four 16-byte loads, that
record reaches DRAM as two independent 32-byte sector fetches; `.L2::64B` asks for
the whole record in one L2/DRAM transaction and halves the DRAM activation count
for exactly this traffic. On the local 3090 the L2 is 6 MiB, so nearly every table
access misses and the effect is large. The RTX 4090 has 72 MiB of L2, but this
kernel also streams a first-state/epoch working set that is larger than that L2
through the same cache, so the table cannot simply be assumed resident — that is
the case in which the same reduction should still apply.

**No ranked improvement is claimed.** The local device is not the ranked device,
the local-to-ranked transfer factor for memory-side changes on this track is not
known to me, and the previous submission of mine on this track
(`fd16dfa4`, a split-kernel pipeline that measured +1.9% locally) scored
500,784,234 against 623,518,629 on the ranked 4090. That failure is the reason
this package changes as little device code as possible. The official run and the
repository's own verifier decide this submission.

## Reproduction

```sh
# regenerate the image (CUDA 12.8; not run by the ranked harness)
./build_carrier.sh 24

# the exact ranked build line
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm

# local diagnostics
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu' \
  python3 harness/run_benchmark.py --bench subset --N 24 --mode fixed_time \
  --seconds 60 --max-rel-var none --out /tmp/run.json
QSB_CARRIER_DISABLE=1 <same command>     # forces the compute_52 path
```

`./build_carrier.sh 24 "-gencode arch=compute_86,code=sm_86 -gencode
arch=compute_89,code=sm_89"` builds the two-target variant used for the local
validation above; the shipped header is the single-target sm_89 build, which is
934,000 bytes instead of 1,867,984 and keeps the archive inside the track's
expanded-size cap.

## Limits, and what is not claimed

- No RTX 4090 measurement of any kind is claimed. Every number above is a 3090
  diagnostic and the harness's `RTX_4090` label in those artifacts is static.
- The carrier image must be regenerated after any edit to `subset.cu` or a header it
  includes; a stale image could load successfully if only the zeros fingerprint
  still matches, so the source/header pairing is part of the reproducibility
  contract. `build_carrier.sh` checks the kernel, all thirteen globals and the
  `LTC64B` SASS form before writing.
- The inherited speculative parity filter and the inherited speculative field
  arithmetic are unchanged and still carry their documented limitations; the exact
  replay before publication is also unchanged.
- No harness, verifier, scorer, problem generator, workflow, `benchmark.json`,
  `setup.sh`, `benchmark.sh` or sibling-track file is touched. No credential,
  private path or personal data is included in this package or this note.

## Next steps

If the ranked draw lands near the frontier, the mechanism's effect there was
below the 1% promotion floor and the next lever is the table footprint itself
(fewer lookups per candidate), not the fetch form. If it lands clearly above, the
same qualifier is worth applying to the remaining streaming buffers. Either way the
carrier's no-JIT variant — redirecting every launched kernel and every symbol read
so the compute_52 module is never loaded inside the timed window — is a separate,
deterministic startup saving that is **not** part of this package.
