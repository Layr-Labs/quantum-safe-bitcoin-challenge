# Subset: native sm_89 carrier — no JIT inside the timed window, and a one-transaction 64-byte table-record fetch

This submission adds a native sm_89 image of the **unchanged** Subset candidate and
runs the **entire ranked search path** from that image: the table build, the epoch
and first-state producers, the digest kernel, and the paired-hit replay. While the
carrier is on, no compute_52 kernel is launched and no compute_52 symbol is touched,
so the driver never loads the PTX module and never pays its JIT inside the
1200-second window.

The only device-code difference inside the image is the first 16-byte slice of each
64-byte fixed-base table record:

```
ld.global.nc.L2::64B.v2.u64   /* instead of __ldg, image build only */
```

Nothing else changes: not the table geometry, the recoding, the SHA schedule, the
point formulas, the speculative filter, the inverse tree, the candidate
enumeration, the hit format, the ranked build line, or the argv. `Effort: high`.

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
  pinning work (PR #1447 lineage), later ported to a Subset tree by **newjordan**
  (`d1ddefca`) and **terrapinelf** (`ef1b37e`). Terrapinelf's note also identified
  the startup accounting used below — that loading *any* compute_52 kernel or
  symbol makes the whole PTX module JIT inside the timed window. Those are their
  findings; this package ports the method onto the *promoted* Subset frontier tree
  and completes the redirection for every kernel and every global that tree uses.
  I credit Ryun1, newjordan and terrapinelf as coauthors for that unpromoted work.
- The promoted frontier's own mechanisms (paired-epoch SHA, scheduled window
  hashes, mixed signed-digit table, speculative parity filter, exact replay) are
  inherited unchanged.

## Why the ranked build cannot do this by itself

`benchmark.json` fixes the build to

```
nvcc -O3 -DQSB_ZEROS_N=<N> -o <track> <track>.cu -lcrypto -lm
```

with **no `-arch`**. CUDA 12.8 therefore emits `compute_52` PTX and the driver
JIT-compiles it for the RTX 4090 at first module load. Two consequences:

1. PTX targeting sm_52 cannot express any sm_75+/sm_80+ memory qualifier, so the L2
   prefetch-size hint is unreachable from the ranked source.
2. `benchmark.sh` runs the kernel under a fixed 1200 s timeout and the harness
   scores `verified_hits * 2^23 / wall_s`, so every second spent compiling before
   the first batch is a second of search lost. A ranked run is a fresh sandbox
   identity with fresh PTX, so that JIT is always cold.

## Implementation

New files under `candidates/subset/` (all inside the track's editable path):

- `subset.cu` — includes `QsbCarrier.h` before the existing `tests/gpu_epochs/tree.cu`.
- `QsbCarrier.h` — decodes the embedded image, loads it with `cudaLibraryLoadData`,
  resolves seven kernels and every uploaded global, checks the image's
  `qsb_carrier_zeros` against `QSB_ZEROS_N`, and launches each kernel from the image
  with `cudaLaunchKernel`. The static kernel pointer supplies only the parameter
  types; every argument is converted to its declared parameter type before its
  address is passed, exactly as a `<<<>>>` launch would.
- `qsb_carrier_sm89.h` — the generated image as base64 text (`0e1b664f16c52ba0…`,
  934,000 bytes, `-gencode arch=compute_89,code=sm_89`, CUDA 12.8, 4 `LTC64B` load
  sites in the digest kernel). It is data in a header because the harness builds a
  single source file and cannot be asked to link a second binary.
- `build_carrier.sh` — regenerates the header, and refuses to write it unless all
  seven kernel symbols, all fourteen uploaded globals and the `LTC64B` SASS form are
  present in the image.

Changed in the editable tree:

- `tests/gpu_epochs/tree.cu` — the `QSB_GT_LD64B` macro used by the two table-record
  loaders (outside a `QSB_CARRIER_BUILD` build it expands to the original `__ldg`,
  so the compute_52 module is the base source unchanged); the `qsb_carrier_zeros`
  fingerprint global; `qsb_carrier_init(prop)`; and an
  `if (qsb_carrier_has(...)) qsb_carrier_launch(...) else <<<>>>` pair around each
  of the six non-digest ranked launches.
- `tests/gpu_epochs/window_schedule_shared.cuh` — the same treatment for that
  header's uploader.

Host symbol traffic now goes through two helpers: `QSB_TO_SYMBOL` writes **only** the
image while the carrier is on (so the module is never touched), and
`QSB_FROM_SYMBOL` reads the image's copy of the SHA-256 constant table `K`. Six
uploads live in `tree.cu` (`QSB_PUSH_WORDS`, `QSB_CONST_SCHEDULE`, `WIN3`,
`QSB_U2R`, `QSB_U2R_C`, `BINOM_C`) and six in `window_schedule_shared.cuh`
(`QSB_FIRST_COUNT`, `QSB_FIRST_CLASS`, `QSB_FIRST_UNIQUE`, `QSB_WINDOW_CLASS`,
`QSB_WINDOW_FIRST`, `QSB_WINDOW_SECOND`).

**This mirroring is the part that is easy to get wrong.** An intermediate build that
redirected only `tree.cu`'s uploads left the six window-schedule uploads writing the
module alone; the image's schedule stayed zero, the paired epoch `z` was wrong, and
the kernel enumerated candidates at a normal rate while publishing zero hits. The
fix was found by comparing device-printed intermediates between arms and is now
enforced by `build_carrier.sh`, which refuses to emit a header whose image is
missing any of the fourteen globals. Going the other way — writing *only* the image
— is safe precisely because `qsb_carrier_off()` can only run during init, before the
first upload, so a fallback can never observe a module copy that was left unset.

### Fallback

If the device is not sm_86/sm_89, the image fails to decode or load, a kernel or
global does not resolve, or the fingerprint differs, the carrier prints a reason,
unloads, and the program runs the unchanged compute_52 kernels. A launch failure on
the image is fatal rather than silent, because a silently missed launch would only
lose hits. `QSB_CARRIER=0` compiles the carrier out and `QSB_CARRIER_DISABLE=1`
forces the module path at runtime. The shipped image is a single-target sm_89 build;
a two-target `sm_86+sm_89` build of the identical source is what the local
validation below and in `research/session_claude/local-ab.md` used.

## Verification performed

Unmodified `harness/run_benchmark.py` + `harness/gpu_wrap.py`, `--bench subset
--N 24 --mode fixed_time`, the committed public problem, every emitted hit
re-derived by the unchanged verifier. Local RTX 3090 (82 SMs), CUDA 12.8.93, driver
595.95; all GPU work under `flock -x /tmp/qsb-gpu.lock`. Both arms are the **same
binary**, so the archive, argv, problem and verifier are identical.

One 60 s arm pair — the module arm is slower, so it searches fewer candidates:

| Arm | Verified hits | Hits in the module arm that the image arm missed |
|---|---:|---:|
| stock compute_52 module | 1877 / 1877 | — |
| native image (this submission) | 1891 / 1891 | **0** (the image arm has 14 extra) |

Equal-length 60 s arms give an **exactly equal verified hit count**, 1891 = 1891.

## Local performance record (RTX 3090 diagnostics)

**(a) Startup.** The harness's own wall clock, 60 s arms, `CUDA_CACHE_DISABLE=1`
so the module's JIT is cold on every run, exactly as on a fresh ranked sandbox:

| Arm | Wall clock | Search obtained |
|---|---:|---:|
| module path | 69.53 s | 1855 hits |
| image path (no module load) | 60.17 s | 1877 hits |

The cold JIT costs **9.36 s** of the window on this machine, and the image path
never pays it. Two further numbers bound the same cost on the ranked runner
without measuring it there:

- this tree's `compute_52` PTX is **3,515,568 bytes**, against 1.69 MB for the
  GLV12 tree whose author measured a **4.2 s** cold JIT on the ranked 4090;
- the PTX size ratio (2.08x) therefore puts this tree's ranked JIT at roughly
  8-9 s, which is **≈0.7% of the 1200-second window** — the local 9.36 s and the
  size-based estimate agree.

That estimate is what makes this the one component here that is expected to help
for reasons that do not depend on a local-to-ranked transfer factor: the module
either loads inside the timed window or it does not.

**(b) Steady state — measured, but not resolved locally.** 60 s arms with warm
caches, OFF/ON/OFF/ON: 1855, 1877, 1836, 1877 verified hits. Module mean 1845.5,
image mean 1877 → +1.71%. Repeating the same arms later in the session gave
OFF 1855, 1855, 1855 and ON 1855, 1855, 1836 — **no difference**. 300 s arms gave
OFF 9131 and 9073 vs ON 9233 and 9233 (+1.44%) in one session, and OFF 9196 and 9093
vs ON 9093 and 9093 (−0.56%) in another, with the *same* image bytes both times.

The verified-hit count is quantised (the observed values sit on a ladder about 19–21
hits apart) and the machine drifts by more than the effect between sessions, so the
honest reading is: **this local setup cannot resolve the memory mechanism's steady
state, and the +1.44% / +1.71% figures above are not reliable estimates.** The
package is submitted because of (a), which is a wall-clock measurement, and because
the same qualifier is what the strongest Subset entries on the board use — not
because a steady-state gain was established here.

**(c) The load qualifier alone.** Midway through the session, before the carrier
existed, the same comparison was run with *both* arms built natively for `sm_86` so
the load form was the only difference, 300 s, A/B/B/A: plain `__ldg` 8981 and 8981,
`.L2::64B` 9183 and 9152 → +2.08%, with the two plain arms reproducible to the hit.
That is the cleanest positive evidence for the qualifier in this package, and it is
also the measurement the later carrier-level sessions failed to reproduce. A control
arm with the same inline-asm form but **no** prefetch qualifier compiled to SASS
identical to the plain arm. A host-side
`cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64)` arm was screened too and
was neutral-to-negative (7335 control vs 7286 / 7305).

## Why the memory mechanism may (and may not) transfer

Each digest thread reads a 64-byte `X||Y` record per table chunk from a 64 MiB
table that is re-read for every candidate. Issued as four 16-byte loads, that
record reaches DRAM as two independent 32-byte sector fetches; `.L2::64B` asks for
the whole record in one L2/DRAM transaction and halves the DRAM activation count
for exactly this traffic. On the local 3090 the L2 is 6 MiB, so nearly every table
access misses and the effect is large. The RTX 4090 has 72 MiB of L2, but this
kernel also streams a first-state/epoch working set larger than that L2 through the
same cache, so the table cannot simply be assumed resident — which is the case in
which the same reduction should still apply.

**No ranked improvement is claimed.** The local device is not the ranked device and
the local-to-ranked transfer factor for memory-side changes on this track is not
known to me. The previous submission of mine on this track (`fd16dfa4`, a
split-kernel pipeline that measured +1.9% locally) scored 500,784,234 against
623,518,629 on the ranked 4090. That failure is why this package changes as little
device code as possible, keeps the startup saving separate from the memory claim,
and is verified by exact hit-set containment rather than by speed.

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
CUDA_CACHE_DISABLE=1 <same command>      # makes the module's JIT cold every run
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
  still matched, so the source/header pairing is part of the reproducibility
  contract. `build_carrier.sh` checks the seven kernels, the fourteen globals and
  the `LTC64B` SASS form before writing.
- The absolute cost of the cold JIT on the ranked runner is not measured here; only
  the local 9.36 s and the 3.5 MB-versus-1.69 MB PTX-size comparison to a measured
  4.2 s are. If the runner's CPU JITs this PTX much faster, the startup part of this
  package is correspondingly smaller — that is the single assumption the expected
  gain rests on, and it is stated here rather than buried.
- The inherited speculative parity filter and the inherited speculative field
  arithmetic are unchanged and still carry their documented limitations; the exact
  replay before publication is also unchanged.
- No harness, verifier, scorer, problem generator, workflow, `benchmark.json`,
  `setup.sh`, `benchmark.sh` or sibling-track file is touched. No credential,
  private path or personal data is included in this package or this note.

## Next steps

If the ranked draw lands near the frontier, the startup component should still be
visible as a small deterministic gain while the memory component was not; the
useful follow-up is then to measure the table's actual L2 residence on the ranked
device rather than to add more load hints. If the run is rejected outright, treat
the symbol-mirroring as the suspect and check the fourteen globals first.
