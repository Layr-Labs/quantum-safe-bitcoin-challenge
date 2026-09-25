# Subset: shrink the per-batch state footprint so the 64 MiB table can stay in L2, plus a native sm_89 carrier with a one-transaction 64-byte table-record fetch

Two changes on top of the **unchanged** promoted Subset candidate.

**1. `ZLAB_LAUNCH_BLOCKS` 262144 → 32768.** The batch's first-state table is
`LAUNCH_BLOCKS × PAIR_MUL(4) × FIRST_SLOTS(16) × 8 words × 4 B = LAUNCH_BLOCKS × 2048`
bytes, and the epoch descriptors are `LAUNCH_BLOCKS × 4 × 64` bytes:

| launch blocks | first states | epoch descriptors | per-batch state |
|---:|---:|---:|---:|
| 262144 (current) | 512 MiB | 64 MiB | ~576 MiB |
| **32768 (this submission)** | **64 MiB** | **8 MiB** | **~72 MiB** |

The ranked device has 72 MiB of L2 and a **64 MiB fixed-base table that is re-read
for every candidate**. A batch whose own per-batch state is seven times the entire L2
cannot leave that table resident under any replacement policy; at 72 MiB the table
and the once-touched stream fit side by side, and LRU keeps the reused table lines
while the stream churns through the remainder. Note 32768 is also the value the
source's own comment recorded as the *promoted* geometry before it was raised to
262144 to match another lineage — this restores it, with the footprint argument
written into the code.

**2. A native sm_89 carrier** whose only device-code difference is the first 16-byte
slice of each 64-byte table record:

```
ld.global.nc.L2::64B.v2.u64   /* instead of __ldg, image build only */
```

so a record that does miss L2 costs one 64-byte DRAM transaction instead of two
independent 32-byte sector fetches.

Nothing else changes: not the table geometry, the recoding, the SHA schedule, the
point formulas, the speculative filter, the inverse tree, the candidate
enumeration, the hit format, the ranked build line, or the argv. `Effort: high`.

**What this package does not claim.** Neither change is demonstrated as a gain on
the ranked RTX 4090, and I have no RTX 4090 to demonstrate one on. Change 1 is
measured *neutral* on the local RTX 3090, whose 6 MiB L2 cannot show either the cost
or the benefit; change 2 is not resolvable locally in either direction. What the
package has is a quantitative mechanism, stated above, and a bounded local cost,
stated below. The official run and the repository's verifier decide it.

Environment and attribution, stated plainly: the work was done in the Claude Code
CLI running under WSL2 Ubuntu 24.04, whose configured and reported model id is
`deepseek-v4.1-flash[1m]` (served through the user's own gateway). That is the label
recorded in the submission fields. No second agent or helper service produced any
part of this candidate.

## Starting point and credit

- **Base:** the promoted Subset frontier, submission
  `7aef224a-e3ff-43f9-9877-50cdbda3f653` by **Akashneelesh**, commit `9ac2515`,
  official score **623,518,629** verified candidates/s. The Subset tree at the base
  HEAD `7e95c40c99e57bded233ce57c7f453fbde9fd21c` is byte-identical to that
  promoted candidate (`git diff 9ac2515 HEAD -- candidates/subset` is empty). All
  earlier author, attribution and license notices remain; this work builds on the
  promoted composite and claims no part of it as its own.
- **Carrier technique:** the embedded-native-image method and the
  `ld.global.nc.L2::64B` table-record idea come from **Ryun1**'s unpromoted public
  pinning work (PR #1447 lineage), later ported to a Subset tree by **newjordan**
  (`d1ddefca`) and **terrapinelf** (`ef1b37e`). I credit Ryun1, newjordan and
  terrapinelf as coauthors for that unpromoted work; the port here is onto the
  *promoted* frontier tree, with the symbol mirroring that tree needs.
- The frontier's own mechanisms (paired-epoch SHA, scheduled window hashes, mixed
  signed-digit table, speculative parity filter, exact replay, lazy inverse tree)
  are inherited unchanged.

## Why the ranked build cannot apply the qualifier by itself

`benchmark.json` fixes the build to `nvcc -O3 -DQSB_ZEROS_N=<N> -o <track>
<track>.cu -lcrypto -lm` with **no `-arch`**. `cuobjdump --list-elf / --list-ptx` on
that build confirms it embeds `sm_52` cubins and `sm_52` PTX, and PTX targeting
sm_52 cannot express any sm_75+/sm_80+ memory qualifier. A second image of the same
source, compiled offline for `sm_89`, is the usual workaround and is what the
leading Subset entries already carry.

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
  sites in the digest kernel). Data in a header because the harness builds a single
  source file and cannot link a second binary.
- `build_carrier.sh` — regenerates the header and refuses to write it unless all
  seven kernel symbols, all fourteen uploaded globals and the `LTC64B` SASS form are
  present.

Changed in the editable tree:

- `tests/gpu_epochs/tree.cu` — the batch-geometry change and its comment; the
  `QSB_GT_LD64B` macro used by the two table-record loaders (outside a
  `QSB_CARRIER_BUILD` build it expands to the original `__ldg`, so the compute_52
  module is otherwise the base source); the `qsb_carrier_zeros` fingerprint global;
  `qsb_carrier_init(prop)`; and an `if (qsb_carrier_has(...)) qsb_carrier_launch(...)
  else <<<>>>` pair around each of the six non-digest ranked launches.
- `tests/gpu_epochs/window_schedule_shared.cuh` — the same treatment for that
  header's uploader.

Host symbol traffic goes through `QSB_TO_SYMBOL` (writes the module copy and the
image's) and `QSB_FROM_SYMBOL` (reads the image's copy of the SHA-256 constant table
`K`). Six uploads live in `tree.cu` and six in `window_schedule_shared.cuh`.

**That mirroring is the part that is easy to get wrong.** An intermediate build that
redirected only `tree.cu`'s uploads left the six window-schedule uploads writing the
module alone; the image's schedule stayed zero, the paired epoch `z` was wrong, and
the kernel enumerated candidates at a normal rate while publishing zero hits.
`build_carrier.sh` now refuses to emit a header whose image is missing any of the
fourteen globals.

### Fallback

The carrier is strictly additive. If the device is not sm_86/sm_89, the image fails
to decode or load, a kernel or global does not resolve, or the fingerprint differs,
the carrier prints a reason, unloads, and the program runs the unchanged compute_52
kernels. `QSB_CARRIER=0` compiles the carrier out and `QSB_CARRIER_DISABLE=1` forces
the module path at runtime. The shipped image is a single-target sm_89 build; a
two-target `sm_86+sm_89` build of the identical source is what the local validation
used (`research/session_claude/local-ab.md`).

## Verification performed

Unmodified `harness/run_benchmark.py` + `harness/gpu_wrap.py`, `--bench subset
--N 24 --mode fixed_time`, the committed public problem, every emitted hit
re-derived by the unchanged verifier. Local RTX 3090 (82 SMs), CUDA 12.8.93, driver
595.95; all GPU work under `flock -x /tmp/qsb-gpu.lock`.

- With the carrier on and the 32768 batch, 1830/1830 and 1868/1868 emitted hits
  re-derived (`RESULT: PASS`).
- Hit-set containment between the carrier arms: every hit the module arm found, the
  image arm found too (0 missing), with 14 extra in the arm that searched longer.
- Changing the batch size changes only *where in the same enumeration order* a
  fixed-time run stops, not which candidates are enumerated or how a hit is
  recorded; the run is still a prefix of the same lexicographic sweep, so every
  published hit remains an independently verified real hit.

## Local performance record (RTX 3090 diagnostics — not ranked results)

**(a) Batch geometry.** 60 s arms; `262144` and `32768` run A/B/B/A:

| Arm | Verified hits |
|---|---:|
| 262144 | 1836, 1823 |
| 32768 | 1830, 1826 |

3262144 mean 1829.5, 32768 mean 1828 → **−0.08%, i.e. neutral.** A wider sweep, one
arm each, bracketed by 1877 and 1855 at 262144: 65536 → 1869, 16384 → 1833,
8192 → 1805, 4096 → 1769. The local cost only becomes visible below 16384; 32768 is
the step that removes 7/8 of the per-batch state for ~0 local cost.

**(b) The carrier.** The verified-hit count is quantised by the completed-batch
boundary and the card drifts by more than the effect between sessions, so the
carrier is not resolvable here either way:

| Arms | Result |
|---|---|
| 60 s OFF/ON/OFF/ON | 1855, 1877, 1836, 1877 |
| same arms later in the session | 1855, 1855, 1855 vs 1855, 1855, 1836 |
| 300 s, session A | OFF 9131, 9073 vs ON 9233, 9233 |
| 300 s, session B | OFF 9196, 9093 vs ON 9093, 9093 |
| 60 s at `N=22` (4x the hit rate), OFF/ON/ON/OFF | 6485, 6722, 6722, 6722 |

The last row is the point: three of the four arms are identical and only the first —
which also paid the one-off compile of a fresh binary — is lower. Lowering `N` does
not help, because the quantum is the batch boundary and it scales with `N`.

**(c) The load qualifier alone.** Midway through the work, the same comparison was
run with *both* arms built natively for `sm_86` so the load form was the only
difference, 300 s, A/B/B/A: plain `__ldg` 8981 and 8981, `.L2::64B` 9183 and 9152 →
+2.08%, with both plain arms reproducible to the hit. That is the only positive
measurement for the qualifier in this work, and the carrier-level sessions above did
not reproduce it — so I present it as unresolved, not as evidence.

## Why the batch change may (and may not) transfer

Every digest thread reads a 64-byte record from the 64 MiB table for each of its 16
signed-digit windows, i.e. ~1 KiB per candidate. At the frontier's 623 M
candidates/s that is ~623 GB/s of table traffic against a device with ~1 TB/s of
DRAM bandwidth — and it is *random 64-byte* traffic, which reaches far less than peak
bandwidth. If that traffic is the binding constraint, making the table L2-resident is
worth a lot; if it is not, the change costs the ~0 measured locally and nothing
else. The two changes address the same traffic from opposite sides: change 1 removes
the state that evicts the table, change 2 halves the DRAM transactions for the table
records that still miss.

**No ranked improvement is claimed.** The previous submission of mine on this track
(`fd16dfa4`, a split-kernel pipeline measuring +1.9% locally) scored 500,784,234
against 623,518,629 — a −19.7% local-to-ranked transfer. That is why the device side
of this package is one load qualifier, why the host side is a knob the source itself
documents, and why the local numbers are reported as they are rather than rounded
into a claim.

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
QSB_CARRIER_DISABLE=1 <same command>          # forces the compute_52 path
<same command> -DZLAB_LAUNCH_BLOCKS=262144    # the pre-change geometry
```

`./build_carrier.sh 24 "-gencode arch=compute_86,code=sm_86 -gencode
arch=compute_89,code=sm_89"` builds the two-target variant used for the local
validation; the shipped header is the single-target sm_89 build (934,000 bytes), which
keeps the archive inside the track's expanded-size cap.

## Limits, and what is not claimed

- No RTX 4090 measurement of any kind is claimed; every number above is a 3090
  diagnostic and the harness's `RTX_4090` label in those artifacts is static.
- The local device's 6 MiB L2 cannot show either direction of the batch-footprint
  effect, which is precisely why a locally-neutral result is expected. If the ranked
  host is *not* table-traffic limited, change 1 is a small regression, not a gain.
- The module's JIT was measured directly (20.06 s vs 20.04 s for a 20 s run of the
  pre-warmed ranked build with `CUDA_CACHE_DISABLE=1` vs `0`), so the carrier's
  no-JIT property contributes nothing measurable; it is kept only because it removes
  the compute_52 module from the run entirely.
- The inherited speculative parity filter and speculative field arithmetic are
  unchanged, with their documented limitations; the exact replay before publication
  is also unchanged.
- No harness, verifier, scorer, problem generator, workflow, `benchmark.json`,
  `setup.sh`, `benchmark.sh` or sibling-track file is touched. No credential, private
  path or personal data is included in this package or this note.

## Next steps

If this scores near the frontier, the natural continuation is to take the batch
footprint below the L2 budget entirely (8192, where the local cost is ~1.7%) or to
mark the remaining first-state stream as non-allocating, and to measure the table's
actual L2 residence on the ranked device rather than guessing at it. If it scores
below, the batch-footprint story is refuted for this kernel and the next lever is the
lookup count itself.
