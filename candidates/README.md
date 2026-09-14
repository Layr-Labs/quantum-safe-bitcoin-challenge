# Candidate kernels — baseline seed, then the promoted implementation

**These CUDA trees are the editable surface, not the scored artifact.** On
the imported commit they are the starting seed and the baseline: that commit
is scored once, and that number is the floor every later submission must beat
by `minScoreImprovementBips`. After a promotion, the same directories hold
the current implementation. You may optimize them, rewrite them, or replace
every file in a track. Two things remain fixed: the I/O contract in
`../spec/PROBLEM.md`, and the build and hit-reporting interface that
`../harness/gpu_wrap.py` drives — one `.cu` per track built with a fixed `nvcc`
line, the kernel's positional argv, and the text hit format the bridge parses.
See "Editable surface" in `../README.md`. A track is a CUDA kernel to optimize,
not a free-form slot for any runtime.

Each track is a self-contained directory (`editablePaths` in
`../benchmark.json`):

| Track | Directory | Contents |
|---|---|---|
| `pinning` | `pinning/` | `pinning.cu`, `GPUHash.h`, `GPUMath.h` |
| `subset` | `subset/` | `subset.cu`, `GPUHash.h`, `GPUMath.h` |

## Licensing

The current implementation in each track uses GPLv3-only code from
VanitySearch. Its `.cu` source, `GPUMath.h`, `GPUHash.h`, and the executable
compiled from them are therefore GPLv3-governed as one unit; the complete
license text is in that track's `COPYING` file. Preserve the existing notices
in the third-party headers and do not add Apache 2.0 labels to GPLv3-governed
candidate files.

A complete replacement that does not copy, modify, include, link to, or derive
from the GPLv3 code is covered by the repository's root Apache 2.0 license
instead. See the root README's **Licensing** section for the full boundary.

Every file under a track directory may be rewritten. The two tracks are
disjoint on purpose: a pinning edit cannot change what the subset track
builds. `../harness/gpu_wrap.py` and this README sit **outside** both
editable surfaces; everything outside the two track directories is the
judge. The benchmark is the spec + verifier + scoring in `../harness/`; a
kernel is only measured by the score its output produces (see
`../spec/PROBLEM.md` for the I/O contract).

`pinning/pinning.cu` and `subset/subset.cu` are the production QSB grinders
provided verbatim as a seed. They already read the synthetic
`../problems/<bench>.bin` unchanged (same loader format). They also carry
production-specific machinery (multi-GPU partitioning, tiles, `easy`/`calibrate`
modes) that is **not** part of the benchmark — ignore or strip it.

## The one benchmark change (already applied)

The DER-signature validity check has been replaced by the benchmark gate
(`gpu_bench_valid` in each file): **`leading_zero_bits(h) ≥ QSB_ZEROS_N`**
(no on-curve check on `h`), matching `../harness/verify.py` exactly. `N` is a compile-time
constant:

```bash
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning/pinning pinning/pinning.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -o subset/subset   subset/subset.cu   -lcrypto -lm
```
(Recompile to change `N`, or wire `N` to runtime yourself. `libssl`/`build-essential`
required; first run builds a ~128 MB secp256k1 G-table cached in `/tmp`.)

## Running the seed directly

```bash
# from candidates/ — the kernels write hits to ./results/ relative to the cwd
./pinning/pinning ../problems/pinning.bin 0 1 0      # bin, gpu_index, total_gpus, global_offset
./subset/subset   ../problems/subset.bin  0 0 0 1 0  # bin, gpu_index, seq, lt, total_gpus, global_offset
```
Run in **normal mode** (no `easy`/`calibrate`) so the leading-zeros gate is used.
The kernels print `…M/s` progress and write hits to `results/*.txt`.

## Plugging into the scoring harness

`harness/gpu_wrap.py` compiles a kernel with `-DQSB_ZEROS_N=N`, runs it under a timeout,
parses throughput + candidate count, collects hits, and writes a run artifact the
verifier scores:

```bash
# from the repository root
python3 harness/run_benchmark.py \
    --grinder "cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu"
```

> `gpu_wrap.py` is **best-effort and untested on this machine (no GPU here)** — in
> particular validate, on your first GPU run at a low `N` (many hits), that the
> harness's independent re-derivation agrees with the kernel (kernel↔verifier
> agreement). Pay attention to the subset **STORAGE index convention** (skip
> index `i` ↔ `dummy_sigs[i]`) when mapping the kernel's hit output.

## Note on faithfulness

The kernels ship precomputed `neg_r_inv` and `u2·R`, so the measured hot path is
`u1·G` (G-table scalar-mult) + point-adds + one batch modular inverse + SHA — the
real QSB grind. The biggest known lever is subset-selection's occupancy (the
~8 KB per-thread preimage buffer); GLV endomorphism and a cache-resident G-table
window are also open. None of that is required — optimize however you like.
