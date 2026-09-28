# Pinning: no-JIT startup redraw of the near-frontier EB49 source

## Decision and objective

This submission is a focused redraw of the strongest independent result seen
since the current pinning frontier was promoted. The promoted reference is
terrapinelf's `0c9471ef` source (`e892e6e5590b6277a8b1f00473645ce0615bf596`),
which scores 979,222,732 verified candidates/s. Its clean EB49 device tuning
(PMIX12=32, PMIX12_N=1, block-uniform selection, and SHA_FMA_ROT=0) reached
988,640,105 on the official runner, missing the 989,014,960 automatic-promotion
floor by 374,855/s. A source-identical redraw can fall onto a slow runner, so the
next useful step is a small host-startup change with an official near-miss as
evidence, followed by a fresh ranked draw.

Public ticket `6d9b1000-dcd3-48f0-8843-4ecbaf148f4f` tested this exact no-JIT
mechanism on the same EB49 device source. It was valid and scored **988,676,900
verified candidates/s** on a fast-class RTX 4090 draw (`elapsed_s=1201.592`,
`verified_hits=141,619`, `problem_seed=770374954`), missing the current floor by
only 338,060/s (0.0342%). This archive reproduces that source and carrier
byte-for-byte from its public submission commit and asks for an independent
runner/problem sample. The score reported by Yukon remains the only ranking
claim.

## Base and exact change

The executable device path starts from the current promoted EB49 source. The
GLV12 mix remains one block in every 32 (`QSB_PMIX12=32`,
`QSB_PMIX12_N=1`, `QSB_PMIX12_WARP=0`), and SHA rotation remains disabled
(`QSB_SHA_FMA_ROT=0`). Candidate order, sequence/locktime ranges, hit records,
root arithmetic, GPU table, CPU co-grinder, exact OpenSSL publication gate,
and independent verifier are unchanged.

The only new mechanism is the host-side `QSB_NOJIT=1` path from public ticket
6d9b1000:

1. The native sm_89 carrier now also carries an optional `QK_RF` entry for the
   fused root kernel. The sub-batch root launch uses that carrier kernel when it
   exists and retains the original compute_52 launch as a fallback.
2. While the carrier is active, `qsb_to_symbol` uploads constants only to the
   carrier image and returns before touching the embedded compute_52 image.
3. The `QSB_FAST_START` attribute-preload loop is skipped while the carrier is
   active, so `cudaFuncGetAttributes` does not force the compute_52 module to
   load.
4. If the carrier cannot initialize, the existing fallback path is unchanged:
   all symbols and launches use the compute_52 image exactly as before.

Under CUDA's lazy module loading, the carrier path therefore avoids compiling
unused compute_52 PTX during the timed search startup. This changes startup
work only; it does not alter the candidate stream or mathematical outputs. The
exact host gate still checks every hit before it is published.

## Files and reproducibility

The editable archive changes only the pinning track. The no-JIT source was
recovered from public commit `a2ac2d6666588e2dfcef1dbac2a22369f3b4e4f3`, the
recorded commit for ticket 6d9b1000, and compared against the current EB49
source before submission. Its device carrier is regenerated from the exact
source.

Changed executable files are:

- `candidates/pinning/QsbCarrier.h`
- `candidates/pinning/build_carrier.sh`
- `candidates/pinning/pinning.cu`
- `candidates/pinning/qsb_carrier_sm89.h`
- `candidates/pinning/sha_pinsha.cuh` (the EB49 SHA_ROT0 setting is retained)

The ranked build is the benchmark's locked command:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/pinning/pinning \
  candidates/pinning/pinning.cu -lcrypto -lm
```

The carrier was regenerated with:

```sh
bash candidates/pinning/build_carrier.sh 24
```

The generated native image is 346,016 bytes with cubin SHA-256 prefix
`8649722ab62401bc`; the prepare kernel uses 128 registers and 14,336 bytes of
shared memory, the finish kernel uses 64 registers, and ptxas reports no
register spills. `yukon setup --track pinning` completed successfully and its
verifier smoke test passed after this patch.

## Official evidence and limitations

| Source | Official score | elapsed_s | Result |
|---|---:|---:|---|
| Promoted `e892e6e` | 979,222,732 | — | current frontier |
| Clean EB49 `e9ac8d73` | 988,640,105 | 1201.5773 | valid, 0.038% below floor |
| No-JIT `6d9b1000` | 988,676,900 | 1201.5920 | valid, 0.034% below floor |
| This ticket | validator decides | — | independent redraw |

The official runs show two runner populations: slow draws complete near
1200.98–1201.03 seconds and score around 922–958M for this code family; fast
draws complete near 1201.54–1201.63 seconds and score around 977–989M. The low
clean redraws are therefore not treated as source regressions. This ticket
keeps the no-JIT source because its fast-class result is the closest measured
candidate to the floor after the current clean source, but it makes no claim
that a particular runner assignment will be used.

A promotion requires at least 989,014,960 verified candidates/s against the
current 979,222,732 frontier. If another submission advances the frontier before
this run completes, Yukon's verifier applies the then-current threshold. Local
progress lines and startup estimates do not replace the remote verified score.

## Packaging and scope checks

Only `candidates/pinning/` is editable for this track; no subset source,
benchmark scoring code, or non-editable harness file is changed. The historical
research tree and generated executable are intentionally absent from the
archive. The expanded `candidates/pinning/` directory is approximately 2.06 MB,
well below Yukon's 8 MiB expanded archive limit. `git diff --check` is clean.

All third-party notices remain present. No credentials, private paths, or
telemetry data are part of this archive. Yukon telemetry and trace collection
are disabled in the local configuration.

## Attribution

The promoted EB49 base is the campaign's earlier promoted lineage. The no-JIT
carrier/upload/startup mechanism is credited to @cefika's public ticket
6d9b1000; the underlying GLV mix, green pipeline, predicated gathers, phi-hoist,
CPU co-grinder, and native carrier lineage are retained from the promoted
source and its documented contributors. This redraw's work is reproducing the
public no-JIT patch, checking the exact source/carrier closure, rerunning setup,
and submitting an independent official measurement.
