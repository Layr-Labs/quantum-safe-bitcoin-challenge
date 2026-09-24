# Subset launch-batch trial, 2026-09-25

Result: retain the existing production candidate. The smaller batch passed
independent hit verification but did not improve this local short comparison.
No submission or production-source change was made.

Base: repository commit `7e95c40`, `subset.cu` including
`tests/gpu_epochs/tree.cu`. The only candidate override is
`ZLAB_LAUNCH_BLOCKS=32768` instead of 262144. Enumeration, field arithmetic,
exact replay, hit encoding, 256-thread blocks, and 128 windows are unchanged.

Hypothesis: reducing batch size could reduce producer/consumer working-set
pressure and the final unpublished batch at timeout. With four epochs per
block, the first-state allocation falls from 512 MiB to 64 MiB; candidates
per complete launch fall from 134,217,728 to 16,777,216. The smaller buffer
still does not fit the RTX 3090 L2, and the tradeoff includes eight times as
many producer launches, exact replay launches, and blocking host readbacks.

Both executables were built before timing with the harness's ordinary
`nvcc -O3 -DQSB_ZEROS_N=24 ... -lcrypto -lm` command. Runs were serial with
the other trial session paused, on the same RTX 3090, CUDA 12.8, WSL.
The unchanged harness generated diagnostic seed 2026092501 independently
for each run and verified every hit. Each grinder had a 20-second limit.

| Measurement | Baseline | Smaller batch |
|---|---:|---:|
| Verified hits | 579 / 579 | 562 / 562 |
| Harness wall time | 20.1088 s | 20.1054 s |
| Hit-derived local throughput | 241.536142 M/s | 234.484547 M/s |
| Self-reported rate at approximately 15 s | 248.8 M/s | 235.1 M/s |
| Completed candidates at termination | 4,831,838,208 | 4,664,066,048 |

Hit-derived throughput fell 2.92%; completed work fell 3.47%. This is a
single ordered short comparison, not a rigorous throughput qualification:
the individual hit samples have approximately 4.2% relative uncertainty,
and startup, launch boundaries, GPU clocks, and ordering can affect it.
The counters corroborate the lack of a win but are diagnostic, not ranked.
No further GPU runs were used to pursue this rejected hypothesis.

The harness artifact inherits the repository's `RTX_4090` label; the actual
device for these results was an RTX 3090. These are unranked local numbers.
The earlier approximately 194 M/s smoke result used another run and cannot
serve as this comparison's baseline.

Reproduce in WSL with CUDA on PATH, from any working directory:

```sh
python3 /mnt/d/kongtaoxing/qsb-subset/candidates/subset/research/batch_trial/trial.py build
python3 /mnt/d/kongtaoxing/qsb-subset/candidates/subset/research/batch_trial/trial.py baseline
python3 /mnt/d/kongtaoxing/qsb-subset/candidates/subset/research/batch_trial/trial.py candidate
```

The wrapper, generated problems, run artifacts, hit records, and harness
logs stay under this directory. Build binaries and generated output are
ignored by the trial's local `.gitignore`. The build cache follows the
official wrapper's source-mtime rule; if changing included headers, rebuild
both wrappers before comparing again.
