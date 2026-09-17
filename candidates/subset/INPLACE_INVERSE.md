# In-place packed inverse experiment

Goal: exceed 495,193,826 verified subset candidates/s on the ranked RTX 4090.
Starting checkout: 372a325; subset frontier: 99ce841 (495,193,826).

## Design and implementation plan

- Add a selectable `ZLAB_TREE=3` inverse implementation with the same field
  arithmetic and level-packed products as variant 2.
- Reuse internal product slots for inverses. A single thread loads both
  children before replacing either, avoiding cross-thread read/write races.
- Preserve the leaf products until every lane computes its final inverse.
- Test the 16 KiB shared-memory budget before implementation, then exercise
  the actual helper with CPU threads and separate warp/block barriers.
- Select variant 3 in the ranked entry point for remote evaluation, as requested
  by the user. Variant 2 remains available through `subset_baseline.cu`.
  GPU correctness and performance remain unverified locally.

The structural change removes the separate 8 KiB inverse allocation. The
optimization hypothesis is that this reduces shared-memory occupancy pressure.
The downward pass uses half as many active threads and two multiplies per
active thread; that can outweigh any occupancy gain. Register allocation and
the full kernel's other shared allocations must be measured with nvcc.

CPU tests replace field operations with arithmetic modulo 2^61-1. They exercise
the actual tree indexing and synchronization source, but do not validate PTX,
secp256k1 arithmetic, CUDA memory behavior, or GPU performance.

## GPU validation required

From the repository root on a CUDA host:

```sh
nvcc -O3 -DZLAB_TREE=3 -DQSB_ZEROS_N=24 \
  candidates/subset/tests/gpu_epochs/tree_audit.cu \
  -o /tmp/qsb-inplace-audit -lcrypto -lm
/tmp/qsb-inplace-audit
compute-sanitizer --tool synccheck --error-exitcode 9 /tmp/qsb-inplace-audit
compute-sanitizer --tool racecheck --error-exitcode 9 /tmp/qsb-inplace-audit
```

Build both entry points before timing (the wrapper records its build stamps):

```sh
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'harness')
from gpu_wrap import compile_kernel
for name in ('subset_baseline', 'subset_inplace'):
    compile_kernel(Path('candidates/subset') / (name + '.cu'), 24)
PY
```

For each entry point, run a separate 60-second warmup and then the same
1200-second measurement; repeat the pair with a second seed. For example:

```sh
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset_inplace.cu' \
QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS=1200 \
QSB_PROBLEM_SEED=1789110211 ./benchmark.sh subset
```

Use `subset_baseline.cu` for the baseline. Save each run's artifacts before the next
invocation overwrites them. These command-grinder runs are local comparisons;
the configured bridge remains authoritative for ranked evaluation.
Header-only edits require rebuilding: the current
wrapper's stamp checks the entry point's timestamp, not its include closure.

Compare variant 2 and variant 3 on the same GPU with the same seed, difficulty,
duration, warmup, and verifier. Include fresh-seed repeats and the ranked run.
Never treat a CPU test, operation count, or advisory kernel rate as a score.
No performance improvement is currently established.

## Local checks performed

- Storage regression test first failed at 24,576 bytes, then passed at 16,384.
- Host execution: 1,440 lane inverses passed, including identity padding,
  near-prime values, and 32/64/128/256-thread blocks.
- The same host test passed with ThreadSanitizer, with no reported data races.
- `git diff --check` passed; all edits are within `candidates/subset/`.
- CUDA build, GPU audits, full hit verification, and GPU score are pending:
  the available host is macOS ARM64 without CUDA or an NVIDIA GPU.

Reproduce local checks from the repository root:

```sh
python3 candidates/subset/tests/gpu_epochs/test_inverse_storage.py
clang++ -std=c++17 -O1 -g -pthread -fsanitize=thread -Wno-unknown-pragmas \
  candidates/subset/tests/gpu_epochs/inverse_schedule_host.cpp \
  -o /tmp/qsb-inverse-schedule-tsan
/tmp/qsb-inverse-schedule-tsan
```
