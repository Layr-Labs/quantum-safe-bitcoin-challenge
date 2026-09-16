# Wide-table load-lookahead experiments

2026-09-16. Supporting work for the substantial ten-window/16 GiB architecture,
not a separate claim of a large improvement. Submitted PR60 and the checked wide
control remain unchanged. `prepare_lookahead.py` stages both variants from the
exact `49ebdc07...` wide control and refuses to overwrite an existing experiment.

`loaded/` decodes and loads the next affine point before the current addition.
`prefetch/` retains only the next index/sign and issues L2 hints for both 32-byte
halves of that future 64-byte point. Both preserve all ten digits, point order,
the first-point seed anchor, deferred-Y updates and final resolved addition.
Neither changes field arithmetic, recovery, the builder or the trusted harness.

| Ranked prepare, CUDA 12.8.93 sm89 | Wide control | Loaded next point | L2 prefetch |
| --- | ---: | ---: | ---: |
| Registers | 128 | 128 | 128 |
| Shared bytes | 24576 | 24576 | 24576 |
| Stack bytes | 8 | 8 | 0 |
| Spill store/load bytes reported by ptxas | 8/8 | 8/8 | 0/0 |
| Static instruction slots | 8392 | 8432 | 8440 |

Ranked finish is unchanged: 80 registers, 24576 shared bytes, no stack/spills.
Both production variants compile with explicit sm89 and official default flags;
the prefetch GPU audit also compiles. Source hashes match the checked candidates.
The ARM Linux host build is not GPU execution or the exact x86 judge executable.

Each variant passes 12769 scalar recodings, 414 OpenSSL curve chains, 822 recovered
key comparisons and 380 actual AoS loader cases including offsets above 4 GiB.
The prefetch audit also matches 3312 prefetched entries to their next actual
logical loads, checking both address halves. PTX hints are replaced only in the
CPU projection; this does not test device cache behavior. The unchanged wide
control still passes the generalized runner. The wrong seed-anchor mutation is
still rejected at scalar 1/base 1 on the prefetch variant.

SASS retains four static `CCTL.E.PF2` instructions (initial pair plus rolled-loop
pair). They are scheduled late in the arithmetic loop. Source placement therefore
does not establish a full point addition of lookahead or enough latency overlap.
No `LDL`/`STL` remains in ranked prepare. See `*-sass-excerpt.txt` and
`comparison.json`; these are static observations, not a runtime speed ratio.

The hint follows [NVIDIA PTX prefetch semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prefetch-prefetchu).
It does not provide a residency or completion guarantee. Extra hints can consume
bandwidth or issue capacity. Retain this as an optional comparison candidate;
do not overwrite the main prototype or advertise spill removal as a large win.

Reproduce CPU checks from the repository:

```sh
python3 -B candidates/subset/research/wide_windows/check_wide.py --source candidates/subset/research/wide_windows/lookahead/loaded --report /tmp/loaded-cpu.json
python3 -B candidates/subset/research/wide_windows/check_wide.py --source candidates/subset/research/wide_windows/lookahead/prefetch --report /tmp/prefetch-cpu.json
```

Use the documented local CUDA compiler helper with `--source` set to either
variant and `--entry subset.cu`; prefetch audit entry is
`tests/gpu_epochs/tree_audit.cu`. The VM was stopped after the builds.

Next substantial decision remains small-table versus wide-table end-to-end
performance. A bounded runtime choice between those architectures would address
the larger cache/memory risk; further instruction tuning cannot settle it.
