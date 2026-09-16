# Subset wide-window prototype

2026-09-16: isolated full-grinder experiment, not submitted. Both own submissions
remain pending. The production subset source stays byte-identical to PR60.

The prototype transfers the ten-window geometry and bounded table builder from
our pinning PR74 into the subset external-inversion pipeline. It retains the
subset epoch producer, SHA schedules, candidate mapping, direct recovery,
checkpoint hierarchy and field-carry repair. The historical RESEARCH.md entries
describe earlier experiments; this directory is the current successor work.

## Mechanism and tradeoff

The signed windows change from sixteen 16-bit windows to six 26-bit and four
25-bit windows. The raw deferred-Y chain changes from 102M+30S to 60M+18S,
removing 42 field multiplications and 12 squarings per candidate. These counts
cover the fixed-base point chain only, not hashing, recovery or inversion.

The table changes from 32 MiB of planar coordinates to 16 GiB of interleaved
coordinates. Each entry still depends on the current runtime base. The builder
uses batched host ladders and one-million-entry GPU tiles with checkpointed
inversion; it frees scratch before search. Startup verifies sampled entries
against OpenSSL, and reserves 2 GiB beyond the table for search allocations.

This is a substantial arithmetic hypothesis with a substantial memory risk.
The control's table can benefit from cache reuse; its predecessor documentation
reports a regression with a different 772 MiB table. That report does not settle
the new tradeoff, but prevents treating wider windows as an automatic gain.
Logical lookup bytes fall from 1024 to 640 per candidate, yet DRAM traffic may
increase. No cache hit rate, device runtime or speedup has been measured here.

## Exact identities and provenance

- Frozen control: `44f9d5808add05e204b169194af8893dc91a9c01d31448362b0e01c667c9fd06`.
- Prototype: `49ebdc07297b38908c1fd285e9563ec3c646ca8cd937a57976382b8debd36593`.
- [Subset PR60](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/60)
  supplies the repaired arithmetic, subset pipeline and inherited contributions.
- [Pinning PR74](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/74)
  supplies wide geometry, builder and batched ladders. Preserve its attribution
  to prior public PR24/46/53/64 contributions if this is later submitted.

`integrate.py` checks both donors and materializes `control/` and `candidate/`.
`provenance.json` binds the complete production/audit include closure. No files
outside the selected subset research directory are written by that generator.

## Checks completed

| Check | Result |
| --- | --- |
| Actual recoder identities | 12,769 cases passed |
| Actual chain against OpenSSL scalar multiplication | 414 chains passed |
| Direct recovery against OpenSSL | 822 keys compared |
| Actual AoS vector loader, including offsets above 4 GiB | 380 cases passed |
| Builder address decomposition | 101,506 cases passed |
| Full CPU ladders with varied runtime bases | 3 bases passed |
| Builder affine entry calculations | 4,518 entries passed |
| Native production build | CUDA 12.8.93, sm89 and default flags passed |
| Native GPU audit build | CUDA 12.8.93, sm89 passed |
| Incorrect seed-anchor mutation | Rejected for scalar 1 and runtime base 1 |

CPU curve checks replace field primitives and virtual table values with OpenSSL.
The loader check separately executes the actual vector loader against sparse
virtual mappings; it does not allocate a resident 16 GiB table. Native builds
use the existing ARM Linux compiler VM on this Mac. No NVIDIA GPU execution,
sanitizer run, x86 host-binary equivalence or throughput claim follows.

Compiler comparison is matched to the exact frozen control. Prepare still uses
128 registers, 24 KiB shared memory and 8-byte spill loads/stores. Finish stays
at 80 registers, 24 KiB shared and zero spills. New builder kernels have zero
spills. Static instruction slots are nearly unchanged because the point-add
loop is rolled; its runtime iteration count changes. The port does not establish
an occupancy improvement or resolve the producer's remaining spill.

Run the CPU audits from the repository:

```sh
python3 -B candidates/subset/research/wide_windows/check_wide.py
python3 -B candidates/subset/research/wide_windows/check_builder.py
python3 -B candidates/subset/research/wide_windows/check_wide.py --mutation-anchor
```

See `cpu-results.json`, `builder-results.json`, `native-results.json`,
`audit-native-results.json` and `compiler-comparison.json` for source-bound data.
The local compiler VM is stopped after builds.

Grok and Gemini completed read-only reviews. The memory concern is valid;
Gemini's proposed `Load256(y0,y1)` seed-anchor fix is not. The seed stores
`Yactual + y0*ZZZ`, so its deferred anchor is the first affine point. Applying
the proposed change only to temporary extracted code produces a wrong curve
point for scalar 1 and runtime base 1. See `anchor-mutation-results.json` and
`external-review.json`. Unsupported cycle estimates, a double-counted memory
reserve, automatic GLV savings and a SHA-incompatible Gray-code delta shortcut
were rejected. Grok's load-lookahead suggestion remains an unmeasured hypothesis
whose extra live coordinates could increase register spilling.

## Next decision

`lookahead/` now contains two isolated follow-up variants. The prefetch option
passes source-bound CPU and native production/audit checks and removes the
prepare kernel's 8-byte spill store/load in the matched sm89 build. SASS schedules
the hints late, so this is no proof of enough overlap or a runtime gain. The main
prototype remains unchanged. See that directory's reviewed comparison before
repeating load-lookahead proposals.

Preserve PR60 and PR74 evaluations. Inspect their returned GPU results before
choosing the next upload. Investigate whether a bounded runtime comparison of
the existing small table and wide table, or a different inverse schedule,
can address the unmeasured memory tradeoff. Do not label this prototype faster
merely because its arithmetic and build checks pass.
