# Four shared point fields in a phase-reused 32KiB arena

Frozen candidate `a4f244206ae33529ace1c139f704612dcf3122e611f9e1670f2d3f4c8271a8ef` combines the checked M/anchor placement c970236c and short-lived ZZ/ZZZ loads from 4b65e089. Both derive from exact ranked/direct-digit 9c812950. No existing frozen candidate, submitted stage, production source or evaluation was changed.

## Native result and decision boundary

Production sm89 and default builds pass: **80 registers, 32KiB shared, 280-byte stack, 404-byte spill stores/284-byte loads, 14,754 non-NOP SASS**. This removes the separate 8KiB SHA allocation as intended. It reduces whole-kernel spill totals relative to the two independent placements, but does not eliminate active-loop spilling.

The repeated 11-add loop is SASS `0x10690..0x16ba0`, with backedge to 0x10690, matching PTX `$L__BB6_10` and c<11. Its 9 unpredicated LDL instructions read 40 logical bytes/lane/iteration, and 8 unpredicated STL instructions write 36 bytes. Total: **836 logical spill bytes per lane over 11 additions**. The corresponding PTX loop has no explicit local accesses, so these are compiler-introduced spills. Seven 32-bit state pairs remain at offsets 0x80..0x9c except 0x94, with a 64-bit temporary at 0x78 and readonly 32-bit slot 0xa4. This does not beat the independently reviewed Z-only 816-byte active-loop total; lower aggregate compiler spill totals are not a sufficient reason to prefer it. The experiment remains a source-correct resource/lifetime control, with no GPU timing or selection claim.

## Exact union views and joins

The kernel owns one union with three typed members:

```cpp
union QsbSharedPhaseArena {
    uint32_t sha[8][256];
    uint64_t fields[4][1024];
    uint64_t inverse[4][512];
};
__shared__ QsbSharedPhaseArena shared_phase;
```

A static assertion requires 32,768 bytes. SHA receives the 256-column view, point-chain helpers receive the 1024-column view, and the inverse receives the 512-column view. These are intentionally different mappings. No 1024-stride pointer is passed to a 512-stride helper. Field slots are M at tid, old affine Y anchor at 256+tid, ZZ at 512+tid and ZZZ at 768+tid, separately in each limb row.

The scheduled SHA scratch helper is byte-identical to the 9c helper after only its name/signature and local shared declaration change. The original ABI wrapper is retained for other callers. Both source branches of the digest kernel call the scratch ABI. After all SHA shared consumers and SHA256d finish, an unconditional CTA join precedes any field writes. This protects cross-lane cached-state readers from early point-phase writers.

Every point-field access is volatile and lane-owned. The seed still uses the first cold point y0 as deferred anchor and publishes seed ZZ/ZZZ. Each intermediate add reads the old anchor near use; the source-exact checked Z helper reloads ZZ or ZZZ only at its immediate multiplication and stores their updates. The final guarded helper is unchanged and receives register-reloaded ZZ/ZZZ and anchor. A second unconditional CTA join separates all completed point-field reads from inverse-tree writes. Inactive/unusable lanes participate with inverse factor 1; their per-lane return remains after inversion. The sole early return before these phases is the original whole-block condition.

The inverse scratch body remains byte-identical to 9c after only signature and local-shared declaration changes; builder/audit callers retain the local-shared wrapper. The original field arithmetic, guarded 14-window geometry, all eleven intermediate additions, final exception handling, recovery, epoch and hit logic remain unchanged.

## Traffic and resource arithmetic

M/anchor placement adds 416 store bytes and 576 load bytes per combination. The Z placement adds 64 seed-store bytes, eleven rounds of 128 load bytes plus 64 store bytes, and 64 final reload bytes: 768 stores and 1472 loads. Combined additional source-level traffic is therefore **1184 store bytes +2048 load bytes =3232 logical shared bytes per combination**, or 808KiB per 256-lane CTA. This excludes unchanged inverse/SHA accesses and global point-table traffic. CUDA can narrow individual loads in SASS; source logical bytes are not hardware transactions.

Compared with 9c there are two extra CTA joins, one at each phase transition. Long-lived M/anchor/ZZ/ZZZ field payload is moved out of registers: 32 physical 32-bit words at the source level. Native allocation and actual spill evidence, not this count, determine what the compiler retained.

At 80 registers/thread, three 256-thread CTAs require 61,440 registers. Three 32KiB allocations plus the documented 1KiB/block reservation use 99KiB, within Ada's 100KiB shared capacity. This is a static upper-bound residency eligibility calculation. A suitable shared carveout is needed and leaves less L1 capacity; achieved residency, latency hiding and throughput are unmeasured. Hardware limits follow the existing [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) reference in the shared-state design.

## Checks

The actual new chain and scoped Z helper execute in the existing independent OpenSSL curve/recovery projection: 628 chains, 1242 recovered keys, 12769 recodes, 532 actual vector-load cases and 7536 prefetch checks pass. All 477 final-guard cases pass, with 159 unguarded-doubling controls caught. A local 32KiB CPU arena replaces the CUDA shared allocation; owner tid cycles through all 256 lanes, and canaries assert other lanes' four-field slots stay untouched. This checks mathematical values/ownership, not concurrent execution.

Actual shared-digit helpers pass 17278 boundary/random scalars, 241892 digit comparisons and 34556 cold-digit comparisons, with wrong-start and wrong-last mutations caught. The scoped Z helper is checked byte-for-byte against 4b65e089 after only row-stride/slot remapping. First-SHA scratch helper equivalence, inverse body equivalence and final guard identity are asserted. Removing either lifetime join independently fails its structural audit; no early per-lane return is permitted before the joins.

`check.py`, `check_digits.py`, `review.py`, source manifests and all reports bind to the exact frozen include closure. The native report additionally checks every compiled production file hash against that closure. Production/default CUDA compilation is static ARM-host code generation using the already-running local CUDA 12.8 VM, not GPU execution. No VM lifecycle operation occurs.

```sh
python3 -B candidates/subset/research/shared_all_state/prepare.py
python3 -B candidates/subset/research/shared_all_state/check.py
python3 -B candidates/subset/research/shared_all_state/check_digits.py
python3 -B candidates/pinning/research/compile_local.py \
  --source candidates/subset/research/shared_all_state/candidate \
  --entry subset.cu --default-build \
  --report candidates/subset/research/shared_all_state/native-results.json
python3 -B candidates/subset/research/shared_all_state/review.py
```

The generator refuses to overwrite its candidate. The original GPL notices/COPYING and existing promoted/unpromoted attribution are preserved: corrected PR77 base, promoted PR120 mechanisms, and substantial earlier unpromoted contributions credited to alvaroborras, MakiRH4, jacklightChen, ercumentyildirim, AbdelStark and IvanLudvig. This isolated experiment adds only arena views, phase joins and state placement.
