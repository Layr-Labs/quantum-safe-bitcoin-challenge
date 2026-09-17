# Publish the next anchor before the deferred add

Frozen candidate **e575d61ad2a0f4d9b65e8901e741c6bbe7514c1a6df6fc60118cfe537653124a** derives from the frozen a4f24420 combined shared-state candidate. The sole source edit moves the four-limb volatile cy store from after the shared-Z addition to immediately after the previous anchor has been fully copied to local registers, before that addition. It changes neither the stored values/addresses nor prefetch, recoding, arithmetic, barriers, epoch mapping or hit handling.

```cpp
for(int k=0;k<4;k++)anchor[k]=arena[k][256+tid];
for(int k=0;k<4;k++)arena[k][256+tid]=cy[k];
qsb_PointAddXYZZ_shared_z_def(X,Y,arena,tid,cx,cy,anchor,true);
```

The helper receives the old anchor by its separate local-array pointer. It reads/writes only ZZ and ZZZ shared slots 512+tid and 768+tid. Each lane owns its anchor slot, so publishing cy early cannot race another lane or change this addition's old-anchor value. The helper's defer_y=true path uses Y2 only when forming its early slope term; eliminating the later store permits cy to die after that use. Final guarded addition remains byte-identical. Both phase joins are unchanged, including the join protecting cross-lane inverse-tree reuse.

## Exact native and CPU evidence

Production sm89 and default builds pass on the authorized already-running local CUDA 12.8 compiler VM. No GPU kernel executed.

| Variant | Registers | Shared | Stack | Ptxas spill stores / loads | Non-NOP |
|---|---:|---:|---:|---:|---:|
| a4 combined late-anchor | 80 | 32 KiB | 280 B | 404 /284 B | 14,754 |
| e575 early-anchor | 80 | 32 KiB | 280 B | 376 /276 B | 14,747 |

The early variant's repeated 11-add loop is SASS 0x10660..0x16a70, matching PTX `$L__BB6_10`. Five unpredicated LDL instructions load 24 logical bytes/lane/iteration; four unpredicated STL instructions store 20 bytes. Thus the repeated loop accesses **484 logical spill bytes per lane**, versus 836 for a4 and the separately reviewed 816 for Z-only. The matching PTX loop has no local loads/stores: these SASS accesses are compiler spills rather than explicit source arrays. The report binds saved SASS/PTX and compiled source hashes. It does not equate logical accesses with DRAM bandwidth or elapsed time, and remaining whole-kernel spills have not all been assigned a runtime frequency.

Actual-source CPU projection passes 628 point chains, 1242 recovered keys, 12769 recodes, 532 sparse vector-load cases and 7536 prefetch checks. All 477 guarded final-add witnesses pass; 159 unguarded-doubling controls are caught. All 256 lane owners are exercised with canaries on every other lane's four-field slots. The source audits verify both phase joins and reject separate omitted-join mutations, preserve the first-SHA scratch body, retain exact checked Z helper algebra after slot mapping, and preserve the 512/1024/256 view strides. See `check-results.json` and `chain_recovery/` for source-bound results.

Direct-digit, field and other helper bodies are byte-identical to the a4 parent. The parent's 17278-scalar/241892-digit UBSan check therefore supplies helper-level evidence; it was not falsely rerun or relabeled as a fresh e575 digit report. The fresh e575 chain check does execute those helpers in the new chain. `prepared-source.json` proves that only the compact device header changed and the generator asserts that the full old anchor load precedes the moved store, which precedes the add.

## Scope and tradeoff

Source shared traffic remains 3232 logical bytes per candidate, shared allocation 32KiB, and two uniform phase joins. At 80 registers/thread, three 256-thread blocks satisfy the nominal register/shared resource arithmetic only with enough shared carveout. This source does not explicitly request that carveout, and launch bounds do not establish achieved occupancy. Relative to the 24KiB Z-only variant, this version reduces repeated spills but adds 992 shared bytes/candidate and needs more shared capacity. There is no measured GPU speedup or guaranteed preference between them.

This is the strongest checked combined-state version in this bounded experiment and a credible exploratory candidate, not an upload decision. Existing pending/production/staged source is untouched. Parent-level frontend/inverse/audit checks and any final stage/package verification remain separate. Original GPL and the a4/9c provenance are retained.

Reproduce after the frozen candidate has been generated:

```sh
python3 -B candidates/subset/research/shared_all_state/check.py \
  --source candidates/subset/research/shared_all_state/early_anchor/candidate \
  --output candidates/subset/research/shared_all_state/early_anchor
python3 -B candidates/pinning/research/compile_local.py \
  --source candidates/subset/research/shared_all_state/early_anchor/candidate \
  --entry subset.cu --default-build \
  --report candidates/subset/research/shared_all_state/early_anchor/native-results.json
python3 -B candidates/subset/research/shared_all_state/review.py \
  --directory candidates/subset/research/shared_all_state/early_anchor
```
