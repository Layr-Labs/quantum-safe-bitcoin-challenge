# Partitioned cache-policy comparison

The complete isolated variant compiles and passes CPU address, curve, recovery,
hit-mapping and collective checks. It remains an experiment with no GPU timing.
It is not selected for submission on the current evidence.

Parent: asymmetric14 source
`230f651785e7bcd9affd8cb188e41287a057a06caa8d50749487c5f7a73dec0a`.
Variant:
`80267671027b602ea51002c533aa05c7972600fdd30e61d619d2099cb7911d0e`.

The first twelve windows retain ordinary128-bit vector loads. The last two
use `ld.global.cs.v2.u64`. Cold-window ordinary L2 prefetches are removed;
the first twelve windows retain address-only lookahead. State records use
128-bit streaming stores/loads and checkpoint nodes use64-bit streaming
stores/loads. Shared checkpoint helpers also affect builder/root kernels.
Root arrays, point formulas, field primitives, table geometry, allocation
policy and synchronization are unchanged. There is no L2 reservation API.

Streaming means evict-first allocation, not L2 bypass or absence of cache fills.
The source has ordinary host load/store branches solely for CPU projections.
Those do not simulate caches or execute the PTX path. The four PTX forms are
reviewed directly and their native64/128-bit instructions are checked.

| Full native CUDA12.8.93sm89 kernel | Parent | Variant |
|---|---:|---:|
| Ranked prepare registers / spills |128 /0|128 /0|
| Ranked prepare shared memory |24KiB|24KiB|
| Ranked prepare static non-NOP instructions |8,428|8,433|
| Ranked finish registers / spills |80 /0|80 /0|
| Ranked finish shared memory |24KiB|24KiB|
| Ranked finish static non-NOP instructions |4,685|4,691|

The default build also passes. Emitted code includes eight static
`LDG.E.EF.128` sites in prepare, eight `STG.E.EF.128` state sites,
four `STG.E.EF.64` checkpoint sites, and the corresponding finish loads.
The rolled point loop has mutually selected ordinary/streaming load paths;
static site counts are not per-candidate traffic or executed instruction counts.
The generic fallback retains its inherited spills. Full differences and
source/report hashes are in `comparison.json`.

CPU evidence:12,769 recodings,414 curve chains,822 recovered keys,532actual
vector loads,4,140hot-prefetch checks;1,282hierarchy leaf inverses,
2,000direct recoveries,768pipeline candidates including3singular lanes and
237verified reduced-gate hit records;13,107builder entries across3bases.
Restoring cold ordinary prefetch in a temporary copy fails the independent
policy/address oracle. CPU passes do not establish GPU scheduling or timing.

Run `prepare.py` to regenerate the candidate. Use the parent's curve command
with this candidate path and `--prefetch-chunks 12`. Run
`candidates/subset/check_pipeline.py --compact --source <this candidate>` and
the parent's integrated builder command with separate reports in this folder.
Use the existing local compiler helper for `subset.cu`, `--default-build`, and
`native-results.json`. Finally run `check_policy.py` to bind all reports and
verify native forms plus the negative control. The local compiler VM is stopped.

Muse14 supplied the geometry-specific policy proposal; GPT6Astra/Codex corrected
its semantics and implemented this variant. Prior public PR101 explored uniform
streaming hints on pinning and was author-cancelled without a score. The
negative official uniform-state-streaming result also prevents treating these
hints as a generally beneficial change. A later submission must revisit source
attribution and current/pending comparisons.

Keep both geometry variants as controls. Further small hint permutations are
not justified without new evidence. The next substantial investigation is
retaining projective state through a final cooperative inversion, with its
increased inversion frequency and possible occupancy loss explicitly costed.
