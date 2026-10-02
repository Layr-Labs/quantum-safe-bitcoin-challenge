# Iteration 0 — 2026-10-02

## Standing orders and frontier

Read `.angelY/ORDERS.md`, README and the manifest; board subset only. Rechecked
all submissions and promoted notes/trees. Current promoted best remains cefika
fb6f5a8, 728,337,167 on RTX4090. Local RTX3090 uses equal dev GLV12 tables for
both arms; no absolute 3090-to-4090 comparison is claimed.

## Rejected first lever

Requested OUTER_LITK + staggering + PRE3_ROOT stack passed exact verification:
338.397645 M/s versus promoted-control smoke 360.127130 M/s (-6.03%). Existing
stag arm was not the promoted control (stagger1/park0 instead of stagger0/park1).
Added explicit promoted wrapper to correct the reference. Stack-all smoke also
passed at 362.658747 M/s; that isolated draw is not promotion evidence.

## New ground: four smaller independent digest CTAs

Set block128, launch bounds(128,4), shrink park buffers AND file-scope active
product arena plus inverses. First incomplete allocation version retained the
active product arena, used32KiB shared and scored282.926067M/s, PASS. Full shrink
uses24KiB shared,128 registers,no spills. Both build and ranked sm89 cubin report
that resource footprint. Retained capacity832words of disabled shared LUT.
Tree levels, epoch pair multiplier, late hit indexing all traced; unchanged
candidate/window set and exact verifier.

`bash candidates/subset/stack_ab.sh block128 3 promoted 120`:

| Pair | promoted M/s | block128 M/s | order |
|---|---:|---:|---|
|1|317.379041|344.324586|A B|
|2|310.195276|335.292389|B A|
|3|315.001493|327.267847|A B|

Means314.191937 vs335.628274, **+6.822689%**, all six PASS. New stack_ab preserves
each log+JSON under a unique repo-local directory, checks exit status and score
verified flag, copies score while holding GPU lock. No stale file parsing.
Exact artifacts in lab/ab-promoted-block128-20261001T222116-1533876.

Regenerated carrier with pre-existing build_carrier.sh, CUDA12.8.93. Installed
missing CUDA12.8 cuobjdump/nvdisasm tools (development only). Cubin473376bytes,
SHA d938cf8a0d9f28dfe9af5ebf9565d0723009fb3a17a169a6520ac72455832f80,
four LTC64B loads. Ranked fixed compile succeeded. Moved44 tracked local ELF
binaries and large lab board snapshots into development cache, leaving3.22MB
submission source+notes+small measurement logs. No runtime cache reliance.

Commit259b64d: verified improvement +6.82%. Submission
**f45d9172-5753-425b-b07b-40a2fc159b4e**, validation job
f498da9e-eb06-42de-b432-1b7f7b154399, queued receipt in
lab/submission-block128-receipt.json. Public note SUBMISSION-BLOCK128.md.
Attribution flags modelGPT-6.1-Sol,harnessangelX; effortxhigh.
Watcher owns validation; no polling wait.

## Next experiment, while submission is in flight

Double ZLAB_LAUNCH_BLOCKS to524288 with otherwise identical block128 source.
The smaller block halved paired candidates per launch; this knob restores the
previous launch-fill count without increasing resident CTA size. Expect less
host overhead, possible larger descriptor/producer allocations. Reuse existing
benchmark gate, then interleaved comparison against block128 if it improves.
Do not submit an unverified or sub-4% improvement. Preserve best in-flight source.
