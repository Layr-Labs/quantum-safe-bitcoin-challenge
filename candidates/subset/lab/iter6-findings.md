# Iteration 6 evidence

## Best up to bat

Current qualified combined best submitted at806b050. CLI issued
414fe58f-5131-4596-97f8-8ccf4fbdcf7f and queued validation job
6ea4aa2f-82ef-4634-8e04-3e0e18e0a0e0. Watcher owns it; no polling or duplicate
submission. Previous93cf failed atBenchmark. Retained workflow diagnostics show
run-preflightPASS then worker exit0 after0.6057s with zero attempts/hits. They
omit raw grinder logs: do not assign an exact failure cause from that artifact.

## Numerator-sum basis: verified but no speedup

Default-off QSB_K2S_SUM_BASIS tests pre3 payload(N1,N1+N2,ZZ) and post3's second
slope reconstruction after parity0, reusing the first slope's four-word storage.
This is a liveness/basis experiment, not fewer field multiplies. The source review
protects front/tail semantics, denominator relocation and center scratch overwrite.

Existing benchmark, N24/fixed120s/seed1789110211, frozen current qualified control:
A405.984212M/s5872/5872hitsPASS, B394.619697M/s5735/5735hitsPASS = -2.799250%.
It is only one screen, not a three-pair established regression; no justification
to adopt or spend more confirmation runs. All verifier hits checked, existing
self-report Poisson warnings remain visible in full logs. GPU label in score
JSON is inherited RTX_4090; actual local GPU is RTX3090.

Native CUDA12.8 build: candidate473632bytes sha801a50a0...,128registers/thread,
24576Bshared,zero spill stores/loads. Shorter source live range does not lower
reported register ceiling. Default-disabled image remains EXACT9c9aab2...,
473504bytes. No production carrier update required or performed.

## Packing8 + launch8: verified negative screen, still capacity-constrained

The old launch8-only experiment could not fit first-state buffers on ranked24GiB.
Packing8 now keeps first-state requests equal to current launch4:2GiB across
two slots. But actual descriptor/map/group requests still grow. Implemented
launch4 request accounting24117.463MiB leaves458.537MiB; pack8launch8 requests
24446.162MiB leaves only129.838MiB before runtime overhead. Existing default-off
combo trim would raise that to201.838MiB, not prove fit. Thus packing fixes the
first-buffer inequality but does NOT establish ranked readiness. Existing local benchmark pair finished: A399.694616M/s (5795/5795hitsPASS),
B393.430942M/s (5780/5780hitsPASS), -1.567115%. One screen, not confirmed
regression. No adoption: neither performance screen nor capacity evidence
qualifies it. Computation is in iter6-packlaunch-memory.json.

Native digest disassembly counts14512instructions vs14503default (+9); neither
source-liveness reduction nor static instruction count is a measured speedup.

Live NVML process snapshot during pack8launch8 existing benchmark:12572MiB.
Known small-table requests12158.162MiB; gap413.838MiB, greater than ranked
129.838MiB headroom (201.838withcombo trim). Native ranked overhead is unmeasured,
but this sharpens why the apparent buffer fix is not enough to call it ready.
