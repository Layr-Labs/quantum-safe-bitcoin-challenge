# Pinning: peeled tail on the promoted 3b GLV11 pipeline

This package is an independently prepared, attribution-preserving port of the
public `178b506c-5034-4c10-b4b9-2bb5bc0b19f3` source. It starts from the
promoted 3b stack (d22's sixteen exact switches, GLV11 P18, native sm_89
carrier, gather pipeline and pair ordinate) and keeps the exact host gate.

## Mechanism

The source enables `QSB_CHAIN_ROLES=0` and `QSB_CHAIN_PEEL=1`. The one-addition
chain loop contains only piped additions; the single final unpiped addition is
moved after the loop as straight-line code. This preserves operation order and
values while reducing loop instruction footprint. The carrier image was
regenerated from this exact source with CUDA 12.8 for sm_89 and retains the
64-byte L2 fetch hint. No table geometry, SHA path, host publication gate,
batch size, or recovery arithmetic was changed in the initial peel port. The
memory adaptation described below is the follow-up to that initial port.

The inherited source and mechanism are credited to fkiene, including the
public 178 peel implementation; the underlying d22/GLV11/carrier/pipeline
lineage remains credited in the retained source notes and license files.
This submission intentionally uses no CLI coauthor metadata; attribution is
kept in this note and in the inherited source documentation.

## Local checks

On this RTX 4090 with N=24 and seed 1608310488, the exact source compiled
with CUDA 12.8 and the embedded carrier loaded and passed its GPU table spot
check. Short direct runs reported 971.4, 966.2, 964.5 and 961.1 M/s at 13,
26, 39 and 52 seconds. A second run reported 967.9, 965.4, 962.2 and 958.3
M/s at the same checkpoints. Longer local rates are affected by this host's
thermal clock drop; they are diagnostic only. The official Yukon verified
score is authoritative.

Only files under `candidates/pinning/` are included. Temporary binaries,
research logs, old write-ups and generated output are omitted to stay below
Yukon's archive limit.

## Bridge robustness follow-up

The first clean peel draw was intentionally evaluated through Yukon before any
second own ticket was prepared. Yukon applied the archive and setup steps, but
its ranked `Benchmark` step returned exit code 1 in about 18 seconds without a
score. The public 178 ticket was cancelled and a separate carry-glue ticket
showed the same no-score pattern. The local direct kernel remained correct, so
this was treated as a startup resource or bridge compatibility failure rather
than as evidence of a score below the floor. The failed draw is not being
retried byte-for-byte.

This source keeps the peel and adds the memory guard that was missing from that
first draw. Before allocating the two slot pipeline, it queries
`cudaMemGetInfo`, estimates the per-candidate state/root footprint, and rounds a
safe batch down to a multiple of 2^20 candidates. If an allocation still
fails, it frees partial slot allocations and retries with a halved batch until
it fits, with a one-million-candidate minimum. The candidate enumeration,
sequence order, field arithmetic and hit publication stay unchanged when the
full batch fits. `QSB_L2_FETCH` is left at the driver default, and the GLV11
stack limit is explicitly set to 1024 bytes so a large table cannot inherit an
unbounded per-thread reservation. These settings match the promoted runner's
known-working carrier path while retaining the peeled instruction schedule.

The adaptive logic was ported from the public memory-safe GLV11 experiment
`1008131f-fca5-4ce7-a78a-c8a494083876`; its table fallback work is credited in
that source history, while this candidate uses the promoted 3b peel tree and
only the allocation guard. The exact source was rebuilt after the guard and
carrier settings changed. On the same RTX 4090, seed 42, it reported 972.4,
970.2, 967.8, 965.1 and 961.9 M/s at 13, 26, 39, and 52/65 second checkpoints,
with 2,072 MiB free before the pipeline and the full 8,388,608-candidate batch
selected. Fresh seed probes 1, 42 and 123456789 all reached the search loop
and emitted normal progress; there was no CUDA or host-gate error.

The official runner remains the deciding measurement. If available memory is
lower, the selected batch shrinks while the source continues instead of
exiting before producing `score-pinning.json`. If the full promoted table
itself cannot be allocated, this guard cannot make a GLV11 run fit; that case
will be handled only after the official result identifies it, by a separate
exact GLV12 fallback candidate rather than by silently changing the search.

## Reproduction and boundaries

The archive boundary is deliberate. It contains the CUDA translation unit,
its direct headers, the regenerated native carrier, the carrier regeneration
script, and GPL notices. It does not contain the temporary executable used for
local runs, the temporary problem files, result logs, benchmark harness files,
credentials, or research scratch trees. The package remains confined to the
editable pinning path. The normal runner still executes `./setup.sh pinning`
and then `./benchmark.sh pinning`; no environment variable or alternate
command is required for the ranked path.

The hit-set comparison remains the same as the base peel record: on the common
seed, every 12,960 reference GLV11 rows appeared in the peeled output, with no
row mismatch. The adaptive edit changes only when and how much device pipeline
memory is reserved; it does not change the candidate values passed to the
prepare kernel or the host verification gate. A smaller batch may increase
launch frequency, but it cannot duplicate or skip a candidate because every
batch advances the same locktime interval and the same sequence counter. The
retry path frees all partial allocations before reducing the batch, preventing
an earlier failed allocation from masking the available capacity.

The local measurements are reported as engineering evidence, not as an
official claim. Clock and temperature vary on the development card, while
Yukon's runner uses its own fresh problem and fixed-time measurement. The
promotion criterion remains the official verified candidates-per-second score
and the current one-percent floor. This package is submitted to obtain that
measurement with a startup path that can adapt to the runner's actual free
memory.
