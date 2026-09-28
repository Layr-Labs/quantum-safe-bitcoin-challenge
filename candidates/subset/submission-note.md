Model: SWE-2 High
Harness: Devin CLI

# Subset: promoted Q_MIX2 host package with adaptive producer handoff

Agent: Devin
Effort: high
Analysis assist: ByteAsk 0.1.15 (`gpt-5.4`), read-only repository review and diff review

This package starts from the current promoted subset source at repository commit `8d07d3ebad41a017dfaa5906b164f883a9b59348`. The subset candidate bytes correspond to accepted public submission `5c7e36c5`, submitted by jacklightChen at candidate commit `6343a38d3dde830b079cb95b0e2e99c7f9a812e9`, with official score `708,411,009` verified candidates/s. The later repository accept commit changed the unrelated `pinning` candidate; the subset candidate under that source ref is the `5c7e36c5` tree.

The goal is a narrow host-only experiment on top of that promoted composition. It does not change candidate generation, exact verification, kernel arguments, kernel code, or the embedded native image. Remote Yukon validation is the only performance test available for this prepared package.

## Functional change

The only implementation change is in `candidates/subset/tests/gpu_epochs/host_producers.h`. The promoted consume-side handoff waited a fixed `QSB_HP_WAIT_MS` interval, default `40 ms`, whenever the host batch needed for the next launch was still in `S_PROD`. If it was not ready when that timeout expired, the batch was abandoned, all three GPU producer kernels were enqueued, and the host work for that batch was discarded.

This candidate adds `QSB_HP_ADAPTIVE_WAIT`, default `1`, and makes the handoff timing-aware:

1. Under the existing host-producer mutex, it counts all unfinished queued chunks through the requested batch.
2. It estimates remaining host time using the existing exponential-moving-average `tchunk` divided by the effective producer-thread count.
3. For `QSB_HP_PLACE=1`, it counts only the pinned producer when the helper is parked (`behind == false`), matching the worker gate. The promoted default remains `QSB_HP_PLACE=0`, where all configured producers participate.
4. `QSB_HP_WAIT_MS` remains a hard cap. If the predicted ready time exceeds that cap, the code skips the fixed wait and falls back immediately. Otherwise it waits approximately `1.2 * ETA + 1 ms`, still capped by `QSB_HP_WAIT_MS`.
5. `QSB_HP_ADAPTIVE_WAIT=0` restores the previous fixed wait behavior without rebuilding or replacing source files.

The intended effect is twofold. Batches that are nearly ready can still arrive through the host path, while batches that cannot finish within the bounded delay no longer burn a fixed `40 ms` launch stall before the GPU producer fallback. This should reduce avoidable idle time and reduce discarded near-ready host batches when the fixed timeout is too long or too short for the actual queue state.

## Why this hypothesis

ByteAsk's read-only review identified that the promoted ring already tracks the measurements needed for a predicted-ready decision (`done_chunks`, `nchunks`, `tchunk`, `tg`, and the helper/behind state), but the consume-side timeout was still a constant inherited from an earlier producer placement. Its diff review found the implemented code C++-valid and identified one concurrency accounting correction, now included: a parked `QSB_HP_PLACE=1` helper is not counted as an active producer.

The prior L2-window experiment from this account was a negative control and is explicitly excluded. That package capped the table policy at 24 MiB and scored `690,113,917` against the `700,953,730` parent frontier. The promoted tree already clips its table policy to the 48 MiB dense hot region; this package does not alter cache policy.

The expected magnitude is uncertain. If host producer fallback is rare, the gain may be small; if fallbacks or near-miss waits are frequent in the official runner, eliminating the fixed stall and preserving more host batches could be worth a few tenths of a percent, with the optimistic bound approaching the requested `1.2%` target. There is no local measurement and no guarantee that this crosses either the `1%` promotion gate or the `1.2%` stretch target.

## Scope and exactness

Only `candidates/subset` is changed. Relative to the promoted `5c7e36c5` candidate content, the functional source change is limited to `tests/gpu_epochs/host_producers.h`. `SOURCE-MANIFEST.json`, `submission-note.md`, and the generated source-hash comment in `qsb_carrier_sm89.h` are refreshed for package accuracy.

The change does not alter:

- Epoch ranking/unranking or candidate ordering.
- SHA-256 message schedules or produced descriptor bytes.
- `kernel_digest`, producer kernel signatures, launch dimensions, or stream ordering.
- Tentative hit records, host re-derivation, exact publication, or score accounting.
- Device source defaults, including `QSB_Q_MIX=2`, `QSB_Y_PAIR`, `QSB_SC_PARK`, and the promoted P18/GLV chain.
- The promoted `sm_89` carrier payload, kernel names, or carrier knob fingerprint.
- The promoted CPU co-grinder or host-built epoch producer algorithms.

The host producer self-check on batch 0 remains the gate for all host-produced batches. If the predicted wait path fails or host work is late, the same GPU producers remain the fallback, and the existing consecutive-fallback watchdog still disables host producers. `QSB_HP_ADAPTIVE_WAIT=0` provides an environment-level rollback to the promoted fixed timeout.

## Carrier compatibility

`host_producers.h` is host-side C++ and is not part of the carrier device-knob fingerprint. The embedded carrier is retained from the promoted package:

- Cubin size: `462,496` bytes
- Cubin SHA-256: `f74548427859ec03273f05e9151c810716c6475596e0213a0db3915e468aa5dc`
- Kernel names and `QSB_CARRIER_KNOBS`: unchanged
- Device defaults needed for the image, including `QSB_Q_MIX=2`: unchanged

The carrier header's source-hash audit comment is updated to describe this package's source tree. The encoded cubin payload is unchanged. No native carrier rebuild was performed or required for this host-only scheduling experiment.

## Validation performed

This machine still lacks `nvcc`, `nvidia-smi`, a usable CUDA GPU, and passwordless `/opt/starkware-challenge/bench-exec.sh` access. The ranked benchmark therefore cannot execute locally, and this note claims no local GPU throughput or unofficial score.

Completed checks:

- `setup.sh subset` completed and the verifier smoke test passed.
- `git diff --check` passed after the implementation.
- The functional diff is confined to `candidates/subset/tests/gpu_epochs/host_producers.h`.
- The adaptive-wait path is under `QSB_HP_ADAPTIVE_WAIT`; `QSB_HP_ADAPTIVE_WAIT=0` restores the promoted behavior.
- The embedded cubin payload and digest remained `f7454842...a5dc` and `462,496` bytes.
- The carrier device-knob list was not modified.
- No benchmark harness, verifier, score file, problem generator, workflow, or `pinning` source was edited.

No unmeasured performance claim is made. The official remote run is the correctness and throughput authority.

## Attribution

The immediate base is jacklightChen's promoted `5c7e36c5` composition. It credits RealAdii's `521075fe` promoted source, i34-9's host-package composition from `4da17ebc`, cefika's host work from `bf001729`, ercumentyildirim's producer package from `a141df2b`, terrapinelf's `QSB_Q_MIX=2` variant and native image from `3e6069ee`, and the underlying contributions of HyeokxC, kshitij-hash, fkiene, Meganpark980320, Ryun1, newjordan, and earlier listed contributors. Their public source and notices remain in the package. Naming them records provenance and does not imply participation, review, or endorsement.

Planning and source analysis used a ByteAsk read-only session (`gpt-5.4`) to inspect the promoted producer loop and review the resulting diff. Implementation, package preparation, and this submission were performed by SWE-2 High in Devin CLI. ByteAsk did not modify repository files.

## Evaluation and rollback

At preparation time the live promoted score was `708,411,009` verified candidates/s. The automatic `1%` promotion threshold is approximately `715,495,119`, and the requested `1.2%` stretch target is approximately `716,911,941`. These are thresholds, not predicted results.

A successful remote result would show the same or better correctness while increasing verified candidates/s and preferably reducing `GPU-built after start-up` batches in the `[HP]` diagnostics without increasing `ready ahead at launch` stalls. A neutral or negative score falsifies the hypothesis that replacing the fixed timeout with measured queue ETA helps the promoted host producer on the ranked machine. In that case, `QSB_HP_ADAPTIVE_WAIT=0` or restoration of the promoted header is the direct rollback.
