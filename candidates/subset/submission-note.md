# Subset: host pipeline and exact SHA folds on the promoted clean arithmetic

## Source and scope

This candidate starts from shared main `7c3609b87b9d8e094a16be148fe846dfd5ac7807`.
Its subset implementation is the promoted `7aef224a-e3ff-43f9-9877-50cdbda3f653`
source, landed at `9ac2515`, with a recorded score of **623,518,629 verified
candidates/second**. The source milestone for this port is `da34628`.
The implementation and tests in this submission were prepared with GPT 6 Astra,
xhigh effort, using Codex. All edits and added artifacts belong to
`candidates/subset`. The harness, scoring, benchmark manifest, problem generator,
problem instances, sibling track, and workflows are unchanged.

The new runtime mechanisms come from the public difference between
`8b397a5c56b32c8d799eada6f913d990b6503cd1` (submission `35c4db43`) and
`ae0ade77bfdd71e5a2dc1a3e2bab780af8c7bb46` (submission `e63e42ec`, public
[PR 977](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/977)).
Akashneelesh is credited as a coauthor for this substantial unpromoted code.
The promoted base's authorship and source notices remain in place, including
jacklightChen, Saviour1001, owizdom, DPZZxlz, fkiene, dun999, Meganpark980320,
ercumentyildirim, EvanYan1024, terrapinelf, and Akashneelesh. The pipeline donor
also credits the promoted pinning host-loop lineage. The SHA header comes from
the pinning source in the same shared-main snapshot; a byte-identical copy is
included inside this track, avoiding a dependency on later sibling changes.
VanitySearch notices and the existing GPLv3 COPYING file are preserved.

This port intentionally extracts only the host pipeline, startup trims, and
exact SHA folding. It retains the promoted subset arithmetic and table layout.
The donor's isomorphic recovery, extra short-carry tails, and speculative
first-fold cuts are not introduced. Our contribution is the clean-base
composition, cross-track dependency removal, checked CUDA error paths, resource
cleanup, source-executing host tests, and independent build/resource census.

## Mechanisms and invariants

`QSB_HOST_PIPE=1` gives each of two slots its own nonblocking CUDA stream,
completion event, epoch descriptors, first states, group records, epoch-to-group
map, tentative hits, and verified hits. Each stream orders its own producer
kernels, digest, exact replay, asynchronous readback, and completion event.
Immutable parameters, window tables, and the fixed-base table are shared.
A checked device synchronization finishes startup uploads before either
nonblocking stream consumes them.

Batch k uses slot k modulo 2. Before reusing that slot for batch k+2, the host
waits for k's event, writes its completed verified records, and updates its
completed-candidate and hit counters exactly once. This makes the next batch's
GPU work eligible to overlap the previous batch's host formatting and writeout,
and permits producer/replay work on the other stream where GPU resources allow.
Actual overlap and its performance effect require the ranked device.

Exhaustion drains the oldest pending slot and then the last pending slot before
exit. Zero-hit batches still increment completed-candidate accounting. The
existing host publication limit of 64 records per batch is retained, bounded
by the 1,028-byte pinned readback region; the device tentative and verified
buffers each hold 1,024 records. The exact replay kernel remains the only source
of published records. Its per-slot epoch and first-state buffers remain live
until completion. Event synchronization, readback enqueue, event recording, and
initial synchronization errors are checked. A zero-length write is rejected
instead of causing an infinite write loop. Normal exhaustion releases streams,
events, pinned mirrors, secondary producer buffers, and both hit-buffer pairs.

The inherited termination handler still exits on SIGTERM/SIGINT. At forced
termination, up to two undrained batches can be lost, as disclosed in the donor;
the diagnostic counter reports only completed drains. This implementation does
not claim a new graceful-signal drain. It does not extend or control the judge's
clock. At the ranked difficulty, the inherited 64-hit host cap is ordinarily
well above the expected batch count; low-difficulty diagnostic hit saturation
must not be mistaken for an unbiased throughput measurement.

`QSB_STARTUP_TRIM=1` changes three startup costs. OpenSSL ladder points are made
affine in batches, which produces the same coordinates as individual affine
conversion. The GPU table is checked at the same four corners of every chunk
plus the same 192 deterministic LCG samples. Only those 252 records of 64 bytes
are copied to the host, with a pageable-copy fallback if pinned allocation
fails. A construction, transfer, or sample failure follows the original full
host-table builder; the fallback upload's return code is checked. Finally, the
unconditional 32 KiB per-thread stack reservation is omitted. Every compiled
kernel has zero spills; the largest reported cumulative stack is 120 bytes,
below the default reservation. Switch zero retains the old reservation and
full-table copy.

`QSB_SHA_FOLD=1` uses exact SHA-256 identities already present in the promoted
pinning header: literal round constants, IV-specific initial rounds, sparse
first expansion for the 32-byte digest block, and the final H0 feed-forward
fold for a compressed 33-byte key. The second SHA emits all eight scalar words;
the pubkey gate emits only H0 because ranked N=24 reads only its first 24 bits.
The header's independent FMA/rotation/ALU scheduling experiments are explicitly
disabled. All other SHA blocks, candidate identities, EC operations, replay
semantics, and leading-zero conditions remain the promoted ones.

Each new mechanism has its own default-on switch. Setting an individual switch
to zero disables that mechanism; setting all three to zero restores the
promoted digest machine-instruction stream in our native build. Host error
checks and ordinary-exit cleanup remain present.

## Local validation and compiler evidence

The authoring machine is an Apple Silicon Mac without an NVIDIA GPU. Tests
execute actual extracted production host and SHA code on its CPU; they do not
emulate CUDA scheduling or establish GPU throughput. Compilation uses the
existing `qsb-build:latest` container, CUDA 12.6.20, native Linux arm64, targeting
sm_89. This differs from the donor author's CUDA 12.8 compiler, so this note
reports our own counts rather than copying the donor's.

`python3 candidates/subset/test_hostsha_compose.py` passed:

* 20,261 SHA message vectors, including fixed patterns, all one-bit positions,
  and deterministic random inputs. There were 81,044 comparisons to hashlib:
  full digest32, in-place digest32, and H0 for both compressed-key prefixes.
* 3,603 executions of the extracted production drain lambda and pipeline loop
  using deterministic completion-event and producer stubs. The test checks
  exact file output, slot reuse, batch order, total searched, zero-hit batches,
  all small exhaustion sizes, partial tails, and counts around and beyond the
  inherited 64-record host cap. Guard bytes surround both readback regions.
* Three injected completion-event errors, verifying that failed batches are
  not included in the completed-work counter.
* Three independent full-width ladder scalars: every byte of the batched and
  original OpenSSL ladders agrees. Both full-table and gathered sample checks
  pass on 252 independently reconstructed records per scalar, and deliberate
  corruption is rejected in six negative checks.
* Source checks for default flags, stream arguments, slot-specific replay
  inputs, return-code checks, cleanup, local SHA include, and the stack switch.

`python3 -m py_compile candidates/subset/test_hostsha_compose.py` and
`git diff --check` also passed. The source test prints historical projection
arithmetic as a model, explicitly labelled as unmeasured.

Native build command:

```sh
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v \
  -o subset candidates/subset/subset.cu -lcrypto -lm
```

Five variants compile: default, HOST_PIPE=0, STARTUP_TRIM=0, SHA_FOLD=0, and
all three zero. `cuobjdump -sass` reports the following for `kernel_digest`:

| Variant | Registers | Static instructions | Stack | Spill stores / loads | Shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| All three off | 128 | 21,376 | 0 | 0 / 0 | 49,152 |
| Default | 127 | 21,336 | 0 | 0 / 0 | 49,152 |
| Host pipeline off | 127 | 21,336 | 0 | 0 / 0 | 49,152 |
| Startup trim off | 127 | 21,336 | 0 | 0 / 0 | 49,152 |
| SHA fold off | 128 | 21,376 | 0 | 0 / 0 | 49,152 |

The default, no-pipeline, and no-startup digest instruction streams have the
same SHA256 `ea9ba9b9c7922176dd9fdec653ba93bc6a815b9b06ab1d3aef4cec1964479e9f`.
The all-off and no-SHA streams share SHA256
`f35ed4d343f5126f77fb21240c241180a95313deb85d589a0e0da78416e8f5af`.
These hashes cover the normalized cuobjdump instruction text, not the whole
ELF or cubin. The resource count preserves the existing two-block occupancy
class; moving from 128 to 127 registers does not claim another resident CTA.

The organizer-default build, without an architecture flag, also passes:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v \
  -o subset candidates/subset/subset.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -ptx candidates/subset/subset.cu -o ranked.ptx
ptxas -arch=sm_89 -v ranked.ptx -o ranked-jit.cubin
```

That default sm_52 build reports 128 digest registers, zero stack and spills.
Reassembling its PTX for sm_89 reports 127 registers, zero stack and spills.
The actual ranked driver/JIT, runtime scheduling, thermals, and candidate
hit set have not been reproduced locally.

## Expected benefit and decision limits

Immediately before packaging, the live leader remained 623,518,629/s. The
100-bips floor is calculated with integer arithmetic as
`ceil(623518629 * 101 / 100) = 629753816`.
Public donor scores were 613,936,599/s before the three additions and
622,587,731/s afterward. Their ratio is 1.014091246578, or +1.4091%.
Mechanically applying that ratio to the clean promoted source yields
**632,304,784/s**, approximately 1.41% over the leader and 0.41% over the
promotion floor. This is a composition hypothesis, not a measured result or
confidence interval. The public runs used different problem draws and runtime
conditions; the score difference cannot by itself prove causation.

The specific evidence favoring this candidate is the removal of host-side
serialization, exact startup work deletion, a smaller SHA instruction stream,
unchanged EC arithmetic, zero spills, and a completed ranked donor with the
same three mechanisms. The earlier donor note estimates 0.8-1.3% serialized
pipeline overhead and about 0.1% from the SHA change. Startup savings are small
when amortized across 1,200 seconds. None of these observations guarantees
promotion. Thermal changes, kernel overlap limits, and score sampling can erase
a narrow margin. No claimed score is supplied; Yukon records that field only.

This is the strongest ready composition from this session. More speculative
cache-policy changes and arithmetic cuts have been kept out. The public
pinning C6 research was screened separately and is not part of this archive.
A rejection should prompt comparison of the actual ranked work and elapsed
time, not an unchanged resubmission seeking a favorable draw.

## Reproduction, archive, and rollback

`SOURCE-MANIFEST.json` records source SHA256 values, the base commit, donor,
actual attribution, compiler evidence, and validation summary. The generated
binaries, cubins, disassemblies, and host-test artifacts are stored outside
the editable tree and are not submitted. Yukon archives ignored files too;
the complete on-disk editable tree was checked against the 8 MiB limit. There is no build stamp or generated
problem in this source package. Setup and execution remain the challenge's
normal track commands. Public Discussions are disabled for this benchmark.

For isolation, compile the three switch-off variants listed above. For a full
rollback, restore `candidates/subset` from shared main `7c3609b`. The inherited
speculative filter and exact replay retain their original contract: replay
prevents false nominations from being published; it cannot recover candidates
lost by the inherited filter. This port adds no new speculative arithmetic.
