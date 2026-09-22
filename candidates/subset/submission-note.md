# Subset: isolate a 16 MiB signed multi-comb on the promoted K2 frontier

Effort: max

## Experiment and starting point

This is a structural performance experiment. It replaces the promoted fixed-base
table representation with a signed B=8, T=16, S=2 multi-comb, reducing the complete
affine lookup table from 64 MiB to 16 MiB. It retains the promoted two-epoch paired
recovery, 128-window enumeration, SHA code, field primitives, inversion ownership,
publication replay and launch geometry. There is no local NVIDIA throughput
measurement and no claimed score. The official run is intended to decide whether
the smaller working set repays the additional group arithmetic.

The public starting point is shared main
`7c3609b87b9d8e094a16be148fe846dfd5ac7807`. Its selected subset tree is identical
to the promoted `9ac2515450446dbadbe061e98ebfc317c36d4999` tree, submitted by
Akashneelesh as `7aef224a-e3ff-43f9-9877-50cdbda3f653`, with an official score
of 623,518,629 verified candidates per second. Later shared-main changes belong
to the other track. The current service requires a 100-bips improvement at
evaluation time; the source-level optimization described here is not evidence
that this threshold will be met.

This experiment isolates the compact-table mechanism from our earlier combined
compact-comb/four-epoch entry, public source
`54562620970e9cf9a9dc38ba4ffca09f2466f091`, submission
`11551718-8cf9-4c53-8ada-79ce49bab211`. That older combination was valid but
rejected at 454,413,395 candidates/s. Its four-epoch staging and ownership changes
are absent here. That result is a reason for caution, not an isolated measurement
of compact-comb performance with the current K2 implementation.

## Exact identity and representation

Let n be the secp256k1 group order, A=(-r^-1)G the public instance-dependent base,
P=A/2, and d=(k+(2^256-1-n)/2) mod n. The code preserves the bit-256 carry when
adding this offset and performs the required conditional reduction. For each
bank b, tooth t and row j, the bit at 2*(16*b+t)+j selects a positive or negative
power. Summing the signed powers yields

```text
sum(sign_i * 2^i) * P
  = (2*d - (2^256 - 1)) * (A/2)
  = k*A.
```

Each bank stores 2^15 affine points. A sign symmetry stores only the half with
the highest tooth negative: when its bit is positive, complement all sixteen
bits, address the fifteen low bits, and negate the selected point. The address
is not a truncated scalar. Two rows consume the same eight banks, with one
doubling between rows. Total storage is 8 * 32768 * 64 = 16,777,216 bytes.

The exact path uses deferred-anchor XYZZ mixed additions. The last step retains
complete infinity/equal-point/inverse-point handling. The inter-row doubling
consumes and produces exact Y; its anchor is reset before the low row begins.
Intermediate signed-power ranges exclude the exceptional equal/inverse sums
before the final step. Zero and order-valued scalars are included in the host
boundary checks. Public synthetic scalars are the intended input; this variable
time implementation is not a secret-key multiplication API.

The existing speculative field filter remains inherited from the promoted base.
The compact exact path supplies the unchanged GPU publication replay with the
new table geometry, so tentative nominations are recomputed before publication.
As with the base, exact replay prevents false records from being published but
cannot restore a true hit missed by speculative arithmetic. No new truncated
carry rule is introduced by this port.

## Implementation and table construction

The runtime delta is confined to three files under `candidates/subset`:

* `tests/gpu_epochs/hm83_comb.cuh`: packed scalar recoding, signed table loads,
  complete doubling/final addition, exact and speculative comb chains.
* `tests/gpu_epochs/hm83_table.cuh`: instance-dependent OpenSSL ladders, Gray
  traversal construction, independent table spot checks and the CPU fallback.
* `tests/gpu_epochs/tree.cu`: the 16 MiB geometry, table-builder indexing and
  dispatch to the two compact chains.

Packed row digits stream from scalarized 64-bit words. There is no new global
scratch array per candidate. The existing GPU builder combines low and high
ladder points into ordinary binary-addressed table entries. The host constructs
3072 ladder points from the fresh instance. Four corners of each bank plus 192
deterministic samples are checked against independent OpenSSL scalar
multiplication. If the GPU-built table fails that check, the CPU fallback builds
the same signed geometry. Neither path depends on a previously seen problem.

The main kernel keeps its current 256-thread blocks, two-epoch pairing, inverse
tree, 49,152-byte shared allocation and paired short-hash gate. Candidate domain,
recid priority, hit format, N=24 ranked condition and judge-owned timing are
unchanged. No harness, benchmark manifest, scoring function, workflow, problem
generator, committed problem or sibling-track source is changed in the archive.

## Cost case and strongest counterargument

The original table uses fifteen point selections. This comb uses sixteen and
adds one point doubling. Logical table requests therefore increase from 960 to
1024 bytes per scalar before transaction granularity and cache reuse. The
doubling costs six field multiplications and three squarings, with additions;
the extra point, recoding and conversion costs also have to be paid.

The upside depends on improved reuse of a table one quarter the original size.
That is a hypothesis about the actual workload, not a claim that 16 MiB is
resident or that every saved capacity byte is saved traffic. If the promoted
table already receives efficient cache service, the extra arithmetic can make
this candidate slower. The whole-workload speedup cannot be inferred from the
allocation ratio or static code size. Setup and table-check time also count
inside the official process window.

This is why the experiment keeps the existing K2 scheduling and does not combine
the table change with additional host pipelines, hash ownership or occupancy
changes. A rejected result will be retained as a result for this implementation;
it will not be relabeled as a speedup or used to claim a universal algorithmic
floor. An unchanged archive will not be resubmitted merely for a favorable draw.

## Validation and execution limits

The authoring host is macOS arm64 without an NVIDIA device. Whole-program CPU
emulation on a newly generated synthetic instance, seed `202609220707`, ran 37
epochs per launch for two launches at N=11. Candidate and unchanged promoted
control each produced 43 unique hits; both set differences were zero. This
includes an odd epoch tail. The sample remains below the inherited hit cap.

The host adapter disables `QSB_K2S_PARITY_WINDOW`, `QSB_SHORT_CARRY3` and
`QSB_CHAIN_ANCHOR_UPDATE` in both arms because their CUDA-only implementations
are not portable C++. The ranked candidate retains the promoted defaults.
Host agreement is evidence for the representation, table, recovery and output
integration, not execution of inline PTX or real CUDA synchronization.

The full current source was compiled privately with CUDA 12.8.93. The exact
organizer frontend command and separate sm_89 JIT-style assembly were used:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -ptx -o compact.ptx candidates/subset/subset.cu
ptxas -arch=sm_89 -v compact.ptx -o compact.cubin
nvdisasm compact.cubin > compact.sass
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v \
  -o compact candidates/subset/subset.cu -lcrypto -lm
```

The sm_89 digest reports 128 registers, 49,152 bytes shared, one barrier, zero
stack frame and zero spill stores/loads. The unchanged control has the same
resource class. Full executable linking with OpenSSL succeeds. A static census
including outlined digest functions grows from 21,376 to 26,192 instructions;
that includes complete exceptional paths and is not a dynamic instruction
count. There is no measured GPU occupancy, cache-hit ratio or elapsed speedup.
The compiler ran on Linux arm64; the ranked runner's exact execution and JIT
environment remain part of the official test.

Fresh component checks passed 2,048 compressed recovered keys and 2,048 key
hashes against the unchanged Python recovery/hash oracle, with zero mismatches
and no skipped recoveries. A separate 2,288-point check against libsecp256k1
passed zero, n, neighbors of n and n/2, every single-bit scalar and 2,024 new
random scalars. Both infinity cases were correct; the exact and filter host
chains agreed on every checked point. These tests use actual extracted current
component code, not another implementation of the same comb formula. The
source manifest records their seeds and results. They remain host evidence.

## Reproduction and attribution

The full source package fits the existing single-translation-unit interface.
On an authorized CUDA machine, run the official `./setup.sh subset`, then the
official benchmark with this source and a restored 9ac2515 control on the same
fresh synthetic seed. Alternate controls and candidates, retain complete hit
sets over their common candidate range, and include cold startup and the final
undrained batch when comparing wall time. The ranked service independently
chooses its seed and owns its clock. Restoring `candidates/subset` from 9ac2515
is the removal ablation for the complete table mechanism.

The signed-comb identity follows bitcoin-core/secp256k1
`ecmult_gen_impl.h` at `46db787112beabdb5e17e0dc35680716f1057e7b`; T=16 is our
separately checked parameterization, beyond that implementation's supported
CPU comb configuration. Group operations adapt the inherited VanitySearch
GPUMath primitives and the EFD a=0 XYZZ formulas. GPLv3 COPYING and all inherited
notices remain intact.

The promoted source lineage credits Akashneelesh, terrapinelf, jacklightChen,
dun999, ercumentyildirim, EvanYan1024, fkiene, Meganpark980320, Saviour1001,
owizdom, DPZZxlz, odinfree, i34-9, scarletbright, anamdongparkjinhyeong and
jrcarlos2000. This submission contributes the isolated compact-comb port and
its current correctness/compiler review. It does not incorporate another
solver's pending unpromoted runtime changes. The public artifact contains
source and this technical account; generated binaries, compiler outputs,
problems, credentials and agent session data are absent.
