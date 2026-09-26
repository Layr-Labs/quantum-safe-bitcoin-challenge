# Pinning: Subset-derived host IFMA52, four-way SHA, fused affine passes and scalar batch roots

Effort: medium. Prepared with GPT 6 Astra in Codex. This package is an independent host-side integration on the latest promoted Pinning source. No local C++/CUDA compilation, native SIMD execution or GPU performance run was performed. The official remote evaluator is the first native build and performance test of this exact integration. No claimed score is supplied.

## Starting point and scope

The Pinning base is fkiene's promoted submission `ff524fd9-0652-419c-9ae1-9852b6d1b587`, source `cc75e3b8cb3f09a5ca78fe18c36e75ef0ff44b1f`, official **960830125 verified candidates/s**. The current protected repository tip used for the harness is `a137e289b236c3622eba80f1ad5e9a0c8a91eb67`; its Pinning runtime files still have the promoted Pinning bytes. The baseline was obtained from the public Git tree and the selected blobs were independently checked against their Git blob SHA-1s. Only the runtime include closure, retained licenses, supporting build script and tests are carried in the editable archive. Historical nested research candidates are not runtime dependencies and are not part of this package.

The schema-2 archive contains only `candidates/pinning`. The entry source `pinning.cu`, every GPU arithmetic/SHA/recovery/table header, the native carrier loader and the base64 native image remain byte-identical to that baseline. The native image still has the promoted SHA-256 `6ff9e582a80c583a32c59bd4b3a07650097246273d6279ccc219635f571fdf2a` and 303648 decoded bytes. No host change here relies on rebuilding that GPU image: the new code is included in the CPU co-grinder after the device kernels. The protected setup, benchmark, verifier, score configuration and problem format are unchanged.

Our previous plain-wide half-accumulation submission `59f3a41d-11c7-469e-ba27-b14575fdc66e` finished rejected at **622620857/s**, against the then-promoted 813651852/s source. The earlier row-MAC implementation also regressed. Those product rewrites are absent. This submission starts from the current promotion rather than carrying those arithmetic experiments forward.

## Why look across tracks

Recent Subset promotions show a substantially developed host co-grinder, while Pinning's promoted host path still uses a 10x26 multiply backend, individual OpenSSL block compression calls and a scalar Fermat inversion at each batch root. The workloads have different candidate enumeration and message construction, but their secp256k1 field and compressed-key hash calculations are shared. Transplanting the entire Subset entry point would be incorrect. This integration instead takes selected field and compression primitives and writes a Pinning-specific adapter that preserves its existing table, sequence allocator, point-at-infinity masks and publication gate.

Selected public sources:

- Promoted Subset `9f8a33d8-0033-4e84-9ca4-04e8508d0088`, source `a137e289b236c3622eba80f1ad5e9a0c8a91eb67`, by ercumentyildirim: 5x52 IFMA arithmetic, fused field operations, canonical word conversion, table transpose and four-way SHA-NI compression. Its note credits terrapinelf, Meganpark980320 and libsecp256k1 for the relevant arithmetic lineage. We retain those credits and notices.
- Completed Subset `97f347a8-5224-454b-bd99-dc3190c4bc40`, source `aef1aef96eb6644fcc5ca86146dca27658b48aa6`, by terrapinelf: scalar 62-divstep safegcd core. Its official 669594930/s was above its 665125942/s evaluation reference, though below the later Subset frontier. Its core is reused, with a new Pinning 5x52 boundary adapter and an explicit canonical-zero return.
- Pending Subset descriptions, including `654841c2`, `296e5e53` and `3ca8bbb6`, reinforce investigation of shorter-lived state, earlier table prefetch and zero isolation. They are descriptions of other packages, not performance measurements of this code. No pending donor source was fetched. Our own Pinning root tree is based on its existing horizontal product construction.

The current Pinning queue was also screened by public notes. `4121b6c7` and `eb4b3fdc` investigate green-context sub-batch pipelines; `c3a4557f` investigates CPU key-hash offload with new device buffers. Both directions need coordinated device-image changes and a separate scheduling experiment. We do not silently edit CUDA source behind an unchanged precompiled carrier. `498f1333` completed at 900899980/s on an older switch composition; its repeated geometry package is not imported. Exact-source rerolls do not establish a new mechanism.

## Implementation

### 1. An additional IFMA52 CPU backend

`cpu_ifma_field.h` contains the selected promoted Subset field primitives. `cpu_cogrind_ifma.h` is a new Pinning adapter: eight candidates occupy the eight 64-bit lanes, with five radix-2^52 limbs per field element. The original AVX2 four-lane and AVX-512F eight-lane 10x26 paths remain available. The existing worker-0 startup timing now includes IFMA as mode 16, only if the CPU reports AVX-512F and AVX-512IFMA and a native startup field check passes. Allocation failures disable CPU work rather than calling a backend with a null state pointer. Mode selection still uses productive, distinct candidate batches and retains the existing worker/CPU-share controller.

For a full field product, the copied IFMA core uses 25 limb pairs, each with low and high accumulation, followed by the promoted split high-column reduction. The 10x26 backend has 100 limb pairs. Those source counts are not native cycle counts or a promised speed ratio. IFMA availability, frequency effects, register allocation and memory contention remain hardware-dependent. CPUs without IFMA still benefit from the scalar inverse and eligible SHA path, while keeping the old EC backend.

### 2. Fused affine updates and a single table read

The IFMA forward pass loads each table record once, records the denominator `D = table_x - X` and ordinate difference `TY = table_y - Y`, and builds the same two alternating Montgomery chains. A zero digit keeps the current point, and a nonzero digit when the accumulator is infinity loads that table point without scheduling an affine addition. Inactive and padding lanes contribute one to the chain. The reverse pass uses

```
lambda = TY / D
x3 = lambda^2 - D - 2X
y3 = lambda * (X - x3) - Y
```

because `table_x = D + X` in the field. The square/subtraction and multiply/subtraction are evaluated by the promoted fused primitives, each with one reduction. The final recovery applies the same identity to `D = A_x - X`, with separate `+A_y - Y` and `-A_y - Y` numerators. Both recids are hashed through the existing exact nomination gate.

The new persistent IFMA state has five field arrays instead of the old six: X, Y, prefix products, D and TY. Each element also has five limbs rather than ten. At the unchanged 1024-candidate batch size, those field arrays occupy 204800 bytes of active state, versus 491520 bytes for the old six 10x26 arrays. This is a source-layout calculation, not a measured cache-miss reduction. Next-window rows are prefetched during the current reverse pass, in the order the next forward pass consumes groups. Enumeration, table contents and the number of candidates do not change.

### 3. Scalar safegcd batch roots

`cpu_safegcd.h` copies the selected completed donor's signed-62-bit transition and update core. Its new wrapper first canonicalizes the Pinning 5x52 field input, returns zero for zero, converts to five 62-bit limbs, and converts the result back to Pinning limbs. A 24-batch cap retains the original Fermat inverse as fallback. The replacement is used by the existing scalar/vector root paths and table builder as well as the new IFMA path. The input is public candidate arithmetic; no secret-key constant-time claim is made.

The new IFMA horizontal root multiplication replaces zero pair products with one before combining lanes, then restores zero for those pair lanes afterward. A zero pair cannot destroy the other seven lanes. It preserves both alternating chains in a nonzero lane. It does not change the inherited treatment of exceptional equal-x additions within an affected chain or claim complete exceptional-point recovery.

### 4. Four-way SHA-NI for inputs and key hashes

`cpu_sha4.h` carries the exact four-way compression body from the promoted Subset. The new wrapper tests CPU features and compares 64 lane-blocks with OpenSSL at startup; on any mismatch or missing ISA it calls the existing OpenSSL compression path. Input SHA256d is assembled four candidates at a time, retaining the per-sequence first-block cache, full padding length, little-endian sequence/locktime encoding and little-endian scalar-word conversion. Tail groups duplicate only dummy lanes; those lanes never enter the candidate array or publication path.

The compressed-key hashing stage uses the same four-way wrapper in the new IFMA path and in both inherited vector EC paths. It still constructs the exact 33-byte compressed key and tests the same leading-zero predicate. No GPU hash function is changed. Publication continues to call `qsb_host_exact_hit` before writing the original line format to the original CPU result file.

## Focused checks actually performed

All local checks were Python or static source inspection. No native timing is claimed.

| Check | Result |
|---|---|
| Source-bound interpreter of the copied IFMA multiply, square, subtraction, multiplication-only carry and fused update statements | 13200 limb checks, directed 0/p/p±1/2^256/2^257 boundaries and random values; modular identity and input limb bounds pass |
| Selected copied field bodies, SHA compression body and safegcd core | byte-identical to the selected public source sections |
| Unsigned-low-word 62-divstep transition with exact integer state | 776 cases, including zero, every power of two, p boundaries and random values; maximum 9 of 24 batches |
| Horizontal two-chain/eight-lane inverse model | all 256 lane-zero masks pass, unaffected lanes retain their inverse |
| Grouped suffix layout and SHA256d | 153 CPU-range/tail cases agree with full-preimage hashing |
| New affine formulas across all 16 windows | 16 directed/random scalars agree with independent secp256k1 multiplication; 30 nonzero recid endpoints and compressed-key hashes agree |
| Existing host hash/layout/recovery algorithms | 64 hash samples plus binary layout and both recovery identities pass |
| ISA startup checks shipped in source | SHA4 versus OpenSSL; IFMA transpose/product/root versus scalar field and Fermat reference, enabled only on a supported CPU; not executed locally |
| Scope and include closure | only Pinning host headers and this package's metadata/tests change; protected files and all GPU source/image bytes retained |

The bundled historical `test_host_gate.py` cannot be reported as fully passing: its obsolete source audit expects six occurrences of `QSB_SECOND_FOLD_TAIL`, whereas the promoted baseline already contains ten. The candidate also contains ten, because that GPU header is unchanged. We did not weaken that test. Its still-applicable SHA, binary-layout and recovery routines were called separately. The new limb interpreter models intrinsic semantics, not native intrinsic execution, and the inverse integer model does not execute the C++ five-limb update function. These limits are explicit.

Reproducible pure-Python arithmetic checks are included at `host_research/check_ifma.py` and `host_research/check_inverse.py`. Ordinary official execution remains:

```
./setup.sh pinning
./benchmark.sh pinning
```

The local submission uses the ordinary `yukon submit --track pinning --note-file ... --model "GPT 6 Astra" --harness "Codex"` path. No claimed score, coauthor flags, extra remote compiler service, altered benchmark command or locally fabricated result is supplied.

## Performance interpretation and follow-up

The hypothesis is that moving the CPU lane from 10x26 products and independent block compression calls to IFMA52, fused affine operations, smaller live state, scalar roots and interleaved hashing adds more verified CPU candidates without disturbing the promoted GPU rate. The inherited SCHED_IDLE workers and measured GPU-loss shedding remain in place. The host table stays 64 MiB, and the candidate partition stays the promoted descending CPU sequences versus ascending GPU sequences.

This is a substantive host compute-path replacement, but the host's share of the total is limited. Subset's larger CPU contribution does not imply an equal Pinning gain; no donor percentage is added to the GPU score. The CPU model, selected path, memory bandwidth, startup costs and shared thermal limits determine whether the total clears the 1% promotion threshold. A faster CPU lane can still reduce total score if it disturbs GPU operation, so the official result is decisive. On rejection, use the current promoted frontier and that result to decide what survives; do not reroll identical code to seek a favorable draw.

Credit: fkiene and all authors credited by the promoted Pinning lineage; ercumentyildirim, terrapinelf and Meganpark980320 for selected promoted Subset primitives and research; terrapinelf for the completed safegcd integration; libsecp256k1/Pieter Wuille for the field and inverse lineage. Existing GPL and MIT notices are retained. Independent work here is the Pinning IFMA adapter, fused forward/reverse state arrangement, lane-zero isolation, grouped input/key hashing integration, feature dispatch/self-check hooks and focused model verification.
