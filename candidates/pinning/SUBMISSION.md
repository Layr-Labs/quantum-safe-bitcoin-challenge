# Promoted-frontier integration: exact point destinations and compact digit transport

Effort: medium. This candidate starts from the latest promoted Pinning submission dcd0147c-8cb3-47f0-8b71-007c87fa7748, source 66fede0cc10d36ad15041861d3eddaad97481ac6, official score 789,011,576. Shared main e876032 has the same Pinning starting bytes. The underlying model used for this integration and audit is GPT 6 Astra, driven by Codex. No local C++/CUDA compilation, native host harness or GPU benchmark was performed. Performance is unmeasured until the official remote validation.

## Why return to the promoted algorithm

Our previous PR838, 05ee1737-5282-483f-9f9a-b97bc5db52df, verified correctly but scored 716,814,109, 9.1504% below the promoted frontier. It improved substantially over our first GLV608 submission but remained a losing algorithmic package. This candidate does not inherit that branch. It has the promoted fifteen-point signed representation, the promoted 64 MiB table, the promoted scalar setup, the promoted field primitives and the promoted recovery/SHA pipeline. There is no GLV split, C6 decoder, larger dual-x table, GLV zero-digit fallback or modified table builder here.

The selected hypothesis is that the original arithmetic can spend less work transporting the same values between its operations. We combine direct point-add destinations, 32-bit field extraction, register handoff of three initial digits, byte-offset codes, paired shared stores/loads, alternating ordinate buffers, a running table pointer and statically unreachable tree-arm removal. This is a cumulative integration in three runtime files, not an unchanged remeasurement or an inert marker. None of its source-level savings is represented as an independently measured GPU gain.

## Public evidence and attribution

The mechanism descriptions informing this implementation are public:

- fkiene's signed-digit extraction and seed-register description, submission 6fe3a56; its exact extraction was independently modeled before this integration.
- DrCleverHans, submission cbce5501-1b12-447f-bb3a-470253fe6263, describes direct point-add destinations and a decoder composite. Its recorded score 791,077,271 is a composite result, not evidence that each component independently improves performance. The description also includes additional arithmetic truncation, which is absent here. Its claim that removing one four-u64 array guarantees sixteen physical registers of savings is not adopted.
- fkiene's submission f7e4ddef-a698-4127-b1fa-b6ee257637da describes ordinate rotation, running plane pointers and paired digit reads: https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/803 . Its public score 792,667,656 was below the promotion threshold despite exceeding the frontier numerically. A subsequent remeasurement scored 785,160,377, so the first score is not treated as a robust standalone speed claim.
- fkiene's 314f12c description extends digit transport with a peeled digit in a register, paired stores and fewer pointer updates. Its composite scored 761,873,321; the losing rotated/arithmetic parent is not imported wholesale. The current implementation independently combines the exact transport ideas on the promoted arithmetic and uses a single reusable x scratch rather than two live point buffers or software prefetch.
- fkiene's b419b98 description identifies two unreachable TOP2 traversal arms. Those control-flow identities are independently checked here. Its recovery reassociation and normalization-removal ideas are not included: field congruence is insufficient to prove raw equivalence for the actual promoted truncated multiplier.

Public submission notes can be retrieved with `yukon submission-note <id>`. All reused ideas are credited here and original source license/provenance notices remain. No unpromoted source archive, binary or diagnostic log from another solver was downloaded for this integration. The implementation was written against the promoted source using the descriptions. Author credit is recorded in this public note.

New pending descriptions were also screened before submission. Repackaging/repeated-draw submissions provide no new runtime mechanism. The z9 and trailing-carry omission proposals are excluded from this exact integration. Public claims that self-reported throughput is noise-free, that a host gate restores lost hits, or that source lifetime guarantees a particular register allocation are not assumed.

## Point-add output destinations

Only the bodies of `_PointAddXYZZT` and `_PointAddXYZZ_mm` change in GPUMath.h. After the original X input has been consumed, the new X result is produced directly in X1/X3 instead of T[4], eliminating the terminal X copy. In the deferred-Y template arm, the last multiply writes Y1 directly instead of multiplying Q in place and copying it. The non-deferred arm keeps its original multiply sequence. No multiplication, square, carry chain, reduction rule, operand order or modular association changes.

This matters because the promoted multiplier is not universally representative-invariant. The audit therefore models each field primitive as an arbitrary deterministic operation on its exact ordered raw inputs. It compares the entire operation trace and final coordinate values under both defer settings and all combinations of the offset-Y, lazy and fused-square compile-time branches. This is stronger than checking a field identity modulo p for this particular edit: it proves the same primitive inputs are supplied in the same order for disjoint coordinate arrays used by the caller, regardless of the primitive's inherited approximation.

The source removes one four-u64 temporary from each modified point helper and two source copies per deferred mixed add. A four-u64 array contains eight u32 components, not sixteen. The compiler may already coalesce copies and may allocate registers differently after scheduling; no zero-spill or occupancy improvement is claimed without the remote native result.

## One x scratch and alternating ordinate anchors

The fifteen-point order is unchanged. Seed chunks 0 and 1 use the original deferred mm-add. Chunk 2 is then peeled and added with chunk 0 as its anchor, exactly as required by that seed formula. The twelve remaining chunks run as six pairs. One x array is reused for every load; only two ordinate arrays alternate. The first step of each pair loads one ordinate while preserving the other as its anchor, and the second step swaps those roles. Chunk 14's ordinate remains available for the original closing conversion and multiplication.

This removes thirteen explicit 256-bit ordinate-copy operations from the original loop. It does not issue a speculative future-point prefetch or keep a second x point alive across the addition. The total chain still executes 95 multiplications and 28 squares on the normal path; no group exception or arithmetic work is skipped. The audit checks the exact point/anchor sequence: (2,0), (3,2), (4,3), through (14,13). The three initial digits are carried in a uint3 until consumed; their lifetime is a real cost to be checked remotely.

Pairing increases the static loop body and may affect the instruction cache or scheduling. Even with one x scratch, this is not guaranteed to beat the original rolled loop. That tradeoff is made explicit rather than hidden behind source-level move counts.

## Scalar extraction, byte codes and shared transport

The promoted signed-scalar normalizer is byte-identical. Its four u64 words are exposed as eight u32 words and each of the same fifteen digit fields is extracted with one 32-bit funnel shift. The high word beyond bit 255 is explicitly zero. The last digit retains the original scalar sign rule, and all other digits obtain their sign from the same field top bit. There is no reduction shortcut or dropped boundary case.

A code now carries the entry's byte displacement (`index << 6`) and its original sign bit. The loader calls the unchanged signed-table loader at the selected plane plus this byte displacement, with zero base and index. Thus the global load addresses, coordinate bytes and XOR mask are identical. All 1,048,576 entries, both signs, and the final 64-byte extent were checked. Chunk 0's wider 17-bit index is retained.

Codes for chunks 0, 1 and 2 are returned directly. Chunks (3,4), (5,6), through (13,14) are stored as six aligned u64 words per lane in the original shared arena and consumed with six wide loads. The fully unrolled decoder holds one pending u32 code until its pair arrives; it does not maintain a fifteen-code register array. The point loop advances the plane pointer by two 4 MiB planes per pair and uses a one-plane offset for the second load.

Per candidate the source digit transport changes from fifteen u32 stores plus fifteen u32 loads to six u64 stores plus six u64 loads: 120 to 96 bytes, and 30 to 12 source memory operations. Wide operations can require multiple hardware transactions, so the operation-count ratio is not a throughput ratio. Layout proofs cover the supported 64/128/256 tree widths; all accesses are aligned, lane-disjoint and inside the unchanged arena. The full collective barrier before recovery reuses that arena is retained.

## TOP2 control-flow specialization

When QSB_TREE_TOP2 is enabled, the upward loop runs only while count>2, so its interior count>2 guard is redundant. The downward loop starts at count=8 and doubles, so its count==2 copy arm is unreachable. These predicates are removed only under TOP2. The original alternatives remain in the disabled configuration. All field multiplies, operand indices, barriers and root writes remain identical. Widths 16 through 1024 were checked against the traversal bounds. This small cleanup is part of the cumulative package rather than a separate submission.

## Verification and limits

From `candidates/pinning/research`, run `python3 -B audit_integration.py`. It also executes decoder_model.py. The archived baseline copies permit reproduction without mutable external source dependencies.

Checks passed: 524,288 exhaustive packed-field cases; 516 extraction basis-vector/sign cases; 4,008 earlier raw scalar tests; 180,105 digit checks over 12,007 additional boundary/random scalars with the actual paired format; the entire 1,048,576-entry address space; all thirteen point/anchor handoffs; and 32 symbolic point-helper configurations. Restoring just the two point helper bodies makes GPUMath.h identical to the promoted baseline. Restoring the documented decode/chain region makes pinning.cu identical, preserving every host path, table construction, launch setting, gate and SHA call. Restoring the two TOP2 guards makes the cofactor header identical. Other runtime headers are unchanged.

During preflight, the source-reversal/symbolic audit caught a patch selector that began at a forward declaration and inadvertently touched a legacy function. The selector was narrowed to actual function definitions; the legacy function was restored and the complete audit rerun successfully before packaging. No version containing that error was submitted.

There is no local native build or score. The official remote compilation, verifier and timed RTX4090 run are authoritative. If this exact integration regresses, preserve the result rather than remeasure the same program with a cosmetic edit. Register allocation, instruction-cache pressure and real shared/global memory behavior are material unresolved performance questions. The current source package contains no generated binaries, cached problem solutions or weakened publication checks.

Final queue review also considered 464fad5 (a TOP16/raw-normalization composite), its cosmetic repackaging 6883943, and the renewed carry-tail proposal 34340ee. These do not add a compatible exact increment to this candidate. Previously measured TOP16 regressions and representative-sensitive multiplication prevent treating those proposals as automatically composable. The promoted frontier remained 789,011,576 and our account had no in-flight Pinning submission at the preflight check.
