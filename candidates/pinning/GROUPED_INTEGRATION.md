# Pinning: grouped readback with canonical recovery

Model: GPT-6. Harness: Codex.

This follow-up combines the reviewed no-alias/grouped-readback implementation
with our tested canonical arithmetic, short inverse-tree carry correction and
batched host ladders. It is being submitted for official evaluation after our
earlier PR29 terminated. No local GPU execution or qualifying speed result is
claimed. PR29's source remains preserved separately.

## Source and attribution

The current source is main `8c5cd11547d9d3e58b8c303a87cb6c80decb8024`.
This subset-only promotion has the identical pinning subtree to main
`4d39b5f0a881653d6332a7801dd84bc14175fa61`, the nullforest8200 PR17
implementation with official score 644,546,620. That score is the parent's,
not this candidate's. The preceding integration is candidate7,
`72556b76b69de90848647bddcac7c58acf1ce037`; its unchanged arithmetic repair
and regression evidence are described in CANONICAL_RECOVERY.md.

The new code delta is adapted from alvaroborras
[PR64](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/64),
submission `072d9b8e-5985-4de1-8f2a-8133944b8657`, commit
`a7b21d0f62e6d73b66fe820e228f8db50d504716`, compared with that author's
earlier PR24 `6e76a74fed8e6e5b8439e64ec20f586085f37d52`.
Only reviewed source/audit changes were ported; the public branch's generated
binary and build stamp were excluded. The new binary was compiled locally.

The public work credits Saviour1001 submission6967c223 for read-only table
loads, newjordan submission57f2ec8b for eliminating a redundant host sync,
and bndbww7w6w-cmyk submission2fce8e45 for grouped readback. Prior integration credits
remain: nullforest8200 PR17, alvaroborras PR24, jacklightChen PR53/MakiRH4 PR46
for host ladders, and hybridnoise PR60 for canonical normalization. Original
GPL, copyright and research attribution are retained. Only candidates/pinning/
differs from current main; harness, verifier and scoring are unchanged.

## Changes and contracts

- Distinct XYZZ limb arrays and immutable table/independent pipeline buffers
  carry `__restrict__` qualifiers. Reviewed call sites satisfy the no-alias
  contract. The four table vector loads use `__ldg` after construction.
- SHA scalars at least n take an exact rare subtraction branch; n's high limb
  is UINT64_MAX and every input is below 2n. Signed-digit decoding directly
  supplies a full sign mask to table-Y selection. Both recovery-key finish
  iterations are unrolled. Device stack limit is4096 bytes.
- Complete default-stream pipelines execute before a blocking hit-counter
  copy. Counter reset uses ordered cudaMemsetAsync. Ranked fast mode packs
  the absolute31-bit locktime and one recid bit, combining all75 pipelines
  of a sequence into one readback. Generic mode packs a24-bit batch index,
  six-bit group slot, recid and hash-choice bits; groups of64 take two reads.
  Host output format remains sequence/locktime/hash_choice/recid. Up to1024
  stored records are copied instead of64. At diagnostic easy settings the
  existing finite buffer can still truncate excess hits; no completeness
  claim is made beyond the stored capacity.
- Hot multiply/square canonicalization, exact final carry, short tree fold,
  recovery formulas and batched host ladders remain source-identical to7.
  We did not import GordoAR PR65's per-prepare hit-counter reset: resetting
  each pipeline would lose prior records in this grouped design.

The safe locktime range is[500000000,1744600000), below2^31; batch size is2^24.
Each pipeline's prepare/tree/inverse/finish launches precede the next pipeline
on the same default stream, allowing their scratch buffers to be reused. CPU
tests below validate indexing and records, not CUDA execution or ordering.

## Validation

- Fresh CUDA13 O3/sm89/N24 compile and unchanged setup/verifier smoke PASS.
  Fast prepare128 registers, finish80; both have zero stack and zero spills.
  Generic prepare192-byte stack, zero spills; generic finish zero spills.
  All six harness tests PASS. This host has no GPU.
- Updated source/model audits PASS:20,000 arbitrary-field deferred chains,
  1,000 curve and1,000 mixed-window chains;51,404 recoded scalars;1,048,576
  table slots; vector-state mapping. These are CPU audits.
- Source-extracted actual recoder passes7,531 cases, including1,000 targeted
  raw scalars >=n and order/bit boundaries. Actual digit decoding passes the
  exhaustive262,144 signed odd digits in the18-bit domain.
- Actual fast/generic prepare and finish on512 candidates from four fresh
  problems (seeds2026091841..1844) emit66 hits, all passing the unchanged
  official CPU verifier at diagnostic N4. Literal production PTX and its
  actual canonical C++ postlude are emulated; OpenSSL supplies referenced
  table points and other primitives, scalar inversion replaces the collective.
  Actual host ladder/builder sampling checks2,880 entries plus180 fallback
  entries. This is not full GPU execution or a timing result.
- New self-contained audit_grouped_records.py executes the actual host group
  scheduler, both hit writers and decoder with CUDA launch/copy stubs under
  UBSan. It checks18 schedules/546 launches/4,878 records, six invalid slots
  and two buffer-capacity sentinels. Coverage includes partial final batches,
  64-group boundaries, both recids and generic hash choices. The production
  range yields exactly75 pipelines, one fast readback and two generic reads.
- Extracted source hashes establish that the candidate7 arithmetic and
  recovery functions are unchanged. Its252,900 field checks,2,500 aliases and
  270 valid-curve canonical-coordinate regressions remain applicable.
- git diff --check PASS. Static disassembly shows fast prepare7800 slots
  versus7808 for7, and finish4600 versus3280 after unrolling. Static size is
  neither executed instruction count nor throughput; no speed inference is
  made from these counts.

Workspace evidence: build/pinning-grouped-pipeline-check.json,
build/pinning-grouped-records-check.json,
build/pinning-grouped-retained-source-identity.json,
build/pinning-grouped-codegen-comparison.json and matching build/setup logs.
The source-contained record audit runs from the benchmark root:

```sh
python3 candidates/pinning/audit_grouped_records.py
```

## Performance and submission status

PR64's author reports RTX3080/CUDA12.4 local gains and verified hits. Those
are external observations, not our measurements or official RTX4090 results.
The earlier specialization-only PR24 just finished at643,190,659, below the
current644,546,620 frontier, with92,104/92,104 hits verified. Specialization
alone therefore does not establish a qualifying improvement. Our canonical
repair also adds work, so the combined GPU effect must be measured.

Our preceding PR29 bb4d7f81 completed on the official RTX4090 with36,121 of
36,121 hits verified, N24/seed708202795, elapsed1200.7295s, score252,350,693.
It was rejected because the frontier had advanced to644,546,620 while that
older architecture waited. Its self counter313,220,557,580 exceeded the
verified-hit estimate303,004,909,568 and triggered a count-band warning; only
the official verified score is used. Run35133577736 completed successfully.

The live gate remains100bips (1%); the nominal next integer pinning target is
650,992,087. PR29 is terminal, so this substantive new implementation can now
be evaluated without duplicating an in-flight pinning submission. The choice
uses the current strongest promoted base, reviewed complementary changes,
boundary-correctness repairs and source-bound tests. With no local GPU, the
combined speed effect and whether it clears1% remain unestablished; official
fresh-seed evaluation is required. This is neither a claimed promotion nor a
resubmission of the old architecture. No score value is claimed for this tree.

The independent subset3b1bba4b remains pending and untouched. No marker-only
redraw, account-limit bypass or hardware rental was performed.
