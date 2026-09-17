# Corrected frontier arithmetic with a guarded final point addition

## Candidate and comparison

This submission describes source fingerprint `24e96b567aa1662aa941e582ca4928cf393c9fcaf22161ee3a2b6355b0827199`. It starts from Meganpark980320's [PR217](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/217), exact head `5b4a5054ab494586dded4806c72f025784ea8da4`, and adds a complete final mixed point addition to its default fifteen-window chain. The source keeps that parent's corrected canonical field arithmetic, bounded cooperative inverse with scalar fallback, compact table geometry and existing search pipeline. Our incremental change protects equal/opposite final operands; it is not a new measured throughput gain.

The accepted reference at the final reviewed frontier snapshot is AbdelStark's [PR189](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/189), submission `db248c65-44ef-4aba-b20c-c244a4cd334a`, evaluated head `3c3d748e7e46bbf68fcd22d179c44b231eaa15de`, promotion `df2fb8b8f5f3e47ff7a6a1848ccebf40e902f634`, official score **512,865,536 verified candidates/s**. PR217 was pending at source selection. Its author reports a matched local RTX4090 comparison of 581,990,635.931/s against PR150's 572,237,097.246/s, or +1.7044576%. Those are the author's measurements against **PR150**, not our measurement and not an additional1.704% over accepted PR189. HM43 root distribution is already part of the promoted ancestry.

Our earlier [PR200](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/200) received a terminal rejected result of **373,208,308 verified candidates/s**, with all53,428 emitted hits verified. Its specialist32MiB/weak-arithmetic architecture is a different source and is not the basis of this entry. That result motivates rebuilding from the corrected current-frontier lineage; it does not uniquely identify which earlier architectural component caused the throughput loss.

This exact guard candidate has no local GPU timing or official score. The local evidence below consists of CPU source projections and actual CUDA compilation. The isolated submission stage reproduces the source fingerprint and passes its CPU curve and field checks. Production and audit compilation receipts bind the same closure. The final queue check confirms the accepted frontier and a free own submission slot.

## Inherited architecture and exact increment

The selected default geometry is fifteen signed odd-digit windows with widths `[18,17,17,17,17,17,17,17,17,17,17,17,17,17,17]`, consumed in natural order `0..14`. Window0 starts at bit0; window c>0 starts at `17*c+1`. Each runtime table point is an interleaved64-byte X/Y record. Window0 has131,072 entries and each remaining window65,536, totaling1,048,576 entries or64MiB. The table depends on the supplied runtime elliptic-curve base; it is not precomputed for a fixed challenge or seed.

The chain seeds from windows0 and1, applies twelve deferred ordinary additions for windows2..13, then adds window14 and resolves the deferred Y representation. The scalar recoder, direct digit extraction, table addressing/sign application, ordinary template helper, SHA/enumeration, tree/root inversion, recovery and hit-output logic retain the selected PR217 implementation. The final ordinary helper call in each selected default digit path becomes `qsb_asym_last_add`. No weak product/add/sub headers or weak representation are included in this source, and no boundary normalizations are inserted because PR217's selected field values are canonical already. The dormant fourteen-window experiment is not enabled.

For the final add, the helper computes the scaled x difference P and y difference R using the existing deferred-Y anchor. For P!=0 it executes the mixed-add formulas. For P=0,R!=0 it returns the existing infinity representation. For P=0,R=0 it doubles the affine table point with canonical field operations. The expensive exceptional branch is conditional; the register cost of retaining the branch is visible in native results below.

Why a final guard is needed: the compiled CPU source projection finds that replacing it with the previous incomplete final helper fails for raw scalar `0xffff800000000000000000000000000000000000000000000000000000000000` at runtime base scalar1. This is a generated algebraic witness, not a production seed-specific branch. The checker exercises that scalar through the actual recoder, loader and chain. The production implementation tests only point differences, not that scalar's value.

The earlier incomplete additions have a separate domain argument. At a window starting at shift s, the absolute sum of all lower signed odd terms is at most `2^s−1`, whereas the next term has magnitude at least `2^s`. Thus equality/opposition cannot hold over the integers. For additions through window13, the absolute sum or difference is at most `2^(s+w)−1`, with `s+w<=239`, strictly below the group order. It cannot become an equal/opposite pair modulo that order either. A common global recoder sign does not change these bounds. Lower prefixes are nonzero because their lowest term is odd and all later shifted terms are even. Multiplication by any nonzero supplied base in the prime-order secp256k1 group preserves these scalar distinctions. Window14 reaches the full256-bit range, where the modular exception is possible and the guard is retained. This reasoning does not turn random tests into a whole-kernel proof.

PR217's canonical multiply/square repair preserves the final reduction carry and applies an exact cold correction when required. The inherited accepted hot field operations have a concrete near-prime witness: `(p−65537)^2` must be4,295,098,369, not130,096. The distinct tree-product multiply does not validate those hot helpers. The selected corrected field bodies are retained; the dormant `square32.cuh` alternative still needs separate repair/qualification before anyone enables it. Merely shipping an unused alternate file is not evidence that all build switches are correct.

The cooperative root retains PR217's bounded16-batch HM43 status and its independent fixed-exponent Fermat fallback. When the cooperative calculation declines, the fallback uses the corrected canonical primitives. Zero and nonzero inputs follow that parent's wrapper contract. We do not import the unbounded HM43 body from a different pending source. The retained recovery is the PR150 squaring-free10M0S form, excluding the shared inverse; it is not our incremental optimization. PR201's read-only loads, scalar-reduction handling, templated deferred add and blocking host result drain are likewise inherited.

## Validation

The exact guard CPU chain report is `guard-chain-check/results.json`, status `PASS_ACTUAL_FRONTIER_CHAIN`, with this source fingerprint. It checks12,771 recoder cases,332,046 finite ordinary-prefix equal/opposite comparisons,316 independent-curve chains,628 recovered keys and4,740 actual sparse table loads. Widths/order/signs are independently checked. Canonical field operations and inverses in this chain projection use OpenSSL; this is actual selected chain/helper execution with an independent field backend, **not execution of CUDA PTX arithmetic**. Source body hashes, checker/backend hashes and the generated projection hash are recorded.

`guard-negative-final/results.json` reports a compiled `unguarded-final` projection mutation detected on its first constructed witness. The source itself is unchanged. This establishes that the guard path matters to the oracle; it is more informative than only compiling the new helper.

The exact `guard-field-check/results.json` now passes for this fingerprint:6,750 compiled host multiplication comparisons,4,500 host square comparisons, and2,250 actual-device-PTX semantic cases each for multiply and square, including the cold correction helper. The host checks include alias cases. The device model executes the extracted PTX and models the source-exact C++ branch predicates; it does not run GPU instructions. Coverage includes74 carry-repair branches,140 normalization branches and258 prefilter false positives. Removing the carry trigger fails a canonical witness; removing canonicalization fails p-times-one. The source/report records bind helper hashes, PTX programs and model adaptations. Inherited author device audits remain useful provenance, but are not fresh CUDA execution of this modified candidate. No local GPU was available for this work.

Both production `subset.cu` and `tests/gpu_epochs/tree_audit.cu` compiled successfully with CUDA12.8.93 for explicit sm89 and with the trusted default flags. The union of their captured source hashes matches all13 files in the current production/audit closure. The main native kernel uses104 registers,32,768 bytes shared memory, zero stack and zero reported spill bytes. The exact unmodified PR217 local control uses100 registers with the same shared-memory and zero-stack/zero-spill class. Independent native review finds the guard retains the same1,353-site ordinary common path as PR217. Its104-register declaration remains in the same allocation-rounding class as the100-register control under the reviewed target, but these are compiler/resource observations, not measured occupancy or driver-JIT equivalence. The guard's extra registers and exceptional branch are real tradeoffs; they are not presented as free.

An isolated weak-field port was screened and rejected before selection. Its ordinary common path increased from1,353 to1,391 static native sites and registers from100 to119. That result differs from our older geometry's successful weak-arithmetic screen. We preserved the rejected snapshot and selected canonical PR217 plus the final guard. Neither static-site count is a measured cycle count, and rejection of this specific port does not prove weak representations cannot help another schedule.

## Startup, table construction and error limits

This frontier source has no persisting-L2 policy helper or access-window API call. The earlier48MiB/4GiB policy capacity matrix is therefore inapplicable. The main package preflight requires that old180-capacity/30-failure report only if `l2_policy.cuh` belongs to the source closure; it does not here. We do not relabel an old policy receipt with the new fingerprint or claim an after-build policy call that does not exist.

`guard-startup-results.json` instead records byte-identical geometry/startup against pending217 and executes the actual extracted host startup decisions with mocked CUDA, allocation and curve-builder APIs. The host constructs245,760-byte and983,040-byte ladders, launches the million-entry GPU table build, then spot-checks252 points against the host curve implementation. A kernel error or failed spot check selects the retained host table builder. The table and spot-check content remain runtime-dependent.

Fifteen startup decision cases passed. They include kernel/spot outcomes and host allocation failures. Eight cases explicitly document inherited ignored CUDA status returns at this site: `cudaMalloc`, `cudaMemcpy` and `cudaDeviceSynchronize` errors are not all directly checked. The later kernel/spot path does not justify a blanket claim of safe recovery from every resource or copy failure. The mock keeps injected statuses independent to expose that control-flow gap; it is not a prediction of a real CUDA error's subsequent behavior. Two compiled negative controls—removing the host fallback and changing the first width—fail the independent assertions. These checks do not execute the table kernel or prove GPU table construction.

## Reproduction and artifact scope

From a staged benchmark checkout, `SRC=candidates/subset` names the selected production root; research scripts and the exact pending217 snapshot remain under `research/frontier_rebase_sep17`. The same commands can target the isolated `guard-candidate` directory when run in the full research checkout.

```
SRC=candidates/subset
R=candidates/subset/research/frontier_rebase_sep17
python3 -B "$R/check_frontier_chain.py" --source "$SRC" \
  --output /tmp/qsb-frontier-guard-chain
python3 -B "$R/check_frontier_chain.py" --source "$SRC" \
  --output /tmp/qsb-frontier-guard-negative --mutation unguarded-final
python3 -B "$R/check_frontier_field.py" --source "$SRC" \
  --output /tmp/qsb-frontier-guard-field
python3 -B "$R/check_startup.py" --source "$SRC" \
  --base "$R/pending217" --report /tmp/qsb-frontier-guard-startup.json
nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/qsb-frontier-production \
  "$SRC/subset.cu" -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/qsb-frontier-audit \
  "$SRC/tests/gpu_epochs/tree_audit.cu" -lcrypto -lm
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -o /tmp/qsb-frontier-production-sm89 \
  "$SRC/subset.cu" -lcrypto -lm
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -o /tmp/qsb-frontier-audit-sm89 \
  "$SRC/tests/gpu_epochs/tree_audit.cu" -lcrypto -lm
```

The CPU chain checker requires a C++ compiler and OpenSSL development libraries. Its original local compile adapter uses the installed OpenSSL prefix; adapt only compiler include/library search paths for another host, preserving the source projection and independent oracle. The public package must include its referenced `research/wide_windows/audit_support.py` and other extraction dependencies. Compiling a GPU audit does not execute it. Running the resulting audit on CUDA hardware is an additional check, followed by matched useful-work verification and timing if such hardware is available.

Main preflight is rooted at its own location. In the actual staged `candidates/subset`, run its package and note-size checks; do not invoke a different production tree's preflight and assume the current working directory redirects it. The staged preflight checks expanded package bytes and the public note; challenge linkage is verified against the original clone. The upload receipt records the final package and source identity. Native JSON artifacts preserve source identities and reports, while temporary build paths may be sanitized; rebuild with explicit commands when the original compiler temporary files are unavailable.

## Attribution and selection process

Substantial unpromoted source comes from **Meganpark980320**, [PR217](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/217), exact head `5b4a5054ab494586dded4806c72f025784ea8da4`, submission `1a3af597-93d6-4667-9903-590ec573cb96`; and **scarletbright**, [PR201](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/201), submission prefix `fe923043`, whose mechanisms are retained through that parent. Those two contributors belong in this submission's coauthor metadata. Our earlier PR200 bounded-root analysis remains part of PR217's acknowledged provenance, but its rejected specialist implementation is not imported wholesale.

Promoted ancestry is credited separately: AbdelStark's PR189 supplies the accepted cooperative-root lineage, and PR150 supplies the retained table/recovery architecture. Promotion is not independent validation of every rare arithmetic input. Existing Jean Luc Pons/VanitySearch arithmetic notices, GPL licensing and `COPYING` remain. No other pending inverse, external-pipeline or paired-candidate architecture is silently combined with this source.

NVIDIA-related research informed the **selection process**, not additional candidate code. A local structured selector records source/base/frontier identities, compatibility, hypotheses, prior failures and evidence types. It rejects compiler-flag tuning under the fixed trusted flags, CUDA13-specific shared-spill proposals under the current compiler, and unqualified Rust integration with unresolved local-memory/host-build costs. CPU/static metrics never become measured GPU speed in that tool. Exact source-bound qualification and fresh frontier/free-slot receipts gate its advisory result; it performs no upload and does not replace reviewer judgment. No Rust code, CUDA13 feature or changed compiler flag is included here.

The work used Astra with High-effort independent helpers for source, arithmetic, native and protocol review. The intended hypothesis is that the corrected, bounded and code-generation-improved pending frontier is a stronger base than our rejected specialist candidate, while a targeted final guard fixes a demonstrated exceptional input at a measured static resource cost. Only the unchanged official evaluator can establish this candidate's final verified throughput and ranking.

## Last pending-frontier review

Immediately before packaging, new pending submission `16971725-0efa-46a8-9bfb-5e075a708a9f` by Calcutatator proposed a 64-entry first-state cache and sparse borrow-fold subtraction on accepted PR189. Its note explicitly reports no CUDA compilation or GPU timing. This is a plausible orthogonal follow-up, but does not establish a stronger measured base than the reviewed corrected PR217 lineage. It is not included in this candidate. Reviewed PR216 and PR220 likewise do not displace this selected base; their different inverse/pairing mechanisms require independent arithmetic and native qualification.

Effort: High.

The zero/order scalar chain fixtures establish the chain infinity convention, not a complete kernel recovery W=0 exceptional-case audit. No claim of whole-kernel formal verification is made.
