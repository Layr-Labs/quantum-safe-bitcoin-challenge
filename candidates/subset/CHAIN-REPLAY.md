# Exact replay for point-chain field boundaries

Parent: canonicaladd 56039952205fb52ff65ecc0244c1fa3499f16a03. All inherited VanitySearch/GPL, fixed-base table, complete final-addition, bounded inverse and SHA attribution remains in place.

Only ordinary fixed-base point-chain addition, multiplication and square helpers defer their cold field correction. Their literal PTX matches the canonical parent; the original boundary prefilter sets a sticky flag instead of dispatching immediately. Subtraction and complete final point addition remain exact parent operations. At the end of the chain, a set flag reruns the original complete chain from the saved scalar and overwrites every output. Scalar digits and table addresses depend only on the saved scalar. The trial has bounded loops and no external side effects.

Before the first marked guard, the trial is bit-for-bit identical to the original. If no guard fires, the entire trial matches. If a guard fires, exact replay replaces all results. This preserves full-domain behavior and does not accept rare arithmetic errors or filter outputs approximately. Saving the input also preserves output/input alias behavior.

The source-bound CPU audit checks 15,060 literal PTX operations and 75,300 sticky-flag cases, with independent arbitrary-precision expected residues. It includes boundaries which fail without replay. Exact fallback body and unchanged parent field source are hash-bound. These are CPU checks; actual CUDA audits and matched solver timing are required separately.

Native CUDA13 sm89 diagnostic: 124 registers, no stack/spills, 32 KiB shared, 19,192 digest instructions versus 13,976 parent. Cold code duplication increases static size by37.3%; this count is not runtime-weighted and is not a speed claim. The normal official build uses CUDA12.8.93 with default architecture. No submission qualification is claimed here.
