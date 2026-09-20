# Canonical helper contracts

This is an optimization of the public QSB benchmark on generated inputs. The generic field add/sub routines retain the complete raw-representative corrections.

`kernel_build_gtable` now normalizes its inversion input and both output coordinates before storing an entry. Host ladder coordinates already come from OpenSSL affine conversion. Each nonzero odd table multiple is finite and has nonzero y in the odd-order subgroup. Signed loads therefore return canonical x and either y or p-y. This construction establishes the canonical-table property instead of relying on spot checks.

`_ModAddCanonicalPair` is used for the sum of the current and previous signed affine table ordinates in the point addition chain. Both are below p. Their sum is below 2p, so a second carry after folding is impossible. The generic `_ModAddLazy` still supports arbitrary raw representatives.

`_ModSubCanonicalRhs` is used for the initial two table-point differences, the recovery denominator after X is explicitly normalized, and five packed-recovery differences. In the direct-distance recovery sequence, the right operands are v, l, m and the two recovered distances d1/d2. u/v are normalized by qsb_recovery_mul. Subtracting canonical u/v gives canonical l; `_ModAdd256` conditionally subtracts p from the sum of canonical u/v, giving canonical m. The same addition gives canonical sum from l/m. Each distance is normalized by qsb_recovery_mul before it becomes the right operand of a-d. The fixed a and c are canonical host values, so both final x coordinates are canonical too. For any raw left operand a<2^256 and canonical right operand b<p, a-b>-p, so the second borrow correction is unnecessary.

The generic functions remain complete outside these narrower call sites. Both helper bodies passed 39,762 CPU actual-PTX modular-oracle cases in their declared ranges; the generic add/sub bodies passed a further26,034 arbitrary-raw cases. The exact source's native helper audit additionally passed 78,102 independently checked outputs with both aliases. Point and pipeline audits also passed; throughput qualification is separate. CPU checks are not GPU or score evidence.

Runtime identity is recorded in SOURCE-MANIFEST.json. The pre-normalization
prototype 3f313002 was never measured or submitted; its performance and caller
contracts are not used as evidence for this source.
