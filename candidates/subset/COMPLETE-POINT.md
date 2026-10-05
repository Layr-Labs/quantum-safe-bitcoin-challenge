# Complete final addition on the ordinary fifteen-term chain

Parent: our `quad32reg`, `3b9196b3df8a2cdbd24349d07a5fac8acb7d057c`. Its short matched measurement was +1.575% against PR189, but a newly reproduced exceptional-input defect means that result alone does not qualify the parent for submission.

PR229 / hybridnoise identified raw scalar `0xffff800000000000000000000000000000000000000000000000000000000000`. Our independent regular-odd recoder gives `sum(first fourteen terms) - final term = -n`. Thus the last two point operands are equal for every nonzero runtime base. The old incomplete mixed-add formula returned infinity instead of their double. The actual submitted PR217 recoder, loader and point formulas also fail this witness in a CPU projection using OpenSSL field operations. PR217 was cancelled on 2026-09-17 at 20:36:59.962 UTC.

This candidate uses our complete deferred-XYZZ last-add helper, independently written earlier for the shifted GLV experiment. It reuses the scaled differences already needed by the ordinary formula. Equal operands take affine doubling; opposite operands return infinity. Earlier additions remain ordinary: before the last chunk, the lower partial sum and the incoming shifted odd term cannot agree or oppose as integers, and their sum/difference magnitudes are below the group order.

The original fields, bounded inverse and fallback, table, SHA, enumeration, recovery and hit recording remain the parent's. Three compile-time digit/table paths call the complete final helper. The default fifteen-term path is the qualification target. No speed improvement is claimed for this correctness repair.

The CPU source projection passes 316 curve chains and 628 recovered-key comparisons, with 12,774 recoder cases. `point_audit.cu` checks the production point chain against OpenSSL at two fresh base scalars. Its fixture includes both modular-doubling witnesses plus the old order/bit/shift edges and random scalars. The same audit can include an unmodified parent by setting `QSB_POINT_TREE` at compile time, providing a negative control. Device completion and performance evidence are recorded separately.

Attribution: hybridnoise's PR229 supplies the ordinary-chain witness and the source-projection checker; our preexisting shifted-GLV helper supplies the repair implementation. All parent field, inverse, SHA and XYZZ notices remain intact. This is not a whole-kernel formal proof; singular recovery denominators retain the parent's handling and are a separate domain to audit.
