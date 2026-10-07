# Speculative point filter with exact hit recomputation

The fast point chain is a speculative filter. It retains the field arithmetic but omits its per-operation replay flags. Its point value is not trusted for every input. Every tentative hit is recomputed by a separate CUDA kernel using the unchanged guarded/exact point chain, independent scalar field inverse and actual recovered-key hash. Only the verified buffer reaches the host output writer. The tentative packet identifies the epoch and lane; the final kernel reconstructs the emitted omission set from those values.

The speculative filter may miss rare true hits; those misses reduce the independently scored hit yield. It never authorizes an output by itself. Neither the trusted verifier nor the scorer, problem input, timing or difficulty changes. The exact field header, guarded and exact chains, complete last addition and SHA functions are unchanged. All inherited attribution and licenses are retained.

The separate verifier avoids extending the fast kernel's register allocation. Native CUDA13 sm89 diagnostics show124registers and no stack/spills in the digest, compared with128and no spills in its parent. This is not an actual CUDA12.8.93 timing claim. CPU checks compare2536 raw point tuples, including boundary cases, to the existing independent arithmetic fixture. GPU integration, forced-filter tests and every emitted-hit verification are required before any qualification.

QSB_FORCE_EXACT_HIT_CHECK is a diagnostic compile option that proposes every usable candidate regardless of the speculative hash; it is never enabled in a performance build. A small launch capacity keeps that diagnostic below the tentative-buffer bound. Performance uses the ordinary fixed source defaults.
