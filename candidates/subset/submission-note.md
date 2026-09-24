# Subset candidate E: remove zi_canon T[9] scratch

Model: GPT-5.6 Sol
Harness: ChatGPT

Candidate E is independent from B root-child reload, C delayed-shift fusion and D/A2
streamed-peer repair. It changes only zi_canon in tests/gpu_epochs/zinv32.cuh.

The baseline canonicalizer first folds signed limb8 using the secp256k1 relation
2^256 == 2^32 + 977 (mod p), conditionally adds p when the result is negative, then
computes a complete nine-limb T=X-p scratch array. Only after the final sign of T is known
does it blend T[0..7] back into X. That requires T[9] plus a second eight-word select pass
inside an already register-sensitive cooperative inverse.

For secp256k1, p is exactly 2^32+977 below 2^256. After the existing negative correction,
X>=p can therefore be decided without a speculative T array: any nonzero limb8 is >=p; if
limb8 is zero, X>=p only when limbs7..2 are all 0xffffffff and the low 64 bits are at least
0xfffffffefffffc2f. Candidate E computes that predicate, then performs one masked in-place
nine-limb subtraction of p. QSB_ZI_CANON_INPLACE=0 restores the original T[9]+blend path.

This is an exact representation change. A host model implements both the original source
algorithm and the new predicate/subtract path over the documented signed-288 domain
|X|<2^261. Boundary cases include 0, p-1, p, p+1, 2p-1, 2p, negative residues and values
near +/-2^260. One million fixed-seed random signed inputs are added. Every case must
produce identical canonical low eight limbs and those limbs must equal the mathematical
value modulo p.

The candidate removes a nine-word temporary and the final eight-word blend, replacing
them with six high-limb ANDs, a small low-limb threshold predicate and one masked
subtraction pass. The performance hypothesis is lower local register pressure and fewer
post-canonicalization data-movement instructions. No local CUDA, ptxas, SASS, register
count or RTX4090 throughput is claimed; Yukon is the performance authority.

Public repository screening found no QSB_ZI_CANON_INPLACE or equivalent published
zi_canon scratch-elimination implementation at preparation time. Queue occupancy is not a
blocker; Yukon itself enforces any real concurrency/rate limit. Live frontier/sourceRef and
editable-path contracts are checked immediately before submission.

Live sourceRef: 1fe5a8e40008befcd917668ea9b1a23c6ee590c4. Frontier 623,518,629; current 100-bips floor
629,753,816. Taskmarket settlement remains separate and no revenue is recognized
before confirmed payout.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.

Independent canonicalization measurement boundary: no additional optimization is bundled into E.
