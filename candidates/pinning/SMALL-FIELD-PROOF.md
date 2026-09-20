# Raw field add/sub boundaries

The public QSB candidate uses p = B-K, where B = 2^256 and K = 2^32+977. This change keeps results congruent modulo p for arbitrary raw representatives in [0,B). Canonical callers preserve their prior outputs. It does not alter the benchmark domain or verifier.

For subtraction, a-b is in (-B,B). A first borrow adds p by subtracting K from the wrapped low result. If that correction borrows again, the remaining low result is B-d with 0<d<K. Subtracting K once more yields p-d and cannot borrow beyond the low 64-bit limb, since 2K < 2^64.

For lazy addition, a+b is at most 2B-2. Folding the first carry by adding K may overflow again. That second overflow leaves a low result below K; adding K a second time fits in the low 64-bit limb. Both carries are now preserved.

The optimized subtraction and lazy-add routines capture the carry or borrow
after the first low 64-bit correction. When it is zero, that correction cannot
change any upper limb, so the upper chain is skipped. When it is nonzero, the
complete upper chain and any second correction execute. The CPU fixture covers
both outcomes: subtraction has 8,603 zero and 75 nonzero cases; lazy addition
has 8,614 zero and 64 nonzero cases. Native checks also cover in-place aliases.

For the one-subtraction addition, retain the signed high limb of a+b-p. When it is 1, the low result lost one B and therefore needs an extra K. That low value is at most K-2, so the correction also fits in one 64-bit limb. An output may be noncanonical for arbitrary raw inputs, but remains congruent; canonical input callers still receive canonical results.

The actual inline PTX was interpreted against Python integers on 8,678 pairs per routine, including densely constructed second-carry/borrow neighborhoods. All 26,034 operations passed. The parent has 36 add, 47 subtract and 36 lazy-add failures on this deliberately adversarial raw fixture. These are primitive-domain findings, not a claim of normal production failure and not grounds to exclude any public competitor.

The final normalized source additionally passed 78,102 raw native add/sub outputs, 21,885 native multiply/square outputs, complete point checks and eight pipeline modes. Original bytes were independently checked locally. See RESEARCH.md for exact source identity and final qualification status.
