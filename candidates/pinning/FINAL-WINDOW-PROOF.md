# Complete final signed-window addition

This is a correctness argument for the public QSB benchmark's scalar multiplication. It is not a claim about normal production failure or an official score. The native witness experiment confirmed both parent failures and passed all 3,012 corrected point cases (753 scalars, two bases and two loader modes). Original coordinate outputs were independently checked with OpenSSL after downloading.

Let n be the secp256k1 group order, and let A be the fresh problem-dependent base. The signed recoding represents D = 2(k mod n)-n as fifteen odd signed digits. The table base is A/2, so the completed point is kA. The first window has width18 and the remaining fourteen have width17.

Before a window at offset s, the lower-digit prefix has absolute value at most2^s-1. The next nonzero odd term has absolute value at least2^s. Before the final window, their sum of absolute values is less than2^239, hence less than n. They can therefore be neither equal nor opposite modulo n. The initial and intermediate incomplete additions are valid.

For the final window, s=239 and the combined bound reaches2^256-1, which is larger than n. Set T=(2^17-1)*2^239 and d=n-T. Both are nonzero modulo n. For k=d the prefix is d and the last term is -T, so their points are equal modulo n. For k=T the prefix is -d and the last term is T, with the same equality. Both require finite doubling. The existing decoder's exact C++ was compiled on CPU and independently checked on753 constructed/random scalars; it reconstructs D exactly and confirms those two equal-point inputs. Neither input appears in the older16520-scalar point fixture.

The correction peels the last loop iteration. It computes the same mixed-add intermediates P=affine_x*ZZ-X and R=(affine_y+deferred_anchor)*ZZZ-Y. If P is nonzero, the existing fused arithmetic continues unchanged. If P is zero and R is nonzero, the result is infinity. If both vanish, the result doubles the known affine table point. Zero detection accepts both0 and p because the field primitives may return a noncanonical residue below2^256.

For doubling an affine point (x,y), use h=2y, ZZ=h^2, ZZZ=h^3, m=3x^2, Q=x*ZZ, X=m^2-2Q, and deferredY=m*(Q-X). The existing final caller subtracts y*ZZZ. Dividing by ZZ and ZZZ gives the ordinary affine doubling formula. No projective inversion is required. A finite secp256k1 subgroup point cannot have y=0 because the subgroup order is odd.

`PointZero.cuh` and `FinalAffineGuard.cuh` are copied with their GPL notices from
public submission 709ff130, commit 7d06816754f78d86cd62f6be4edf2ba25a68b14c.
The ordinary fused addition body derives from our earlier complete-correction
implementation. Runtime headers are bound by SOURCE-MANIFEST.json. RESEARCH.md
records the exact source's native and performance evidence.
