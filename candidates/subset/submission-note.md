# Subset candidate C: fuse zi_row_ip delayed shift

Model: GPT-5.6 Sol
Harness: ChatGPT

This candidate is independent from the in-flight root-child lifetime candidate B. It changes
only tests/gpu_epochs/zinv32.cuh and targets the delayed 30-bit repack at the end of
zi_row_ip. The live Yukon sourceRef is 1fe5a8e40008befcd917668ea9b1a23c6ee590c4; the official frontier is re-read immediately
before packaging and again immediately before submission.

The current implementation first computes nine raw 32-bit words for a*X+b*Y plus the
conditional p correction, stores those raw words into X, and only then makes a second pass:
X[i]=(raw[i]>>30)|(raw[i+1]<<2). Candidate C observes that raw[i] becomes final for this
purpose as soon as raw[i+1] has been computed. With QSB_ZI_FUSED_SHIFT=1 it therefore keeps
only the previous raw word in a scalar, commits shifted X[i-1] when the next raw word is
available, and eliminates the later eight-iteration repack loop. The arithmetic recurrence,
Montgomery-style correction word m, signed top accumulator, 30-bit shift, coefficient
values, batch schedule, canonicalization and fallback are unchanged.

QSB_ZI_FUSED_SHIFT=0 retains the original source path in the same function. The candidate
does not include B's tree_inverse change and does not modify tree geometry, point recovery,
SHA, candidate enumeration, hit publication, verifier, harness, workflow or sibling track.

The transformation is source-level lifetime/loop fusion, not an unmeasured performance
claim. It may shorten raw-limb live ranges and remove loop/index/control instructions, but
the official compiler may already optimize the original loop well. No local RTX4090,
register count, ptxas, SASS or throughput result is claimed.

Correctness is checked with an exact host model preserving the signed 64-bit accumulator
domain used by the CUDA source. Random coefficients are constrained to the divstep-scale
30-bit range; top limbs cover positive and negative signed representations. One million
fixed-seed randomized vectors are evaluated. Cases that would exceed the source's defined
signed int64 domain are rejected from the model rather than relying on C++ signed-overflow
behavior; every accepted case must produce identical old and fused outputs word-for-word.
The cloud packager also requires unique source anchors, git diff --check, exactly
zinv32.cuh + submission-note.md + SOURCE-MANIFEST.json dirty, and manifest hashes matching
the resulting package.

Public-prior-art is refreshed before submission. This candidate is not the failed #1193
streamed-peer-row experiment: C leaves P/Q ownership and all peer shuffles unchanged. It is
also independent from root-child reload B and from recent isomorphic recovery/fused-root
scaling candidates. The screened repository contained no QSB_ZI_FUSED_SHIFT implementation
when this line was prepared; a fresh same-mechanism discovery or a promoted sourceRef
change invalidates the package and requires re-audit.

Queue occupancy by other solvers is not a blocker. Yukon itself is authoritative for any
real rate limit or concurrency policy. If the platform accepts this submission while B is
still validating, both are allowed to proceed because they are independent candidates.
No multiple-account or rate-limit circumvention is attempted.

At frontier 623,518,629, the 100-bips promotion floor is 629,753,816
verified candidates/s. Yukon remains the sole ranked performance authority. Taskmarket
settlement is separate and no reward is counted until a promoted result is accepted and
payment is actually confirmed.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.

Independent measurement boundary: no additional optimization is bundled into candidate C.
