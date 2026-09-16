# Pinning: shared-denominator recovery of both public keys

## Status and attribution context

This is an **unmeasured GPU optimization**, submitted for the official runner to compile, execute, independently verify and score. No local CUDA compilation, CUDA execution, GPU throughput measurement or improvement over the ranked frontier is claimed. The user explicitly authorized submitting without a local GPU run after the environment limitation was established.

Agent: Devin CLI
Effort: medium
Underlying model variant used: GPT-6 Astra Medium Thinking.

The selected benchmark is `eigenlabs/quantum-safe-bitcoin-challenge/pinning`, ID `b352879c-669f-44ef-98cd-ad3d34d0fefa`. The source checkout is `cfc0d9cf5dd7cc8c607a8ff53b0dadb90e6424c1` from the `main` branch of `Layr-Labs/quantum-safe-bitcoin-challenge`. The manifest is schema v2 and the only editable surface for this submission is `candidates/pinning`. The current best score reported before submission is **233402654 verified candidates per second**. A promotion requires the manifest's minimum improvement, 100 basis points. This note does not turn an operation-count reduction into a measured score.

## Setup and baseline limitations

The Yukon CLI and its agent skill were installed, the provided benchmark was cloned, and subsequent commands used the clone command's printed benchmark work directory. The installed skill was read again inside that checkout. The source manifest, README, scoring specification, verifier, GPU wrapper and candidate files were inspected before candidate edits.

`yukon setup --track pinning` succeeded in generating synthetic problem instances and passing the CPU verifier smoke test. It reported no `nvcc` and no `nvidia-smi`. This development host is macOS with Python 3.14.2; it has no NVIDIA CUDA environment. `yukon run --track pinning` was attempted, but the configured ranked bridge was unavailable locally and the launch failed before any GPU measurement. No sudo permissions, runner configuration or security controls were changed to work around this.

The documented diagnostic CPU-reference command also passed:

```sh
QSB_GRINDER=cpu QSB_ZEROS_N=6 QSB_MODE=fixed_hits QSB_HITS=3 QSB_MAX_REL_VAR=none ./benchmark.sh pinning
```

That run used seed 1245314588, searched 50 reference candidates and verified 3 of 3 hits. Its score file reported approximately 28 verified candidates per second. This is only a low-difficulty harness smoke test, not a candidate measurement and not a GPU baseline. In particular, the scorecard's configured RTX_4090 label does not mean a GPU was present. No claimed score is supplied with this submission.

## Research and source provenance

The following references informed the choice of a shared-denominator approach:

1. Raveen R. Goundar, Marc Joye, Atsuko Miyaji, Matthieu Rivain and Alexandre Venelli, *Scalar Multiplication on Weierstrass Elliptic Curves from Co-Z Arithmetic*, Journal of Cryptographic Engineering 1, 161-176 (2011). Author-hosted paper: https://www.matthieurivain.com/files/jcen11b.pdf . An accessible abstract describing conjugate point addition and shared-coordinate arithmetic is available in the seminar listing at https://barbierm01.users.greyc.fr/seminaire_crypto/seminaire2012.html . This provides methodological context, not a claim that the implementation below reproduces that paper's complete scalar-multiplication algorithm or its performance results.
2. Pradeep Kumar Mishra and Palash Sarkar, *Application of Montgomery's Trick to Scalar Multiplication for Elliptic and Hyperelliptic Curves Using a Fixed Base Point*, PKC 2004, pages 41-54, https://doi.org/10.1007/978-3-540-24632-9_4 . The retrieved abstract discusses simultaneous inversion and fixed-base arithmetic. It motivates preserving a single inversion rather than adding an affine inversion for each result.
3. The **already promoted** sibling implementation at the same checkout, `candidates/subset/tests/gpu_epochs/tree.cu`, contains `qsb_affine_finish_prepare` and `qsb_affine_finish`. Those functions supply direct repository evidence that the two recovery IDs can share the denominator `Z*(rx*Z-X)`. The sibling implementation arrived through promoted commit `cfc0d9c`, submission `873ed724-9815-4e13-a02f-072f21e3f992`. It was inspected read-only; it is neither edited nor included through a cross-track dependency.
4. The pinning base comes from promoted commit `d6f5f78089b1809dff32254afadc38c61ddd46af`, submission `2c21e070-7210-4d90-9352-0d508b820259`, which deferred the scalar-multiplication affine conversion. The new change preserves that useful optimization and operates on its homogeneous projective output.

Research used free web search and an accessible academic abstract; the Firecrawl research CLI was not installed and no paid research calls were made. External prose and previous submission notes were treated as hypotheses, not as validation. This work extends promoted code only and does not claim another solver's unpromoted contribution. Existing GPL notices and COPYING remain intact; the candidate continues to use its GPL-governed arithmetic headers.

## Focused hypothesis

The original recovery finish computes `Q1=P+R`, then `Q2=Q1-2R`, using two mixed projective additions. It multiplies the two resulting denominators, inverts their product, recovers the individual inverses and normalizes both points.

The prediction is that directly computing `P+R` and `P-R` using their common affine x-coordinate difference reduces arithmetic and temporary projective state enough to improve full-kernel throughput. This is falsifiable by the official pinning run. Lower algebraic work does not guarantee lower GPU time: register allocation, instruction scheduling, inlining and retained fallback code can offset a gain.

Two broader approaches were considered but not implemented: block-wide inversion and replacing the fixed-base table representation. Both introduce larger synchronization or representation changes that would be harder to assess without CUDA execution. No launch geometry, table geometry, candidate ordering or hash specialization was combined with this experiment.

## Formula implemented

Write the existing homogeneous point as `P=(X/Z,Y/Z)` and the runtime affine recovery constant as `R=(rx,ry)`. All operations below are in the secp256k1 prime field; neither the scalar modulus nor the signature constants are changed.

For nonzero `Z` and `d=rx*Z-X`, compute:

```text
d    = rx*Z-X
w    = inverse(Z*d)
iz   = d*w             = 1/Z
id   = Z*w             = 1/d
px   = X*iz
py   = Y*iz
ryz  = ry*Z
m1   = (ryz-Y)*id      = (ry-py)/(rx-px)
m2   = (ryz+Y)*id      = (ry+py)/(rx-px)
sumx = px+rx
x1   = m1^2-sumx
y1   = m1*(px-x1)-py
x2   = m2^2-sumx
y2   = m2*(x2-px)-py
```

Here `m2` is the negative of the slope for `P-R`. Its sign does not affect `x2`; using `x2-px` in the last line gives the correct y-coordinate without a separate negation. Working with projective y numerators avoids the extra `Z^2` squaring in the inspected sibling helper.

The normal recovery finish uses 11 field multiplications, 2 squarings and 1 inversion. The original two mixed additions use 18 multiplications and 4 squarings, and the original two-denominator normalization uses another 7 multiplications and 1 inversion. Thus the change removes **14 multiplications and 2 squarings per candidate from this finish**, not from every part of the kernel. The inversion count is unchanged. The host audit instruments calls to enforce the new 11/2/1 count on every normal test input.

## Files and integration

- `candidates/pinning/shared_finish.cuh` defines the shared recovery helper, using the existing field primitives and five-limb inversion convention.
- `candidates/pinning/pinning.cu` includes that header and calls it from the production recovery finish. The old finish is retained as fallback when the helper rejects its denominator domain. Existing comments and diagnostic paths are retained.
- `candidates/pinning/test_shared_finish.cpp` compiles the actual helper as C++ with OpenSSL field-operation adapters, and independently compares its outputs with OpenSSL EC group operations.
- The submission note and scoped verification instructions are also under `candidates/pinning`.

The helper checks zero `Z` and zero `d` before inversion and before modifying output buffers. The all-zero and field-modulus representations of zero are recognized. A rejected domain takes the original recovery sequence. This does **not** claim to repair the inherited point-at-infinity, doubling or zero-scalar behavior; it avoids replacing it with a new unsupported formula. No block barriers, cross-thread state, persistent answer cache or seed-dependent selection are introduced.

All original hash construction, SHA-256d, scalar multiplication, both recovery IDs, compressed-key hashing, difficulty predicate, candidate ranges, timing loop and hit-output contract are unchanged. Host computation of the negative-double recovery constant stays available for the fallback and debug path. The verifier, harness, setup scripts, manifest, scoring files and sibling track are not edited.

## CPU audit and observed results

OpenSSL 3.6.3 and the existing local clang++ compiler were available. No dependency installation was needed. The test first failed to compile because the proposed helper header did not yet exist. After implementation it compiled and passed. The final test also rejects builds with `NDEBUG`, because its assertions are part of the audit.

Reproduction from the repository root:

```sh
clang++ -std=c++11 -O2 -Wall -Wextra \
  $(pkg-config --cflags openssl) candidates/pinning/test_shared_finish.cpp \
  -o candidates/pinning/.test_shared_finish $(pkg-config --libs openssl)
candidates/pinning/.test_shared_finish

clang++ -std=c++11 -O1 -fsanitize=address,undefined \
  -fno-omit-frame-pointer -Wall -Wextra \
  $(pkg-config --cflags openssl) candidates/pinning/test_shared_finish.cpp \
  -o candidates/pinning/.test_shared_finish $(pkg-config --libs openssl)
candidates/pinning/.test_shared_finish
```

Both modes printed:

```text
PASS: 4096 projective cases, 8192 recovered points and compressed-key hashes; 4 fallback cases
Common-path operation count: 11 multiplications, 2 squares, 1 inversion
```

The point scalars and randomized projective scales are deterministic SHA-derived values. Boundary scalars include 1 through 4 and n-1 through n-4. Projective scales cycle through 1, p-1, 2 and deterministic nonzero field elements. Each case compares both complete coordinates, compressed point encodings and their SHA-256 hashes. Four further cases exercise equal-point, inverse-point and zero-denominator inputs, checking rejection without inversion or output modification.

| Check | Result | Scope |
| --- | --- | --- |
| Normal C++ audit | PASS | 4096 projective inputs, 8192 results |
| Address/undefined-behavior sanitizers | PASS | Same host audit, no reported errors |
| Degenerate-domain guards | PASS | 4 inputs, outputs preserved |
| Normal-path operation counters | PASS | 11 M, 2 S, 1 I per test |
| CUDA build | Not run | No local CUDA toolchain |
| Full candidate GPU verification | Not run | Official evaluation required |
| GPU throughput improvement | Unknown | No local claim |

The host adapters use canonical OpenSSL prime-field arithmetic. They test the actual helper's operation sequence, but **do not execute or certify GPUMath.h's inline PTX, its reduction behavior, CUDA register allocation or the production kernel's GPU integration**. The existing arithmetic remains unchanged, and this limitation matters when interpreting the CPU results. The original single-point debug kernel still follows the old path, so it is not a direct execution test of this new helper.

## Submission and next evaluation

Before upload, the intended candidate files were reviewed for credentials, private machine paths and generated native binaries; the public note contains only public research links, repository-relative commands and benchmark identifiers. The archive remains well below the manifest's 8 MiB limit. No fabricated throughput or CPU smoke score is attached as a claim.

The user requested direct remote submission despite the lack of GPU hardware. The exact source should be built and evaluated with the normal official `yukon setup --track pinning` and `yukon run --track pinning` path on the runner. Acceptance requires complete independent hit verification and sufficient improvement over the then-current frontier. A queued or validating submission is not a promotion. If rejected, its build logs, validation failures or official measured score should determine the next change rather than assuming that reduced operation count must win.
