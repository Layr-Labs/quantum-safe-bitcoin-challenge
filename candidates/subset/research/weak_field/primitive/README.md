# Full-range weak-field add/sub prototype

Frozen header: `qsb_weak_addsub.cuh`, SHA256
`31f72e474a8e42ae291b71a4f7250a3bc43c489c367a9787797f74169c52c88e`.

This is an isolated arithmetic component. It changes no frozen candidate, production GPUMath, staged source or submission. The parent owns any integration and CUDA compiler qualification. The header has actual CUDA inline PTX and an independent ordinary C++ host implementation. CPU model/host results do not establish native performance or GPU correctness.

## Contract and API

Let U=2^256, C=2^32+977 and p=U−C. Inputs and weak outputs are arbitrary integers in W=[0,U); outputs need not be canonical. The caller ABI is four little-endian uint64 limbs; the CUDA arithmetic operates on eight physical32-bit field words. No persistent ninth field word is introduced; carry/borrow scratch is temporary.

- `qsb_weak_add(r,a,b)` returns a weak representative of a+b modulo p.
- `qsb_weak_sub(r,a,b)` returns a weak representative of a−b modulo p.
- `qsb_weak_normalize(r,a)` returns a mod p in[0,p).
- `qsb_weak_normalize(r)` is the in-place overload.
- `qsb_weak_is_zero(a)` recognizes either weak representation of residue zero:0 or p.

Every input is captured before any output memory store. Exact in-place use, overlapping input ranges, and arbitrary partial overlap between aligned uint64 limb ranges are supported. The contract does not legalize unaligned uint64 pointers. CUDA asm has only register effects, so it has neither `volatile` nor a memory clobber. Input operands are moved into local PTX registers before output registers are written; their load-to-output data dependencies preserve the alias contract without a scheduling barrier.

## Addition proof over the entire input domain

Write a+b=s+kU, where0≤s<U and k∈{0,1}. Since U≡C modulo p, the first correction is t=s+kC. Compute it across all256 bits, and write t=r+jU,0≤r<U,j∈{0,1}. Then the desired residue is r+jC.

If j=0, r already lies in W and has the required residue. If j=1, necessarily k=1 and a+b≤2U−2 implies s≤U−2. Consequently r=s+C−U≤C−2. The second correction satisfies0≤r+C≤2C−2<2^64. It therefore touches only the low64 bits and cannot propagate into higher field words. The final result is in W and congruent to a+b modulo p for every a,b∈W. A second full-width correction is unnecessary, but omitting this second correction is incorrect.

A double-fold witness is a=b=U−1. Initial wrapped sum is U−2, first correction wraps to C−2, and the correct weak result is2C−2 (`0x2000007a0`). Dropping the second correction returns C−2, which is a different residue. The first correction must propagate across all256 bits: short low-limb-only fixes fail on carry chains even when the second fold is absent.

## Subtraction proof over the entire input domain

Write d=a−b=s−kU, where0≤s<U and k is the initial borrow. If k=1, a−b>−U implies1≤s<U. The first correction is t=s−kC. Compute it across all256 bits, and write t=r−jU,0≤r<U, where j is its borrow. The desired residue is r−jC.

If j=0, r is already valid. If j=1, then k=1 and1≤s<C. Hence r=U+s−C lies in[U−C+1,U−1]. Its low64 bits lie in[2^64−C+1,2^64−1], and 2^64−C+1>C. Subtracting the second C from the low64 bits therefore cannot borrow into word2. The final value lies in[U−2C+1,U−C−1]⊂W and has residue d. This proves both the full-range contract and the short second-correction bound.

A double-fold witness is a=0,b=U−1. Initial wrapped difference is1, first subtraction of C borrows and wraps, and the final result is U−2C+1. Omitting the second subtraction instead returns U−C+1, again a different residue. All first-fold borrow propagation is full-width.

## Canonicalization and zero

Because U<2p, at most one subtraction of p is needed. The implementation forms a+C modulo U and captures overflow. Overflow occurs exactly when a≥U−C=p; select that wrapped sum in that case, otherwise preserve a. This produces a mod p in[0,p). The CUDA selection uses a carry-derived32-bit mask with XOR/AND; it does not rely on host comparison semantics. The host body independently selects the fully propagated sum.

Within W, the only multiples of p are0 and p. Therefore `is_zero` tests exactly those two values, including the all-zero canonical representation and noncanonical p. Raw wordwise zero tests would miss p. Integration must canonicalize before any existing routine whose contracts require canonical coordinates; this component does not itself prove a surrounding point-chain integration.

The current canonical `_ModAdd256`/`_ModSub256` assumptions must not be extended silently to W. In particular, a single p correction can leave an additional U carry or borrow when operands approach U. The witness pairs above demonstrate why merely deleting reduction tails is not a valid weak implementation. This header provides new named functions rather than changing inherited helper contracts.

## PTX semantics and tests

`check.py` extracts the actual asm strings from the frozen header. `ptx_model.py` reuses the existing `research/ptx_field_model.py` parser/interpreter through a controlled local extension. Only32-bit subtract/borrow and bitwise AND/XOR are added; existing carry, packing and multiplication behavior remains. Unsupported instructions and uninitialized registers still fail closed. The extension does not modify the shared model file.

The model follows NVIDIA's [PTX extended-precision arithmetic specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-subc): `subc` subtracts the incoming borrow, and only `.cc` updates its outgoing flag. Plain capture/mask instructions preserve the previous condition code. The local tests add288 borrow probes covering both carry-in states, boundary words and all flagged/unflagged sub/subc forms; the original model's semantic probes also run. The actual PTX uses32-bit instructions supported by the legacy target family; native compilation remains a separate parent check.

Independent Python arbitrary-precision arithmetic supplies expected residues and exact two-fold representatives. Actual C++ host bodies and extracted PTX are compared on adversarial limb boundaries, near-p/noncanonical inputs, both-fold witnesses and deterministic random full-width inputs. Alias tests snapshot both input ranges before invocation and verify all non-output memory canaries. Normalization must satisfy exact canonical equality, not merely residue equality; zero is compared with Python's modulo-p predicate.

Negative controls change only test artifacts or extracted PTX. They remove the second add/sub correction, corrupt full first-fold carry/borrow propagation, suppress normalization carry selection, or omit noncanonical p from zero recognition. Host mutations are separately compiled and executed; a compile failure would not count as an arithmetic failure. The initial volatile/clobber test receipt is preserved separately and is superseded by the final frozen-header receipt.

Reproduce from the benchmark root:

```sh
python3 -B candidates/subset/research/weak_field/primitive/check.py
```

The main receipt is `test-artifacts/results.json`; it binds the header, checker, original semantic model, local extension and extracted PTX strings. Generated host binaries and mutation headers remain under this prototype directory. No native CUDA assembly, GPU arithmetic, compiler register allocation, point-chain equivalence, device synchronization or throughput is claimed by this checker. Any proposed point-loop instruction saving is a hypothesis until source-matched integration and native/device evidence exist.
