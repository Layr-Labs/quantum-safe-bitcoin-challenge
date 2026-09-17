# Bounded root outlining screen

Prepared candidate `a36071b7a5b06c22d16172f8f29ec0d86eb1cd90620e3716526b40349a890047` changes exactly one declaration qualifier relative to boundedf47:

```cpp
__device__ __noinline__ bool hm43_warp_inverse(uint64_t result[5], int lane)
```

The former declaration used `__forceinline__`. All function-body bytes, full-mask collectives, batch cap, boolean publication contract, caller fallback, subgroup joins, point arithmetic and other18 source files are identical. The19-file closure, original COPYING and predecessor identities are bound in `prepared-source.json`; `prepare.py` refuses to overwrite an existing candidate. Ready3ac and boundedf47 remain preserved.

The motivation is concrete: the bounded inlined root changes whole-kernel allocation and the parent's native review observes repeated local traffic in the ordinary point-add loop. Separating the root's compilation region might permit a different allocation that avoids that repeated traffic. This is an allocation experiment supporting the same substantial cooperative-inverse mechanism, not an arithmetic or synchronization change.

## CUDA semantics and limits

[NVIDIA's CUDA12.8 programming guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#noinline-and-forceinline) documents `__noinline__` as a hint against inlining and forbids combining it with `__forceinline__`. The new declaration has neither C++ `inline` nor the conflicting qualifier. It remains a normal device-to-device direct call in the same translation unit; no function pointer, indirect target, recursion, device launch or separate-compilation change is introduced. A native call must still be verified rather than inferred from the annotation.

Outlining does **not** give the root warp its own runtime register allocation or automatically restore the old kernel allocation. NVIDIA's [compiler-toolchain explanation](https://developer.nvidia.com/blog/programming-efficiently-with-the-cuda-11-3-compiler-toolchain/) shows how device callee register requirements propagate to kernels in separate compilation. That example is not a numeric prediction for this same-translation-unit build, but it rules out treating a function boundary as resource independence.

The current call site is warp-uniform for physical threads64..95 and passes relative lanes0..31. Keeping that site and all helper control flow unchanged preserves the required participation structurally. CUDA shuffles/ballots remain inside the called body; all32 lanes must enter it. The existing whole-subgroup join still waits until ecid0 publishes the result or runs the scalar fallback. No new barrier or memory-order assumption is introduced.

## Costs to inspect in the native screen

The root parameter is a pointer to five uint64 words. Without inlining, the compiler may materialize that40-byte caller array in local memory and pass its address, introducing loads/stores on the selected root warp. It may also save values live across the call, use return/parameter stack space, and lose interprocedural scheduling opportunities. Conversely, shorter allocation regions could improve the earlier point-add loop. These are source-level possibilities, not measured costs or a guarantee that all40 bytes become traffic.

Compare against both boundedf47 and ready3ac:

- Confirm an out-of-line root symbol and actual call/return in sm89 and the official default build.
- Count the thirteen-add loop's executed LDL/STL operands, separately from stack allocation and compiler aggregate spill counters.
- Locate root-array materialization, call-boundary saves/restores, fallback traffic and the callee's own local operations. One root warp pays these costs per192-lane batch; point-add spills may repeat thirteen times per candidate.
- Check total registers/shared memory, launch feasibility, producer code and polling code. Separate function reports do not imply separate resident resources.

The prior bounded helper/tree arithmetic evidence transfers only through the exact unchanged body and caller. No fresh outlined host execution, CUDA build, GPU execution or throughput claim is made by preparation. A host rerun must map the CUDA `__noinline__` attribute in its projection; it must not edit the candidate to accommodate the host compiler. Parent-owned native/audit checks and any later package/selection decision remain separate. Retain AbdelStark attribution and existing GPL notices if this mechanism is used.
