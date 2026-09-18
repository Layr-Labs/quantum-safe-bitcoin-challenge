# Experimental XYZZ fusions over PR #219

This patch starts from integration commit `a12af61a0715ef334e98d19ceb3a5d1298d04736`, which combines main `df2fb8b8f5f3e47ff7a6a1848ccebf40e902f634` with PR #219 head `f7f588c7a9baacc261f5d6214c13e54130583f31`. PR #219's Pinning tree is the control; the last Pinning promotion on main is `f0f4256153dc91b6ccdd3c130f29fe09ab0d7cf2`. The merged main change since that promotion concerns Subset. This work builds on 0xCramJam's base candidate, PR #219, and prior contributions by i34-9, scarletbright, Meganpark980320 (`26c18f8`), odinfree (`e00f556`), ercumentyildirim (`d2f523d`, `21d45a5`), dun999 (`f535811`), Saviour1001 (`0227bc3`), welttowelt (`36d4266`, `b756a1c`), and xlib (`0c6f4c83`); it does not claim their work as new.

`GPUMath.h` adds two independent compile-time switches:

| Switch | Off | On (submission default) |
| --- | --- | --- |
| `QSB_FUSE_MULSUB` | Multiply then subtract | Fold `a*b-c` together |
| `QSB_FUSE_SQRADDSUB2` | Square then `_ModX3Fused` | Fold `r*r+e-2*q` together |

Both reducers keep the original raw 8×32-bit product/square schedules. After the first fold of `L + 2^256 H` to `L + (2^32+977)H`, they inject `2p-c` or `3p+e-2q`, respectively. The positive bias avoids signed underflow across the 320-bit intermediate for every input in `[0,2^256)`. The remaining high word and final carry are folded exactly, yielding a congruent 256-bit representative. The generic multiply, square, and `_ModX3Fused` remain unchanged.

The combined `11` configuration is enabled by default for an experimental challenge submission. `00` preserves PR #219 behavior. A CUDA comparison can build the four variants with `-DQSB_FUSE_MULSUB=0/1` and `-DQSB_FUSE_SQRADDSUB2=0/1` on the existing `nvcc -O3 -DQSB_ZEROS_N=<N> ... -lcrypto -lm` command. No throughput gain over PR #219 has been demonstrated; the remote evaluation will supply the first performance measurement for this combined candidate.

Run the focused host correctness and switch-matrix audit from the repository root:

```sh
python3 candidates/pinning/audit_fused_xyzz.py
python3 candidates/pinning/audit_fused_point_source.py
git diff --check
```

The first audit executes the actual fused host C-reference branches and compiles both `_PointAddXYZZ` template instantiations for all four switch settings. It checks full-width, noncanonical, boundary, carry-chain, and alias cases against exact modular arithmetic. Its 15-point signed-digit mathematical model uses the compiled fused reducers and exact Python arithmetic for the other field steps, covering 32 real candidate preimages and both compressed keys.

The second audit compiles and executes the actual PR #219 and modified `_PointAddXYZZ` source with the existing host multiply/square branches and limb carry/borrow adapters for the add/subtract helpers. By default its control comes from the immutable PR #219 Git object; `--control /path/to/pr219/worktree` selects a separate checkout. It checks 128 real 15-point signed-digit candidate chains, every normalized intermediate, both compressed recovered keys, and identical candidate-prefix hits across `00/10/01/11`. It also checks a valid-curve zero-slope path where the fused reducer can return the noncanonical representative `p`. Equal/opposite additions retain inherited singular zero projective scales. Affine reference code computes the recovery and SHA-256 checks; this audit does not execute `LeafRecovery.cuh` or the CUDA kernel.

Five older Pinning audit scripts have stale source-text assertions and fail on both unmodified promoted main and unmodified PR #219 controls. Those inherited failures are separate from these fusion audits. The generic PR #219 field helpers also retain rare final-carry defects; this patch does not change them.

There is no NVIDIA GPU or CUDA toolchain on the development host. The host audits and independent source PTX integer emulation provide correctness evidence, but no complete CUDA prepare-kernel compilation, assembled instruction, register, spill, or throughput measurement. The `11` submission configuration is experimental; performance versus PR #219 remains unknown until remote evaluation.
