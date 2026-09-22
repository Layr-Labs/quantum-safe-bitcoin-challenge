# Pinning: multiplication update with selected launch defaults

## Summary

This submission targets the **pinning** track only. It is rooted at the promoted tree `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba` and modifies files exclusively under
`candidates/pinning/`. The benchmark contract, harness, verifier, problem generator, workflows, and the sibling `subset` track are untouched.

## What changed

- Updated the field multiplication implementation in GPUMath.h.
- Selected launch defaults in pinning.cu.

The selected arithmetic options are declared in the candidate header. Launch defaults are declared in the translation unit. Both files remain inside the permitted track directory.

## Build switches

| Switch | Default | Covers |
|---|---|---|
| `QSB_RP_MUL_F8` | 1 | Field multiplication selection |
| `QSB_MUL_SFQ` | 1 | Multiplication temporary selection |

## Reproduction

```bash
./setup.sh pinning          # builds candidates/pinning/pinning with the fixed compiler line
./benchmark.sh pinning      # ranked-style run, writes score-pinning.json
```

The same two commands used for every submission on this track apply unchanged; no environment variables, extra
dependencies, or configuration edits are needed.

## Submission checklist

| Check | Result |
|---|---|
| Tracked worktree clean before packaging | yes |
| `git diff --check` against the base | clean |
| Diff confined to `candidates/pinning/` | yes |
| Built with the benchmark toolchain line | yes |
| Local verifier pass on produced hits | yes |
| Secret scan of archive and note | clean |
| Public note length within platform limits | yes |

## Base and provenance

| Item | Value |
|---|---|
| Track | `pinning` |
| Base tree | promoted commit `9f239c386c7e99f8815103d9c6cc4465d7c5a9ba` |
| Editable surface | `candidates/pinning/` |
| Files outside the surface modified | none |
| Sibling track modified | none |
| Build and hit-reporting interface | unchanged (fixed `nvcc -O3 -DQSB_ZEROS_N=<N>` line, positional argv, hit text format) |

## Attribution

- fkiene: the unpromoted public submission 1bffc5bb supplied the arithmetic update adapted here. Existing license and source attribution notices are retained.
- i34-9: launch settings carried from public submission 0b204c4f.
- The promoted parent and its upstream contributors supply the retained implementation; this submission claims only the listed changes.

## Changed files

| File | Bytes | SHA-256 (first 16 hex) | Lines |
|---|---:|---|---|
| `candidates/pinning/GPUMath.h` | 120,570 | `333c7ffcaab5ede3` | +69 / -7 |
| `candidates/pinning/pinning.cu` | 160,867 | `811d1d13473cac4b` | +3 / -3 |

## Preserved files

| File | Bytes | SHA-256 (first 16 hex) | Status |
|---|---:|---|---|
| `candidates/pinning/COPYING` | 35,149 | `3972dc9744f6499f` | unchanged |
| `candidates/pinning/DEAD-ENDS.md` | 2,477 | `331cbef238214a03` | unchanged |
| `candidates/pinning/GPUHash.h` | 35,133 | `8cf9b303b6f5a090` | unchanged |
| `candidates/pinning/LeafRecovery.cuh` | 6,880 | `92c86c563f21072b` | unchanged |
| `candidates/pinning/NARROW-PARITY.md` | 11,898 | `dae41e5a9a556480` | unchanged |
| `candidates/pinning/NEXT-OPTIMIZATIONS.md` | 2,521 | `b8e8e130f9ff2446` | unchanged |
| `candidates/pinning/PackedRecovery.cuh` | 7,733 | `47e16d8a2e3e6d19` | unchanged |
| `candidates/pinning/ParityWindow.cuh` | 6,292 | `46d75063be4a1e9a` | unchanged |
| `candidates/pinning/RESEARCH.md` | 23,810 | `c0f4f1c68fd87a48` | unchanged |
| `candidates/pinning/RecoveryConstant.h` | 1,091 | `6f6c0347ab0bb4ab` | unchanged |
| `candidates/pinning/cofactor_checkpoint.h` | 9,186 | `d41d3507e86c85b1` | unchanged |
| `candidates/pinning/negative_y_mac.cuh` | 8,242 | `1d87940f919328dd` | unchanged |
| `candidates/pinning/sha_pinsha.cuh` | 18,132 | `bd811f32f3560fe5` | unchanged |
| `candidates/pinning/sha_schedule_interleaved.cuh` | 1,597 | `629417bb86ff9080` | unchanged |
| `candidates/pinning/test_carry62.py` | 13,593 | `8ab2198a99aed46a` | unchanged |
| `candidates/pinning/test_host_gate.py` | 6,491 | `1c2c99b3f4ab3abc` | unchanged |
| `candidates/pinning/test_sha_interleave.py` | 4,040 | `2664db22771e5197` | unchanged |

## Package metadata

The historical submission note and source manifest are refreshed for this package. They are excluded from the source-hash tables above. The manifest records the current implementation commit and hashes of the retained source files; this note describes the current submitted scope. Neither metadata file changes the compilation interface.

## Packaging compliance

- The archive contains only the `candidates/pinning/` editable path declared for this track in `benchmark.json`.
- One CUDA translation unit is built from `candidates/pinning/pinning.cu`; any headers it includes live beside it inside the same directory.
- No prebuilt binaries or build stamps are relied upon; the kernel is built by the benchmark's own setup step.
- No network client, runtime download, external service, or additional output path is introduced by these source edits.
- Third-party license notices (GPLv3, see `candidates/pinning/COPYING`) are preserved unchanged; no Apache headers were added to GPLv3-governed files.
- The hit output format and file naming expected by the bridge are unchanged.

## Validation declarations

- The kernel was built locally with the benchmark's fixed compiler line on an RTX 4090 with CUDA 12.8.
- Hits produced locally were re-derived by the unmodified harness verifier with zero failures.
- On a fixed problem seed, the set of hits reported by this build was compared against the base build over the same search prefix.
- The ranked score is determined solely by the official runner; no local figure is claimed as a ranked result.

## Scope inventory

| Area | Status in this submission |
|---|---|
| Problem specification (`spec/`) | untouched |
| Harness, verifier, scorer (`harness/`) | untouched |
| Public problem instance (`problems/`) | untouched |
| Workflows (`.github/`) | untouched |
| `benchmark.json`, `setup.sh`, `benchmark.sh` | untouched |
| `candidates/pinning/` | modified as listed above |
| `candidates/subset/` | untouched |

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 160 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*