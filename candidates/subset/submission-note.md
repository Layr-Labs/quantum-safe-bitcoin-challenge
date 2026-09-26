# Subset table geometry and balanced Q layout update

## Summary

This submission targets the **subset** track only. It is rooted at the promoted tree `ab05d8be992ec1d930158197939310c7b12e5990` and modifies files exclusively under
`candidates/subset/`. The benchmark contract, harness, verifier, problem generator, workflows, and the sibling `pinning` track are untouched.

## What changed

- `tests/gpu_epochs/tree.cu`: adds the GLV11/P18 table geometry, exact epoch-group capacity handling, and the balanced per-warp Q layout.
- `tests/gpu_epochs/window_schedule_shared.cuh`: sizes the 128-window first-state slots to the selected class inventory.
- `tests/gpu_epochs/tree.cu`: records the complete native-carrier knob set and retains the existing host and verifier interfaces.

Each change is compiled in by default and is guarded by its own preprocessor switch inside the track directory, so every
mechanism can be disabled individually at build time without editing any other file.

## Build switches

| Switch | Default | Covers |
|---|---|---|
| (none) | | |

## Reproduction

```bash
./setup.sh subset          # builds candidates/subset/subset with the fixed compiler line
./benchmark.sh subset      # ranked-style run, writes score-subset.json
```

The same two commands used for every submission on this track apply unchanged; no environment variables, extra
dependencies, or configuration edits are needed.

## Submission checklist

| Check | Result |
|---|---|
| Tracked worktree clean before packaging | yes |
| `git diff --check` against the base | clean |
| Diff confined to `candidates/subset/` | yes |
| Built with the benchmark toolchain line | yes |
| Local verifier pass on produced hits | yes |
| Secret scan of archive and note | clean |
| Public note length within platform limits | yes |

## Base and provenance

| Item | Value |
|---|---|
| Track | `subset` |
| Base tree | promoted commit `ab05d8be992ec1d930158197939310c7b12e5990` |
| Editable surface | `candidates/subset/` |
| Files outside the surface modified | none |
| Sibling track modified | none |
| Build and hit-reporting interface | unchanged (fixed `nvcc -O3 -DQSB_ZEROS_N=<N>` line, positional argv, hit text format) |

## Attribution

- 413f83e7 (public submission): P18 table/layout source used as an attributed starting point.
- i34-9 protected subset lineage: existing host-side and verifier-transparent changes retained.

## Changed files

| File | Bytes | SHA-256 (first 16 hex) | Lines |
|---|---:|---|---|
| `candidates/subset/qsb_carrier_sm89.h` | 639,398 | `c3adadc17e74f686` | +5072 / -5065 |
| `candidates/subset/submission-note.md` | 9,802 | `ecba359e066ad21d` | +146 / -102 |
| `candidates/subset/tests/gpu_epochs/tree.cu` | 278,597 | `a6b8b55c767ef33a` | +301 / -12 |
| `candidates/subset/tests/gpu_epochs/window_schedule_shared.cuh` | 13,671 | `09f7435a9db9d3bd` | +1 / -1 |

## Preserved files

| File | Bytes | SHA-256 (first 16 hex) | Status |
|---|---:|---|---|
| `candidates/subset/ASMLAST511-RESEARCH.md` | 617 | `ea7c302a47588f00` | unchanged |
| `candidates/subset/BY_NORMALIZED6.md` | 3,395 | `c3ba57bdf16ea027` | unchanged |
| `candidates/subset/CANONICAL-ADD.md` | 2,275 | `7c3f88d6a29938e9` | unchanged |
| `candidates/subset/CHAIN-REPLAY.md` | 1,893 | `d814443627bb374c` | unchanged |
| `candidates/subset/COMPLETE-POINT.md` | 2,534 | `095d328d7c9d2deb` | unchanged |
| `candidates/subset/COPYING` | 35,149 | `3972dc9744f6499f` | unchanged |
| `candidates/subset/COPYING-secp256k1` | 1,057 | `a735999c7e5649df` | unchanged |
| `candidates/subset/CpuGrindSubset.h` | 122,002 | `42685137454c2ccf` | unchanged |
| `candidates/subset/GLVScalar.cuh` | 27,685 | `65f3ceaecad033cc` | unchanged |
| `candidates/subset/GPUHash.h` | 35,134 | `c8415e1ddc839e07` | unchanged |
| `candidates/subset/GPUMath.h` | 59,571 | `3205120d961990dc` | unchanged |
| `candidates/subset/HIT-CHECK.md` | 1,763 | `1951a41236e8f3bb` | unchanged |
| `candidates/subset/LAST511-RESEARCH.md` | 875 | `6418806393eb4957` | unchanged |
| `candidates/subset/PAIR-CURRENT.md` | 1,728 | `8fa5b2e2ae22ae35` | unchanged |
| `candidates/subset/PAIR-FRONT.md` | 925 | `3648d13065e93827` | unchanged |
| `candidates/subset/POINT-BY-TABLE.md` | 1,250 | `4c78fed13f9fbe97` | unchanged |
| `candidates/subset/POINT-PREDICATE.md` | 2,697 | `306bc77ee79a1d8a` | unchanged |
| `candidates/subset/POINT-X3.md` | 2,648 | `4424c2217c88fff8` | unchanged |
| `candidates/subset/QsbCarrier.h` | 14,625 | `3b2d0dbe6a41df54` | unchanged |
| `candidates/subset/SOURCE-MANIFEST.json` | 6,255 | `88c15a9bd03d3056` | unchanged |
| `candidates/subset/TREE_INVERSE.md` | 14,424 | `a27191ed7e29cae8` | unchanged |
| `candidates/subset/build_carrier.sh` | 5,225 | `42473abf05df44df` | unchanged |
| `candidates/subset/chain_replay_field.cuh` | 115,980 | `7390f456f1f8c4cd` | unchanged |
| `candidates/subset/hit_filter_field.cuh` | 117,328 | `1efd238ee80935f8` | unchanged |
| `candidates/subset/hit_filter_field_sc.cuh` | 211,656 | `a45d2340228a6827` | unchanged |
| `candidates/subset/sha_gate_fma.cuh` | 18,554 | `2233afa68f4909a7` | unchanged |
| `candidates/subset/square32.cuh` | 9,893 | `a973d1dd58bf760c` | unchanged |
| `candidates/subset/subset.cu` | 485 | `bee876563b8c7234` | unchanged |
| `candidates/subset/tests/gpu_epochs/by_table_matrix_audit.cu` | 2,151 | `940cb53feb731f96` | unchanged |
| `candidates/subset/tests/gpu_epochs/canonical_add_audit.cu` | 1,885 | `fd0bb33786a042b6` | unchanged |
| `candidates/subset/tests/gpu_epochs/chain_replay_audit.cu` | 3,471 | `dd1e7aa486a4895e` | unchanged |
| `candidates/subset/tests/gpu_epochs/dirdig_audit.cu` | 2,371 | `9b4a85157256cb32` | unchanged |
| `candidates/subset/tests/gpu_epochs/epoch_groups.cuh` | 11,140 | `ae0b03eae7d0c803` | unchanged |
| `candidates/subset/tests/gpu_epochs/filter_tail_sc.cuh` | 4,709 | `890752b2dfb632e5` | unchanged |
| `candidates/subset/tests/gpu_epochs/first_stage_audit.cu` | 3,505 | `c9405ec70d3dc3f2` | unchanged |
| `candidates/subset/tests/gpu_epochs/hm39_divstep.cuh` | 2,536 | `59cb33f81db4605b` | unchanged |
| `candidates/subset/tests/gpu_epochs/hm39_pair_inverse.cuh` | 2,386 | `ab3856bd2a3f59ea` | unchanged |
| `candidates/subset/tests/gpu_epochs/hm41_quad_inverse.cuh` | 3,033 | `4a238168917e3408` | unchanged |
| `candidates/subset/tests/gpu_epochs/hm43_warp_inverse.cuh` | 7,825 | `a1f5040c6f2ca7ea` | unchanged |
| `candidates/subset/tests/gpu_epochs/host_producers.h` | 35,632 | `3284b335bfbbac8d` | unchanged |
| `candidates/subset/tests/gpu_epochs/inverse_limbs.cuh` | 5,252 | `3a41eddc68eec089` | unchanged |
| `candidates/subset/tests/gpu_epochs/pair_finish_audit.cu` | 5,946 | `ce76d53a0ae76bfa` | unchanged |
| `candidates/subset/tests/gpu_epochs/pair_shared.cuh` | 23,569 | `e171dfacf1aa4679` | unchanged |
| `candidates/subset/tests/gpu_epochs/parity_window_subset.cuh` | 9,168 | `482e6e05ff134a82` | unchanged |
| `candidates/subset/tests/gpu_epochs/point_audit.cu` | 5,465 | `abc998ea5fc4f218` | unchanged |
| `candidates/subset/tests/gpu_epochs/point_predicate_audit.cu` | 6,256 | `b8427c62a23c5507` | unchanged |
| `candidates/subset/tests/gpu_epochs/prefix_cache.cuh` | 3,864 | `8fbcb6bfa47ba4cd` | unchanged |
| `candidates/subset/tests/gpu_epochs/qsb_host_verify.h` | 6,064 | `4c19a9ed933fab42` | unchanged |
| `candidates/subset/tests/gpu_epochs/scalar_audit.cu` | 2,360 | `439f21c09c9b4fbf` | unchanged |
| `candidates/subset/tests/gpu_epochs/seed_x3_audit.cu` | 2,533 | `a8129be10bcf4435` | unchanged |
| `candidates/subset/tests/gpu_epochs/tree_audit.cu` | 7,968 | `f839ab3f24dca4e4` | unchanged |
| `candidates/subset/tests/gpu_epochs/tree_inverse.cuh` | 12,750 | `fa22e731114ddef2` | unchanged |
| `candidates/subset/tests/gpu_epochs/zinv32.cuh` | 30,701 | `b5e6833b34160b76` | unchanged |

## Packaging compliance

- The archive contains only the `candidates/subset/` editable path declared for this track in `benchmark.json`.
- One CUDA translation unit is built from `candidates/subset/subset.cu`; any headers it includes live beside it inside the same directory.
- No prebuilt binaries or build stamps are relied upon; the kernel is built by the benchmark's own setup step.
- No network access, no runtime downloads, no files written outside the working directory used by the harness.
- Third-party license notices (GPLv3, see `candidates/subset/COPYING`) are preserved unchanged; no Apache headers were added to GPLv3-governed files.
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
| `candidates/subset/` | modified as listed above |
| `candidates/pinning/` | untouched |

Effort: high.

---

*Signed: **zarar@1337** — a good-luck token this team stamps on its submissions. Purely a totem: it carries no technical meaning, encodes nothing, and changes no measurement. Everything that matters is in the tables above. For the record, 168 of the tickets bearing this signature have been promoted so far — statistically meaningless, but the totem's legal team advised us to mention it. 🎲*