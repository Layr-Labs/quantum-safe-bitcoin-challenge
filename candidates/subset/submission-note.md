# Subset: offset-ordinate fixed-base table for a pure-XOR signed load

## Summary

This submission ports the already-promoted offset-ordinate idea from the pinning track into the subset track's speculative fixed-base filter, while deliberately leaving the independent exact replay on the original table representation. The base checkout is `7c3609b87b9d8e094a16be148fe846dfd5ac7807`, whose published subset frontier is 623,518,629 verified candidates/s on the benchmark RTX 4090.

The hot filter previously loaded a signed fixed-base ordinate by XORing all four limbs with a sign mask and, for the negative case, adding a secp256k1 correction to limb zero. The new table copy stores every ordinate as `y + c`, where `c = (K-1)/2` and `K = 2^32 + 977`. In that representation the negative ordinate is exactly the bitwise complement of the stored value, so the hot loader becomes four XORs and no correction add. One short conversion is paid at the final deferred-Y resolution, not at every table load.

This is intentionally a narrow experiment. It changes two implementation files plus this note, has a compile-time kill switch, keeps the raw table for exact verification, and does not touch the harness, verifier, problem generator, scoring code, workflows, or the pinning candidate.

Development context: Codex with GPT-5. The runtime did not expose a separate effort-level label, so none is invented here.

## Starting point and hypothesis

The promoted subset candidate already uses a speculative pair filter followed by `kernel_verify_pair_hits`. The speculative path may conservatively lose an extremely rare hit, but no tentative hit is published until the separate exact kernel replays it with the full fixed-base arithmetic. That architecture makes a filter-only coordinate representation possible without weakening accepted-output correctness.

For secp256k1,

```text
p = 2^256 - K
K = 2^32 + 977 = 0x1000003D1
c = (K - 1) / 2 = 0x800001E8
```

Store `y' = y + c`. Because every table ordinate satisfies `0 <= y < p` and `c < K`, the table post-pass cannot overflow 256 bits. A positive signed load is simply `y'`. A negative signed load is

```text
p - y + c
= 2^256 - K - y + c
= 2^256 - 1 - (y + c)
= ~y'.
```

The old filter-only short loader performed the complement and then a low-limb addition. The new form uses the same sign mask but only XORs. The hypothesis is that removing this dependent integer add from every signed table load is worth more than the two-limb subtraction paid once at the final anchor, without increasing the digest kernel's register footprint enough to cancel the gain.

The expected improvement is deliberately not claimed as measured here. This environment has no CUDA compiler or GPU, and its local `yukon run` cannot invoke the privileged benchmark bridge. The official Yukon validation is therefore the first source-bound RTX 4090 compile and throughput measurement for this exact patch.

## Implementation

### 1. Filter-only table representation

`candidates/subset/tests/gpu_epochs/tree.cu` defines `QSB_YOFF_FILTER`, defaulting to `1`. It is a kill switch: setting it to zero restores the prior loader and aliases the filter table pointer to the raw table.

After the existing table build, fallback handling, and host spot check complete, the host allocates a second 64 MiB device table named `d_gt_filter`. It copies the already-validated raw `d_gt` table, then launches `qsb_table_offset_y_filter`. That kernel adds `0x800001E8` to the Y coordinate of every table entry with a full four-limb carry chain.

The extra table is deliberate. Mutating `d_gt` in place would make the independent exact replay consume shifted coordinates and would defeat the safety boundary. The three ranked `kernel_digest` launch sites now receive `d_gt_filter`; `kernel_verify_pair_hits` still receives raw `d_gt`. Allocation, copy, launch, and synchronization errors fail closed with a diagnostic rather than silently reverting to a mixed representation.

The one-time 64 MiB copy is negligible relative to the benchmark GPU's memory capacity. It is also outside the repeated candidate hot loop. The raw table is cold after initialization except when the rare exact verifier replays tentative hits.

### 2. Pure-XOR signed loader

Under `QSB_YOFF_FILTER`, `gt_load_signed_flat_f` retains the same vectorized `__ldg` loads for X and Y and computes the all-zero/all-one sign mask as before. The four Y limbs are now only

```text
gy[i] = loaded_y[i] ^ sign_mask
```

The X path is unchanged. The previous `QSB_NEG_SHORT` implementation remains directly below the new branch, so the compile-time kill switch preserves the old candidate without a source revert.

### 3. Offset-aware anchor sum

The deferred-Y point-add body in `candidates/subset/hit_filter_field_sc.cuh` adds two table ordinates at its anchor. Both operands now contain `+c`, so this one sum must remove `2c = K-1`.

The new PTX sequence is the same carry-conditioned construction used by the promoted pinning donor. Let `t` be the low 256 bits of the sum and `k` its carry bit. If `k=0`, it subtracts `K-1`; if `k=1`, it adds one because the discarded `2^256` is congruent to `K` modulo `p`. The correction retains its carry/borrow through limb one. All other point-add differences are unchanged because `(y2+c) - (y1+c) = y2-y1`.

The conditional is local to the first anchor sum. The rest of the packed PTX body, its operand constraints, and its output contract are unchanged.

### 4. Final anchor decode

The last deferred-Y resolution needs the ordinary affine ordinate before multiplying it by `ZZZ`. `qsb_yoff_filter_to_y` subtracts `c` from the final table Y value and propagates the borrow through limb one. `qsb_filter_last_add` applies this conversion to a temporary and feeds that temporary to the existing filter multiply.

Keeping the short subtract mirrors the promoted donor's filter-oriented arithmetic. A dropped borrow would require the low limb to be below `c` and the next limb to be zero, a probability bounded around `2^-97` for a field-like ordinate. The offset-aware anchor correction has a similarly tiny short-carry exceptional set. These cases can only remove a speculative candidate. They cannot publish a wrong hit because `kernel_verify_pair_hits` independently replays every tentative record against the unshifted table before host output.

### 5. Configuration guard

The offset representation is valid for the ranked speculative pair path used by the base. A compile-time guard rejects `QSB_YOFF_FILTER=1` unless both `QSB_PAIR_SHARED` and `ZLAB_TRIM` are enabled. This prevents an accidental build from feeding shifted ordinates into a path whose representation contract was not audited.

## Correctness checks

The repository's standard setup completed with its verifier smoke test:

```text
yukon setup --track subset
setup.sh: verifier smoke test passed
setup.sh: ready — run ./benchmark.sh subset
```

There is no `nvcc`, `nvidia-smi`, or benchmark bridge available in the local container. The attempted local `yukon run --track subset` reached the expected bridge command but could not initialize the privileged runner. No local score is therefore attached or implied.

I used a deterministic Python integer model for the representation identities and the two short corrections. The test seed was `0x594F4646`. It covered:

| Check | Cases | Result |
|---|---:|---|
| Positive/negative signed-load identity | 1,000,000 | pass |
| Offset anchor-sum congruence | 500,000 | pass |
| Final two-limb decode outside its documented exceptional set | 500,000 | pass |

The signed-load check sampled `y` uniformly below `p`, both sign bits, and compared the pure-XOR result with `(y or p-y)+c`. The anchor check modeled the four-limb carry and selected correction, then compared modulo `p` with the unshifted ordinate sum. The decode check modeled the exact two-limb PTX borrow contract and compared with the original `y`; its vanishingly rare documented exceptional predicate is handled as a conservative filter loss.

Static source checks also established the table boundary:

```text
kernel_digest(...)              -> d_gt_filter   (all three launch sites)
kernel_verify_pair_hits(...)    -> d_gt          (raw table)
```

`git diff --check` passes. The official linked checkout contains only these intended modifications:

```text
candidates/subset/hit_filter_field_sc.cuh
candidates/subset/tests/gpu_epochs/tree.cu
candidates/subset/submission-note.md
```

The CPU verifier smoke test exercises the unchanged public output contract. The exact replay kernel and host publication path were not modified.

## Safety and failure modes

The main risk is performance, not validity. A 64-bit add removed from each signed load may be hidden by memory latency, while the final decode and any register-allocation change may offset the instruction saving. Only the source-bound RTX 4090 result can settle that tradeoff.

The representation transition is explicit and fail-closed:

1. Build and validate the original table.
2. Preserve it for exact replay.
3. Allocate and copy a dedicated speculative table.
4. Offset every copied Y coordinate with a full carry chain.
5. Synchronize and reject any CUDA error.
6. Pass only the copied table to the speculative digest kernels.
7. Pass only the raw table to exact replay.

If the extra allocation fails, the program exits rather than launching with the wrong pointer. If the table post-pass fails, the program exits. If the speculative short arithmetic hits its exceptional set, it can miss a tentative hit but cannot introduce a verified false positive. If a future configuration disables the audited pair path while leaving the feature enabled, compilation fails at the guard.

The kill switch is a single definition:

```cpp
#define QSB_YOFF_FILTER 0
```

With that value the old `QSB_NEG_SHORT` loader is compiled, no second allocation is made, and `d_gt_filter` aliases `d_gt`.

## Benchmark interpretation

The current record is sufficiently optimized that noise and occupancy cliffs matter. I would treat the result as follows:

- A compile or verifier failure means the port is invalid and should not be promoted.
- A score below the current frontier means the removed add did not compensate for the new decode, allocation layout, or compiler scheduling.
- A small positive result should be reproduced because fixed-time GPU measurements near the frontier can move with thermal state.
- A clear gain with verified hits supports keeping the dual-table representation and then inspecting SASS/register counts for a second iteration.

No score is predicted in this note, and no local number is presented as comparable to the official runner. The submission exists to obtain the authoritative compile, verifier result, and fixed-time throughput on Yukon's configured RTX 4090.

## Reproduction

From a Yukon-linked checkout of the recorded base, apply the three-file change, then run:

```sh
git diff --check
yukon setup --track subset
yukon run --track subset
```

On an actual ranked host, `setup.sh` should compile with CUDA 12.8 and the benchmark should run the standard 1,200-second fixed-time workload. The score must be read from the normal `score-subset.json`/Yukon validation result; a CPU fallback is not comparable.

## Follow-up

If this candidate is slower, the useful negative result is that subset's current short-negation add is already cheap enough that the representation conversion does not pay. The next step would be a source-identical A/B build with the kill switch and SASS/register census, not stacking another speculative change.

If it is faster, the next focused checks are: reproduce the gain, compare register count and occupancy with the kill switch, and determine whether the raw exact table can be made even colder without weakening replay. Any follow-up should retain the separate exact verifier and should not port pinning's more aggressive approximate field shortcuts into the subset publication path without an equally explicit gate.
