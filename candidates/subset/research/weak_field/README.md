# Weak uint256 field contract: bounded feasibility

**Proceed with the narrow thirteen-add prototype, not global normalization deletion.** The existing eight-word representation can carry residues in `[0,2^256)` without a magnitude word, wider radix, additional raw products or persistent field storage. Correct multiply/square overflow folds remain mandatory. Existing add/sub/neg and representation-sensitive boundaries cannot simply consume the widened domain unchanged.

This review binds frozen cd3d `cd3d11f24e6fd42459eed3ba3a431c7f952de506cb1d22dfddc9b291e4c98667`. No candidate, prepared stage, evaluation, VM or native build was changed. `contract-results.json` records the complete source identity, checker/model identities, actual PTX operation counts and counterexamples. The separate parent normalization-elision build is an intentionally invalid cost screen, not correctness evidence for this proposal.

## Representation and multiplication

Write `B=2^256`, `C=2^32+977`, `p=B-C`. A weak value is an ordinary unsigned integer `0<=a<B` representing `a mod p`. Canonicalization is a single conditional subtraction of p. Only residues `0..C-1` have two encodings; zero is encoded by either 0 or p. Storage remains four uint64 limbs/eight physical 32-bit words. This is not carry-free limb arithmetic: each helper must return an exact uint256 representative.

The current `_ModMultCore` and `qsb_square32` already compute a weak representative inside their PTX and canonicalize afterward in C++. For product `T=a*b<B²`, define:

```
u = (T mod B) + C*floor(T/B)
v = (u mod B) + C*floor(u/B)
w = (v mod B) + C*floor(v/B)
```

We have `floor(u/B)<=C` and `v<=B-1+C²`. If `v>=B`, its wrapped low part is less than `C²`; hence the last correction gives `w<C²+C<2^96`. Otherwise `w=v<B`. This proves the existing final captured carry can be folded through only the first three 32-bit limbs. **It does not justify dropping that carry or its `.cc` producer.** The resulting `w` is in `[0,B)` and congruent to the product for all uint256 inputs. Removing only the external `>=p` tail retains this contract. The supplied checker executes both unchanged raw PTX bodies and confirms exact equality to this integer schedule.

The tails in cd3d are `GPUMath.h:822` and `square32.cuh:233`. The old comment in square32 claiming extremely rare inputs justify deletion is not a correctness argument: canonical inputs `(p-1)²` directly produce weak `p+1`, whose canonical value is one. Host implementations of a future weak square must call a weak core too; the existing host square calls canonical `_ModMultCore`, so deleting only the square's final C++ condition would leave different host/device representatives.

## Low-state add/sub recipes

The following are contracts/schedules, not implemented candidate code. They require no persistent fifth limb. Carry/borrow bits and one masked 33-bit constant are short-lived primitive temporaries. All in-place/overlap claims still require actual implementation tests.

**Weak addition:** form `u=(a+b) mod B` and capture `c=floor((a+b)/B)`. Compute `v=(u+c*C) mod B` with a complete carry chain, capturing its carry `d`. Return `v+d*C`. If `d=1`, then `v<=C-2`; the last correction is less than `2C` and needs only the low64 limb, with no further carry. If `d=0`, it changes nothing. Thus two full 256-bit addition chains plus one bounded low64 correction suffice. Canonical subtraction and its selection are absent.

**Weak subtraction:** form `u=(a-b) mod B` and capture borrow `c`. Compute `v=(u-c*C) mod B` with a complete borrow chain, capturing borrow `d`. Return `v-d*C`. The sign is **minus** because wrapping a negative difference added B, equivalent to C modulo p. If `d=1`, the wrapped value lies in `[B-C,B)` and its low64 word is at least `2^64-C`, so subtracting C cannot borrow from the next limb. Two full subtraction chains plus one bounded low64 correction suffice. There is no p-sized temporary mask array or canonical output requirement.

**Weak negation, for a wider future scope:** form `v=(p-a) mod B`; if that subtraction borrows, subtract C from the low64 word. For borrowed cases `a>p`, the low word is near `2^64`, so this correction does not borrow. This may represent negative zero as p. It is unnecessary in the first loop-only prototype: table Y inputs remain canonical and the original table negation remains unchanged.

The repaired add/sub schedules may cost more or less than current canonical helpers after lowering. Source chain counts and the lack of persistent new state are bounds, not SASS counts, allocation guarantees or cycle estimates. Retain all full-range carry witnesses even if random curve tests never exercise the narrow noncanonical band.

## Concrete failures of unqualified deletion

| Case | Existing helper result | Correct canonical result |
|---|---|---|
| add `B-1,B-1` | `C-2` | `2C-2` |
| subtract `0,B-1` | `p+1`, residue1 | `p-C+1` |
| negate `p+1` | `B-1`, residue`C-1` | `p-1` |
| raw nonzero test on p | nonzero | zero |
| raw parity on p+1 | even | odd (value1) |
| serialize x=p+1 | wrong 32-byte key coordinate | x=1 |

These are not merely inputs impossible for the new multiply. `weak_square(p-1)=p+1`. Also `weak_square(p-65536)=p+2^32`; adding two such outputs with the existing `_ModAdd256` gives `2^32-977` rather than `2^33`. These are actual primitive compositions starting from canonical field inputs, not claimed curve-coordinate witnesses.

Existing `_ModAdd256` subtracts p at most once and truncates: it is residue-correct on weak inputs only when `a+b<B+p`. Existing `_ModSub256` adds p on a borrow: it fails when `b-a>p`. One canonical input does not generally make subtraction safe. No probability argument or strict valid-prefix theorem repairs these arithmetic defects.

## Recommended first region and exact boundary

Keep canonical `_PointAddXYZZ_mm_def`, `qsb_asym_last_add`, table loads/negation, recovery, inverse tree, builder, scalar recoding and packed digits unchanged. Copy only `qsb_ec192_PointAddXYZZ_shared_z_def` into a new helper specialized for its actual `defer_y=true` call. In that helper:

- Replace seven multiplies and two squares with weak variants preserving the exact corrected raw reduction.
- Keep `S2=Y2+Yoff` at `compact_table_device.cuh:323` canonical: both inputs are unchanged affine table ordinates.
- Replace only `T=R²+PPP` at line345 with weak addition, and the five subtractions at lines330,331,346,347,357 with weak subtraction.
- Preserve all existing scoped volatile ZZ/ZZZ loads and immediate stores, the anchor lifetime and deferred-Y convention. Do not widen shared slots or point state.

The actual thirteen-iteration caller is at lines491–507, with its call at506. The canonical seed at486 initializes four weak-valid state elements. Inductively, each replacement performs the same polynomial map over the field and returns four weak-valid representatives; the affine anchor remains canonical. The intermediate helper contains no field-zero, parity, integer ordering or serialization decisions. Existing all-input prefix bounds exclude incomplete-add exceptions independently of the chosen representative, conditional on the same valid nonzero runtime base and correct table construction.

After the loop, the caller loads the final affine point and anchor, then reloads `ZZ` and `ZZZ` at line514. **Canonicalize X, deferred Y, ZZ and ZZZ between this load and the unchanged final guarded call at line515.** Each is normalized once, sequentially in place. Deferred Y must be normalized even though it is not the affine y-coordinate: the original final helper subtracts it directly. Canonicalizing only X/Z is insufficient. No writeback to the old shared slots is necessary: the final helper consumes the local arrays and the loop will not resume. Existing later phase joins still protect shared-arena reuse.

This boundary reinstates the original final P/R zero tests (`compact_table_device.cuh:79–80`) and their equal/opposite behavior. The final helper emits the original canonical ordinary/doubled result or explicit zero sentinel. Recovery therefore retains its existing contracts: prepare at `tree.cu:730`, usable-factor check at1176, canonical nonzero inputs/identity padding to `qsb_ec192_inverse_tree_scratch`, and key parity at770/775 followed by canonical x serialization. Scalars0, n and the two constructed final-add witnesses remain covered by the same guard. No weak-zero API, inverse modification or changed parity rule is needed for this first scope.

Exact ordinary-path budget: **117 M/S canonical tails removed; four boundary normalizations added; thirteen weak additions and sixty-five weak subtractions introduced.** Thirteen canonical affine-Y additions remain. No M/S operation, raw product, table load, shared allocation, inverse, barrier or SHA round is removed. The parent elision build is a comparison/cost screen, not a performance bound: the corrected add/sub schedule also changes work and compiler lowering, so the invalid screen does not bound the corrected prototype's instruction count or runtime.

For a later whole-chain region, at minimum final P/R tests must recognize 0 or p, W must be canonical/nonzero before the existing inverse contract, and recovered x/parity must be canonical. Those extra boundary changes are deliberately excluded now.

## Evidence and implementation gate

`python3 -B candidates/subset/research/weak_field/check_contract.py` passes 4,809 full-width add/sub/neg cases and 4,809 actual PTX executions each for multiply and square. It exercises second addition carries, second subtraction borrows and final multiply carries, exhausts262,144 smaller-radix add/sub pairs, and rejects both the stale-carry-flag and omitted-overflow-fold mutations. Witnesses are persisted in hexadecimal. This combines universal integer bounds with finite actual-PTX semantic checks; it is not a GPU run, compiled weak-add/sub test or actual weak-curve execution.

The primitive implementation should test output aliasing with either operand; zero, p, p±1, B−1; carry/borrow across all limb boundaries; the two actual-weak-square composition witnesses; and both second-fold corrections. Source-extracted loop/curve checks must execute weak primitives rather than substitute canonical OpenSSL helpers that erase the representation difference. Add an explicit boundary mutation that omits deferred-Y normalization and injects weak representatives at the final helper boundary. Re-run final equal/opposite witnesses, complete recovery and actual field/PTX carry tests. Then inspect the corrected native repeated loop, including local accesses and producer resources. Preserve canonical builder/inverse APIs and default-target compatibility.

The earlier lazy-field README is under `candidates/pinning/research/lazy_field/README.md`, not the subset research directory; it was read without changing sibling work. Its source-magnitude review covers5x52/10x26 (ten physical words),9x29 range issues and extra raw products; none is reused here. The seven product-difference screens in `product_difference/README.md` and `spill-review.md` fuse two raw products/reductions with extra temporary pressure; this proposal preserves one raw product at a time and defers only final canonical selection. Those historical costs neither prove nor disprove this distinct eight-word contract.

## Executable ordinary-helper check

`python3 -B candidates/subset/research/weak_field/ordinary_region/check_helper.py` passes on corrected prototype `3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912`. It extracts the actual ordinary weak helper and includes the unchanged weak product/add/sub headers in host C++. Across1,591 synthetic states and7,682 calls, an independent Python field polynomial agrees for both deferred and completed Y, all192 shared-arena owners, and six affine-input alias configurations. Guard values and every non-owned arena element remain unchanged. There are694 calls with noncanonical raw output, all correctly normalized by the actual normalization helper. Input classes include p as zero, near-2^256 values, equivalent p+x lifts, and canonical affine inputs.

Three successfully compiled mutations fail with explicit numerical diagnostics: dropping the affine Y anchor, omitting the shared ZZ update, and omitting returned Y normalization. The last fails on the concrete state `(X,Y,ZZ,ZZZ)=(0,1,0,0)` and zero affine inputs: raw returned Y is p+1, while the canonical result must be1. This is a synthetic polynomial witness, not a valid-curve claim. `ordinary_region/helper-results.json` binds all source and checker hashes. The check does not execute the final guarded helper, complete recovery, GPU arithmetic, or synchronization; full-chain qualification remains separate.
