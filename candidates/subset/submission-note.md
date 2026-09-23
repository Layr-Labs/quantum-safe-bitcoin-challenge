# Subset: paired H0 gate and B-first recovery on the PR #1137 source

Effort: xhigh. Prepared with GPT 6 Sol in Codex for the subset track. The
starting source is public PR #1137 by hybridnoise, commit
`d5d2283e47fdb0108ff36b3e6acc5d16c115d8ad`. Its official RTX 4090 run
scored 624,752,385 verified candidates/s and reported a 731.0M/s peak. The
promoted subset score at package preparation was 623,518,629, and the 100-bip
promotion floor was 629,753,816. PR #1137 exceeded the promoted score but
missed that floor. This is one follow-up experiment on that stronger public
source; it does not claim a throughput improvement before the organizer
measures it.

## Changes and correctness boundaries

Only `candidates/subset/` is changed. The executable differences from PR
#1137 are in `tests/gpu_epochs/tree.cu`, `tests/gpu_epochs/pair_shared.cuh`,
and `hit_filter_field_sc.cuh`. The candidate enumeration, fixed-base table,
launch geometry, GPU inverse, exact host publication gate, scorer, verifier,
and benchmark scripts retain their inherited behavior. The three compile-time
controls can be disabled independently for an equal-source comparison.

`QSB_TAIL_B_FIRST=1` evaluates epoch B's recovery tail before reloading epoch
A's parked front words from shared memory. The tails have independent front
words, inverse leaves, epoch identities, window lanes, and recovery IDs. Each
tentative hit stores its own identifiers, which the host gate uses when it
recomputes the candidate. The atomic arrival order of hit slots may change;
the candidate associated with a slot does not. Setting
`QSB_TAIL_B_FIRST=0` restores the PR #1137 order. The source does not change
the count of candidates enumerated or the formula for either tail.

`QSB_GATE_H0_FMA=0` selects the active paired-ALU implementation of the two
H0 SHA-256 gates. The inherited `=1` path uses FMA-folded additions. The
gate's actual source body was extracted into a host differential executable;
20,004 digests, including edge cases and random inputs, matched OpenSSL. The
switch changes scheduling and instruction placement, not the hash definition,
qualifying target, or publication check. Earlier local work found the
FMA-folded gate favorable in some cases, so a shorter static instruction
listing is not itself a speed claim.

`QSB_DROP_Z2_EARLY=1` removes one early `addc.u32` at nine lean field-square
fold sites while retaining the later carry propagation. This is a **rare,
lossy** speculative-filter cut, not an exact field rewrite. It was also
present in our earlier PR #1211 source. A literal-preprocessed-PTX point
model had no mismatch in 1,000 directed and random point updates; that sample
does not bound a rare carry. The exact inherited host publication gate rejects
false tentative hits, but it cannot recover a genuine hit missed by the GPU
filter. Any such miss reduces the official verified score. Setting
`QSB_DROP_Z2_EARLY=0` restores the earlier carry path. The cut is included
because it removes the 4-byte spill load and store that otherwise remain in
this specific native `sm_89` composition; its error budget must be weighed
against measured throughput.

The host gate, hit record layout, and output file format are inherited from
PR #1137. This candidate does not introduce a new host overlap path, reorder
the candidate stream, change hit-buffer capacity, or publish a result without
the existing exact verification. The SHA differential and algebraic tail
independence support correctness of the changes, but this Mac cannot run the
end-to-end CUDA hit list. The ranked run is the authority for both verified
hits and sustained speed.

## Compiler and comparison evidence

I built the candidate with CUDA 12.6 using the organizer-style command
`nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm`, and separately
with `-arch=sm_89 -Xptxas=-v`. Build artifacts stayed outside the editable
candidate directory. The organizer-style `sm_52` digest kernel uses 127
registers, 49,152 bytes shared memory, zero stack frame, and zero spill loads
or stores. The native `sm_89` `kernel_digest` uses 128 registers, 49,152 bytes
shared memory, zero stack frame, and zero spills. No local NVIDIA GPU or
driver was available for throughput measurements.

The native target's static `kernel_digest` SASS contains 14,536 instructions.
The matched PR #1137 source configuration, obtained by disabling all three
new switches, contains 14,896. With only the field cut disabled, this
candidate has an 8-byte frame, 4-byte spill stores, 4-byte spill loads, and
14,544 static instructions. The field cut therefore removes a real spill in
this compiler configuration. The static reduction from the paired gate and
tail order does not directly predict execution time, power, clock behavior,
or the 20-minute verified score. These local builds used CUDA 12.6, while the
ranked runner may use another toolkit version. The `sm_89` resource report is
a viability check, not a promise of occupancy or speed.

Our preceding PR #1211 combined a different base-A recode, SHA-fold path,
and this field cut. It reported a 731.2M/s peak but scored only 615,555,939
verified candidates/s, below PR #1137. That result is why this package
starts again from PR #1137 and omits the base-A and SHA-fold additions. It
does not prove which PR #1211 change or runner condition caused the score
difference. The official runs used different problem seeds and are not
hit-for-hit comparisons. This branch should be assessed by its own verified
candidate rate and hit count, not by subtracting static SASS from the PR
#1137 score.

## Reproduction and attribution

Build from `candidates/subset/` with `nvcc -O3 -DQSB_ZEROS_N=24 -o subset
subset.cu -lcrypto -lm`; use `-arch=sm_89 -Xptxas=-v` for native resource
inspection. For a matched source control, add
`-DQSB_TAIL_B_FIRST=0 -DQSB_GATE_H0_FMA=1 -DQSB_DROP_Z2_EARLY=0`.
The controls are independent, so a fixed-work RTX 4090 A/B can isolate each
change while checking the same problem and verified hit set. Any missing hit
on the field-cut arm must count against it, even if its self-reported rate
rises.

The substantial inherited implementation is credited to hybridnoise's PR
#1137 and its public PR #1088/PR #1054/PR #1022/PR #996/PR #973 lineage,
including terrapinelf, DrCleverHans, Akashneelesh, dun999,
Meganpark980320, ercumentyildirim, EvanYan1024, owizdom, DPZZxlz,
fkiene, jacklightChen, mitchuski, and Babbaragga as named in the inherited
notes. Their licensing notices are retained in the source. The B-first order,
paired-ALU selection, and nine-site field-cut composition were prepared
here. No donor's earlier local or official speed is attributed to these new
changes.
