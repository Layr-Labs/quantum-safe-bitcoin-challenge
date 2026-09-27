# Affine batching research screen

This is a new architecture hypothesis, not a production implementation.
The existing PR24 pipeline batches only the final recovery denominator. The
proposal would batch denominators during fixed-base point accumulation itself.

For 15 table points, 14 ordinary affine additions cost 28M+14S plus inversion.
A work-efficient simultaneous inverse adds approximately 42M, yielding
70M+14S versus PR24's deferred XYZZ 95M+28S. This is a substantial arithmetic
difference, but does not include memory, synchronization, launches, exceptional
cases, or final recovery. Both serial accumulation and a balanced tournament
perform 14 additions; the tournament has four dependency depths (7,4,2,1).

The gECC authors investigate affine batch operations with Montgomery inversion
and explicitly address global-memory costs through selective recomputation,
data placement and kernel fusion. Their measurements concern other workloads
and hardware; they do not establish a pinning speedup. This supports further
architecture analysis, not importing their headline gains into our forecast.
[gECC paper](https://arxiv.org/html/2501.03245v1).

## What the first source comparison rules out

Naively repeating PR24's full 320-byte/candidate checkpoint arrangement for
14 rounds would transfer roughly 4,480 bytes/candidate before additional
table/state costs. At 650 million candidates/s that is 2.9 TB/s of logical
traffic. This is an arithmetic scenario, not a measured bandwidth requirement
or proof that every affine implementation fails. Changing retained state,
kernel fusion, batch size or caching changes the traffic.

The tournament reduces dependency rounds but has seven pair denominators per
candidate at its first level and eight points after the first round. Keeping
those eight affine points for a 16M batch uses 8 GiB. This exceeds PR24's 2 GiB
state allocation but is not, by itself, proof that a 24 GiB GPU cannot fit the
entire design. A complete peak-lifetime allocation calculation is required.

Grok's advice to reject any extra inverse round or any state larger than the
existing allocation was too broad. Its concrete warning about multiplied
checkpoint traffic is useful. Gemini's barrier-based rejection likewise did
not establish hardware timing or a general impossibility result.

## Next falsifiable design gate

1. Specify exactly which coordinate values survive each kernel boundary.
   Fuse one level's affine completion with the next level's denominator
   preparation where data dependencies and lane ownership allow it.
2. Derive total logical traffic and peak storage from that schedule, including
   all point state, inverse products, root arrays, table reads and tails.
   Analyze smaller batches without assuming cache residency or free launches.
3. Test the actual addition/checkpoint expressions against independent curve
   arithmetic. Handle infinity, equal points and opposite points explicitly;
   feeding identity into an inverse collective does not itself finish a
   degenerate point addition correctly.
4. Only implement a full candidate if the resulting cost argument supports a
   credible overall win over the strongest pending source. A 25M+14S saving
   with unbounded memory cost is insufficient.

Do not repeat the rejected naive 14-pass design or dismiss all affine designs
based on it. The field-corrected source base and PTX semantic checks now provide
a better foundation for whichever architecture survives this gate.

## Resident cooperative alternative, September16

A one-candidate-per-thread persistent cooperative kernel could keep affine
coordinates and recoding state in registers, exchanging only CTA roots. This
avoids the naive global point-checkpoint traffic. It still requires fourteen
dependent inverse rounds for accumulation, and likely a fifteenth for final
recovery. With a hypothetical128SM,2CTA/SM,256thread layout, the resident tile
contains65,536 candidates. Cooperative launch must use actual occupancy and
device capability checks; those block counts are not guaranteed by compiling.
[NVIDIA cooperative launch constraints](https://nvidia.github.io/cccl/unstable/libcudacxx/runtime/launch.html).

At644,546,620/s the tile corresponds to101.68us. A25% rate gain needs81.34us.
Even assuming a30% whole-runtime arithmetic saving leaves only10.17us for all
extra costs, including root inversion and synchronization. Thirty global
barriers alone would each have at most0.339us if every other extra cost were
zero. This is a conditional budget, not a measured latency or impossibility
proof. A four-depth tournament trades fewer rounds for multiple live points
per candidate or low lane utilization at later levels.

The actual Grok CLI review (backend grok-4.6-build; CLI model name grok-4.6)
correctly highlighted repeated serial root inversions and exceptions. Its
worked partial-affine arithmetic example has ambiguous add counts and is not
accepted as an exact bound. Gemini returned an ERROR/503 with a partial draft;
that draft substituted a Fermat addition chain for the actual safegcd inverse,
invented microsecond timings and asserted zero spills without compiling.
Those numeric latency/resource claims and categorical rejection are discarded.

PR41 also reports an affine-every-step prototype at64M/s versus its204M/s old
baseline, with128registers and164B spills. Its described shuffle-scan/uniform
inverse mechanism executes more work than a work-efficient hierarchical tree;
that result rejects its particular implementation, not all affine batching.
[Public experiment notes](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/41).

Current decision: park the resident-affine schedule because no concrete
critical-path budget supports the required gain. Prioritize the ten-window
projective prototype in `wide_windows/`, which already passes source-derived
recoding/curve tests and native point-chain compilation. Do not repeat generic
affine reviews without a new schedule that addresses the counted costs.
