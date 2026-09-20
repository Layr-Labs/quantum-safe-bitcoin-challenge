# Local scalar SHA round constants

Parent gatefix734 448b773cf494d73a5b7bc22fedc1f804941c87d6 retains
the public734 chain and the preceding paired-gate literal IV/K experiment.
Credit dun999 PR734, ercumentyildirim PR624, Jean Luc PONS / VanitySearch
for GPUHash.h, earlier point/SHA contributors, and the inherited research
attributions. All original notices remain.

The only runtime change is a function-local constexpr K[64] declaration in
_SHA256Transform, containing the exact existing SHA256 constants. It shadows
the global array for this function's macro-expanded constant indices only.
All 64 rounds, input chaining state, eight feed-forward words, initialization,
packing, message schedule and callers remain unchanged. The global K and I
arrays remain unchanged for other consumers. Arbitrary chaining states remain
supported. This is an ordinary constant-folding experiment.

This source is a local contingency, not an installed or selected GPU arm.
The already queued gatepair734 experiment and any qualified submission have
priority. Native instruction changes alone are not speed evidence. All
inherited speculative arithmetic limitations and exact output replay remain.
