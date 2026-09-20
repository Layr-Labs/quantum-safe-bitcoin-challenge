# Local literal SHA256 round constants in the paired gate

Parent: local gateiv734 d4b72bfbfe6b835007b85d01dfcfa122c551ac73,
based on dun999 PR734 434b9573d01d7083666ec80f2f0605c54d50f43a.
Retain its attribution to ercumentyildirim PR624, earlier SHA/point contributors,
and the low64 correction from our PR654. Inspecting Saviour1001 PR756 prompted
the inherited IV representation probe; this extension still computes all eight
digest words for both streams.

Add a function-local constexpr array with the exact unchanged GPUHash.h K words
and replace only the paired round macro's two K references with that array.
All macro-expanded indices are compile-time constants. Keep all 64 rounds,
word schedule, packing, padding, feed-forward, consumers and other SHA functions.
The parent standard-IV literals remain. No custom-IV or custom-K extension is
intended. This enables constant folding; instruction count and speed must be
measured separately. There is no performance claim from source changes alone.

Local-only contingency. The installed chainu2734 experiment has priority.
Do not install, select or queue this probe while that experiment can qualify.
All inherited speculative arithmetic boundaries and exact hit rechecks remain.
