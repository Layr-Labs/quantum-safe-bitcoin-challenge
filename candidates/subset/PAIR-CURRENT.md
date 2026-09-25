# Corrected current point and inverse with a compact paired pipeline

The parent is fddf0ba86cae66c9ef9195d287cf0c02d8cb9cb9: complete point replay and final addition, canonical field operations, a normalized six-step table inverse, a32-batch bound and an independent fallback. Those arithmetic sources and the entire point-chain block are retained byte-for-byte.

The two-epoch consumer, first-state producer, odd-tail guards, exclusive descriptor bounds and exact epoch accounting come from our audited pairfirst f28e0be2f88ab7c4377d5cdf98beeca2f54d6a78. Public mechanisms are attributed to dun999 PR212/PR258 and odinfree's SHA schedule. The compact in-place tree is byte-identical to our audited2c7326496d597a98cfbbc1c43423d97dc2663057, inspired by Akashneelesh PR157. The new composition keeps only two epochs and16KiB parked finish state plus16KiB inverse-tree storage. All existing notices remain.

The compact tree reads the entire old level before overwriting it. Uniform read/write barriers cover cross-warp accesses, and cooperative root lanes finish reading both children before replacing them. The first-state tables are read-only during consumption; an absent odd-tail epoch aliases valid storage, uses an identity denominator and cannot emit a hit.

This is a new composition. Source identity binds earlier actual mapping/point/inverse audits; it does not establish the new executable's correctness or speed. Official CUDA12.8.93 GPU first-state, finish, treeLOCAL_PATH_OMITTED and finite whole-solver checks are required before performance comparison. Production retains65536physical blocks; diagnostic finite limits are not performance settings. No measured improvement or submission qualification is claimed here.
