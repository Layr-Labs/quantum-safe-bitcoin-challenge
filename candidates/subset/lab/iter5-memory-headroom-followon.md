# Next host-memory direction: unused generic combo buffers

The full self-check alias removes a required 640 MiB duplicate allocation.
However the ranked request accounting still leaves only 458.537 MiB before
runtime overhead, even without that duplicate. A live local sample during the
verified final integration held 12,244 MiB on the GPU; known major requests for
its small table sum to 11,829.463 MiB. The 414.537 MiB gap is runtime/module
allocation plus granularity and other requests, not a ranked runtime estimate.
Substituting the table delta alone would leave only about 44 MiB. Native sm89
may differ and the desktop owns unrelated VRAM, so this is a risk signal, not a
proof of ranked fit or OOM.

Source audit: main always allocates BATCH=8,388,608 * t_sel=9 bytes for both
h_combos and d_combos: 72 MiB each. The short-epoch path only consumes epoch
arrays and tentative hit records. All references that fill/use d_combos are
inside the `#else` of ZLAB_TRIM, whereas ranked ZLAB_TRIM=1 rejects any shape
outside the short-epoch branch. `free(h_combos)` accepts null in every exit.

Proposal: an explicit default-off host-only switch QSB_TRIM_COMBO_ALLOC avoids
these two allocations only when ZLAB_TRIM=1; no generic path loses its input
buffers. Full benchmark on the resulting production wrapper must still pass.
Record its local GPU-footprint delta against 12,244 MiB, with binary/source
provenance, and distinguish removed requests from observed runtime footprint.
Do not assert throughput gain: this repairs additional ranked headroom only.
Do not change the pending frozen combined gate or device arithmetic.


## Result and disposition

One existing benchmark entry point ran the frozen enabled wrapper at N=24,
120 seconds, seed 1789110211: **394.115136 M verified candidates/s**, all
**5,709 hits verified**, PASS. Source, dependency and executable hashes are
in `iter5-trimcombo-manifest.json`; score and log are retained beside it.
No new proof fixture was added for this edit and no overlapping follow-up
verification was stacked on the pass. The process exited before a useful
live-footprint snapshot, so there is no measured footprint delta. The exact
removed allocation requests are 72 MiB each on host and device. This is not a
paired speed claim or proof of ranked fit. The probe remains default-off;
production qualification remains the combined +7.906507% gate above.
