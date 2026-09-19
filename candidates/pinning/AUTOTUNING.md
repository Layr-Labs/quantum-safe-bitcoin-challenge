## Integration update

This revision uses32M production batches (PR563 @terrapinelf). Each timing sample covers64M candidates, rounded to full slot cycles. With two slots this is2 batches per sample, not the donor8. Both slots receive one warmup batch. Everything else below documents inherited PR552 behavior; its native measurements belong to that donor source, not this composite.

# Selecting the ordinary-window size on the actual GPU

The six precompiled scalar kernels use11,12,13,14,15,16 signed affine terms
with3584,1024,352,144,64,32MiB tables respectively. Fewer point additions compete
with larger random table traffic; cache and compute/bandwidth balance differ
between GPUs. The finish kernel, SHA, inverse hierarchy and host loop are shared.

Startup builds fresh problem-dependent tables, independently checks them,
warms two full batches, then measures three eight-batch full-pipeline intervals
on the normal slot streams. The smallest median elapsed wall time wins. All
startup work is inside the grinder command and therefore the official root
timer. The program invokes no compiler at runtime; normal driver PTX JIT
remains part of root elapsed time. Tables that cannot be allocated
are skipped; a table mismatch stops the program before producing search hits.

The tuning batches compute the real problem's full recovery and SHA path.
Their hits are discarded and never enter output files or candidate counters.
The ordinary search then restarts its unchanged sequence/locktime domain once,
resetting hit counters before every batch. Timing diagnostics are milliseconds,
not reported candidate throughput. QSB_WINDOW_COUNT=11..16 is an optional
diagnostic override to reproduce each geometry with the same executable.

Preserved foundations and credit: the promoted pinning implementation, its
shared-plane digit decoding, deferred-Y XYZZ chain, slot streams, cofactor
inverse tree and interleaved pubkey SHA. All inherited source notices remain.
Direct digit extraction builds on the promoted work attributed to @dun999
(f535811); rare raw-hash reduction follows @scarletbright (e7a648c7). The
slot pipeline retains the source's @draheemking (11ba7e43) attribution.
This candidate adds balanced wider geometries, a complete final affine guard,
the common table builder and actual-device window selection. See the public
submission note for exact baseline refs, experiments and measured limitations.

The11-term geometry uses three24-bit and eight23-bit windows. GPU table
construction is divided into launches of at most1,048,576 threads, bounding
individual kernel duration. The2KiB per-thread CUDA stack limit avoids the
unnecessary32KiB reservation inherited from the baseline. The native production
maximum compiled frame is120B; direct point audits execute the inversion paths
under the same2KiB limit before this variant is benchmarked.

Each completed trial retains only the best table seen so far. A losing table
is freed after restoring the access-policy range to the still-live winner;
an improved winner frees the previous winner after all trial streams finish.
Thus at most two trial tables are resident, reducing irrelevant memory pressure
during selection. The table builder and all device arithmetic are unchanged.
