# PR138 leaf-pair inverse component with the n=64 join repaired

Prepared candidate fingerprint:
`94e46e120d921d87273fd9e50584cc2294ba58c57f9de6e7c8aa67b102a14b5f`.
Base: exact fused small48
`b832d7ba4ec2b2554aa5279e46858a5163e83e3ad7e714e9b108551165c95f7b`.
Only `tests/gpu_epochs/tree_inverse.cuh` changes. All field, geometry, builder,
frontend and other source files remain byte-identical to that base.

The donor is AbdelStark's public PR138 commit
`2dc49dc4e4083050ffc34b9be61eb027eb8c291f`. Its exact header snapshot is
`pr138-tree_inverse.cuh`; `prepared-source.json` records its SHA-256. If this
substantial unpromoted work is used in a submission before promotion, credit
`AbdelStark` as a coauthor in addition to the candidate's existing contributors.
Original source notices and GPL `COPYING` are preserved.

The public header forms each leaf pair using a neighbor warp shuffle, builds
and inverts the smaller shared product tree, then lets every lane multiply the
pair inverse by its original sibling. It retains original leaves in shared
memory instead of keeping them in registers across root inversion. Arithmetic
work remains exactly `3n-3` field multiplies and one inverse. This is a supporting
inverse component, not a standalone submission or demonstrated speedup.

The public final descent misses a CTA join at n=64: warp 0 writes final pair
inverses 32..63 while warp 1 may read 48..63 before the writes. The repaired
variant adds uniform `if(n==64)__syncthreads();` immediately before final
all-lane expansion. See `../two_lane/pr138-race-review.md` for the independent
producer/consumer analysis and all-leaves-2 stale-result witness. The donor's
general claim about two outer block barriers applies to the 128/256 cases;
the repaired 64 case does not save a CTA join over the base.

| Block size | Multiplications | Inverses | Repaired CTA joins | Base CTA joins |
|---|---:|---:|---:|---:|
| 32 | 93 | 1 | 2 | 2 |
| 64 | 189 | 1 | 3 | 3 |
| 128 | 381 | 1 | 3 | 5 |
| 256 | 765 | 1 | 5 | 7 |

`check_leaf_pair.py` executes the actual candidate header with the existing
`check_candidate.BACKEND`: CPU threads emulate warp/block joins and shuffles,
and OpenSSL replaces field math. Eight batches at each size pass: all identity,
all two (race witness), near-p, inactive identity tail and four full random
batches, totaling **3840 correct inverse outputs**. Every batch asserts exact
multiply/inverse counts and all threads' matching CTA-join count. Actual output
fifth words are zero and each result is independently checked with modular
inversion.

A separate focused source-checked join audit derives final producer/consumer
warp ownership and required joins for n=32/64/128/256. Removing the n=64 join
fails that audit at consumer 32 / producer 8; the original public header also
fails. This catches the missing synchronization independent of CPU scheduling.
It is a narrow analysis of this exact loop structure, not a general CUDA race
verifier. Full-mask shuffle correctness for blocks below 32 is not claimed.

Reproduce from the benchmark root:

```sh
python3 -B candidates/subset/research/leaf_pair/check_leaf_pair.py
```

`prepare.py` verifies the exact base and refuses to overwrite its frozen output.
Do not rerun it over this candidate. CPU evidence does not execute native field
PTX, CUDA shuffles or device barriers and does not establish speed. Native
production/audit compilation and full integrated builder checks are separate
requirements because this header serves search and builder inversion. Parent
coordination owns those checks. No production/stage source was modified and no
submission was made by this component experiment.
