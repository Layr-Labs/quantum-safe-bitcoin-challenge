# Tail occupancy experiment

The prepared four-stage candidate's unscored RTX 3090 profile placed 11.39%
of batch time in the tail kernel. This experiment changed only that kernel's
launch bounds from `(256,2)` to `(256,3)`, retaining all arithmetic and record
logic. Offline sm_86/sm_89 compilation reduced registers from 127 to 80 with
no stack or spills; default production PTX was unchanged.

Four serialized 60-second N=24 runs used the unchanged harness and synthetic
seed 2026092501:

| Run | Verified hits | Verified M candidates/s |
| --- | ---: | ---: |
| Baseline before | 1814 | 252.974440 |
| Candidate first use | 1729 | 241.092664 |
| Candidate warm | 1800 | 250.948491 |
| Baseline after | 1791 | 249.572463 |

All hits passed verification. The warm candidate was 0.8009% slower than the
first baseline and 0.5514% faster than the last. Completed work changed by
-0.8969% and +0.4545% respectively. Both common-prefix hit sets were exactly
equal. These results do not establish a reliable 1% improvement. Reject the
variant and retain the prepared four-stage production candidate.

The temporary launch-bound option was removed from production. Its exact
header and candidate wrapper remain in `rejected-source-overlay.tar.xz`;
apply that overlay to an isolated copy of the prepared `candidates/subset`
tree before rebuilding this experiment. The plain candidate wrapper was
removed to prevent accidentally benchmarking the restored baseline as the
rejected variant. All raw runs remain losslessly compressed, and terminal
binaries were removed after hashing. No external submission was made.
