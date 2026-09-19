# Packed first-state producer on PR407

Base: ercumentyildirim public PR407, 3fa1a9f29ce7946eaa9a7758b2e4a1f8e14b5598, including its shared slope-scale finish and all prior notices. The packed producer is our c29f9c8e first-state experiment on PR363. Four 64-slot epochs now share one 256-thread producer block, with explicit epoch and class bounds. All SHA arithmetic and the existing state layout are unchanged. Consumer arithmetic, exact output verification, enumeration and trusted harness remain unchanged from PR407. The first-stage audit exercises the new launch geometry.

No throughput improvement is claimed until matched device measurements. Tracked generated binary/build-stamp artifacts are omitted; builds use the normal source path.
