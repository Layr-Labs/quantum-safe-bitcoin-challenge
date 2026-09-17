# Corrected thirteen-window startup policy

The current corrected source is a03c3c1ca5f07be955c32af76f34e5a58cc81237d7b9e4c685ee56cfe5353bda.
The earlier e80c shared13 and5d17 regular13 passed device/curve checks but would
exit before search: their host L2 assertion still expected window12 at48MiB.
Thirteen geometry starts its cold tables at9. This caught defect withdraws their
old readiness conclusions. The unchanged fourteen-window e575 remains valid.

Only l2_policy.cuh changes in a03: boundary9 and its diagnostic. Actual geometry
and policy execute with the existing fault-injected CUDA API, covering180
capacity/limit combinations and30 API failures. Frozen e80c reproduces the old
abort. The independent expected widths also check every table offset/shift and
the total size; no geometry stub substitutes the expected boundary.

Fresh actual chain/recovery/frontend checks pass for a03 and its staged root.
Production/audit sm89/default builds match all16 include files. All25,464 parsed
device instruction strings and kernel resource records equal olde80c, preserving
80registers,32KiBshared,280Bstack,344/256compiler spill totals,14761nonNOP and40B
repeated-loop local loads/lane. Host compilation alone would not catch the old
runtime assertion. No GPU execution or performance measurement occurred.

check.py accepts --source and --output for a snapshot or stage. check_policy.py
accepts --source, --report and independently expected --widths; its default is13.
The old fourteen-window fallback passes the same policy matrix with its widths.
preflight.py --package-check now requires startup-policy-validation.json matching
the complete current source, actual after-build call, and capacity/failure matrix.
Missing and wrong-source reports are rejected by explicit negative controls.

Evidence is in public-validation.json, native-equivalence.json, policy-host-results.json,
old-boundary-regression.json, package-gate-results.json and the fresh stage reports.
The cold-prefetch experiment was paused before implementation to correct this
qualification gap. There is still no performance ordering between corrected
shared13 and the preserved shared14 option.
