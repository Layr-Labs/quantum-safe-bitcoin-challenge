# Local SHA-only follow-up

Literal SHA IV/K constants with the repaired first-stage audit.

Retain the exact paired SHA header from parent bda7c3f76664a54f784941e2f0e9d8b35033b1ed.
Keep public734 tree.cu byte-identical, including chain unroll1. Use the
repaired first_stage_audit.cu from chainfix734, which calls the active flat
producer with production geometry. Existing research notes describe the
historical parents; this note describes the resulting composition.

Credit dun999 PR734, ercumentyildirim PR624, our PR654 correction work,
the preceding point/SHA contributors, Saviour1001 PR756 for the H0 idea,
and jacklightChen PR768 for the scoped separate H0 implementation where
used. Retain all inherited notices. No preparation/final-Y approximation
from751/768 is imported. Exact replay remains.

The existing source-bound host SHA oracle is reused by exact header and
GPUHash byte equality. No GPU speed or correctness result is claimed for
this new commit; it is local only until a separately bound experiment.
