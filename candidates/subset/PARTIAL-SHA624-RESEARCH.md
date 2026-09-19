# Partial constant-round unrolling on the rolled PR624 composition

This private ablation keeps the outer constant-block loop rolled and expands the
inner loop by a factor of2. Each inner iteration already contains8SHA rounds for
each of the two states, so this tests an intermediate code-size choice between
roll624 and the parent's fully unrolled constant blocks. Round order and message
schedule bytes are unchanged. No arithmetic or exact publication code is changed.

Credit: ercumentyildirim PR624 and dukemawex/terrapinelf for the paired SHA and
parent composition, plus all retained frontier notices. No GPU validation,
timing, installation, queue or submission is claimed for this private source.
The ongoing frozen roll624/public624 full comparison is unchanged.
