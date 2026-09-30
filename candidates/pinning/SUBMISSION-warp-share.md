# Per-warp GLV12 share selection in the pinning candidate path

## Result

**Not measurable in this environment.** The official benchmark runs on the
challenge runner (GitHub Actions, NVIDIA RTX 4090). This machine has an AMD
GPU, so I compiled for the target architecture and read the compiler's own
resource report, but I did not run the kernel and I am not quoting a
throughput number I did not measure.

What I *did* measure, and can be checked, is the resource cost of the change
at the ptxas level:

| build | registers | stack | shared memory |
|---|---|---|---|
| default | 128 | 0 B | 14336 B |
| this change | 128 | **8 B** | 14336 B |

The image hash changes, so the edit does reach the device code — a hash that
stayed identical after an edit would have meant the edit never got compiled
in.

## What I changed, and why

The candidate decoder has two ways to choose the GLV12 share for the P path:
per block, or per warp. The per-warp path is gated by a compile-time switch
whose comment calls it a kill switch defaulting to on, while the
`#define` below it sets it to zero. Those two statements disagree, and nothing
in the track had ever measured what the disagreement costs.

I enabled the per-warp selection, which is a one-line change to that switch.
The measured effect is an 8-byte cumulative stack allocation at 128 registers
— the extra live value has nowhere to go, so it spills. Eight bytes is a
small, bounded cost, not a wall, which is why it is worth putting in front of
the official runner rather than discarding on an estimate.

## Why this and not a geometry change

A previous pass proposed changing the number of GLV terms per component. That
proposal was refuted on the source, with line numbers, for four independent
reasons — among them that the chain loops are explicitly not unrolled, so
additional terms are additional iterations of one body rather than additional
live registers, and that the chunk widths are pinned by a radix constant
asserted in two places rather than being a free parameter.

So the register-pressure argument that closed that proposal does not hold.
That does not make the proposal good — it removes the objection. A cost
estimate built on an instruction count that is not the count of the executed
instructions is not an estimate, and it should not decide a direction.

## How the measurement was established

Two things had to be true before any of the above was worth reading, and
neither is obvious from the source.

**First: the compiled image is a real build input, not a decoy.** The
candidate ships its device code as a prebuilt cubin embedded in a generated
header, with its own checksum recorded in that header and re-checked at load
time. Editing the device source without regenerating that header therefore
produces a clean compile and a score that is a redraw of the previous image.
Any comparison against "the previous best" made without confirming the
regeneration path measures nothing at all.

**Second: the regeneration is bit-reproducible with the toolchain I have.**
Rebuilding from unmodified source reproduced the shipped checksum exactly,
byte for byte. That is what licenses the rest of this submission: it means a
checksum that changes after my edit is a consequence of the edit, and a
checksum that does not change means the edit did not arrive.

The procedure is therefore three commands: rebuild the embedded image from
the candidate directory, read the checksum the header now records, and compare
it to the unmodified one. I did this once with the edit applied and once after
restoring the source, and the checksum returned to its original value — so the
change is reversible and the measurement is not an artifact of a dirty tree.

## The cost estimate that motivated the direction, and its limit

The one trustworthy calibration available in this track comes from two
official promotions of the same project with a single variable changed: a
build with four cold 64-byte record reads scored about 1.2% above a build with
six, for one fewer add. That puts the marginal cost of a scattered cold read
at a fraction of a percent of total runtime — considerably cheaper than an
earlier figure of roughly 67% per scattered read that circulates in the notes.
Those two numbers cannot both describe this program; the same-project pair is
the one I would act on, and the larger figure was priced on a different
geometry by a marginal perturbation. I flag it because any argument in this
track resting on the 67% number is resting on something I could not reproduce.

The practical consequence is that this kernel is compute-bound rather than
latency-bound: a direct attempt to prefetch the chain's table reads scored
about 37% *worse*, which is the opposite of what a latency-bound program would
do. So the win is unlikely to come from moving memory traffic around. It is
more likely to come from doing less serial arithmetic, or from letting each
warp pick a share that suits its own slice of the candidate range.

That is exactly what this change touches, which is why I am submitting it
rather than theorising further: it is the smallest step in the direction the
measurements actually point at, and its cost is now known rather than
estimated.

## What I did not verify

- Throughput. I cannot run this kernel here; only the official runner times
  it.
- Whether 8 bytes of stack matters at the table traffic this kernel sustains.
  That is precisely the question the runner answers, and I am not going to
  pre-empt it with a number.
- Whether the per-warp share is faster or slower once the table is resident.
  The switch is called a kill switch, which suggests someone expected it to
  help; the comment and the `#define` disagree about its default, and I have
  no measurement that would settle which is right.
- Whether the spill is on the hot path. The 8 bytes could sit in a path that
  is already cold, in which case it costs nothing, or in the serial chain, in
  which case it costs something. `ptxas -v` reports the total, not the site,
  and I did not disassemble the two builds to attribute it.
