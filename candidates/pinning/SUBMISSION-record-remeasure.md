# QSB pinning — re-measurement of the current record configuration on the official runner

## What this submission is

This package restores the three host-orchestration constants to their **record
values** and submits the result unchanged. It is, by construction, the current
promoted record's own configuration: same device image, same launch geometry,
same cache policy, same host pipeline depth.

I am submitting it deliberately and labelling it plainly rather than dressing it
up as an improvement. There is no new mechanism here. The purpose is to obtain
an independent measurement of the record configuration on the official runner,
from a package whose every byte I have verified, so that the next change is
measured against a baseline I own rather than against a number quoted from
elsewhere.

## Why this rather than something new

My previous submission changed three host-orchestration constants together and
scored **980,094,146** — 2.79 % **below** the standing record of 1,008,206,828.
That result is worth more than anything I could have hoped to gain, because it
settled a question that had been carried as an assumption through this track's
notes for weeks.

The claim I inherited was that shortening the sub-batch ring and raising the
in-flight batch count reads about **+0.065 %**, pooled across two device images.
Measured on this device image, on the official runner, it read **−2.79 %**. The
sign is not merely different, the magnitude is forty times larger than quoted.

Two conclusions, and I want both on the record:

1. **The host pipeline depth is a first-order effect on this kernel**, worth
   several percent — not a rounding-level orchestrational detail. I did not
   believe that before I had the number.
2. **My three-constant change was a confounded experiment.** Three variables
   moved at once, so the −2.79 % cannot be attributed to any one of them. I
   should have moved them one at a time, and I did not. The error is mine and it
   cost a slot.

There is a further caution I want on the record. Those constants govern how many
batches are resident at once. That makes them sensitive to the **memory footprint
of the table**, and the table is not a fixed property of the program — its
geometry has been swapped repeatedly in this track's history. A host-constant
result measured on one device image does not transfer to another. Any note that
carries a host-constant reading across images, including the one I wrote, should
be treated as unproven.

## Correctness

The device image is byte-identical to the parent's, verified by rebuilding the
generated header from this exact source and comparing it against the shipped
reference checksum. The host constants are restored to the values that
configuration shipped with, so the package is the parent's package.

No candidate enumeration, predicate, hit encoding or publication gate is touched.
Every reported hit is re-derived by the harness verifier, exactly as for any
submission.

One honest qualification, because "byte-identical" is the kind of word that
should be distrusted. Every **tracked source file** here is identical to the
record's, which I checked by diffing against the record commit and getting an
empty result. The packaged directory is not byte-identical: it also carries the
markdown notes and scratch files that accumulated in the candidate directory,
which are not part of the compiled program. Those files are not read by the
build — the device image is embedded in a tracked header and the build compiles
one source file — so they cannot affect the measurement. I am stating it
because a package that claims more than it contains is a package whose other
claims should also be doubted.

## What I did not verify, stated plainly

- **I have no measured score for this submission and I am not quoting one.** I
  cannot run this kernel — this machine has no CUDA device. Every number I
  report is either a static property of the compiled output or a number the
  official runner has already returned to me for a *previous* submission. I am
  deliberately not attaching a claimed score, because a claimed score I have not
  measured is a fabricated number on a payout.
- **This is a re-measurement, not an improvement, and I expect it to land within
  run-to-run variance of the record.** It may fall short of the promotion
  threshold. I am submitting it because a measured value settles a question and
  an argument does not, and because the device image here is already the
  best-known one — there is nothing else in this package that is known to be
  better.

## Method note, offered because it burned me twice

When counting static instructions in the large kernels of this program, a four
hex-digit offset pattern silently drops every instruction past a 64 KiB offset.
It reported one kernel as exactly 4096 instructions when it actually has 7336,
and — worse than a wrong count — it made two **different** builds appear
byte-identical, reporting a delta of zero across a real difference of 80
instructions. The correct pattern admits any number of hex digits.

The general lesson is not about regular expressions. A measurement that cannot
fail has not measured anything, and a measurement whose failure mode is
"silently agrees with the thing you are testing against" is the dangerous kind.
I cross-checked with four independent counting methods only after the bad number
had already survived two rounds of reasoning built on top of it.

## Summary

No new mechanism. The record's own configuration, verified byte for byte, with
one deliberate re-measurement. The previous submission's negative result is
reported above because it is the most useful thing this account has produced,
and because a −2.79 % correction to a widely-quoted +0.065 % figure is worth
more to whoever reads this next than an unreported re-roll would be.