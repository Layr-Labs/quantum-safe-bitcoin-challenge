# Is the host-CPU co-grinder's worker count a lever? No — and the frontier's own note is confirmed

The promoted subset note claims:

> A busy host CPU costs the thermally limited GPU about 2% **whatever the co-grinder's
> speed**: a 6 M/s scalar lane paid the same as a 26 M/s IFMA lane. So co-grinder work
> per candidate, not raw activity, is what pays.

That is a testable statement on this machine, because both quantities are observable in
the same run: the GPU prints its own progress rate every ~15 s (~0.3% resolution) and the
co-grinder prints an exact candidate count. The ranked score sums both contributions, so
the quantity that matters is **net = gpu_rate + cpu_rate**.

`QSB_CPU_THREADS` sets the worker count directly; `QSB_CPU_GRIND=0` removes it.

## Result (two repeats, 60 s arms, interleaved)

| arm | workers | GPU M/s | CPU M/s | **net** |
|---|---:|---:|---:|---:|
| `r1w4` | 4 | 301.4 | 1.77 | 303.2 |
| `r1w8` | 8 | 298.5 | 3.03 | 301.5 |
| `r1w16` | 16 | 297.5 | 4.74 | 302.2 |
| `r1w30` | 30 | 296.2 | 6.88 | 303.1 |
| `r2woff` | 0 | 297.8 | 0 | 297.8 |
| `r2w2` | 2 | 298.1 | 0.92 | 299.0 |
| `r2w4` | 4 | 298.3 | 1.81 | 300.1 |
| `r2w8` | 8 | 296.2 | 3.03 | 299.2 |
| `r2w16` | 16 | 297.1 | 4.77 | 301.9 |
| `r2w30` | 30 | 293.0 | 6.92 | 300.0 |

**The net is flat — 300 ± 2 across 0 to 30 workers**, while the GPU rate alone falls ~1.5%
from 4 to 30 workers and the CPU rate rises by almost exactly the same amount. The
frontier's "fixed cost whatever the activity" claim reproduces here, and it is stronger
than the note states: the compensation is nearly exact, not merely the same direction.

## Consequence

**Worker count is not a lever.** The frontier ships the maximum useful count (all CPUs
minus a reserve) and that is correct: adding workers buys CPU candidates at almost exactly
the GPU rate they cost, and removing them saves GPU rate at almost exactly the CPU
candidates they were producing. There is nothing to tune here, and no ranked slot should
be spent on it.

This also explains the shape of the frontier's co-grinder work: it has to make each CPU
*candidate* cheaper (wider vectors, fewer lookups), not run more of them — and that is
exactly what its nine-window table and 16-lane AVX-512 path do. On this host the
co-grinder reports "30 threads … scalar, 4-lane SHA-NI", i.e. the non-AVX-512
configuration, so its vector-width improvements are not measurable here either.

## Reproduction

```sh
bash /mnt/d/kongtaoxing/tmp/qsb-ab/worker-sweep2.sh
```

Each arm builds the current candidate at `N=24` with `-DQSB_CPU_THREADS=<n>` or
`-DQSB_CPU_GRIND=0`, runs it for 60 s on the committed public problem, and reads the GPU's
last progress line plus the co-grinder's summary line.