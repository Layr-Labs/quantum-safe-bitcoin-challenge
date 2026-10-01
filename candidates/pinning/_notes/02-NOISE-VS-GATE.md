# FINDING 5 — the 1% promotion gate is half of one standard deviation

The brief says not to produce a seventh re-roll, and is right. This note
quantifies *why*, so the next worker does not have to re-derive it, and so the
size of a mechanism that would actually be worth submitting can be stated.

## Two independent same-bytes series, from this track's own records

**Series A — v19 bytes** (`ITERATIONS.md:1480-1483`, four official draws of one
byte-identical archive):

| ticket | official |
|---|---:|
| v19  d27f252e | 829,084,805 |
| v19b ad10074d | 791,023,360 |
| v19c 3cfd54c5 | 827,827,523 |
| v19d b270b779 | 794,321,999 |

mean 810,564,422 · population σ 17.93 M (**2.21 %**) · peak-to-peak 38.06 M
(**4.70 %** of mean) · best-minus-worst 4.70 %

**Series B — v20 bytes** (`SUBMISSION-v27.md:69-74` + `ITERATIONS.md:1602`, five
official draws of one byte-identical archive):

| ticket | official |
|---|---:|
| v20  4fe6a084 | 882,096,418 |
| v20b 56186ef3 | 855,462,909 |
| v24  d8e37a96 | 879,640,393 |
| v25  62651c13 | 878,283,220 |
| v26  3847355e | 844,911,816 |

mean 868,078,951 · population σ 15.03 M (**1.73 %**) · peak-to-peak 37.19 M
(**4.28 %** of mean)

The two series were drawn on different devices/seeds and agree: **σ ≈ 1.7–2.2 %
of the mean, peak-to-peak ≈ 4.3–4.7 %.** The brief's own six v20 draws span
37 M, matching.

## The gate, against that noise

`minScoreImprovementBips = 100` → promotion needs **+1.00 %** over the record.

    gate / σ  =  1.00 % / 1.94 %  ≈  0.52 σ

A 1 % effect is **half of one standard deviation of a single draw.** No single
submission can resolve it. Concretely, on the v20 series:

- the best draw ever obtained (882,096,418) is +1.61 % over its own mean, and it
  still **missed** the then-floor of 890,086,137 by 8.0 M, because the floor sat
  2.53 % above the mean. Drawing the top of the distribution is not sufficient
  when the gate is set above it.
- the largest single mechanism this track has ever validated is NEG_Y_MAC at
  **+0.79 % official** (`ITERATIONS.md:1315`) — half the noise σ.
- every other validated mechanism is smaller: PR #743 +0.997 % (and it *missed*
  its floor by 23,694/s, `DEAD-ENDS.md:17`), K32 +0.26 % local, CHAIN_PIPE
  +0.8–1.4 % self-rate.

So the promotion rule demands a **>1 %** effect that must be located inside a
**4.3–4.7 %** spread, with one ticket in flight per account and ~75–95 min per
ticket (`ITERATIONS.md:1346`). Even a genuine +2 % mechanism has P(promote) near
0.5 on any single draw.

## The arithmetic that actually decides this track

Compare the two series directly. The v19 bytes and the v20 bytes are the *same
program lineage* with different luck, and the record at each point in time moved
by far more than any mechanism anyone has shipped:

    record 881,273,403  (2026-09-23, anamdongparkjinhyeong pr1259)
    record 904,971,814  (F package)
    record 948,943,797  (fkiene 3b423554, eleven-term geometry)
    record 960,830,125  (fkiene ff524fd9, GLV12 GPU)
    record 1,008,206,828  (current, per the brief)

That is **+14.4 % in seven days**, entirely from whole-geometry and whole-pipeline
replacements by the field. The floor rises 1 % per promotion, so the floor's own
drift (+1 %/promotion, several promotions per day) is a *larger* term than any
mechanism this track has shipped in its entire 31-version history.

For a fixed byte-string the achievable score is roughly

    score ~ self-rate(draw) x hit-density(draw),   both ~N(mean, sigma~2 %)

while the target is `1.01^t x record_0` with `t` promotions by others happening
in the meantime. **The floor compounds away from any frozen byte-string faster
than a 1–2 % mechanism can be found, packaged, and drawn.**

## What this rules in

A mechanism is worth a ticket only if its expected effect is **larger than the
noise**, i.e. **≥ ~5 %**, and larger than the floor's drift over the validation
queue (~1–2 %) plus the draw's own σ (~2 %). Call it **≥ 6 %** to be safe.
Nothing in this track's history reaches 6 %. The entire measured field of
mechanisms is:

| family | best measured | verdict |
|---|---:|---|
| 2^-31/2^-62-class carry tails | +0.1–0.6 % | below noise |
| tail truncation (PR #743) | +0.997 % | below noise; missed its floor |
| K32 field-row corrections | +0.26 % | below noise |
| lane-form / issue-order / register steering | −2.7 % … +0.85 % | noise or worse |
| prefetch (L1 / TBL_PREFETCH / cold-table) | −37 % | dead |
| Karatsuba | −6.1 % | dead |
| fused prepare+finish / fused inverse / fused root tail | −9.2 % / −0.3 % / −1.4 % | dead |
| table geometry (GLV10 / grouped GLV) | −55 % | dead |
| init dead-time strip (v8, v12) | −4.1 % … 0 % | dead |
| CPU co-grind | +0.3–0.5 % of total score | below noise, and real |

The single lever that ever moved this track by more than noise was **whole
geometry replacement** (GLV14 → GLV11 → GLV12, i.e. −40 M → +48 M → +12 M).
That is the only class with the right magnitude, and it is exactly the class the
carrier block in note 01 makes un-shippable from a host without CUDA.
