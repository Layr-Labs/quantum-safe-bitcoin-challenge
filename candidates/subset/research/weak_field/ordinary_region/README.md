# Ordinary weak-field region qualification

Source `3ac276e408a4bb399ea805a56fc9e27831a7bbd6da8df5159f16e95f311c5912` is qualified by the available CPU and native compilation evidence. GPU correctness, scheduling and performance remain unmeasured. Package and public-note review are separate pending steps; this work neither stages nor submits source, and preserves cd3d and the pending evaluation.

Run from the benchmark root:

```sh
python3 -B candidates/subset/research/weak_field/ordinary_region/qualify.py
```

This read-only aggregator writes `qualification.json`; it checks identities and existing results, rather than rerunning their workloads. It verifies the production/audit union of all18 source files, with sm89 and default-target compilation passing for both entries. It binds the weak primitive proof and actual host/PTX tests, product test, synthetic helper test, complete-chain test, startup-policy matrix and native review. Original `prepared-source.json` and candidate prototype marker describe the earlier unqualified preparation; they remain historical artifacts.

The source delta is narrowly checked. Removing the two includes and new helper, restoring the packed-chain call and removing its four boundary normalizations reconstructs the cd3d compact header byte-for-byte. The other15 original files and COPYING are identical. Reversing the weak helper's field calls reconstructs the original shared-Z helper exactly, including the canonical affine-Y sum. Multiply and square retain identical corrected PTX; their full bodies differ only by naming, host square fallback, comments and removed canonical tails. All full-range carry repairs remain.

Fresh evidence includes:

- 60,540 host multiplies,40,360 host squares and2,180 extracted PTX cases each, including rejected stale-carry mutation.
- 13,024 weak primitive binary pairs,17,824 overlap cases, host/PTX normalization and full-range proof; compiled incorrect-fold/normalization controls are detected.
- 7,682 actual-host helper executions, all192 arena owners, both defer modes and six aliases;694 calls yield noncanonical raw outputs. Three compiled mutations fail numerically.
- 628 actual weak chains against an independent curve oracle,1,242 recovered keys and2,512 executed boundary normalizations;477 final-guard cases,159 unguarded-doubling negative controls, all four constructed scalar exceptions and the source-bound prefix argument. A compiled missing-ZZ-update chain mutation fails against the independent curve oracle.
- Actual host policy:180 capacity cases,30 injected API failures,8 extra paths and five detected mutations.

The complete-chain projection executes actual weak host arithmetic. Its canonical seed, final guard and recovery use the inherited OpenSSL field backend; it does not execute GPU arithmetic. Product PTX is checked separately by a semantic model. The prefix argument remains conditional on a valid nonzero runtime base and correct table construction.

Protocol, SHA, inverse, packed-digit and builder evidence is explicitly transferred from cd3d's qualification and its exact component reports. Whole tree, inverse, SHA and builder headers remain unchanged; all packed helper hashes match the earlier actual-digit test. This is not a fresh whole-control execution on3ac.

Native sm89 evidence:80 registers,32,832 shared bytes,352 stack bytes,552/384 aggregate spill-store/load bytes and15,455 non-NOP instructions. The thirteen-add loop has1,427 instruction sites versus1,594 on cd3d, with no local loads/stores versus104B+104B logical local traffic per candidate. Producer instructions match after relocation; leader polling is retained. The boundary/final region gains86 static sites and44B/60B static local operand bytes, while later consumer code also gains local traffic. These regional counts and aggregate spill totals are not a dynamic whole-candidate cost or speed prediction. The invalid normalization-elision build is a comparison screen, not a performance bound.
