# Early-anchor readiness review

**Ready as a checked experimental successor**, fingerprint
`e575d61ad2a0f4d9b65e8901e741c6bbe7514c1a6df6fc60118cfe537653124a`.
Preserve the currently validating b273a801 submission. Parent owns the frontier,
lean stage, package checks and eventual submission decision.

I recommend this candidate over the checked Z-only and late-anchor four-field
variants. It combines the structural phase-arena redesign with earlier
publication of the next affine anchor after loading the old one. In its
native eleven-add loop0x10660..0x16a70, unpredicated local traffic is20B
stores+24B loads per iteration, totaling484B per lane, versus816B forZ-only
and836B forlate-anchor. Native allocation is80registers/32KiBshared/280Bstack,
376Bstore/276Bload static spill totals and14747non-NOP instructions.

This is a credible experiment, not a measured winner. Against Z-only, it
adds992 logical shared bytes per candidate, another transition join and8KiB
shared capacity. Lower local spill traffic does not prove those costs are
repaid. Three resident CTAs remain an eligibility hypothesis rather than an
observed residency or speedup.

## Fresh evidence

- Unmodified `check_candidate.py` with explicit source, packed selection and
  leaf-pair inverse passed3840 inverse values,6144 SHA256d digests,10262
  recodings,8086 unrankings and256 actual packed window selections. Inverse
  cases use32/64/128/256 lanes, including the repaired64-lane synchronization.
  The retained wrappers invoke the actual new scratch helper bodies.
- The actual shared-digit oracle, with only input/output paths adapted in
  memory, passed17278 scalars,241892 digit comparisons and34556 cold-digit
  comparisons. Wrong-start and wrong-last negative controls each failed as
  expected on all17278 inputs. `check_digits.py` itself has no CLI parser;
  do not pass it purported `--source`/`--output` options.
- Fresh native audit compilation passed sm89 and default flags in the existing
  local VM, at `the original compiler build directory`. Production
  and audit reports together cover all16 closure files; every hash matches the
  candidate and its corresponding compiler staging tree.
- Independently verified that old anchor load precedes new anchor publication
  and then the deferred add. All shared-digit, Z-state and final-guard helper
  bodies match their checked predecessor. Both phase transitions have an
  immediate unconditional CTA join; the unusable per-lane return stays after
  inversion. Existing628-chain/1242-key and477-final-case reports retain their
  source binding and were not overwritten.

The candidate remained byte-identical throughout. Reports are
`frontend-inverse-results.json`, `digits-readiness-results.json`,
`audit-native-results.json` and `readiness-review.json`; the latter preserves
full source and compiler bindings.

## Compatible commands and scope

Run from the benchmark root, choosing a fresh report/output destination to
preserve prior receipts:

```
python3 -B candidates/subset/check_candidate.py \
  --source candidates/subset/research/shared_all_state/early_anchor/candidate \
  --selection packed --inverse-layout leaf-pair --report REPORT.json
python3 -B candidates/subset/research/shared_all_state/check.py \
  --source candidates/subset/research/shared_all_state/early_anchor/candidate \
  --output OUTPUT_DIRECTORY
```

The digit adaptation replaces only the fixed source path and two report paths
in the existing `shared_all_state/check_digits.py`; its actual helper bodies
and independent oracle are unchanged. Exact adaptation instructions are in
`readiness-review.json`.

CPU frontend/inverse checks emulate synchronization and use OpenSSL field
arithmetic. They execute the scratch bodies through wrappers with their own
storage; concurrent reuse of the union across SHA/EC/inverse phases is
structurally reviewed, not GPU-executed. Shared-digit tests use CPU+UBSan.
Native audit compilation does not run the audit or Compute Sanitizer. Its
default arithmetic target is the tree multiplier and inverse, not an execution
of the full shared-state chain or every hot primitive. The unchanged corrected
hot math retains separately hash-bound prior host/PTX evidence.

No device timing, cache/bank-conflict measurement, sanitizer run or achieved
occupancy is claimed. No further unavailable GPU validation is imposed as an
approval prerequisite under the user's authorized experimental-submit rule.
