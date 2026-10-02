# Logical review before the existing benchmark

- New enable0 includes no helper and emits an empty carrier token; qualified
  behavior and fingerprint remain unchanged. enable1 requires mode2, so no
  early-offset or non-centered tail silently adopts it.
- For short arithmetic the asm copies a,b,c into PTX temporaries before outputs
  are assigned, so p2 may alias either input just as the existing square does.
  Current call uses separate p2 and p1; p1 is unchanged for x1 publication.
- Numerator construction and opposite-denominator relocation are unchanged.
  Pair mode2 calls offset AFTER first parity, BEFORE p1 becomes x1. m2 is live
  and read after offset. sum has no subsequent consumer. Sum-basis mode would
  reconstruct m2 before the destructive offset and remains default-off.
- First-fold square value is a 288-bit residue. Subtract each 256-bit b,c with
  full eight-limb borrow and signed high extension z9. Second fold uses
  z8+(977*z9), carries z9 into the next word, and sign-extends into z3, matching
  the inherited signed-X3 path. The shortened tail is speculative, not exact
  all-domain proof: hit publication uses unchanged exact recovery/verifier.
- No work counts, enumeration, descriptors, buffer allocations, events, cleanup
  or failure paths change. Zero counts take existing unchanged host branches.
- Host compilation initially rejected const pointer arguments in the unused
  exact fallback. Fixed only by explicit casts required by inherited ModSub
  signatures; neither function mutates b,c. The failed compile had no GPU run.
- Build will generate local executable/stamp only after successful compilation;
  the queued pair requires this stamp. Native image is retained separately and
  is NOT written over production qsb_carrier_sm89.h.

## Same-tool census correction

Earlier static14503 value was NOT counted by the same cuobjdump regex as probe
counts. Re-disassembling the byte-identical default with the identical installed
CUDA12.8 cuobjdump and count pattern gives14520, signed14512(-8), unsigned14504
(-16). LDS/STS/CALL counts unchanged; unsigned has3fewerIMAD and4fewerIADD3.
This is native image evidence, not a throughput claim. Corrected notebook and
JSON records; retained exact scores do not change. Generated-helper scripts
now only emit their named primitive and do not reapply source integrations.
