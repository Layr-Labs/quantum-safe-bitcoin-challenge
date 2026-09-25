# Separate SHA and point front

Split the submitted front kernel into a paired SHA producer and a one-candidate-per-thread point producer. Temporarily reuse denominator slots for SHA z, then replace them with the same front result. Every thread reads its own four z limbs before writing its own sixteen fields. No added allocation. Default-stream ordering, candidate indexing, paired SHA, all arithmetic, inverse topology, finish and exact verification stay unchanged. Check default and compact point launch bounds (256,2)/(256,3), the latter only if spill evidence improves. Production is untouched. Local RTX 3090 only; official submission remains pending.

## Dependency and boundary review

SHA assigns the same epoch pair and lane as the submitted front. It stores A for every active epoch and B only when present. The point grid covers exactly stride=epochs*windows entries, with the final block guarded before memory access. Each point thread reads its own four SHA limbs into scalar arguments before qsb_split_store overwrites that same index; no other thread reads this record. The next default-stream launch starts only after those writes complete. The inverse and tail kernels are unchanged. These are source-level checks, not a substitute for pending GPU verification.

Compiled resources: SHA 72 registers and zero stack; ordinary point 118 registers and zero stack. The compact point variant still uses 144 bytes stack at 80 registers, so it is not selected for GPU trials after the preceding spill-heavy regression.

## Qualification and submission state

Warm candidate: 1784/1784 hits verified, 248.735239 M/s. Fresh preceding three-stage baseline: 1747/1747 verified, 243.494215 M/s. All baseline hits including recovery identifier occur in candidate; gain +2.1524%. Production PTX matches tested candidate SHA256 62d52f6a19ef01e9f8b5b835ca84539cde1b6ec485543decc64dbacb0dd6bcaf. All local sessions are terminal; binaries removed after recording hashes. Production is the new four-stage version and the note/manifest are complete. New submission was rejected by automatic approval review before command execution, including a same-command retry with authorization clarification. No new submission ID exists. Explicit user authorization was requested; do not bypass the rejection. Existing fd16dfa4 official evaluation remains independently pending. See submission-approval-status.json and ../split_pipeline/official-status.json.
