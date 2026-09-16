/*
 * Sequence-dependent SHA-256 prefix preparation for the QSB pinning candidate.
 * Part of the GPLv3-governed candidate; see COPYING in this directory.
 */
#ifndef QSB_SEQUENCE_MIDSTATE_H
#define QSB_SEQUENCE_MIDSTATE_H

#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>

struct PinningSequencePlan {
    uint32_t midstate[8];
    int suffix_skip;
    int seq_offset;
    int lt_offset;
};

/* The first suffix block can be shared by every locktime in a sequence when
 * it contains all four sequence bytes and none of the locktime bytes. The
 * device still pads using the original TOTAL preimage length. Other layouts
 * keep the original midstate, suffix, and device patching behavior.
 */
static inline PinningSequencePlan prepare_sequence_midstate(
    const uint32_t midstate[8], const uint8_t *suffix, int suffix_len,
    int seq_offset, int lt_offset, uint32_t sequence)
{
    PinningSequencePlan plan;
    memcpy(plan.midstate, midstate, sizeof(plan.midstate));
    plan.suffix_skip = 0;
    plan.seq_offset = seq_offset;
    plan.lt_offset = lt_offset;
    if (suffix_len < 68 || seq_offset < 0 || seq_offset > 60 ||
        lt_offset < 64 || lt_offset > suffix_len - 4)
        return plan;

    uint8_t block[64];
    memcpy(block, suffix, sizeof(block));
    for (int i = 0; i < 4; ++i)
        block[seq_offset + i] = (uint8_t)(sequence >> (8 * i));

    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    for (int i = 0; i < 8; ++i) ctx.h[i] = midstate[i];
    SHA256_Transform(&ctx, block);
    for (int i = 0; i < 8; ++i) plan.midstate[i] = ctx.h[i];
    plan.suffix_skip = 64;
    plan.seq_offset = -1;  /* the sequence bytes are already in the midstate */
    plan.lt_offset -= 64;
    return plan;
}

#endif
