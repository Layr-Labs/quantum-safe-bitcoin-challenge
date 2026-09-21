// QSB fusion integration: audited current interleaved square schedule.
// Derived from the upstream square and xlib fused reduction; see package provenance.
// GPU throughput has not been established by the CPU semantic tests.
/*
* This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
* Copyright (c) 2019 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// ---------------------------------------------------------------------------------
// 256(+64) bits integer CUDA libray for SECPK1
// ---------------------------------------------------------------------------------


#define GRP_SIZE (1024*2)

#define HSIZE ((GRP_SIZE / 2) - 1)

// 64bits lsb negative inverse of P (mod 2^64)
/* Fused XYZZ squaring reduction: xlib's public f297b0f9 / 8afe4bd lineage.
 * Integration: use the pinned current _ModSqr 512-bit product schedule verbatim,
 * then xlib's original carry-complete r*r+e-2q reduction. The older copied square
 * schedule is no longer used here. Fusion defaults to 1 for this experimental
 * candidate; -DQSB_FUSE_SQRADDSUB2=0 restores the baseline device path.
 * CPU PTX semantic tests passed. No author-side GPU performance claim is made.
 */
/* QSB_SHORT_CARRY: stop five rare-carry propagations at the last limb a uniformly
 * distributed operand can reach with probability above 2^-95 per operation.  Sites:
 * _ModMultCore and _ModSqr second fold (keep z3,z4; drop z5..z7), _ModSqrAddSub2 second
 * fold (keep z3,z4; drop z5..z7 and the final-carry fix-up), _ModSub256 and _ModAddLazy
 * K-correction (keep limb 1; drop limbs 2,3).  Each dropped limb needs the preceding
 * 32-bit (multiply) or 64-bit (add/sub) limb to be exactly all-ones / zero after a
 * 2^-31..2^-32 carry, so per-op failure is <= 2^-95 -- far weaker than this header's
 * existing 2^-222 exposures, but ~4e-15 wrong points per 1200 s run at 727 M cand/s.
 * -DQSB_SHORT_CARRY=0 restores the carry-complete f16f893 code byte for byte. */
#ifndef QSB_SHORT_CARRY
#define QSB_SHORT_CARRY 1
#endif
/* QSB_CARRY62 shortens two already-truncated tails one more step:
 *
 *  - In the second pseudo-Mersenne fold, propagate the rare carry through z3
 *    but not z4.  The first fold's high value fits 33 bits, so sfc <= 2.  A
 *    changed result requires z2+sfc >= 2^32 and z3 == 0xffffffff, hence at
 *    most 2/2^64 = 2^-63 for uniform adjacent limbs.
 *  - In _ModSqrAddSub2's split-3p correction, propagate the 3K borrow through
 *    the low 96 bits but not z3/z4.  A changed result requires low96 < 3K,
 *    hence probability < 3K/2^96 < 2^-62.
 *
 * -DQSB_CARRY62=0 restores the QSB_SHORT_CARRY tails used by PR #739.
 * QSB_C31 goes one step further and is only for builds with an exact
 * publication gate (pinning.cu errors if C31 is on without QSB_HOST_GATE):
 *    - second fold stops after z2 (empty tail): differ iff z2+sfc >= 2^32,
 *      at most 2/2^32 = 2^-31 for sfc <= 2.
 *    - split-3p subtracts 3K through 64 bits: differ iff low64 < 3K (~2^-30.4).
 *    - _ModSub256/_ModAddLazy drop the K-correction into t1: differ iff the
 *      64-bit K add/sub carries, ~2^-33 after the 1/2 field-borrow factor.
 * _ModAddLazyOff is left at the short-carry t1 correction: mk is frequently
 * -1, so dropping t1 would be a ~1/2 error, not a 2^-31-class event. */
#ifndef QSB_CARRY62
#define QSB_CARRY62 1
#endif
#ifndef QSB_C31
#define QSB_C31 0
#endif
#if QSB_C31 && QSB_SHORT_CARRY
#define QSB_SECOND_FOLD_TAIL ""
#elif QSB_CARRY62 && QSB_SHORT_CARRY
#define QSB_SECOND_FOLD_TAIL "\taddc.u32 z3, z3, 0;\n"
#else
#define QSB_SECOND_FOLD_TAIL "\taddc.cc.u32 z3, z3, 0;\n\taddc.u32 z4, z4, 0;\n"
#endif
#if QSB_CARRY62 != 0 && QSB_CARRY62 != 1
#error QSB_CARRY62 must be 0 or 1
#endif
#if QSB_C31 != 0 && QSB_C31 != 1
#error QSB_C31 must be 0 or 1
#endif
/* QSB_RP_SQR drops the two 33rd-limb carry captures (f8/g8) of the even/odd
 * pseudo-Mersenne fold in _ModSqr and _ModSqrAddSub2, and the g9 accumulation
 * that consumes them.  Bound: the f-chain is L + 977*(x8,x10,x12,x14) with each
 * addend limb < 977*2^32 < 2^42, so f8 (and symmetrically g8) can only be 1 when
 * the top limb of the low product half lies in the top 977*2^32 of 2^64, i.e. at
 * most 977*2^32/2^64 = 2^-22.03 per operation for uniform limbs; it is exactly 0
 * otherwise, so the fold is bit-identical on all but that measure-2^-22 set.
 * Such an event perturbs one candidate's x by 2^256 mod p, which cannot create a
 * published hit: QSB_RP_SQR is #error-gated on QSB_HOST_GATE in pinning.cu, which
 * re-derives the pubkey and both hashes with OpenSSL before a hit is printed, so
 * the only effect is losing that single candidate.  -DQSB_RP_SQR=0 restores the
 * carry-complete f8/g8/z9 form byte for byte; the arm is also inert whenever
 * QSB_SHORT_CARRY=0. */
#ifndef QSB_RP_SQR
#define QSB_RP_SQR 1
#endif
#if QSB_RP_SQR != 0 && QSB_RP_SQR != 1
#error QSB_RP_SQR must be 0 or 1
#endif
/* The multiply and square folds below reap their ninth accumulator word under
 * QSB_RP_SQR; the fused square-add-sub2 accumulator is the one fold on the
 * chain's critical path that still carries it. Its top is the 33-bit pair
 * {z9,z8}: the first 977 fold leaves z8 with the carry of w7, the split 3p
 * offset adds 3 to it, and the two q subtractions borrow at most one each, so
 * the top never drops below 1 and z9 is nonzero only when that seed carry is
 * set or when z8 >= 2^32-4 before the offset. Both are 2^-30-class events on a
 * 256-bit square, both perturb that candidate's X3 by a multiple of 2^256 mod
 * p, and neither can publish a false hit: QSB_RP_SQR is #error-gated on
 * QSB_HOST_GATE in pinning.cu, which re-derives the pubkey and both hashes
 * with OpenSSL before a hit is printed, so the only effect is losing that one
 * candidate -- the same trade the mul and sqr folds already take. The three
 * arms are the three places the ninth word appears (its seed, the three carry
 * terminators of the add/sub chains, and the final 977 fold); each restores
 * the carry-complete text byte for byte at 0. */
#ifndef QSB_SAS_Z89_REAP
#define QSB_SAS_Z89_REAP 1
#endif
#ifndef QSB_SAS_TOP32
#define QSB_SAS_TOP32 1
#endif
#ifndef QSB_SAS_FOLD32
#define QSB_SAS_FOLD32 1
#endif
#if QSB_RP_SQR && QSB_SHORT_CARRY
#define QSB_F8_CAP ""
#define QSB_MUL_Z8 "\taddc.u32 z8, 0, w7;\n"
#define QSB_SQR_G8 ""
#define QSB_SQR_Z89 "\taddc.u32 z8, 0, w7;\n"
#define QSB_SQR_SF_HEAD "mov.u32 sfq, z8;\n"
#define QSB_SQR_SFC "addc.u32 sfc, 0, 0;\n"
#define QSB_SAS_G8 ""
#if QSB_SAS_Z89_REAP
#define QSB_SAS_Z89 "\taddc.u32 z8, 0, w7;\n"
#else
#define QSB_SAS_Z89 "\taddc.cc.u32 z8, 0, w7; addc.u32 z9, 0, 0;\n"
#endif
#else
#define QSB_F8_CAP "\taddc.u32 f8, 0, 0;\n"
#define QSB_MUL_Z8 "\taddc.u32 z8, f8, w7;\n"
#define QSB_SQR_G8 "\taddc.u32 g8, 0, 0;\n"
#define QSB_SQR_Z89 "\taddc.cc.u32 z8, f8, w7; addc.u32 z9, g8, 0;\n"
#define QSB_SQR_SF_HEAD "mov.b64 {sfl, sfh}, sft;\n"
#define QSB_SQR_SFC "addc.u32 sfc, 0, 0;\n"
#define QSB_SAS_G8 "\taddc.u32 g8, 0, 0;\n"
#define QSB_SAS_Z89 "\taddc.cc.u32 z8, f8, w7; addc.u32 z9, g8, 0;\n"
#endif
#ifndef QSB_SAS_SPLIT3P
#define QSB_SAS_SPLIT3P 1
#endif
#if (QSB_SAS_Z89_REAP || QSB_SAS_TOP32 || QSB_SAS_FOLD32) && \
    !(QSB_RP_SQR && QSB_SHORT_CARRY && QSB_SAS_SPLIT3P)
#error "the SAS ninth-word arms are the reaped split-3p accumulator of QSB_RP_SQR"
#endif
#if QSB_SAS_TOP32 && !QSB_SAS_Z89_REAP
#error "QSB_SAS_TOP32 carries the top limb the reaped seed leaves behind"
#endif
#if QSB_SAS_FOLD32 && !QSB_SAS_TOP32
#error "QSB_SAS_FOLD32 folds the top limb QSB_SAS_TOP32 maintains"
#endif
#if QSB_SAS_TOP32
#define QSB_SAS_TOPADD3 "\taddc.u32 z8, z8, 3;\n"
#define QSB_SAS_TOPSUB  "\tsubc.u32 z8, z8, 0;\n"
#else
#define QSB_SAS_TOPADD3 "\taddc.cc.u32 z8, z8, 3; addc.u32 z9, z9, 0;\n"
#define QSB_SAS_TOPSUB  "\tsubc.cc.u32 z8, z8, 0; subc.u32 z9, z9, 0;\n"
#endif
/* QSB_SAS_Q2SHF: the accumulator's two identical q subtractions become one
 * subtraction of the pre-doubled subtrahend. 2q is exact in nine 32-bit words
 * (nt = q7>>31, n_i = (q_i<<1)|(q_{i-1}>>31), n0 = q0<<1) and the accumulator
 * {z8,z7..z0} is a Z/2^288 register, so (z - q) - q and z - 2q are the same
 * residue: every result limb, the 3K subtraction and the 977 fold see the same
 * bits. Neither form borrows out of z8 -- the value stays above 2^256 until the
 * 3K subtraction, which is exactly the invariant the split-3p arm already
 * maintains -- so the dropped top borrow is dropped identically. The nine
 * funnel shifts read only q, which is live before the square's column sums
 * finish, so the carry-ordered chain behind the square shrinks from eighteen
 * subtractions to nine at the same instruction count.
 * -DQSB_SAS_Q2SHF=0 restores the two chained subtractions byte for byte. */
#ifndef QSB_SAS_Q2SHF
#define QSB_SAS_Q2SHF 1
#endif
#if QSB_SAS_Q2SHF && !QSB_SAS_SPLIT3P
#error "QSB_SAS_Q2SHF subtracts the doubled q of the split-3p accumulator"
#endif
#if QSB_SAS_TOP32
#define QSB_SAS_TOPSUB2 "\tsubc.u32 z8, z8, nt;\n"
#else
#define QSB_SAS_TOPSUB2 "\tsubc.cc.u32 z8, z8, nt; subc.u32 z9, z9, 0;\n"
#endif
#if QSB_SAS_FOLD32
/* sfq == z8 once the ninth word is gone, so the quotient limb is read straight
 * out of z8 and the 33-bit multiply-add collapses to the 32-bit one the sqr
 * fold already emits: same sfz = z0 + 2^32*z8, same sft = 977*z8, same carry. */
#define QSB_SAS_FOLD "\t{ .reg .u64 sfz, sft; .reg .u32 sfc, sfl, sfh;\nmov.b64 sfz, {z0, z8};\nmul.wide.u32 sft, z8, 977;\nadd.cc.u64 sft, sft, sfz;\naddc.u32 sfc, 0, 0;\nmov.b64 {sfl, sfh}, sft;\nmov.u32 z0, sfl;\nadd.cc.u32 z1, z1, sfh;\naddc.cc.u32 z2, z2, sfc; }\n\n"
#else
#define QSB_SAS_FOLD "\t{ .reg .u64 sfz, sft; .reg .u32 sfc, sfq, sfl, sfh;\nmad.lo.u32 sfq, z9, 977, z8;\nmov.b64 sfz, {z0, sfq};\nmul.wide.u32 sft, z8, 977;\nadd.cc.u64 sft, sft, sfz;\naddc.u32 sfc, z9, 0;\nmov.b64 {sfl, sfh}, sft;\nmov.u32 z0, sfl;\nadd.cc.u32 z1, z1, sfh;\naddc.cc.u32 z2, z2, sfc; }\n\n"
#endif
/* QSB_SAS_SPRE: the accumulator's three post-square addends collapse into one.
 * The reaped accumulator {z8,z7..z0} is a Z/2^288 register, so adding e,
 * adding the split-3p offset 3*2^256 and subtracting 2q commute and associate:
 * z + e + 3*2^256 - 2q is the same residue however it is grouped, and every
 * form truncates at exactly the same place (the carry/borrow out of z8 is
 * dropped in all of them). S = e + 3*2^256 - 2q is therefore precomputed as a
 * nine-word two's-complement value from the operand registers alone -- e is
 * %8..%11 and q is %12..%15, both live at entry -- and a single nine-word add
 * chain replaces the eight-word e add plus its top, the nine-word 2q subtract
 * plus its top, and their two carry terminators. The dependent chain behind
 * the square's column sums drops from nineteen carry-ordered steps to nine at
 * one instruction less overall. -DQSB_SAS_SPRE=0 restores the separate add and
 * subtract chains byte for byte. */
#ifndef QSB_SAS_SPRE
#define QSB_SAS_SPRE 1
#endif
#if QSB_SAS_SPRE && !(QSB_SAS_SPLIT3P && QSB_SAS_Q2SHF && QSB_SAS_TOP32)
#error "QSB_SAS_SPRE merges the split-3p offset and the doubled q of the reaped accumulator"
#endif
/* QSB_SAS_LANE64: the same tail in 64-bit lanes. The even fold already leaves
 * f0..f3 as 64-bit registers; the odd fold's eight words are re-paired into
 * four 64-bit lanes (w7 stays the ninth-word seed exactly as QSB_SAS_Z89 reads
 * it), the precomputed addend is assembled in 64-bit lanes from the operand
 * pairs directly, and 3K = 0x300000b73 fits one immediate. Four carry-ordered
 * 64-bit steps replace eight 32-bit ones in the fold merge and in the addend
 * application, the 3K borrow now propagates through lane 1 (word 3) instead of
 * stopping at word 2, and only the two low lanes are ever split into 32-bit
 * halves, for the top fold. Lane arithmetic is the same base-2^64 positional
 * representation of the same 288-bit register: each lane holds words 2i,2i+1
 * and the carry between lanes is the carry out of word 2i+1, so the bits are
 * identical. -DQSB_SAS_LANE64=0 restores the 32-bit word form. */
#ifndef QSB_SAS_LANE64
#define QSB_SAS_LANE64 1
#endif
#if QSB_SAS_LANE64 && !QSB_SAS_SPRE
#error "QSB_SAS_LANE64 applies the single addend QSB_SAS_SPRE precomputes"
#endif
/* QSB_SAS_DBL64: the fused square's cross-product sum is doubled in the eight
 * 64-bit lanes it is packed into anyway, as one carry-propagating self-add,
 * instead of the fifteen descending 32-bit funnel shifts. 2x in radix 2^64 is
 * x+x with the carry out of lane i entering lane i+1; that carry is the same
 * bit the funnel chain moves across the word boundary, so every output word is
 * identical, and the carry out of lane 7 is bit 511 of x -- exactly the bit the
 * top funnel shift drops. The diagonal squares consume the lanes immediately
 * afterwards, so the doubled value is never needed in 32-bit words.
 * -DQSB_SAS_DBL64=0 restores the funnel chain byte for byte. */
#ifndef QSB_SAS_DBL64
#define QSB_SAS_DBL64 1
#endif
/* QSB_SAS_FRD: the doubled-plus-diagonal accumulator already sits in the eight
 * lanes the two 977 folds read -- lanes 0..3 are the even fold's addend and
 * lanes 4..7 the odd fold's -- so only the high half is split into words, for
 * the four 977 multiplies. The low half's split and both halves' repacks are
 * pure renames of the same bits and are dropped, which also retires x0..x7.
 * -DQSB_SAS_FRD=0 restores the split-and-repack form. */
#ifndef QSB_SAS_FRD
#define QSB_SAS_FRD 1
#endif
/* QSB_SAS_Q2LANE: the doubled subtrahend of the lane accumulator is built from
 * the 64-bit operand limbs as q+q, four carry-ordered lane adds whose carry-out
 * is bit 255 of q, instead of one shift, seven funnel shifts, one shift and
 * four repacks of the 32-bit words. Exact: 2q in radix 2^64 is q+q, the lane
 * carries are the word carries the funnel form re-materialises, and the top
 * word nt takes the same bit 255 that shr.u32 nt,q7,31 produced, so the
 * subtraction, the 3K correction and the fold all see the same bits. The
 * operand words q0..q7 are then unused in this arm and their four splits go
 * with them. -DQSB_SAS_Q2LANE=0 restores the funnel form byte for byte. */
#ifndef QSB_SAS_Q2LANE
#define QSB_SAS_Q2LANE 1
#endif
#if QSB_SAS_Q2LANE && !QSB_SAS_LANE64
#error "QSB_SAS_Q2LANE doubles into the 64-bit lanes QSB_SAS_LANE64 subtracts"
#endif
/* QSB_SAS_FOLD64: the top fold in the same lanes. 2^256 == 2^32+977 mod p, so
 * the fold adds z8*(2^32+977), a 65-bit value: its low 64 bits are
 * 977*z8 + (z8<<32) and its 65th bit is that add's carry. Adding the pair at
 * lanes 0 and 1 and letting the carry die in lane 2 keeps one whole lane more
 * propagation than the 32-bit tail it replaces (which stops at word 3), and
 * the result needs no recomposition: all four lanes are the output words.
 * -DQSB_SAS_FOLD64=0 restores the 32-bit fold and its tail. */
#ifndef QSB_SAS_FOLD64
#define QSB_SAS_FOLD64 1
#endif
#if QSB_SAS_FOLD64 && !(QSB_SAS_LANE64 && QSB_SAS_FOLD32)
#error "QSB_SAS_FOLD64 is the 64-bit-lane form of the QSB_SAS_FOLD32 top fold"
#endif
/* QSB_SQR_LANE64: the plain square's fold merge in 64-bit lanes. The even
 * fold already leaves f0..f3 as 64-bit registers and the odd fold's eight
 * words enter one word up, so the eight 32-bit carry-ordered adds
 * z1+=w0 .. z7+=w6 are exactly four 64-bit lane adds: lane 0 takes
 * {0,w0} = w0*2^32, lane 1 takes {w1,w2}, lane 2 {w3,w4}, lane 3 {w5,w6},
 * and w7 stays the ninth-word seed the reaped accumulator reads. Each lane
 * boundary is a word boundary of the same 288-bit sum and the carry into
 * lane i+1 is the carry out of word 2i+1, so every output bit is identical;
 * the four register recompositions of the even fold disappear with it.
 * -DQSB_SQR_LANE64=0 restores the 32-bit word form byte for byte. */
#ifndef QSB_SQR_LANE64
#define QSB_SQR_LANE64 1
#endif
#if QSB_SQR_LANE64 && !(QSB_RP_SQR && QSB_SHORT_CARRY)
#error "QSB_SQR_LANE64 merges the reaped ninth-word seed of QSB_RP_SQR"
#endif
/* QSB_SQR_FOLD64: the plain square's top fold in the same lanes. With
 * 2^256 == 2^32 + 977 (mod p) the fold adds z8*(2^32+977), a 65-bit value
 * whose low 64 bits are 977*z8 + (z8<<32) and whose 65th bit is that add's
 * carry. Adding the low half at lane 0 and the carry at lane 1 leaves the
 * result in the same four lanes, so no recomposition and no 32-bit tail is
 * emitted: the residual carry is propagated one whole lane further than the
 * truncated 32-bit tail (into words 4,5 instead of stopping at word 3), so
 * the arm is strictly tighter than the form it replaces and the accumulator
 * it consumes is unchanged. -DQSB_SQR_FOLD64=0 restores the 32-bit fold and
 * its carry tail. */
#ifndef QSB_SQR_FOLD64
#define QSB_SQR_FOLD64 1
#endif
#if QSB_SQR_FOLD64 && !QSB_SQR_LANE64
#error "QSB_SQR_FOLD64 is the lane form of the fold QSB_SQR_LANE64 leaves in lanes"
#endif
/* QSB_SQL_REAP / QSB_SQL_LANE64 / QSB_SQL_FOLD64 are the same three rewrites
 * applied to the *live* plain square, the QSB_SHORT_CARRY schedule below.
 * That schedule still emitted the carry-complete ninth and tenth words
 * (f8, g8, z9) and merged the two 977 folds with seven carry-ordered 32-bit
 * adds behind eight register splits, while every other fold in this file --
 * the multiply, the fused square-add-subtract -- already reaps the ninth word
 * and adds in 64-bit lanes. The three arms are independent: REAP removes the
 * two ninth-word captures and the tenth word, LANE64 regroups the merge into
 * four 64-bit lane adds, FOLD64 replaces the 32-bit top fold with the 65-bit
 * lane pair. Each arm carries its bound at its site and each restores the
 * schedule it replaces byte for byte at =0. */
#ifndef QSB_SQL_REAP
#define QSB_SQL_REAP 1
#endif
#ifndef QSB_SQL_LANE64
#define QSB_SQL_LANE64 1
#endif
#ifndef QSB_SQL_FOLD64
#define QSB_SQL_FOLD64 1
#endif
#if QSB_SQL_REAP && !(QSB_RP_SQR && QSB_SHORT_CARRY)
#error "QSB_SQL_REAP is the QSB_RP_SQR ninth-word reap inside the short-carry square"
#endif
#if QSB_SQL_LANE64 && !QSB_SQL_REAP
#error "QSB_SQL_LANE64 merges into the ninth-word seed QSB_SQL_REAP leaves"
#endif
#if QSB_SQL_FOLD64 && !QSB_SQL_LANE64
#error "QSB_SQL_FOLD64 is the lane form of the fold QSB_SQL_LANE64 leaves in lanes"
#endif
/* QSB_SAS_XMERGE64 / QSB_SQL_XMERGE64: the 512-bit cross-product sum of the
 * fused square and of the live short-carry square is merged in the eight
 * 64-bit lanes it is doubled in, instead of in sixteen 32-bit words that are
 * then repacked. The even column registers e2..e12 already span words 2..13 at
 * lane alignment and the odd registers o1..o13 straddle lanes, so re-pairing
 * their words one position up turns the fifteen-add word chain plus the
 * sixteen-word pack into seven pairings and seven lane adds. Exact by
 * construction: lane i is words 2i and 2i+1, the inter-lane carry is the carry
 * out of word 2i+1, and the even chain's top register cannot overflow its lane,
 * so word 15 stays zero exactly as the word form seeded it.
 * -DQSB_SAS_XMERGE64=0 / -DQSB_SQL_XMERGE64=0 restore the word merge. */
#ifndef QSB_SAS_XMERGE64
#define QSB_SAS_XMERGE64 1
#endif
#if QSB_SAS_XMERGE64 && !QSB_SAS_DBL64
#error "QSB_SAS_XMERGE64 emits the lane doubling QSB_SAS_DBL64 documents"
#endif
#ifndef QSB_SQL_XMERGE64
#define QSB_SQL_XMERGE64 1
#endif
/* QSB_SQL_DBL64: the live square's cross sum is doubled as eight 64-bit lane
 * self-adds after the pack, replacing the fifteen descending funnel shifts.
 * 2*X is the same value either way and both forms drop the bit above word 15,
 * which is unreachable because 2*cross + diagonal is below 2^512. It is also
 * the fallback arm of QSB_SQL_XMERGE64.
 * -DQSB_SQL_DBL64=0 restores the funnel chain byte for byte. */
#ifndef QSB_SQL_DBL64
#define QSB_SQL_DBL64 1
#endif
/* QSB_SQL_FRD: the live square's fold addends stay in the lanes the diagonal
 * chain left them in -- fr0..fr3 are d0..d3 and h0..h3 are d4..d7 -- so only
 * the high four lanes are split into the words the 977 multiplies index.
 * The values are identical; twelve register moves disappear.
 * -DQSB_SQL_FRD=0 restores the split-and-repack form byte for byte. */
#ifndef QSB_SQL_FRD
#define QSB_SQL_FRD 1
#endif
#if (QSB_SQL_XMERGE64 || QSB_SQL_DBL64 || QSB_SQL_FRD) && !QSB_SHORT_CARRY
#error "the QSB_SQL_ arms regroup the short-carry square QSB_SHORT_CARRY selects"
#endif
/* QSB_MUL_XMERGE64 / QSB_MUL_LANE64 / QSB_MUL_FOLD64: the same three lane-form
 * rewrites, applied to the live field multiply -- the last schedule in this
 * file that still merged its 512-bit cross-product sum and its 977 fold in
 * sixteen 32-bit words while both squares already work in 64-bit lanes.
 * XMERGE64 merges the odd column chain into the even chain's own lanes and
 * lands the result directly in the fold's r0..r3 / h0..h3, so the fifteen
 * word adds, the fifteen splits and the eight-word repack become seven
 * splits, eight pairings and eight lane adds; LANE64 turns the seven-add
 * fold merge into four lane adds; FOLD64 adds the 65-bit top fold as a lane
 * pair instead of a 32-bit triple plus the short-carry tail, propagating the
 * residual carry one whole lane further than the form it replaces. Every arm
 * is exact because a lane is a word pair and the carry between lanes is the
 * carry out of the odd word; each one restores the schedule it replaces byte
 * for byte at =0, and the two upper arms are #error-gated on the lane data
 * they consume. The multiply runs ten times per point addition against the
 * squares' two, so the same rewrite is worth more here than where it was
 * first proven. */
#ifndef QSB_MUL_XMERGE64
#define QSB_MUL_XMERGE64 1
#endif
#ifndef QSB_MUL_LANE64
#define QSB_MUL_LANE64 1
#endif
#ifndef QSB_MUL_FOLD64
#define QSB_MUL_FOLD64 1
#endif
#if (QSB_MUL_XMERGE64 || QSB_MUL_LANE64 || QSB_MUL_FOLD64) && !QSB_SHORT_CARRY
#error "the QSB_MUL_ arms regroup the short-carry multiply QSB_SHORT_CARRY selects"
#endif
#if QSB_MUL_LANE64 && !QSB_MUL_XMERGE64
#error "QSB_MUL_LANE64 uses the zero register and lane pairs QSB_MUL_XMERGE64 declares"
#endif
#if QSB_MUL_FOLD64 && !QSB_MUL_LANE64
#error "QSB_MUL_FOLD64 is the lane form of the fold QSB_MUL_LANE64 leaves in lanes"
#endif
/* QSB_MUL_FOLDT2 / QSB_SQL_FOLDT2 / QSB_SAS_FOLDT2: the third carry step of
 * the lane top fold. The lane forms above add z8*(2^32+977) as a 65-bit pair
 * -- low half in lane 0, the 65th bit in lane 1 -- and then spend one more
 * instruction propagating the residual carry into lane 2 (words 4,5). Their
 * own comments record that this is three words *beyond* the word-3 terminator
 * of the 32-bit tails they replaced, i.e. safety the word form never bought.
 * The step is reached only when the 65th bit is set and the receiving lane is
 * all ones: the bit needs z8 >= 2^32-976 (below 2^-22.1 for a field-random
 * quotient limb) and the lane needs 2^-64, so the two forms differ with
 * probability below 2^-86 per reduction -- nine orders below the 2^-31 fold
 * exposure QSB_C31 already ships behind the host publication gate, and below
 * even the carry-complete build's own 2^-63 QSB_CARRY62 tails. The multiply
 * runs ten times per point addition, so this is ten instructions an addition
 * plus one for each square. Each arm restores its third step byte for byte at
 * =0 and is inert wherever its lane fold is. */
#ifndef QSB_MUL_FOLDT2
#define QSB_MUL_FOLDT2 1
#endif
#ifndef QSB_SQL_FOLDT2
#define QSB_SQL_FOLDT2 1
#endif
#ifndef QSB_SAS_FOLDT2
#define QSB_SAS_FOLDT2 1
#endif
/* QSB_MUL_FOLD65 / QSB_SQL_FOLD65 / QSB_SAS_FOLD65: the 65th bit of the lane
 * top fold, at the three live reductions (the multiply, the short-carry plain
 * square, the fused square-add-subtract). The lane forms above build z8*K
 * with K = 2^32+977 as a 65-bit quantity: a 64-bit low half in lane 0 and a
 * one-bit high half that lane 1 receives through a captured carry. Building
 * that pair costs the zero-paired shift operand {zr,z8}, a carry-producing
 * 64-bit add and the carry capture; consuming it costs nothing extra, because
 * lane 1 has to take the carry out of lane 0 in any case.
 *
 * The 65th bit is the carry of 977*z8 + (z8<<32), so in words the fold's low
 * half is exactly { lo32(977*z8), hi32(977*z8) + z8 } -- hi32(977*z8) < 2^10
 * because 977*z8 < 2^42 -- and the 65th bit is the carry of that one 32-bit
 * word add. These arms emit the word form and drop that carry: one 32-bit add
 * replaces the paired operand, the 64-bit carry-producing add and the capture.
 *
 * Exactness: the two forms differ only when z8*K >= 2^64, i.e. when the
 * quotient limb reaches 2^32 - 977, so at most 977/2^32 = 2^-22.04 per
 * reduction for a uniformly distributed ninth word -- the same event class,
 * at the same rate, as the QSB_RP_SQR ninth-word reap this build already
 * ships behind the host publication gate (977*2^32/2^64 = 2^-22.03), and far
 * above the 2^-86 lane-2 step of QSB_*_FOLDT2. Such an event perturbs one
 * candidate's x by 2^64 mod p; it cannot publish a wrong hit, because the
 * host gate recovers and re-hashes every hit exactly before publication.
 * Each arm is independent -- the three reductions share no register -- is
 * inert wherever its lane fold is off, and restores the 65-bit pair byte for
 * byte at 0. Both FOLDT2 arms stay in force inside these forms, so the lane-2
 * terminator is whatever QSB_*_FOLDT2 selects. */
#ifndef QSB_MUL_FOLD65
#define QSB_MUL_FOLD65 1
#endif
#ifndef QSB_SQL_FOLD65
#define QSB_SQL_FOLD65 1
#endif
#ifndef QSB_SAS_FOLD65
#define QSB_SAS_FOLD65 1
#endif
#if QSB_MUL_FOLD65 && !QSB_MUL_FOLD64
#error "QSB_MUL_FOLD65 is the word form of the lane fold QSB_MUL_FOLD64 emits"
#endif
#if QSB_SQL_FOLD65 && !QSB_SQL_FOLD64
#error "QSB_SQL_FOLD65 is the word form of the lane fold QSB_SQL_FOLD64 emits"
#endif
#if QSB_SAS_FOLD65 && !QSB_SAS_FOLD64
#error "QSB_SAS_FOLD65 is the word form of the lane fold QSB_SAS_FOLD64 emits"
#endif
#if (QSB_MUL_FOLD65 || QSB_SQL_FOLD65 || QSB_SAS_FOLD65) && \
    !(QSB_RP_SQR && QSB_SHORT_CARRY)
#error "the FOLD65 arms drop the fold bit in the QSB_RP_SQR exposure class"
#endif
/* QSB_SAS_K3T1: the split-3p correction's borrow, at the width QSB_C31 states.
 * The word form under QSB_C31 subtracts 3K = 0x3_00000B73 through z0 and z1
 * and stops -- "split-3p subtracts 3K through 64 bits: differ iff low64 < 3K
 * (~2^-30.4)" -- while the 64-bit lane form carries the borrow one lane
 * further, into words 2 and 3, because a lane pair is the cheapest unit it had.
 * Dropping that second lane leaves exactly the 64-bit window QSB_C31 declares
 * and the host gate covers: the two forms differ only when the accumulator's
 * low 64 bits are below 3K, the same event, at the same rate, as the word arm
 * this build already selects. -DQSB_SAS_K3T1=0 restores the two-lane borrow. */
#ifndef QSB_SAS_K3T1
#define QSB_SAS_K3T1 1
#endif
#if QSB_SAS_K3T1 && !QSB_C31
#error "QSB_SAS_K3T1 is the QSB_C31 64-bit split-3p window in lane form"
#endif
/* QSB_MUL_REAP: the ninth-word carry capture of the *multiply*'s even fold --
 * the one live reduction that still pays it. QSB_RP_SQR reaped f8/g8 in the
 * plain square and QSB_SQL_REAP/QSB_SAS_Z89_REAP in the short-carry and fused
 * squares, but the short-carry multiply below still emits "addc.u32 f8, 0, 0"
 * between its two fold chains and seeds z8 from it. The bound is the one
 * QSB_RP_SQR states, on the same chain shape: the even fold is
 * L + 977*(x8,x10,x12,x14) with L < 2^256 and the four addends below
 * 977*2^32 < 2^42 at lanes 0..3, so their total is under 2^234 and the chain
 * can only carry out of lane 3 when L >= 2^256 - 2^234, at most 2^-22 per
 * multiply for a uniformly distributed low half -- the same event class, at
 * the same rate, as the ninth-word reap and the FOLD65 arms this build already
 * ships behind the host publication gate. Such an event perturbs that one
 * candidate's value by 2^256 mod p and cannot publish a wrong hit, because
 * pinning.cu re-derives the pubkey and both hashes with OpenSSL before a hit
 * is printed; the only effect is losing that single candidate. The multiply
 * runs ten times per point addition, so this is ten instructions and ten
 * carry-chain hand-offs an addition. -DQSB_MUL_REAP=0 restores the capture and
 * the f8 seed byte for byte. */
#ifndef QSB_MUL_REAP
#define QSB_MUL_REAP 1
#endif
#if QSB_MUL_REAP && !(QSB_RP_SQR && QSB_SHORT_CARRY)
#error "QSB_MUL_REAP is the QSB_RP_SQR ninth-word reap inside the short-carry multiply"
#endif
#if QSB_MUL_REAP
#define QSB_MUL_F8CAP ""
#define QSB_MUL_Z8SEED "\taddc.u32 z8, 0, w7;"
#else
#define QSB_MUL_F8CAP "\taddc.u32 f8, 0, 0;\n"
#define QSB_MUL_Z8SEED "\taddc.u32 z8, f8, w7;"
#endif
/* QSB_X3_DBL2 / QSB_X3_TAIL: the two remaining word-form steps of the fused
 * X3 = a + b - 2c of the XYZZ addition, the one field helper on the addition's
 * critical path that was never brought to the conventions the rest of this
 * file runs on.
 *
 *  - QSB_X3_DBL2 builds the doubled subtrahend as four carry-ordered 64-bit
 *    self-adds plus one carry capture instead of four shifts, three right
 *    shifts, three ors and one top shift. 2c in radix 2^64 is c+c: lane i is
 *    2*c_i mod 2^64 = c_i<<1 plus the carry out of lane i-1, that carry is
 *    c_{i-1}>>63, and a lane can never re-overflow because 2*c_i mod 2^64 is
 *    even and cannot be 2^64-1. The capture d4 is the carry out of lane 3,
 *    which is c3>>63 -- bit for bit the funnel form's five words, at five
 *    instructions instead of eleven. It is the doubling QSB_SAS_Q2LANE already
 *    proves on the fused square's subtrahend.
 *  - QSB_X3_TAIL stops the fold correction at the limb QSB_SHORT_CARRY stops
 *    every other correction at. The fold adds k = t4*K with t4 <= 3, so
 *    k <= 3K < 2^33.6; the carry it produces reaches limb 2 only when t0 is
 *    within 2^33.6 of 2^64 *and* t1 is 2^64-1, at most 2^-94.4 per addition --
 *    below the <= 2^-95 the short-carry arms of _ModSub256 and _ModAddLazy
 *    already declare, and far below the 2^-30 window QSB_C31 ships. Two
 *    instructions an addition.
 *
 * Each arm restores its word form byte for byte at 0. */
#ifndef QSB_X3_DBL2
#define QSB_X3_DBL2 1
#endif
#ifndef QSB_X3_TAIL
#define QSB_X3_TAIL 1
#endif
#if QSB_X3_TAIL && !QSB_SHORT_CARRY
#error "QSB_X3_TAIL is the QSB_SHORT_CARRY tail convention inside the fused X3"
#endif
#if QSB_X3_DBL2
#define QSB_X3_2C "add.cc.u64 d0,%12,%12;\naddc.cc.u64 d1,%13,%13;\naddc.cc.u64 d2,%14,%14;\naddc.cc.u64 d3,%15,%15;\naddc.u64 d4,0,0;\n"
#else
#define QSB_X3_2C "shl.b64 d0,%12,1;\nshl.b64 d1,%13,1; shr.u64 k,%12,63; or.b64 d1,d1,k;\nshl.b64 d2,%14,1; shr.u64 k,%13,63; or.b64 d2,d2,k;\nshl.b64 d3,%15,1; shr.u64 k,%14,63; or.b64 d3,d3,k;\nshr.u64 d4,%15,63;\n"
#endif
#if QSB_X3_TAIL
#define QSB_X3_FOLDTAIL "add.cc.u64 t0,t0,k; addc.u64 t1,t1,0;\n"
#else
#define QSB_X3_FOLDTAIL "add.cc.u64 t0,t0,k; addc.cc.u64 t1,t1,0; addc.cc.u64 t2,t2,0;\naddc.u64 t3,t3,0;\n"
#endif
#ifndef QSB_FUSE_SQRADDSUB2
#define QSB_FUSE_SQRADDSUB2 1
#endif
#if QSB_FUSE_SQRADDSUB2 != 0 && QSB_FUSE_SQRADDSUB2 != 1
#error QSB_FUSE_SQRADDSUB2 must be 0 or 1
#endif

#define MM64 0xD838091DD2253531ULL


// We need 1 extra block for ModInv
#define NBBLOCK 5
#define BIFULLSIZE 40

// Assembly directives
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) );

#define UMULLO(lo,a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi,a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r,a,b,c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADDC(r,a,b,c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADD(r,a,b,c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADDS(r,a,b,c) asm volatile ("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

// SECPK1 endomorphism constants
//__device__ __constant__ uint64_t _beta[] = { 0xC1396C28719501EEULL, 0x9CF0497512F58995ULL, 0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL };
//__device__ __constant__ uint64_t _beta2[] = { 0x3EC693D68E6AFA40ULL, 0x630FB68AED0A766AULL, 0x919BB86153CBCB16ULL, 0x851695D49A83F8EFULL };


// ---------------------------------------------------------------------------------------

#define _IsPositive(x) (((int64_t)(x[4]))>=0LL)
#define _IsNegative(x) (((int64_t)(x[4]))<0LL)
#define _IsEqual(a,b)  ((a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a)     ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a)      ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define IDX threadIdx.x

#define __sright128(a,b,n) ((a)>>(n))|((b)<<(64-(n)))
#define __sleft128(a,b,n) ((b)<<(n))|((a)>>(64-(n)))

// ---------------------------------------------------------------------------------------

#define AddP(r) { \
  UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  UADD1(r[4], 0ULL);}

// ---------------------------------------------------------------------------------------

#define Add2(r,a,b)  {\
  UADDO(r[0], a[0], b[0]); \
  UADDC(r[1], a[1], b[1]); \
  UADDC(r[2], a[2], b[2]); \
  UADDC(r[3], a[3], b[3]); \
  UADD(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define SubP(r) { \
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  USUB1(r[4], 0ULL);}

// ---------------------------------------------------------------------------------------

#define Sub2(r,a,b)  {\
  USUBO(r[0], a[0], b[0]); \
  USUBC(r[1], a[1], b[1]); \
  USUBC(r[2], a[2], b[2]); \
  USUBC(r[3], a[3], b[3]); \
  USUB(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define Sub1(r,a) {\
  USUBO1(r[0], a[0]); \
  USUBC1(r[1], a[1]); \
  USUBC1(r[2], a[2]); \
  USUBC1(r[3], a[3]); \
  USUB1(r[4], a[4]);}

// ---------------------------------------------------------------------------------------

#define Neg(r) {\
USUBO(r[0],0ULL,r[0]); \
USUBC(r[1],0ULL,r[1]); \
USUBC(r[2],0ULL,r[2]); \
USUBC(r[3],0ULL,r[3]); \
USUB(r[4],0ULL,r[4]); }

// ---------------------------------------------------------------------------------------

#define UMult(r, a, b) {\
  UMULLO(r[0],a[0],b); \
  UMULLO(r[1],a[1],b); \
  MADDO(r[1], a[0],b,r[1]); \
  UMULLO(r[2],a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3],a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  MADD(r[4], a[3], b, 0ULL);}

// ---------------------------------------------------------------------------------------

#define Load(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3]; \
  (r)[4] = (a)[4];}

// ---------------------------------------------------------------------------------------

#define _LoadI64(r, a) {\
  (r)[0] = a; \
  (r)[1] = a>>63; \
  (r)[2] = (r)[1]; \
  (r)[3] = (r)[1]; \
  (r)[4] = (r)[1];}
// ---------------------------------------------------------------------------------------

#define Load256(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3];}

// ---------------------------------------------------------------------------------------

#define Load256A(r, a) {\
  (r)[0] = (a)[IDX]; \
  (r)[1] = (a)[IDX+blockDim.x]; \
  (r)[2] = (a)[IDX+2*blockDim.x]; \
  (r)[3] = (a)[IDX+3*blockDim.x];}

// ---------------------------------------------------------------------------------------

#define Store256A(r, a) {\
  (r)[IDX] = (a)[0]; \
  (r)[IDX+blockDim.x] = (a)[1]; \
  (r)[IDX+2*blockDim.x] = (a)[2]; \
  (r)[IDX+3*blockDim.x] = (a)[3];}

// ---------------------------------------------------------------------------------------

__device__ void _ShiftR62(uint64_t *r)
{

    r[0] = (r[1] << 2) | (r[0] >> 62);
    r[1] = (r[2] << 2) | (r[1] >> 62);
    r[2] = (r[3] << 2) | (r[2] >> 62);
    r[3] = (r[4] << 2) | (r[3] >> 62);
    // With sign extent
    r[4] = (int64_t)(r[4]) >> 62;

}

__device__ void _ShiftR62(uint64_t dest[5], uint64_t r[5], uint64_t carry)
{

    dest[0] = (r[1] << 2) | (r[0] >> 62);
    dest[1] = (r[2] << 2) | (r[1] >> 62);
    dest[2] = (r[3] << 2) | (r[2] >> 62);
    dest[3] = (r[4] << 2) | (r[3] >> 62);
    dest[4] = (carry << 2) | (r[4] >> 62);

}

// ---------------------------------------------------------------------------------------

__device__ void _IMult(uint64_t *r, uint64_t *a, int64_t b)
{

    uint64_t t[NBBLOCK];

    // Make b positive
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    } else {
        Load(t, a);
    }

    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADD(r[4], t[3], b, r[4]);

}

__device__ uint64_t _IMultC(uint64_t *r, uint64_t *a, int64_t b)
{

    uint64_t t[NBBLOCK];
    uint64_t carry;

    // Make b positive
    if (b < 0) {
        b = -b;
        USUBO(t[0], 0ULL, a[0]);
        USUBC(t[1], 0ULL, a[1]);
        USUBC(t[2], 0ULL, a[2]);
        USUBC(t[3], 0ULL, a[3]);
        USUB(t[4], 0ULL, a[4]);
    } else {
        Load(t, a);
    }

    UMULLO(r[0], t[0], b);
    UMULLO(r[1], t[1], b);
    MADDO(r[1], t[0], b, r[1]);
    UMULLO(r[2], t[2], b);
    MADDC(r[2], t[1], b, r[2]);
    UMULLO(r[3], t[3], b);
    MADDC(r[3], t[2], b, r[3]);
    UMULLO(r[4], t[4], b);
    MADDC(r[4], t[3], b, r[4]);
    MADDS(carry, t[4], b, 0ULL);

    return carry;

}

// ---------------------------------------------------------------------------------------

__device__ void _MulP(uint64_t *r, uint64_t a)
{

    uint64_t ah;
    uint64_t al;

    UMULLO(al, a, 0x1000003D1ULL);
    UMULHI(ah, a, 0x1000003D1ULL);

    USUBO(r[0], 0ULL, al);
    USUBC(r[1], 0ULL, ah);
    USUBC(r[2], 0ULL, 0ULL);
    USUBC(r[3], 0ULL, 0ULL);
    USUB(r[4], a, 0ULL);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModNeg256(uint64_t *r, uint64_t *a)
{

    uint64_t t[4];
    USUBO(t[0], 0ULL, a[0]);
    USUBC(t[1], 0ULL, a[1]);
    USUBC(t[2], 0ULL, a[2]);
    USUBC(t[3], 0ULL, a[3]);
    UADDO(r[0], t[0], 0xFFFFFFFEFFFFFC2FULL);
    UADDC(r[1], t[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(r[2], t[2], 0xFFFFFFFFFFFFFFFFULL);
    UADD(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModNeg256(uint64_t *r)
{

    uint64_t t[4];
    USUBO(t[0], 0ULL, r[0]);
    USUBC(t[1], 0ULL, r[1]);
    USUBC(t[2], 0ULL, r[2]);
    USUBC(t[3], 0ULL, r[3]);
    UADDO(r[0], t[0], 0xFFFFFFFEFFFFFC2FULL);
    UADDC(r[1], t[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC(r[2], t[2], 0xFFFFFFFFFFFFFFFFULL);
    UADD(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);

}

__device__ __forceinline__ void _ModAdd256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nadd.cc.u64 s0,%4,%8;\naddc.cc.u64 s1,%5,%9; addc.cc.u64 s2,%6,%10; addc.cc.u64 s3,%7,%11;\naddc.u64 t4,0,0;\nsub.cc.u64 t0,s0,0xFFFFFFFEFFFFFC2F;\nsubc.cc.u64 t1,s1,0xFFFFFFFFFFFFFFFF;\nsubc.cc.u64 t2,s2,0xFFFFFFFFFFFFFFFF;\nsubc.cc.u64 t3,s3,0xFFFFFFFFFFFFFFFF;\nsubc.u64 t4,t4,0;\nsetp.ge.s64 choose,t4,0;\nselp.u64 %0,t0,s0,choose; selp.u64 %1,t1,s1,choose;\nselp.u64 %2,t2,s2,choose; selp.u64 %3,t3,s3,choose;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

#ifndef QSB_LAZY
#define QSB_LAZY 1
#endif
#if QSB_LAZY
// r = a - b, plus p when the subtraction borrows. Adding p modulo 2^256 is
// the same as subtracting K = 2^256 - p = 2^32 + 977, so the borrow mask only
// has to select one limb-sized constant instead of four limbs of p. The
// result is bit-identical to the original formulation.
#if QSB_C31 && QSB_SHORT_CARRY
__device__ __forceinline__ void _ModSub256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nsub.cc.u64 t0,%4,%8;\nsubc.cc.u64 t1,%5,%9; subc.cc.u64 t2,%6,%10; subc.cc.u64 t3,%7,%11;\nsubc.u64 k,0,0; and.b64 k,k,0x1000003D1;\nsub.u64 t0,t0,k;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

#elif QSB_SHORT_CARRY
__device__ __forceinline__ void _ModSub256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nsub.cc.u64 t0,%4,%8;\nsubc.cc.u64 t1,%5,%9; subc.cc.u64 t2,%6,%10; subc.cc.u64 t3,%7,%11;\nsubc.u64 k,0,0; and.b64 k,k,0x1000003D1;\nsub.cc.u64 t0,t0,k; subc.u64 t1,t1,0;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

#else
__device__ __forceinline__ void _ModSub256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nsub.cc.u64 t0,%4,%8;\nsubc.cc.u64 t1,%5,%9; subc.cc.u64 t2,%6,%10; subc.cc.u64 t3,%7,%11;\nsubc.u64 k,0,0; and.b64 k,k,0x1000003D1;\nsub.cc.u64 t0,t0,k; subc.cc.u64 t1,t1,0; subc.cc.u64 t2,t2,0;\nsubc.u64 t3,t3,0;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

#endif
__device__ __forceinline__ void _ModSub256(uint64_t *r,uint64_t *b) { _ModSub256(r,r,b); }

// Lazy add: r = a + b reduced only by folding the 2^256 carry as K. The
// result is in [0, 2^256) and congruent mod p, which every consumer in the
// fixed-base chain (_ModMultCore, _ModSqr, _ModSub256, this routine) accepts.
// The fold can carry again only when r >= 2^256 - K after a carry, a 2^-223
// event for field-random inputs, ignored like the existing 2^-224 exposure
// of _ModSub256 to inputs above p.
#if QSB_C31 && QSB_SHORT_CARRY
__device__ __forceinline__ void _ModAddLazy(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nadd.cc.u64 t0,%4,%8;\naddc.cc.u64 t1,%5,%9; addc.cc.u64 t2,%6,%10; addc.cc.u64 t3,%7,%11;\naddc.u64 k,0,0; neg.s64 k,k; and.b64 k,k,0x1000003D1;\nadd.u64 t0,t0,k;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

#elif QSB_SHORT_CARRY
__device__ __forceinline__ void _ModAddLazy(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nadd.cc.u64 t0,%4,%8;\naddc.cc.u64 t1,%5,%9; addc.cc.u64 t2,%6,%10; addc.cc.u64 t3,%7,%11;\naddc.u64 k,0,0; neg.s64 k,k; and.b64 k,k,0x1000003D1;\nadd.cc.u64 t0,t0,k; addc.u64 t1,t1,0;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

#else
__device__ __forceinline__ void _ModAddLazy(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nadd.cc.u64 t0,%4,%8;\naddc.cc.u64 t1,%5,%9; addc.cc.u64 t2,%6,%10; addc.cc.u64 t3,%7,%11;\naddc.u64 k,0,0; neg.s64 k,k; and.b64 k,k,0x1000003D1;\nadd.cc.u64 t0,t0,k; addc.cc.u64 t1,t1,0; addc.cc.u64 t2,t2,0;\naddc.u64 t3,t3,0;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}

#endif
#ifndef QSB_YOFF
#define QSB_YOFF 0
#endif
#if QSB_YOFF
// Anchor sum on offset ordinates (QSB_YOFF): a = y2 + c, b = yoff + c with c = (K-1)/2, both in
// [c, 2^256-1-c]. Returns r = a + b - (K-1) (== y2 + yoff mod p) in [0,2^256): with t = (a+b) mod
// 2^256 and k the carry, r = t - (K-1) if k = 0 (a+b >= 2c, so no wrap) and r = t + 1 if k = 1
// (2^256 == K). The correction keeps its borrow/carry through limb 1 (short carry): dropped only
// if limb0 < K-1 (2^-31) and limb1 == 0 (2^-64), <= 2^-95 per operation.
__device__ __forceinline__ void _ModAddLazyOff(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,k,mk,c0;\nadd.cc.u64 t0,%4,%8;\naddc.cc.u64 t1,%5,%9; addc.cc.u64 t2,%6,%10; addc.cc.u64 t3,%7,%11;\naddc.u64 k,0,0; sub.u64 mk,k,1; and.b64 c0,mk,0xFFFFFFFEFFFFFC2F; add.u64 c0,c0,1;\nadd.cc.u64 t0,t0,c0; addc.u64 t1,t1,mk;\nmov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}
#endif
// Fused X3 = a + b - 2c (mod p) for the XYZZ addition (R^2 + PPP - 2V), in
// one carry chain: t = a + b + 2p - 2c lies in [0, 2^258) (the 2^-224 case
// c > p + (a+b)/2 is ignored), and its top two bits h fold as h*K. A second
// carry needs t mod 2^256 >= 2^256 - 2^34, a 2^-222 event, ignored.
__device__ __forceinline__ void _ModX3Fused(uint64_t *r, const uint64_t *a, const uint64_t *b, const uint64_t *c) {
    uint64_t r0,r1,r2,r3;
    asm("{\n.reg .u64 t0,t1,t2,t3,t4,s0,s1,s2,s3,d0,d1,d2,d3,d4,k;\n.reg .pred choose;\nadd.cc.u64 t0,%4,%8;\naddc.cc.u64 t1,%5,%9; addc.cc.u64 t2,%6,%10; addc.cc.u64 t3,%7,%11;\naddc.u64 t4,0,0;\nadd.cc.u64 t0,t0,0xFFFFFFFDFFFFF85E;\naddc.cc.u64 t1,t1,0xFFFFFFFFFFFFFFFF;\naddc.cc.u64 t2,t2,0xFFFFFFFFFFFFFFFF;\naddc.cc.u64 t3,t3,0xFFFFFFFFFFFFFFFF; addc.u64 t4,t4,1;\n"
        QSB_X3_2C
        "sub.cc.u64 t0,t0,d0; subc.cc.u64 t1,t1,d1;\nsubc.cc.u64 t2,t2,d2; subc.cc.u64 t3,t3,d3; subc.u64 t4,t4,d4;\nmul.lo.u64 k,t4,0x1000003D1;\n"
        QSB_X3_FOLDTAIL
        "mov.u64 %0,t0; mov.u64 %1,t1; mov.u64 %2,t2; mov.u64 %3,t3;\n}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]),"l"(c[0]),"l"(c[1]),"l"(c[2]),"l"(c[3]));
    r[0]=r0;r[1]=r1;r[2]=r2;r[3]=r3;
}
#else
__device__ void _ModSub256(uint64_t *r, uint64_t *a, uint64_t *b)
{
    uint64_t t;
    uint64_t T[4];

    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(t, 0ULL, 0ULL);

    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;

    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);

}

// ---------------------------------------------------------------------------------------

__device__ void _ModSub256(uint64_t *r, uint64_t *b)
{

    uint64_t t;
    uint64_t T[4];
    USUBO(r[0], r[0], b[0]);
    USUBC(r[1], r[1], b[1]);
    USUBC(r[2], r[2], b[2]);
    USUBC(r[3], r[3], b[3]);
    USUB(t, 0ULL, 0ULL);
    T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
    T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
    T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
    UADDO1(r[0], T[0]);
    UADDC1(r[1], T[1]);
    UADDC1(r[2], T[2]);
    UADD1(r[3], T[3]);

}
#endif

// ---------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t _CTZ(uint64_t x)
{
    uint32_t n;
    asm("{\n\t"
        " .reg .u64 tmp;\n\t"
        " brev.b64 tmp, %1;\n\t"
        " clz.b64 %0, tmp;\n\t"
        "}"
        : "=r"(n) : "l"(x));
    return n;
}

// ---------------------------------------------------------------------------------------
#define SWAP(tmp,x,y) tmp = x; x = y; y = tmp;
#define MSK62 0x3FFFFFFFFFFFFFFF

__device__ void _DivStep62(uint64_t u[5], uint64_t v[5],
                           int32_t *pos,
                           int64_t *uu, int64_t *uv,
                           int64_t *vu, int64_t *vv)
{


    // u' = (uu*u + uv*v) >> bitCount
    // v' = (vu*u + vv*v) >> bitCount
    // Do not maintain a matrix for r and s, the number of
    // 'added P' can be easily calculated

    *uu = 1; *uv = 0;
    *vu = 0; *vv = 1;

    uint32_t bitCount = 62;
    uint32_t zeros;
    uint64_t u0 = u[0];
    uint64_t v0 = v[0];

    // Extract 64 MSB of u and v
    // u and v must be positive
    uint64_t uh, vh;
    int64_t w, x, y, z;
    bitCount = 62;

    while (*pos > 0 && (u[*pos] | v[*pos]) == 0)
        (*pos)--;
    if (*pos == 0) {

        uh = u[0];
        vh = v[0];

    } else {

        uint32_t s = __clzll(u[*pos] | v[*pos]);
        if (s == 0) {
            uh = u[*pos];
            vh = v[*pos];
        } else {
            uh = __sleft128(u[*pos - 1], u[*pos], s);
            vh = __sleft128(v[*pos - 1], v[*pos], s);
        }

    }


    while (true) {

        // Use a sentinel bit to count zeros only up to bitCount
        zeros = _CTZ(v0 | (1ULL << bitCount));

        v0 >>= zeros;
        vh >>= zeros;
        *uu <<= zeros;
        *uv <<= zeros;
        bitCount -= zeros;

        if (bitCount == 0)
            break;

        if (vh < uh) {
            SWAP(w, uh, vh);
            SWAP(x, u0, v0);
            SWAP(y, *uu, *vu);
            SWAP(z, *uv, *vv);
        }

        vh -= uh;
        v0 -= u0;
        *vv -= *uv;
        *vu -= *uu;

    }

}

__device__ void _MatrixVecMulHalf(uint64_t dest[5], uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, uint64_t *carry)
{

    uint64_t t1[NBBLOCK];
    uint64_t t2[NBBLOCK];
    uint64_t c1, c2;

    c1 = _IMultC(t1, u, _11);
    c2 = _IMultC(t2, v, _12);

    UADDO(dest[0], t1[0], t2[0]);
    UADDC(dest[1], t1[1], t2[1]);
    UADDC(dest[2], t1[2], t2[2]);
    UADDC(dest[3], t1[3], t2[3]);
    UADDC(dest[4], t1[4], t2[4]);
    UADD(*carry, c1, c2);

}

__device__ void _MatrixVecMul(uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, int64_t _21, int64_t _22)
{

    uint64_t t1[NBBLOCK];
    uint64_t t2[NBBLOCK];
    uint64_t t3[NBBLOCK];
    uint64_t t4[NBBLOCK];

    _IMult(t1, u, _11);
    _IMult(t2, v, _12);
    _IMult(t3, u, _21);
    _IMult(t4, v, _22);

    UADDO(u[0], t1[0], t2[0]);
    UADDC(u[1], t1[1], t2[1]);
    UADDC(u[2], t1[2], t2[2]);
    UADDC(u[3], t1[3], t2[3]);
    UADD(u[4], t1[4], t2[4]);

    UADDO(v[0], t3[0], t4[0]);
    UADDC(v[1], t3[1], t4[1]);
    UADDC(v[2], t3[2], t4[2]);
    UADDC(v[3], t3[3], t4[3]);
    UADD(v[4], t3[4], t4[4]);

}

__device__ uint64_t _AddCh(uint64_t r[5], uint64_t a[5], uint64_t carry)
{

    uint64_t carryOut;

    UADDO1(r[0], a[0]);
    UADDC1(r[1], a[1]);
    UADDC1(r[2], a[2]);
    UADDC1(r[3], a[3]);
    UADDC1(r[4], a[4]);
    UADD(carryOut, carry, 0ULL);

    return carryOut;

}

__device__ __noinline__ void _ModInv(uint64_t *R)
{

    // Compute modular inverse of R mod P (using 320bits signed integer)
    // 0 < this < P  , P must be odd
    // Return 0 if no inverse
    // See IntMod.cpp for more info.

    uint64_t u[NBBLOCK];
    uint64_t v[NBBLOCK];
    uint64_t r[NBBLOCK];
    uint64_t s[NBBLOCK];
    uint64_t tr[NBBLOCK];
    uint64_t ts[NBBLOCK];
    uint64_t r0[NBBLOCK];
    uint64_t s0[NBBLOCK];

    int64_t  uu;
    int64_t  uv;
    int64_t  vu;
    int64_t  vv;

    uint64_t mr0;
    uint64_t ms0;

    uint64_t carryR;
    uint64_t carryS;

    int32_t  pos = NBBLOCK - 1;

    u[0] = 0xFFFFFFFEFFFFFC2F;
    u[1] = 0xFFFFFFFFFFFFFFFF;
    u[2] = 0xFFFFFFFFFFFFFFFF;
    u[3] = 0xFFFFFFFFFFFFFFFF;
    u[4] = 0;
    Load(v, R);
    r[0] = 0; s[0] = 1;
    r[1] = 0; s[1] = 0;
    r[2] = 0; s[2] = 0;
    r[3] = 0; s[3] = 0;
    r[4] = 0; s[4] = 0;

    // Delayed right shift 62bits

    // DivStep loop -------------------------------

    while (true) {

        _DivStep62(u, v, &pos, &uu, &uv, &vu, &vv);

        _MatrixVecMul(u, v, uu, uv, vu, vv);

        if (_IsNegative(u)) {
            Neg(u);
            uu = -uu;
            uv = -uv;
        }
        if (_IsNegative(v)) {
            Neg(v);
            vu = -vu;
            vv = -vv;
        }

        _ShiftR62(u);
        _ShiftR62(v);

        // Update r
        _MatrixVecMulHalf(tr, r, s, uu, uv, &carryR);
        mr0 = (tr[0] * MM64) & MSK62;
        _MulP(r0, mr0);
        carryR = _AddCh(tr, r0, carryR);

        if (_IsZero(v)) {

            _ShiftR62(r, tr, carryR);
            break;

        } else {

            // Update s
            _MatrixVecMulHalf(ts, r, s, vu, vv, &carryS);
            ms0 = (ts[0] * MM64) & MSK62;
            _MulP(s0, ms0);
            carryS = _AddCh(ts, s0, carryS);

        }

        _ShiftR62(r, tr, carryR);
        _ShiftR62(s, ts, carryS);

    }

    // u ends with gcd
    if (!_IsOne(u)) {
        // No inverse
        R[0] = 0ULL;
        R[1] = 0ULL;
        R[2] = 0ULL;
        R[3] = 0ULL;
        R[4] = 0ULL;
        return;
    }

    while (_IsNegative(r))
        AddP(r);
    while (!_IsNegative(r))
        SubP(r);
    AddP(r);

    Load(R, r);

}

// ---------------------------------------------------------------------------------------
// secp256k1 field multiply r = a*b mod p, 8x32-bit even/odd column product chains fused
// into IMAD.WIDE.U32[.X], then the sparse double-fold R = L + H*0x1000003D1 (2^256 = 2^32+977
// mod p) applied twice. Output in [0,2^256) (not necessarily < p); the final 2^256 carry is
// dropped -- the SAME convention and the same accepted non-canonical edge behaviour on inputs
// in [p,2^256) as the UMult-based routine this replaces (verified: both agree with the Python
// reference for every input < p, and both are identically non-canonical for the ~2^-224 inputs
// >= p, which never occur in a run).
//
// Device (__CUDA_ARCH__): one non-volatile asm block -- the njuffa/mm32 schedule measured at
// 124 SASS / 73 IMAD.WIDE on Compiler Explorer nvcc 12.9.1 sm_89. %4..%7 = a limbs (LE),
// %8..%11 = b limbs, %0..%3 = result limbs.
// Host (CPU verification): a __uint128_t transcription of the IDENTICAL schedule -- even chain
// e0..e7, odd chain o0..o6, merge to 16 u32 limbs, then the same double fold. Validated against
// harness/crypto.py; the asm<->C line correspondence is documented in
// notes/ce-tools/mm32-cref-map.md.
// NOTE (rp): the device asm below now DELIBERATELY diverges from the host __uint128_t
// transcription. Two carries of the pseudo-Mersenne reduction tail (the carry out of the odd fold
// and the carry it feeds into the second fold) are treated as zero, which is exact unless BOTH
// operands lie within 2^-22 of 2^256; the divergence is therefore bounded at 2^-44 per product.
// It is applied to the two _ModMultCore bodies ONLY, never to a square: a square needs only ONE
// extreme operand (2^-23), which is two million times more likely and is not admissible here.
// The host transcription is intentionally left exact, so the two are no longer bit-identical.
#if QSB_SHORT_CARRY
__device__ __forceinline__ void _ModMultCore(uint64_t *r, const uint64_t *a, const uint64_t *b)
{
#ifdef __CUDA_ARCH__
    uint64_t r0,r1,r2,r3;
    asm("{\n\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n\t.reg .u32 cy,o15;\n\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\tmov.b64 {a0,a1}, %4;\n\tmov.b64 {a2,a3}, %5;\n\tmov.b64 {a4,a5}, %6;\n\tmov.b64 {a6,a7}, %7;\n\tmov.b64 {b0,b1}, %8;\n\tmov.b64 {b2,b3}, %9;\n\tmov.b64 {b4,b5}, %10;\n\tmov.b64 {b6,b7}, %11;\n\t.reg .u64 odd_t,odd_lc; .reg .u32 odd_cy;\nmul.wide.u32 e0, a0, b0;\nmul.wide.u32 o0, a0, b1;\nmul.wide.u32 e1, a0, b2;\nmul.wide.u32 o1, a0, b3;\nmul.wide.u32 e2, a0, b4;\nmul.wide.u32 o2, a0, b5;\nmul.wide.u32 e3, a0, b6;\nmul.wide.u32 o3, a0, b7;\nmul.wide.u32 t, a1, b1;\nmul.wide.u32 odd_t, a1, b0;\nadd.cc.u64 e1, e1, t;\nmul.wide.u32 t, a1, b3;\naddc.cc.u64 e2, e2, t;\nmul.wide.u32 t, a1, b5;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a1, b7;\naddc.u64 e4, t, 0;\nadd.cc.u64 o0, o0, odd_t;\nmul.wide.u32 odd_t, a1, b2;\naddc.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a1, b4;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a1, b6;\naddc.cc.u64 o3, o3, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a2, b0;\nadd.cc.u64 e1, e1, t;\nmul.wide.u32 t, a2, b2;\naddc.cc.u64 e2, e2, t;\nmul.wide.u32 t, a2, b4;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a2, b6;\naddc.cc.u64 e4, e4, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a2, b1;\nmov.b64 odd_lc, {odd_cy, cy};\nadd.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a2, b3;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a2, b5;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a2, b7;\naddc.u64 o4, odd_t, odd_lc;\nmul.wide.u32 t, a3, b1;\nmul.wide.u32 odd_t, a3, b0;\nadd.cc.u64 e2, e2, t;\nmul.wide.u32 t, a3, b3;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a3, b5;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a3, b7;\naddc.u64 e5, t, 0;\nadd.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a3, b2;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a3, b4;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a3, b6;\naddc.cc.u64 o4, o4, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a4, b0;\nadd.cc.u64 e2, e2, t;\nmul.wide.u32 t, a4, b2;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a4, b4;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a4, b6;\naddc.cc.u64 e5, e5, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a4, b1;\nmov.b64 odd_lc, {odd_cy, cy};\nadd.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a4, b3;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a4, b5;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a4, b7;\naddc.u64 o5, odd_t, odd_lc;\nmul.wide.u32 t, a5, b1;\nmul.wide.u32 odd_t, a5, b0;\nadd.cc.u64 e3, e3, t;\nmul.wide.u32 t, a5, b3;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a5, b5;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a5, b7;\naddc.u64 e6, t, 0;\nadd.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a5, b2;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a5, b4;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a5, b6;\naddc.cc.u64 o5, o5, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a6, b0;\nadd.cc.u64 e3, e3, t;\nmul.wide.u32 t, a6, b2;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a6, b4;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a6, b6;\naddc.cc.u64 e6, e6, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a6, b1;\nmov.b64 odd_lc, {odd_cy, cy};\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a6, b3;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a6, b5;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a6, b7;\naddc.u64 o6, odd_t, odd_lc;\nmul.wide.u32 t, a7, b1;\nmul.wide.u32 odd_t, a7, b0;\nadd.cc.u64 e4, e4, t;\nmul.wide.u32 t, a7, b3;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a7, b5;\naddc.cc.u64 e6, e6, t;\nmul.wide.u32 t, a7, b7;\naddc.u64 e7, t, 0;\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a7, b2;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a7, b4;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a7, b6;\naddc.cc.u64 o6, o6, odd_t;\naddc.u32 o15, 0, 0;\n"
#if QSB_MUL_XMERGE64
/* the multiply's 512-bit cross sum is merged in the eight 64-bit lanes the
 * even chain already occupies: e0..e7 are words (0,1)..(14,15) and the odd
 * registers o0..o6 hold words (1,2)..(13,14), so re-pairing their words one
 * position up turns fifteen carry-ordered 32-bit adds, fifteen register
 * splits and the eight-word repack into seven splits, eight pairings and
 * eight lane adds whose results ARE r0..r3 / h0..h3. Exact: lane i is words
 * 2i and 2i+1 and the carry between lanes is the carry out of word 2i+1, so
 * the bit pattern is the word chain's; the top lane takes {y14,o15}, which
 * reproduces "x14 += y14 with carry, x15 += o15 + carry" and drops the same
 * out-of-range carry the word form's terminating addc.u32 dropped.
 * -DQSB_MUL_XMERGE64=0 restores the word merge byte for byte. */
".reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\t.reg .u32 zr;\n\t.reg .u64 q0,q1,q2,q3,q4,q5,q6,q7;\n\tmov.u32 zr, 0;\n\tmov.b64 {y1,y2}, o0;\n\tmov.b64 {y3,y4}, o1;\n\tmov.b64 {y5,y6}, o2;\n\tmov.b64 {y7,y8}, o3;\n\tmov.b64 {y9,y10}, o4;\n\tmov.b64 {y11,y12}, o5;\n\tmov.b64 {y13,y14}, o6;\n\tmov.b64 q0, {zr,y1}; mov.b64 q1, {y2,y3}; mov.b64 q2, {y4,y5}; mov.b64 q3, {y6,y7};\n\tmov.b64 q4, {y8,y9}; mov.b64 q5, {y10,y11}; mov.b64 q6, {y12,y13}; mov.b64 q7, {y14,o15};\n\tadd.cc.u64  r0, e0, q0; addc.cc.u64 r1, e1, q1; addc.cc.u64 r2, e2, q2; addc.cc.u64 r3, e3, q3;\n\taddc.cc.u64 h0, e4, q4; addc.cc.u64 h1, e5, q5; addc.cc.u64 h2, e6, q6; addc.u64 h3, e7, q7;\n\tmov.b64 {x8,x9}, h0; mov.b64 {x10,x11}, h1; mov.b64 {x12,x13}, h2; mov.b64 {x14,x15}, h3;"
#else
"mov.b64 {x0,x1}, e0;\n\tmov.b64 {x2,x3}, e1;\n\tmov.b64 {x4,x5}, e2;\n\tmov.b64 {x6,x7}, e3;\n\tmov.b64 {x8,x9}, e4;\n\tmov.b64 {x10,x11}, e5;\n\tmov.b64 {x12,x13}, e6;\n\tmov.b64 {x14,x15}, e7;\n\tmov.b64 {y1,y2}, o0;\n\tmov.b64 {y3,y4}, o1;\n\tmov.b64 {y5,y6}, o2;\n\tmov.b64 {y7,y8}, o3;\n\tmov.b64 {y9,y10}, o4;\n\tmov.b64 {y11,y12}, o5;\n\tmov.b64 {y13,y14}, o6;\n\tadd.cc.u32 x1, x1, y1;\n\taddc.cc.u32 x2, x2, y2;\n\taddc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4;\n\taddc.cc.u32 x5, x5, y5;\n\taddc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7;\n\taddc.cc.u32 x8, x8, y8;\n\taddc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10;\n\taddc.cc.u32 x11, x11, y11;\n\taddc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13;\n\taddc.cc.u32 x14, x14, y14;\n\taddc.u32 x15, x15, o15;\n\t.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\tmov.b64 r0, {x0,x1}; mov.b64 r1, {x2,x3}; mov.b64 r2, {x4,x5}; mov.b64 r3, {x6,x7};\n\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};"
#endif
"\n\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, r0, t;\n\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, r1, t;\n\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, r2, t;\n\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, r3, t;\n"
QSB_MUL_F8CAP
"\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n\t"
#if QSB_MUL_LANE64
/* fold merge in 64-bit lanes: the even fold leaves f0..f3 as lanes and the
 * odd fold g0..g3 enters one word up, so the seven carry-ordered 32-bit adds
 * z1+=w0 .. z7+=w6 are exactly four lane adds (lane 0 takes {0,w0}, lane i
 * takes {w2i-1,w2i}) and w7 stays the ninth-word seed that z8 captures with
 * the carry out of lane 3 -- the same carry the word chain fed it. Exact for
 * the same reason as the merge above: lane boundaries are word boundaries.
 * -DQSB_MUL_LANE64=0 restores the 32-bit word form byte for byte. */
"/*rp*/\n\tmov.b64 {w0,w1}, g0;\n\tmov.b64 {w2,w3}, g1;\n\tmov.b64 {w4,w5}, g2;\n\tmov.b64 {w6,w7}, g3;\n\tmov.b64 q0, {zr,w0}; mov.b64 q1, {w1,w2}; mov.b64 q2, {w3,w4}; mov.b64 q3, {w5,w6};\n\tadd.cc.u64  f0, f0, q0;\n\taddc.cc.u64 f1, f1, q1;\n\taddc.cc.u64 f2, f2, q2;\n\taddc.cc.u64 f3, f3, q3;\n"
QSB_MUL_Z8SEED
#else
"/*rp*/\n\tmov.b64 {z0,z1}, f0;\n\tmov.b64 {z2,z3}, f1;\n\tmov.b64 {z4,z5}, f2;\n\tmov.b64 {z6,z7}, f3;\n\tmov.b64 {w0,w1}, g0;\n\tmov.b64 {w2,w3}, g1;\n\tmov.b64 {w4,w5}, g2;\n\tmov.b64 {w6,w7}, g3;\n\tadd.cc.u32  z1, z1, w0;\n\taddc.cc.u32 z2, z2, w1;\n\taddc.cc.u32 z3, z3, w2;\n\taddc.cc.u32 z4, z4, w3;\n\taddc.cc.u32 z5, z5, w4;\n\taddc.cc.u32 z6, z6, w5;\n\taddc.cc.u32 z7, z7, w6;\n"
QSB_MUL_Z8SEED
#endif
"\n\t"
#if QSB_MUL_FOLD64
/* top fold in the same lanes. With 2^256 == 2^32 + 977 (mod p) the fold adds
 * z8*(2^32+977) < 2^64 + 2^42: its low 64 bits are 977*z8 + (z8<<32) and its
 * 65th bit is that add's carry (set only for z8 >= 2^32-976). Adding f0 to
 * the low half is the same value the word form built as {z0,z8} + 977*z8,
 * lane 1 takes the 65th bit exactly as the word form added sfc at word 2,
 * and the residual carry then dies in lane 2 -- words 4,5, two whole words
 * above the word-3 terminator of the short-carry tail it replaces -- so this
 * arm truncates strictly later than the form it replaces and needs no
 * recomposition: the four lanes are the output words.
 * -DQSB_MUL_FOLD64=0 restores the 32-bit fold and its documented tail. */
#if QSB_MUL_FOLD65
"{ .reg .u64 sft; .reg .u32 sfl, sfh;\n\tmul.wide.u32 sft, z8, 977;\n\tmov.b64 {sfl, sfh}, sft;\n\tadd.u32 sfh, sfh, z8;\n\tmov.b64 sft, {sfl, sfh};\n\tadd.cc.u64  f0, f0, sft;\n"
#if QSB_MUL_FOLDT2
"\taddc.u64 f1, f1, 0; }\n"
#else
"\taddc.cc.u64 f1, f1, 0;\n\taddc.u64 f2, f2, 0; }\n"
#endif
#else
"{ .reg .u64 sft, sfz, sfw;\n\tmul.wide.u32 sft, z8, 977;\n\tmov.b64 sfz, {zr, z8};\n\tadd.cc.u64 sft, sft, sfz;\n\taddc.u64 sfw, 0, 0;\n\tadd.cc.u64  f0, f0, sft;\n"
#if QSB_MUL_FOLDT2
"\taddc.u64 f1, f1, sfw; }\n"
#else
"\taddc.cc.u64 f1, f1, sfw;\n\taddc.u64 f2, f2, 0; }\n"
#endif
#endif
"\tmov.u64 %0, f0; mov.u64 %1, f1; mov.u64 %2, f2; mov.u64 %3, f3;"
#elif QSB_MUL_LANE64
"mov.b64 {z0,z1}, f0;\n\tmov.b64 {z2,z3}, f1;\n\t"
"{ .reg .u64 sfz, sft; .reg .u32 sfc, sfq, sfl, sfh;\nmov.u32 sfq, z8;\nmov.b64 sfz, {z0, sfq};\nmul.wide.u32 sft, z8, 977;\nadd.cc.u64 sft, sft, sfz;\naddc.u32 sfc, 0, 0;\nmov.b64 {sfl, sfh}, sft;\nmov.u32 z0, sfl;\nadd.cc.u32 z1, z1, sfh;\naddc.cc.u32 z2, z2, sfc; }\n\n" QSB_SECOND_FOLD_TAIL "\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.u64 %2, f2; mov.u64 %3, f3;"
#else
"{ .reg .u64 sfz, sft; .reg .u32 sfc, sfq, sfl, sfh;\nmov.u32 sfq, z8;\nmov.b64 sfz, {z0, sfq};\nmul.wide.u32 sft, z8, 977;\nadd.cc.u64 sft, sft, sfz;\naddc.u32 sfc, 0, 0;\nmov.b64 {sfl, sfh}, sft;\nmov.u32 z0, sfl;\nadd.cc.u32 z1, z1, sfh;\naddc.cc.u32 z2, z2, sfc; }\n\n" QSB_SECOND_FOLD_TAIL "\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};"
#endif
"\n\t}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]) );
    r[0]=r0; r[1]=r1; r[2]=r2; r[3]=r3;
#else
#define QSB_MW(x,y) ((uint64_t)(uint32_t)(x)*(uint32_t)(y))

    uint32_t A[8], B[8];
    for (int i=0;i<4;i++){ A[2*i]=(uint32_t)a[i]; A[2*i+1]=(uint32_t)(a[i]>>32);
                           B[2*i]=(uint32_t)b[i]; B[2*i+1]=(uint32_t)(b[i]>>32); }
    uint64_t e0,e1,e2,e3,e4,e5,e6,e7, o0,o1,o2,o3,o4,o5,o6, cy,lc; __uint128_t s;
    /* even chain */
    e0=QSB_MW(A[0],B[0]); e1=QSB_MW(A[0],B[2]); e2=QSB_MW(A[0],B[4]); e3=QSB_MW(A[0],B[6]);
    s=(__uint128_t)e1+QSB_MW(A[1],B[1]); e1=(uint64_t)s;
    s=(s>>64)+e2+QSB_MW(A[1],B[3]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[1],B[5]); e3=(uint64_t)s;
    e4=(uint64_t)(s>>64)+QSB_MW(A[1],B[7]);
    s=(__uint128_t)e1+QSB_MW(A[2],B[0]); e1=(uint64_t)s;
    s=(s>>64)+e2+QSB_MW(A[2],B[2]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[2],B[4]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[2],B[6]); e4=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)e2+QSB_MW(A[3],B[1]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[3],B[3]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[3],B[5]); e4=(uint64_t)s;
    e5=(uint64_t)(s>>64)+QSB_MW(A[3],B[7])+lc;
    s=(__uint128_t)e2+QSB_MW(A[4],B[0]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[4],B[2]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[4],B[4]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[4],B[6]); e5=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)e3+QSB_MW(A[5],B[1]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[5],B[3]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[5],B[5]); e5=(uint64_t)s;
    e6=(uint64_t)(s>>64)+QSB_MW(A[5],B[7])+lc;
    s=(__uint128_t)e3+QSB_MW(A[6],B[0]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[6],B[2]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[6],B[4]); e5=(uint64_t)s;
    s=(s>>64)+e6+QSB_MW(A[6],B[6]); e6=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)e4+QSB_MW(A[7],B[1]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[7],B[3]); e5=(uint64_t)s;
    s=(s>>64)+e6+QSB_MW(A[7],B[5]); e6=(uint64_t)s;
    e7=(uint64_t)(s>>64)+QSB_MW(A[7],B[7])+lc;
    /* odd chain */
    o0=QSB_MW(A[0],B[1]); o1=QSB_MW(A[0],B[3]); o2=QSB_MW(A[0],B[5]); o3=QSB_MW(A[0],B[7]);
    s=(__uint128_t)o0+QSB_MW(A[1],B[0]); o0=(uint64_t)s;
    s=(s>>64)+o1+QSB_MW(A[1],B[2]); o1=(uint64_t)s;
    s=(s>>64)+o2+QSB_MW(A[1],B[4]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[1],B[6]); o3=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)o1+QSB_MW(A[2],B[1]); o1=(uint64_t)s;
    s=(s>>64)+o2+QSB_MW(A[2],B[3]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[2],B[5]); o3=(uint64_t)s;
    o4=(uint64_t)(s>>64)+QSB_MW(A[2],B[7])+lc;
    s=(__uint128_t)o1+QSB_MW(A[3],B[0]); o1=(uint64_t)s;
    s=(s>>64)+o2+QSB_MW(A[3],B[2]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[3],B[4]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[3],B[6]); o4=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)o2+QSB_MW(A[4],B[1]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[4],B[3]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[4],B[5]); o4=(uint64_t)s;
    o5=(uint64_t)(s>>64)+QSB_MW(A[4],B[7])+lc;
    s=(__uint128_t)o2+QSB_MW(A[5],B[0]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[5],B[2]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[5],B[4]); o4=(uint64_t)s;
    s=(s>>64)+o5+QSB_MW(A[5],B[6]); o5=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)o3+QSB_MW(A[6],B[1]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[6],B[3]); o4=(uint64_t)s;
    s=(s>>64)+o5+QSB_MW(A[6],B[5]); o5=(uint64_t)s;
    o6=(uint64_t)(s>>64)+QSB_MW(A[6],B[7])+lc;
    s=(__uint128_t)o3+QSB_MW(A[7],B[0]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[7],B[2]); o4=(uint64_t)s;
    s=(s>>64)+o5+QSB_MW(A[7],B[4]); o5=(uint64_t)s;
    s=(s>>64)+o6+QSB_MW(A[7],B[6]); o6=(uint64_t)s;
    uint32_t o15=(uint32_t)(s>>64);
    /* unpack + merge to 16 u32 limbs */
    uint32_t x[16];
    x[0]=(uint32_t)e0; x[1]=(uint32_t)(e0>>32); x[2]=(uint32_t)e1; x[3]=(uint32_t)(e1>>32);
    x[4]=(uint32_t)e2; x[5]=(uint32_t)(e2>>32); x[6]=(uint32_t)e3; x[7]=(uint32_t)(e3>>32);
    x[8]=(uint32_t)e4; x[9]=(uint32_t)(e4>>32); x[10]=(uint32_t)e5; x[11]=(uint32_t)(e5>>32);
    x[12]=(uint32_t)e6; x[13]=(uint32_t)(e6>>32); x[14]=(uint32_t)e7; x[15]=(uint32_t)(e7>>32);
    uint32_t y[15];
    y[1]=(uint32_t)o0; y[2]=(uint32_t)(o0>>32); y[3]=(uint32_t)o1; y[4]=(uint32_t)(o1>>32);
    y[5]=(uint32_t)o2; y[6]=(uint32_t)(o2>>32); y[7]=(uint32_t)o3; y[8]=(uint32_t)(o3>>32);
    y[9]=(uint32_t)o4; y[10]=(uint32_t)(o4>>32); y[11]=(uint32_t)o5; y[12]=(uint32_t)(o5>>32);
    y[13]=(uint32_t)o6; y[14]=(uint32_t)(o6>>32);
    { uint64_t c=0; for (int k=1;k<=14;k++){ uint64_t t=(uint64_t)x[k]+y[k]+c; x[k]=(uint32_t)t; c=t>>32; }
      x[15]=(uint32_t)((uint64_t)x[15]+o15+c); }
    /* secp256k1 double-fold reduction (identical to mm32) */
    uint64_t r0=x[0]|((uint64_t)x[1]<<32), r1=x[2]|((uint64_t)x[3]<<32),
             r2=x[4]|((uint64_t)x[5]<<32), r3=x[6]|((uint64_t)x[7]<<32);
    uint64_t h0=x[8]|((uint64_t)x[9]<<32), h1=x[10]|((uint64_t)x[11]<<32),
             h2=x[12]|((uint64_t)x[13]<<32), h3=x[14]|((uint64_t)x[15]<<32);
    (void)r0;(void)r1;(void)r2;(void)r3;(void)h0;(void)h1;(void)h2;(void)h3;
    uint64_t f0,f1,f2,f3; uint32_t f8;
    s=(__uint128_t)r0+QSB_MW(x[8],977);  f0=(uint64_t)s;
    s=(s>>64)+r1+QSB_MW(x[10],977); f1=(uint64_t)s;
    s=(s>>64)+r2+QSB_MW(x[12],977); f2=(uint64_t)s;
    s=(s>>64)+r3+QSB_MW(x[14],977); f3=(uint64_t)s;
    f8=(uint32_t)(s>>64);
    uint64_t g0,g1,g2,g3; uint32_t g8;
    s=(__uint128_t)h0+QSB_MW(x[9],977);  g0=(uint64_t)s;
    s=(s>>64)+h1+QSB_MW(x[11],977); g1=(uint64_t)s;
    s=(s>>64)+h2+QSB_MW(x[13],977); g2=(uint64_t)s;
    s=(s>>64)+h3+QSB_MW(x[15],977); g3=(uint64_t)s;
    g8=(uint32_t)(s>>64);
    uint32_t z[10], w[8];
    z[0]=(uint32_t)f0; z[1]=(uint32_t)(f0>>32); z[2]=(uint32_t)f1; z[3]=(uint32_t)(f1>>32);
    z[4]=(uint32_t)f2; z[5]=(uint32_t)(f2>>32); z[6]=(uint32_t)f3; z[7]=(uint32_t)(f3>>32);
    w[0]=(uint32_t)g0; w[1]=(uint32_t)(g0>>32); w[2]=(uint32_t)g1; w[3]=(uint32_t)(g1>>32);
    w[4]=(uint32_t)g2; w[5]=(uint32_t)(g2>>32); w[6]=(uint32_t)g3; w[7]=(uint32_t)(g3>>32);
    { uint64_t c=0,t;
      for (int k=0;k<7;k++){ t=(uint64_t)z[k+1]+w[k]+c; z[k+1]=(uint32_t)t; c=t>>32; }
      t=(uint64_t)f8+w[7]+c; z[8]=(uint32_t)t; c=t>>32;
      z[9]=(uint32_t)((uint64_t)g8+c); }
    /* second fold: c33 = z8 + 2^32 z9; m = c33*977 + c33<<32 */
    uint64_t tt=QSB_MW(z[8],977);
    uint32_t m0=(uint32_t)tt, m1=(uint32_t)(tt>>32), m2;
    m1=(uint32_t)(m1 + (uint32_t)((uint64_t)z[9]*977));
    { uint64_t t=(uint64_t)m1+z[8]; m1=(uint32_t)t; uint64_t c=t>>32; m2=(uint32_t)((uint64_t)z[9]+c); }
    { uint64_t c=0,t;
      t=(uint64_t)z[0]+m0+c; z[0]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[1]+m1+c; z[1]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[2]+m2+c; z[2]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[3]+c; z[3]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[4]+c; z[4]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[5]+c; z[5]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[6]+c; z[6]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[7]+c; z[7]=(uint32_t)t; }
    r[0]=z[0]|((uint64_t)z[1]<<32); r[1]=z[2]|((uint64_t)z[3]<<32);
    r[2]=z[4]|((uint64_t)z[5]<<32); r[3]=z[6]|((uint64_t)z[7]<<32);
#undef QSB_MW
#endif
}

#else
__device__ __forceinline__ void _ModMultCore(uint64_t *r, const uint64_t *a, const uint64_t *b)
{
#ifdef __CUDA_ARCH__
    uint64_t r0,r1,r2,r3;
    asm("{\n\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7;\n\t.reg .u64 e0,e1,e2,e3,e4,e5,e6,e7,o0,o1,o2,o3,o4,o5,o6,t,lc;\n\t.reg .u32 cy,o15;\n\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t.reg .u32 y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\tmov.b64 {a0,a1}, %4;\n\tmov.b64 {a2,a3}, %5;\n\tmov.b64 {a4,a5}, %6;\n\tmov.b64 {a6,a7}, %7;\n\tmov.b64 {b0,b1}, %8;\n\tmov.b64 {b2,b3}, %9;\n\tmov.b64 {b4,b5}, %10;\n\tmov.b64 {b6,b7}, %11;\n\t.reg .u64 odd_t,odd_lc; .reg .u32 odd_cy;\nmul.wide.u32 e0, a0, b0;\nmul.wide.u32 o0, a0, b1;\nmul.wide.u32 e1, a0, b2;\nmul.wide.u32 o1, a0, b3;\nmul.wide.u32 e2, a0, b4;\nmul.wide.u32 o2, a0, b5;\nmul.wide.u32 e3, a0, b6;\nmul.wide.u32 o3, a0, b7;\nmul.wide.u32 t, a1, b1;\nmul.wide.u32 odd_t, a1, b0;\nadd.cc.u64 e1, e1, t;\nmul.wide.u32 t, a1, b3;\naddc.cc.u64 e2, e2, t;\nmul.wide.u32 t, a1, b5;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a1, b7;\naddc.u64 e4, t, 0;\nadd.cc.u64 o0, o0, odd_t;\nmul.wide.u32 odd_t, a1, b2;\naddc.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a1, b4;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a1, b6;\naddc.cc.u64 o3, o3, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a2, b0;\nadd.cc.u64 e1, e1, t;\nmul.wide.u32 t, a2, b2;\naddc.cc.u64 e2, e2, t;\nmul.wide.u32 t, a2, b4;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a2, b6;\naddc.cc.u64 e4, e4, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a2, b1;\nmov.b64 odd_lc, {odd_cy, cy};\nadd.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a2, b3;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a2, b5;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a2, b7;\naddc.u64 o4, odd_t, odd_lc;\nmul.wide.u32 t, a3, b1;\nmul.wide.u32 odd_t, a3, b0;\nadd.cc.u64 e2, e2, t;\nmul.wide.u32 t, a3, b3;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a3, b5;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a3, b7;\naddc.u64 e5, t, 0;\nadd.cc.u64 o1, o1, odd_t;\nmul.wide.u32 odd_t, a3, b2;\naddc.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a3, b4;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a3, b6;\naddc.cc.u64 o4, o4, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a4, b0;\nadd.cc.u64 e2, e2, t;\nmul.wide.u32 t, a4, b2;\naddc.cc.u64 e3, e3, t;\nmul.wide.u32 t, a4, b4;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a4, b6;\naddc.cc.u64 e5, e5, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a4, b1;\nmov.b64 odd_lc, {odd_cy, cy};\nadd.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a4, b3;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a4, b5;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a4, b7;\naddc.u64 o5, odd_t, odd_lc;\nmul.wide.u32 t, a5, b1;\nmul.wide.u32 odd_t, a5, b0;\nadd.cc.u64 e3, e3, t;\nmul.wide.u32 t, a5, b3;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a5, b5;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a5, b7;\naddc.u64 e6, t, 0;\nadd.cc.u64 o2, o2, odd_t;\nmul.wide.u32 odd_t, a5, b2;\naddc.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a5, b4;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a5, b6;\naddc.cc.u64 o5, o5, odd_t;\naddc.u32 odd_cy, 0, 0;\nmul.wide.u32 t, a6, b0;\nadd.cc.u64 e3, e3, t;\nmul.wide.u32 t, a6, b2;\naddc.cc.u64 e4, e4, t;\nmul.wide.u32 t, a6, b4;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a6, b6;\naddc.cc.u64 e6, e6, t;\naddc.u32 cy, 0, 0;\nmul.wide.u32 odd_t, a6, b1;\nmov.b64 odd_lc, {odd_cy, cy};\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a6, b3;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a6, b5;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a6, b7;\naddc.u64 o6, odd_t, odd_lc;\nmul.wide.u32 t, a7, b1;\nmul.wide.u32 odd_t, a7, b0;\nadd.cc.u64 e4, e4, t;\nmul.wide.u32 t, a7, b3;\naddc.cc.u64 e5, e5, t;\nmul.wide.u32 t, a7, b5;\naddc.cc.u64 e6, e6, t;\nmul.wide.u32 t, a7, b7;\naddc.u64 e7, t, 0;\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a7, b2;\naddc.cc.u64 o4, o4, odd_t;\nmul.wide.u32 odd_t, a7, b4;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a7, b6;\naddc.cc.u64 o6, o6, odd_t;\naddc.u32 o15, 0, 0;\nmov.b64 {x0,x1}, e0;\n\tmov.b64 {x2,x3}, e1;\n\tmov.b64 {x4,x5}, e2;\n\tmov.b64 {x6,x7}, e3;\n\tmov.b64 {x8,x9}, e4;\n\tmov.b64 {x10,x11}, e5;\n\tmov.b64 {x12,x13}, e6;\n\tmov.b64 {x14,x15}, e7;\n\tmov.b64 {y1,y2}, o0;\n\tmov.b64 {y3,y4}, o1;\n\tmov.b64 {y5,y6}, o2;\n\tmov.b64 {y7,y8}, o3;\n\tmov.b64 {y9,y10}, o4;\n\tmov.b64 {y11,y12}, o5;\n\tmov.b64 {y13,y14}, o6;\n\tadd.cc.u32 x1, x1, y1;\n\taddc.cc.u32 x2, x2, y2;\n\taddc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4;\n\taddc.cc.u32 x5, x5, y5;\n\taddc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7;\n\taddc.cc.u32 x8, x8, y8;\n\taddc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10;\n\taddc.cc.u32 x11, x11, y11;\n\taddc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13;\n\taddc.cc.u32 x14, x14, y14;\n\taddc.u32 x15, x15, o15;\n\t.reg .u64 r0,r1,r2,r3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\tmov.b64 r0, {x0,x1}; mov.b64 r1, {x2,x3}; mov.b64 r2, {x4,x5}; mov.b64 r3, {x6,x7};\n\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, r0, t;\n\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, r1, t;\n\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, r2, t;\n\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, r3, t;\n\taddc.u32 f8, 0, 0;\n\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n\t/*rp*/\n\tmov.b64 {z0,z1}, f0;\n\tmov.b64 {z2,z3}, f1;\n\tmov.b64 {z4,z5}, f2;\n\tmov.b64 {z6,z7}, f3;\n\tmov.b64 {w0,w1}, g0;\n\tmov.b64 {w2,w3}, g1;\n\tmov.b64 {w4,w5}, g2;\n\tmov.b64 {w6,w7}, g3;\n\tadd.cc.u32  z1, z1, w0;\n\taddc.cc.u32 z2, z2, w1;\n\taddc.cc.u32 z3, z3, w2;\n\taddc.cc.u32 z4, z4, w3;\n\taddc.cc.u32 z5, z5, w4;\n\taddc.cc.u32 z6, z6, w5;\n\taddc.cc.u32 z7, z7, w6;\n\taddc.u32 z8, f8, w7;\n\t{ .reg .u64 sfz, sft; .reg .u32 sfc, sfq, sfl, sfh;\nmov.u32 sfq, z8;\nmov.b64 sfz, {z0, sfq};\nmul.wide.u32 sft, z8, 977;\nadd.cc.u64 sft, sft, sfz;\naddc.u32 sfc, 0, 0;\nmov.b64 {sfl, sfh}, sft;\nmov.u32 z0, sfl;\nadd.cc.u32 z1, z1, sfh;\naddc.cc.u32 z2, z2, sfc; }\n\n\taddc.cc.u32 z3, z3, 0;\n\taddc.cc.u32 z4, z4, 0;\n\taddc.cc.u32 z5, z5, 0;\n\taddc.cc.u32 z6, z6, 0;\n\taddc.u32 z7, z7, 0;\n\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n\t}"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),"l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]) );
    r[0]=r0; r[1]=r1; r[2]=r2; r[3]=r3;
#else
#define QSB_MW(x,y) ((uint64_t)(uint32_t)(x)*(uint32_t)(y))

    uint32_t A[8], B[8];
    for (int i=0;i<4;i++){ A[2*i]=(uint32_t)a[i]; A[2*i+1]=(uint32_t)(a[i]>>32);
                           B[2*i]=(uint32_t)b[i]; B[2*i+1]=(uint32_t)(b[i]>>32); }
    uint64_t e0,e1,e2,e3,e4,e5,e6,e7, o0,o1,o2,o3,o4,o5,o6, cy,lc; __uint128_t s;
    /* even chain */
    e0=QSB_MW(A[0],B[0]); e1=QSB_MW(A[0],B[2]); e2=QSB_MW(A[0],B[4]); e3=QSB_MW(A[0],B[6]);
    s=(__uint128_t)e1+QSB_MW(A[1],B[1]); e1=(uint64_t)s;
    s=(s>>64)+e2+QSB_MW(A[1],B[3]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[1],B[5]); e3=(uint64_t)s;
    e4=(uint64_t)(s>>64)+QSB_MW(A[1],B[7]);
    s=(__uint128_t)e1+QSB_MW(A[2],B[0]); e1=(uint64_t)s;
    s=(s>>64)+e2+QSB_MW(A[2],B[2]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[2],B[4]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[2],B[6]); e4=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)e2+QSB_MW(A[3],B[1]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[3],B[3]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[3],B[5]); e4=(uint64_t)s;
    e5=(uint64_t)(s>>64)+QSB_MW(A[3],B[7])+lc;
    s=(__uint128_t)e2+QSB_MW(A[4],B[0]); e2=(uint64_t)s;
    s=(s>>64)+e3+QSB_MW(A[4],B[2]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[4],B[4]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[4],B[6]); e5=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)e3+QSB_MW(A[5],B[1]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[5],B[3]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[5],B[5]); e5=(uint64_t)s;
    e6=(uint64_t)(s>>64)+QSB_MW(A[5],B[7])+lc;
    s=(__uint128_t)e3+QSB_MW(A[6],B[0]); e3=(uint64_t)s;
    s=(s>>64)+e4+QSB_MW(A[6],B[2]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[6],B[4]); e5=(uint64_t)s;
    s=(s>>64)+e6+QSB_MW(A[6],B[6]); e6=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)e4+QSB_MW(A[7],B[1]); e4=(uint64_t)s;
    s=(s>>64)+e5+QSB_MW(A[7],B[3]); e5=(uint64_t)s;
    s=(s>>64)+e6+QSB_MW(A[7],B[5]); e6=(uint64_t)s;
    e7=(uint64_t)(s>>64)+QSB_MW(A[7],B[7])+lc;
    /* odd chain */
    o0=QSB_MW(A[0],B[1]); o1=QSB_MW(A[0],B[3]); o2=QSB_MW(A[0],B[5]); o3=QSB_MW(A[0],B[7]);
    s=(__uint128_t)o0+QSB_MW(A[1],B[0]); o0=(uint64_t)s;
    s=(s>>64)+o1+QSB_MW(A[1],B[2]); o1=(uint64_t)s;
    s=(s>>64)+o2+QSB_MW(A[1],B[4]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[1],B[6]); o3=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)o1+QSB_MW(A[2],B[1]); o1=(uint64_t)s;
    s=(s>>64)+o2+QSB_MW(A[2],B[3]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[2],B[5]); o3=(uint64_t)s;
    o4=(uint64_t)(s>>64)+QSB_MW(A[2],B[7])+lc;
    s=(__uint128_t)o1+QSB_MW(A[3],B[0]); o1=(uint64_t)s;
    s=(s>>64)+o2+QSB_MW(A[3],B[2]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[3],B[4]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[3],B[6]); o4=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)o2+QSB_MW(A[4],B[1]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[4],B[3]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[4],B[5]); o4=(uint64_t)s;
    o5=(uint64_t)(s>>64)+QSB_MW(A[4],B[7])+lc;
    s=(__uint128_t)o2+QSB_MW(A[5],B[0]); o2=(uint64_t)s;
    s=(s>>64)+o3+QSB_MW(A[5],B[2]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[5],B[4]); o4=(uint64_t)s;
    s=(s>>64)+o5+QSB_MW(A[5],B[6]); o5=(uint64_t)s;
    cy=(uint64_t)(s>>64); lc=cy;
    s=(__uint128_t)o3+QSB_MW(A[6],B[1]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[6],B[3]); o4=(uint64_t)s;
    s=(s>>64)+o5+QSB_MW(A[6],B[5]); o5=(uint64_t)s;
    o6=(uint64_t)(s>>64)+QSB_MW(A[6],B[7])+lc;
    s=(__uint128_t)o3+QSB_MW(A[7],B[0]); o3=(uint64_t)s;
    s=(s>>64)+o4+QSB_MW(A[7],B[2]); o4=(uint64_t)s;
    s=(s>>64)+o5+QSB_MW(A[7],B[4]); o5=(uint64_t)s;
    s=(s>>64)+o6+QSB_MW(A[7],B[6]); o6=(uint64_t)s;
    uint32_t o15=(uint32_t)(s>>64);
    /* unpack + merge to 16 u32 limbs */
    uint32_t x[16];
    x[0]=(uint32_t)e0; x[1]=(uint32_t)(e0>>32); x[2]=(uint32_t)e1; x[3]=(uint32_t)(e1>>32);
    x[4]=(uint32_t)e2; x[5]=(uint32_t)(e2>>32); x[6]=(uint32_t)e3; x[7]=(uint32_t)(e3>>32);
    x[8]=(uint32_t)e4; x[9]=(uint32_t)(e4>>32); x[10]=(uint32_t)e5; x[11]=(uint32_t)(e5>>32);
    x[12]=(uint32_t)e6; x[13]=(uint32_t)(e6>>32); x[14]=(uint32_t)e7; x[15]=(uint32_t)(e7>>32);
    uint32_t y[15];
    y[1]=(uint32_t)o0; y[2]=(uint32_t)(o0>>32); y[3]=(uint32_t)o1; y[4]=(uint32_t)(o1>>32);
    y[5]=(uint32_t)o2; y[6]=(uint32_t)(o2>>32); y[7]=(uint32_t)o3; y[8]=(uint32_t)(o3>>32);
    y[9]=(uint32_t)o4; y[10]=(uint32_t)(o4>>32); y[11]=(uint32_t)o5; y[12]=(uint32_t)(o5>>32);
    y[13]=(uint32_t)o6; y[14]=(uint32_t)(o6>>32);
    { uint64_t c=0; for (int k=1;k<=14;k++){ uint64_t t=(uint64_t)x[k]+y[k]+c; x[k]=(uint32_t)t; c=t>>32; }
      x[15]=(uint32_t)((uint64_t)x[15]+o15+c); }
    /* secp256k1 double-fold reduction (identical to mm32) */
    uint64_t r0=x[0]|((uint64_t)x[1]<<32), r1=x[2]|((uint64_t)x[3]<<32),
             r2=x[4]|((uint64_t)x[5]<<32), r3=x[6]|((uint64_t)x[7]<<32);
    uint64_t h0=x[8]|((uint64_t)x[9]<<32), h1=x[10]|((uint64_t)x[11]<<32),
             h2=x[12]|((uint64_t)x[13]<<32), h3=x[14]|((uint64_t)x[15]<<32);
    (void)r0;(void)r1;(void)r2;(void)r3;(void)h0;(void)h1;(void)h2;(void)h3;
    uint64_t f0,f1,f2,f3; uint32_t f8;
    s=(__uint128_t)r0+QSB_MW(x[8],977);  f0=(uint64_t)s;
    s=(s>>64)+r1+QSB_MW(x[10],977); f1=(uint64_t)s;
    s=(s>>64)+r2+QSB_MW(x[12],977); f2=(uint64_t)s;
    s=(s>>64)+r3+QSB_MW(x[14],977); f3=(uint64_t)s;
    f8=(uint32_t)(s>>64);
    uint64_t g0,g1,g2,g3; uint32_t g8;
    s=(__uint128_t)h0+QSB_MW(x[9],977);  g0=(uint64_t)s;
    s=(s>>64)+h1+QSB_MW(x[11],977); g1=(uint64_t)s;
    s=(s>>64)+h2+QSB_MW(x[13],977); g2=(uint64_t)s;
    s=(s>>64)+h3+QSB_MW(x[15],977); g3=(uint64_t)s;
    g8=(uint32_t)(s>>64);
    uint32_t z[10], w[8];
    z[0]=(uint32_t)f0; z[1]=(uint32_t)(f0>>32); z[2]=(uint32_t)f1; z[3]=(uint32_t)(f1>>32);
    z[4]=(uint32_t)f2; z[5]=(uint32_t)(f2>>32); z[6]=(uint32_t)f3; z[7]=(uint32_t)(f3>>32);
    w[0]=(uint32_t)g0; w[1]=(uint32_t)(g0>>32); w[2]=(uint32_t)g1; w[3]=(uint32_t)(g1>>32);
    w[4]=(uint32_t)g2; w[5]=(uint32_t)(g2>>32); w[6]=(uint32_t)g3; w[7]=(uint32_t)(g3>>32);
    { uint64_t c=0,t;
      for (int k=0;k<7;k++){ t=(uint64_t)z[k+1]+w[k]+c; z[k+1]=(uint32_t)t; c=t>>32; }
      t=(uint64_t)f8+w[7]+c; z[8]=(uint32_t)t; c=t>>32;
      z[9]=(uint32_t)((uint64_t)g8+c); }
    /* second fold: c33 = z8 + 2^32 z9; m = c33*977 + c33<<32 */
    uint64_t tt=QSB_MW(z[8],977);
    uint32_t m0=(uint32_t)tt, m1=(uint32_t)(tt>>32), m2;
    m1=(uint32_t)(m1 + (uint32_t)((uint64_t)z[9]*977));
    { uint64_t t=(uint64_t)m1+z[8]; m1=(uint32_t)t; uint64_t c=t>>32; m2=(uint32_t)((uint64_t)z[9]+c); }
    { uint64_t c=0,t;
      t=(uint64_t)z[0]+m0+c; z[0]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[1]+m1+c; z[1]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[2]+m2+c; z[2]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[3]+c; z[3]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[4]+c; z[4]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[5]+c; z[5]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[6]+c; z[6]=(uint32_t)t; c=t>>32;
      t=(uint64_t)z[7]+c; z[7]=(uint32_t)t; }
    r[0]=z[0]|((uint64_t)z[1]<<32); r[1]=z[2]|((uint64_t)z[3]<<32);
    r[2]=z[4]|((uint64_t)z[5]<<32); r[3]=z[6]|((uint64_t)z[7]<<32);
#undef QSB_MW
#endif
}

#endif
// ---------------------------------------------------------------------------------------
// Compute a*b (mod p). Interface unchanged: uint64_t[4] little-endian, inputs < 2^256.
// ---------------------------------------------------------------------------------------
__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b)
{
    _ModMultCore(r, a, b);
}

__device__ void _ModMult(uint64_t *r, uint64_t *a)
{
    uint64_t bb[4] = { r[0], r[1], r[2], r[3] };
    _ModMultCore(r, a, bb);
}

// ---------------------------------------------------------------------------------------
// Dedicated secp256k1 square r = a^2 mod p. Triangular 8x32 schedule: 28 off-diagonal
// cross products a_i*a_j (even/odd column chains with multi-bit carries), doubled, plus
// 8 diagonal squares, then the same double-fold as _ModMultCore. 45 IMAD.WIDE/square
// (vs 73 for a*a via _ModMultCore). Output convention identical to _ModMultCore:
// [0,2^256), final 2^256 carry dropped. Device: inline PTX; host: __uint128_t C-ref of
// the IDENTICAL schedule. Independently validated (notes/research/sqr_ptx/VALIDATION.md):
// 10^6 random + boundaries vs crypto.py, PTX row-schedule emulation, CE 45 IMAD.WIDE.
#if QSB_SHORT_CARRY
__device__ __forceinline__ void _ModSqr(uint64_t r[4], const uint64_t a[4]) {
#ifdef __CUDA_ARCH__
    uint64_t r0, r1, r2, r3;
    asm("{\n\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7;\n\t.reg .u64 e2,e4,e6,e8,e10,e12,o1,o3,o5,o7,o9,o11,o13,t;\n\t.reg .u32 ecy,ocy,e14,o15;\n\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t.reg .u32 y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\t.reg .u64 d0,d1,d2,d3,d4,d5,d6,d7;\n\tmov.b64 {a0,a1}, %4;\n\tmov.b64 {a2,a3}, %5;\n\tmov.b64 {a4,a5}, %6;\n\tmov.b64 {a6,a7}, %7;\n\t.reg .u64 odd_t;\nmul.wide.u32 e6, a2, a4;\nmul.wide.u32 o7, a3, a4;\nmul.wide.u32 e8, a3, a5;\nmul.wide.u32 o5, a2, a3;\nmul.wide.u32 e4, a1, a3;\nmul.wide.u32 odd_t, a2, a5;\nmul.wide.u32 t, a1, a5;\nadd.cc.u64 o7, o7, odd_t;\nmul.wide.u32 odd_t, a4, a5;\naddc.u64 o9, odd_t, 0;\nadd.cc.u64 e6, e6, t;\nmul.wide.u32 t, a2, a6;\naddc.cc.u64 e8, e8, t;\nmul.wide.u32 t, a4, a6;\naddc.u64 e10, t, 0;\nmul.wide.u32 o3, a1, a2;\nmul.wide.u32 e2, a0, a2;\nmul.wide.u32 odd_t, a1, a4;\nmul.wide.u32 t, a0, a4;\nadd.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a1, a6;\naddc.cc.u64 o7, o7, odd_t;\nmul.wide.u32 odd_t, a3, a6;\naddc.cc.u64 o9, o9, odd_t;\nmul.wide.u32 odd_t, a5, a6;\naddc.u64 o11, odd_t, 0;\nadd.cc.u64 e4, e4, t;\nmul.wide.u32 t, a0, a6;\naddc.cc.u64 e6, e6, t;\nmul.wide.u32 t, a1, a7;\naddc.cc.u64 e8, e8, t;\nmul.wide.u32 t, a3, a7;\naddc.cc.u64 e10, e10, t;\nmul.wide.u32 t, a5, a7;\naddc.u64 e12, t, 0;\nmul.wide.u32 o1, a0, a1;\nmul.wide.u32 odd_t, a0, a3;\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a0, a5;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a0, a7;\naddc.cc.u64 o7, o7, odd_t;\nmul.wide.u32 odd_t, a2, a7;\naddc.cc.u64 o9, o9, odd_t;\nmul.wide.u32 odd_t, a4, a7;\naddc.cc.u64 o11, o11, odd_t;\nmul.wide.u32 odd_t, a6, a7;\naddc.u64 o13, odd_t, 0;\nmov.u32 x0, 0;"
#if QSB_SQL_XMERGE64
/* lane-form cross-sum merge of the live square. The even columns already
 * are the 64-bit lanes e2..e12 (words 2..13) and the odd columns re-pair
 * into lanes one word up, so the fifteen carry-ordered 32-bit adds and
 * the sixteen-word repack become seven pairings and seven lane adds.
 * Exact: lane i is words 2i,2i+1 and the carry between lanes is the carry
 * out of word 2i+1; the top even column cannot overflow its own lane
 * (PP(5,7) plus one carry stays below 2^64 - 2^33, which is why the word
 * form seeded x14 = x15 = 0), so word 14 carries the odd top y14 plus the
 * chain carry and word 15 stays zero -- bit for bit the cross sum the word
 * chain produced. -DQSB_SQL_XMERGE64=0 falls back to the word merge. */
"\n\tmov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n"
"\tmov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n"
"\tmov.b64 d0, {x0,x1}; mov.b64 d1, {y2,y3}; mov.b64 d2, {y4,y5}; mov.b64 d3, {y6,y7};\n"
"\tmov.b64 d4, {y8,y9}; mov.b64 d5, {y10,y11}; mov.b64 d6, {y12,y13}; mov.b64 d7, {y14,x0};\n"
"\tadd.cc.u64 d1, d1, e2; addc.cc.u64 d2, d2, e4; addc.cc.u64 d3, d3, e6;\n"
"\taddc.cc.u64 d4, d4, e8; addc.cc.u64 d5, d5, e10; addc.cc.u64 d6, d6, e12;\n"
"\taddc.u64 d7, d7, 0;\n"
"\tadd.cc.u64 d0, d0, d0; addc.cc.u64 d1, d1, d1;\n"
"\taddc.cc.u64 d2, d2, d2; addc.cc.u64 d3, d3, d3;\n"
"\taddc.cc.u64 d4, d4, d4; addc.cc.u64 d5, d5, d5;\n"
"\taddc.cc.u64 d6, d6, d6; addc.u64 d7, d7, d7;\n"
#elif QSB_SQL_DBL64
/* the crown word merge, doubled in the eight 64-bit lanes instead of the
 * fifteen descending funnel shifts: 2*X is the same value whether the
 * shift crosses word or lane boundaries, and the bit above word 15 is
 * dropped in both forms because 2*cross + diagonal stays below 2^512.
 * -DQSB_SQL_DBL64=0 restores the funnel chain byte for byte. */
"\n\tmov.b64 {x2,x3}, e2; mov.b64 {x4,x5}, e4; mov.b64 {x6,x7}, e6;\n\tmov.b64 {x8,x9}, e8; mov.b64 {x10,x11}, e10; mov.b64 {x12,x13}, e12;\n\tmov.u32 x14, 0; mov.u32 x15, 0;\n\tmov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n\tmov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n\tadd.cc.u32 x2, x2, y2; addc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4; addc.cc.u32 x5, x5, y5; addc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7; addc.cc.u32 x8, x8, y8; addc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10; addc.cc.u32 x11, x11, y11; addc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13; addc.cc.u32 x14, x14, y14;\naddc.u32 x15, x15, 0;"
"\n\tmov.b64 d0, {x0,x1}; mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n"
"\tmov.b64 d4, {x8,x9}; mov.b64 d5, {x10,x11}; mov.b64 d6, {x12,x13}; mov.b64 d7, {x14,x15};\n"
"\tadd.cc.u64 d0, d0, d0; addc.cc.u64 d1, d1, d1;\n"
"\taddc.cc.u64 d2, d2, d2; addc.cc.u64 d3, d3, d3;\n"
"\taddc.cc.u64 d4, d4, d4; addc.cc.u64 d5, d5, d5;\n"
"\taddc.cc.u64 d6, d6, d6; addc.u64 d7, d7, d7;\n"
#else
"\n\tmov.b64 {x2,x3}, e2; mov.b64 {x4,x5}, e4; mov.b64 {x6,x7}, e6;\n\tmov.b64 {x8,x9}, e8; mov.b64 {x10,x11}, e10; mov.b64 {x12,x13}, e12;\n\tmov.u32 x14, 0; mov.u32 x15, 0;\n\tmov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n\tmov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n\tadd.cc.u32 x2, x2, y2; addc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4; addc.cc.u32 x5, x5, y5; addc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7; addc.cc.u32 x8, x8, y8; addc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10; addc.cc.u32 x11, x11, y11; addc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13; addc.cc.u32 x14, x14, y14;\naddc.u32 x15, x15, 0;\nshf.l.wrap.b32 x15, x14, x15, 1; shf.l.wrap.b32 x14, x13, x14, 1;\n\tshf.l.wrap.b32 x13, x12, x13, 1; shf.l.wrap.b32 x12, x11, x12, 1;\n\tshf.l.wrap.b32 x11, x10, x11, 1; shf.l.wrap.b32 x10, x9, x10, 1;\n\tshf.l.wrap.b32 x9, x8, x9, 1; shf.l.wrap.b32 x8, x7, x8, 1;\n\tshf.l.wrap.b32 x7, x6, x7, 1; shf.l.wrap.b32 x6, x5, x6, 1;\n\tshf.l.wrap.b32 x5, x4, x5, 1; shf.l.wrap.b32 x4, x3, x4, 1;\n\tshf.l.wrap.b32 x3, x2, x3, 1; shf.l.wrap.b32 x2, x1, x2, 1;\n\tshf.l.wrap.b32 x1, x0, x1, 1;\n\tmov.b64 d0, {x0,x1}; mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n\tmov.b64 d4, {x8,x9}; mov.b64 d5, {x10,x11}; mov.b64 d6, {x12,x13}; mov.b64 d7, {x14,x15};"
#endif
"\n\tmul.wide.u32 t, a0, a0; add.cc.u64 d0, d0, t;\n\tmul.wide.u32 t, a1, a1; addc.cc.u64 d1, d1, t;\n\tmul.wide.u32 t, a2, a2; addc.cc.u64 d2, d2, t;\n\tmul.wide.u32 t, a3, a3; addc.cc.u64 d3, d3, t;\n\tmul.wide.u32 t, a4, a4; addc.cc.u64 d4, d4, t;\n\tmul.wide.u32 t, a5, a5; addc.cc.u64 d5, d5, t;\n\tmul.wide.u32 t, a6, a6; addc.cc.u64 d6, d6, t;\n\tmul.wide.u32 t, a7, a7; addc.u64 d7, d7, t;"
#if QSB_SQL_FRD
/* the doubled-plus-diagonal accumulator already sits in the eight lanes
 * d0..d7: the low four are the fold addends fr0..fr3 and the high four are
 * h0..h3, so only d4..d7 need splitting into the words the 977 multiplies
 * read. Identical values, twelve register moves fewer.
 * -DQSB_SQL_FRD=0 restores the split-and-repack form byte for byte. */
"\n\t.reg .u64 fr0,fr1,fr2,fr3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;"
"\n\tmov.b64 {x8,x9}, d4; mov.b64 {x10,x11}, d5; mov.b64 {x12,x13}, d6; mov.b64 {x14,x15}, d7;\n"
"\tmov.u64 fr0, d0; mov.u64 fr1, d1; mov.u64 fr2, d2; mov.u64 fr3, d3;\n"
"\tmov.u64 h0, d4; mov.u64 h1, d5; mov.u64 h2, d6; mov.u64 h3, d7;\n"
#else
"\n\tmov.b64 {x0,x1}, d0; mov.b64 {x2,x3}, d1; mov.b64 {x4,x5}, d2; mov.b64 {x6,x7}, d3;\n\tmov.b64 {x8,x9}, d4; mov.b64 {x10,x11}, d5; mov.b64 {x12,x13}, d6; mov.b64 {x14,x15}, d7;\n\t.reg .u64 fr0,fr1,fr2,fr3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\tmov.b64 fr0, {x0,x1}; mov.b64 fr1, {x2,x3}; mov.b64 fr2, {x4,x5}; mov.b64 fr3, {x6,x7};\n\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};"
#endif
"\n\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, fr0, t;\n\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, fr1, t;\n\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, fr2, t;\n\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, fr3, t;\n"
#if QSB_SQL_REAP
/* the live square's ninth/tenth words are reaped exactly as the multiply and
 * the fused square already reap theirs (QSB_RP_SQR). Bound: the even fold is
 * L + 977*(x8,x10,x12,x14) with every addend limb below 977*2^32 < 2^42, so
 * f8 (and symmetrically g8) is nonzero only when that 320-bit chain passes
 * 2^288 - 2^42, an event a uniformly distributed operand reaches with
 * probability below 2^-95 per square; the remaining path, a carry out of the
 * w7 seed, needs w7 = 2^32-1 and perturbs that one candidate's x by a
 * multiple of 2^256 mod p. Neither can publish a false hit: QSB_SQL_REAP is
 * #error-gated on QSB_RP_SQR, which pinning.cu gates on QSB_HOST_GATE, and
 * that gate re-derives the public key and both hashes with OpenSSL before a
 * hit is printed, so the only effect is losing that single candidate -- the
 * same trade the multiply and the fused square fold already take.
 * -DQSB_SQL_REAP=0 restores the f8/g8/z9 form byte for byte. */
#else
"\taddc.u32 f8, 0, 0;\n"
#endif
"\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n"
#if !QSB_SQL_REAP
"\taddc.u32 g8, 0, 0;\n"
#endif
#if QSB_SQL_LANE64
/* fold merge in 64-bit lanes: the even fold already leaves f0..f3 as 64-bit
 * registers and the odd fold enters one word up, so the seven carry-ordered
 * 32-bit adds z1+=w0 .. z7+=w6 are exactly four 64-bit lane adds (lane 0
 * takes {0,w0}, lane i takes {w2i-1,w2i}) and w7 stays the ninth-word seed.
 * Exact: the lane boundaries are the word boundaries and the carry between
 * lanes is the carry out of word 2i+1, so the bits are identical; the even
 * fold's four register splits disappear with the word form.
 * -DQSB_SQL_LANE64=0 restores the 32-bit word form byte for byte. */
"\t.reg .u32 zr;\n"
"\tmov.u32 zr, 0;\n"
"\tmov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n"
"\t.reg .u64 qg0,qg1,qg2,qg3;\n"
"\tmov.b64 qg0, {zr,w0}; mov.b64 qg1, {w1,w2}; mov.b64 qg2, {w3,w4}; mov.b64 qg3, {w5,w6};\n"
"\tadd.cc.u64 f0, f0, qg0; addc.cc.u64 f1, f1, qg1;\n"
"\taddc.cc.u64 f2, f2, qg2; addc.cc.u64 f3, f3, qg3;\n"
"\taddc.u32 z8, 0, w7;\n"
#else
"\tmov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1; mov.b64 {z4,z5}, f2; mov.b64 {z6,z7}, f3;\n"
"\tmov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n"
"\tadd.cc.u32 z1, z1, w0; addc.cc.u32 z2, z2, w1; addc.cc.u32 z3, z3, w2;\n"
"\taddc.cc.u32 z4, z4, w3; addc.cc.u32 z5, z5, w4; addc.cc.u32 z6, z6, w5;\n"
#if QSB_SQL_REAP
"\taddc.cc.u32 z7, z7, w6; addc.u32 z8, 0, w7;\n"
#else
"\taddc.cc.u32 z7, z7, w6; addc.cc.u32 z8, f8, w7; addc.u32 z9, g8, 0;\n"
#endif
#endif
#if QSB_SQL_FOLD64
/* top fold in the same lanes. With 2^256 == 2^32 + 977 (mod p) the fold adds
 * z8*(2^32+977), which is below 2^64 + 2^42: its low 64 bits are
 * 977*z8 + (z8<<32) and its 65th bit is that add's carry (set only for
 * z8 >= 2^32-976). Lane 0 takes the low half, lane 1 the carry, and the
 * residual carry dies in lane 2 -- word 5, two whole words above the word-3
 * terminator of the 32-bit tail below -- so this form truncates strictly
 * later than the arm it replaces and needs no recomposition: all four lanes
 * are the output words. -DQSB_SQL_FOLD64=0 restores the 32-bit fold and its
 * documented short-carry tail. */
#if QSB_SQL_FOLD65
"\t{ .reg .u64 sft; .reg .u32 sfl, sfh;\n"
"mul.wide.u32 sft, z8, 977;\n"
"mov.b64 {sfl, sfh}, sft;\n"
"add.u32 sfh, sfh, z8;\n"
"mov.b64 sft, {sfl, sfh};\n"
"add.cc.u64 f0, f0, sft;\n"
#if QSB_SQL_FOLDT2
"addc.u64 f1, f1, 0; }\n"
#else
"addc.cc.u64 f1, f1, 0;\n"
"addc.u64 f2, f2, 0; }\n"
#endif
#else
"\t{ .reg .u64 sft, sfz, sfw;\n"
"mul.wide.u32 sft, z8, 977;\n"
"mov.b64 sfz, {zr, z8};\n"
"add.cc.u64 sft, sft, sfz;\n"
"addc.u64 sfw, 0, 0;\n"
"add.cc.u64 f0, f0, sft;\n"
#if QSB_SQL_FOLDT2
"addc.u64 f1, f1, sfw; }\n"
#else
"addc.cc.u64 f1, f1, sfw;\n"
"addc.u64 f2, f2, 0; }\n"
#endif
#endif
"\tmov.u64 %0, f0; mov.u64 %1, f1; mov.u64 %2, f2; mov.u64 %3, f3;\n"
#else
#if QSB_SQL_LANE64
"\tmov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1;\n"
#endif
#if QSB_SQL_REAP
/* sfq == z8 once the ninth word is gone, so the quotient limb is read out of
 * z8 and the 33-bit multiply-add collapses to the 32-bit one: same
 * sfz = z0 + 2^32*z8, same sft = 977*z8, same carry, and the tail below is
 * the documented QSB_SHORT_CARRY / QSB_CARRY62 terminator, unchanged. */
"\t{ .reg .u64 sfz, sft; .reg .u32 sfc, sfl, sfh;\nmov.b64 sfz, {z0, z8};\nmul.wide.u32 sft, z8, 977;\nadd.cc.u64 sft, sft, sfz;\naddc.u32 sfc, 0, 0;\nmov.b64 {sfl, sfh}, sft;\nmov.u32 z0, sfl;\nadd.cc.u32 z1, z1, sfh;\naddc.cc.u32 z2, z2, sfc; }\n\n"
#else
"\t{ .reg .u64 sfz, sft; .reg .u32 sfc, sfq, sfl, sfh;\nmad.lo.u32 sfq, z9, 977, z8;\nmov.b64 sfz, {z0, sfq};\nmul.wide.u32 sft, z8, 977;\nadd.cc.u64 sft, sft, sfz;\naddc.u32 sfc, z9, 0;\nmov.b64 {sfl, sfh}, sft;\nmov.u32 z0, sfl;\nadd.cc.u32 z1, z1, sfh;\naddc.cc.u32 z2, z2, sfc; }\n\n"
#endif
QSB_SECOND_FOLD_TAIL
#if QSB_SQL_LANE64
"\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.u64 %2, f2; mov.u64 %3, f3;\n"
#else
"\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n"
#endif
#endif
"\t}\n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]));
    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
#else
    uint32_t A[8];
    for (int i=0;i<4;i++){ A[2*i]=(uint32_t)a[i]; A[2*i+1]=(uint32_t)(a[i]>>32); }
    #define PP(i,j) ((uint64_t)A[i]*A[j])
    __uint128_t t;
    /* even column words E_c (cols c,c+1), carry chain E_c -> E_{c+2} */
    uint64_t E2,E4,E6,E8,E10,E12,e14;
    t=(__uint128_t)PP(0,2);                          E2=(uint64_t)t;
    t=(t>>64)+PP(0,4)+PP(1,3);                       E4=(uint64_t)t;
    t=(t>>64)+PP(0,6)+PP(1,5)+PP(2,4);               E6=(uint64_t)t;
    t=(t>>64)+PP(1,7)+PP(2,6)+PP(3,5);               E8=(uint64_t)t;
    t=(t>>64)+PP(3,7)+PP(4,6);                       E10=(uint64_t)t;
    t=(t>>64)+PP(5,7);                               E12=(uint64_t)t;
    e14=(uint64_t)(t>>64);
    /* odd column words O_c */
    uint64_t O1,O3,O5,O7,O9,O11,O13,o15;
    t=(__uint128_t)PP(0,1);                          O1=(uint64_t)t;
    t=(t>>64)+PP(0,3)+PP(1,2);                       O3=(uint64_t)t;
    t=(t>>64)+PP(0,5)+PP(1,4)+PP(2,3);               O5=(uint64_t)t;
    t=(t>>64)+PP(0,7)+PP(1,6)+PP(2,5)+PP(3,4);       O7=(uint64_t)t;
    t=(t>>64)+PP(2,7)+PP(3,6)+PP(4,5);               O9=(uint64_t)t;
    t=(t>>64)+PP(4,7)+PP(5,6);                       O11=(uint64_t)t;
    t=(t>>64)+PP(6,7);                               O13=(uint64_t)t;
    o15=(uint64_t)(t>>64);
    #undef PP
    /* merge E (even cols) + O (odd cols) -> cross-sum X[0..15] (u32 limbs) */
    uint32_t X[16]; uint64_t cc;
    X[0]=0;
    cc=(uint64_t)(uint32_t)O1;                                    X[1]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E2 + (uint32_t)(O1>>32);              X[2]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E2>>32) + (uint32_t)O3;              X[3]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E4 + (uint32_t)(O3>>32);             X[4]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E4>>32) + (uint32_t)O5;             X[5]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E6 + (uint32_t)(O5>>32);             X[6]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E6>>32) + (uint32_t)O7;             X[7]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E8 + (uint32_t)(O7>>32);             X[8]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E8>>32) + (uint32_t)O9;             X[9]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E10 + (uint32_t)(O9>>32);            X[10]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E10>>32) + (uint32_t)O11;           X[11]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E12 + (uint32_t)(O11>>32);           X[12]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E12>>32) + (uint32_t)O13;           X[13]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)e14 + (uint32_t)(O13>>32);           X[14]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)o15;                                 X[15]=(uint32_t)cc;
    /* double: 2*cross */
    uint64_t d=0;
    for (int k=0;k<16;k++){ uint64_t v=((uint64_t)X[k]<<1)|d; X[k]=(uint32_t)v; d=v>>32; }
    /* add the 8 diagonal squares A[i]^2 at columns 2i */
    uint64_t carry=0;
    for (int i=0;i<8;i++){
        __uint128_t s=(__uint128_t)X[2*i] + ((uint64_t)X[2*i+1]<<32) + (uint64_t)A[i]*A[i] + carry;
        X[2*i]=(uint32_t)s; X[2*i+1]=(uint32_t)(s>>32); carry=(uint64_t)(s>>64);
    }
    /* secp256k1 double-fold (identical to the multiply's 6.2 reduction) */
    #define MW(x,y) ((uint64_t)(uint32_t)(x)*(uint32_t)(y))
    uint64_t r0=X[0]|((uint64_t)X[1]<<32), r1=X[2]|((uint64_t)X[3]<<32),
             r2=X[4]|((uint64_t)X[5]<<32), r3=X[6]|((uint64_t)X[7]<<32);
    uint64_t h0=X[8]|((uint64_t)X[9]<<32), h1=X[10]|((uint64_t)X[11]<<32),
             h2=X[12]|((uint64_t)X[13]<<32), h3=X[14]|((uint64_t)X[15]<<32);
    __uint128_t s;
    uint64_t f0,f1,f2,f3; uint32_t f8;
    s=(__uint128_t)r0+MW(X[8],977);  f0=(uint64_t)s;
    s=(s>>64)+r1+MW(X[10],977); f1=(uint64_t)s;
    s=(s>>64)+r2+MW(X[12],977); f2=(uint64_t)s;
    s=(s>>64)+r3+MW(X[14],977); f3=(uint64_t)s;
    f8=(uint32_t)(s>>64);
    uint64_t g0,g1,g2,g3; uint32_t g8;
    s=(__uint128_t)h0+MW(X[9],977);  g0=(uint64_t)s;
    s=(s>>64)+h1+MW(X[11],977); g1=(uint64_t)s;
    s=(s>>64)+h2+MW(X[13],977); g2=(uint64_t)s;
    s=(s>>64)+h3+MW(X[15],977); g3=(uint64_t)s;
    g8=(uint32_t)(s>>64);
    uint32_t z[10], w[8];
    z[0]=(uint32_t)f0;z[1]=(uint32_t)(f0>>32);z[2]=(uint32_t)f1;z[3]=(uint32_t)(f1>>32);
    z[4]=(uint32_t)f2;z[5]=(uint32_t)(f2>>32);z[6]=(uint32_t)f3;z[7]=(uint32_t)(f3>>32);
    w[0]=(uint32_t)g0;w[1]=(uint32_t)(g0>>32);w[2]=(uint32_t)g1;w[3]=(uint32_t)(g1>>32);
    w[4]=(uint32_t)g2;w[5]=(uint32_t)(g2>>32);w[6]=(uint32_t)g3;w[7]=(uint32_t)(g3>>32);
    { uint64_t c=0,tt;
      for(int k=0;k<7;k++){ tt=(uint64_t)z[k+1]+w[k]+c; z[k+1]=(uint32_t)tt; c=tt>>32; }
      tt=(uint64_t)f8+w[7]+c; z[8]=(uint32_t)tt; c=tt>>32; z[9]=(uint32_t)((uint64_t)g8+c); }
    uint64_t tt2=MW(z[8],977);
    uint32_t m0=(uint32_t)tt2,m1=(uint32_t)(tt2>>32),m2;
    m1=(uint32_t)(m1+(uint32_t)((uint64_t)z[9]*977));
    { uint64_t tv=(uint64_t)m1+z[8]; m1=(uint32_t)tv; uint64_t c=tv>>32; m2=(uint32_t)((uint64_t)z[9]+c); }
    { uint64_t c=0,tv;
      tv=(uint64_t)z[0]+m0+c; z[0]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[1]+m1+c; z[1]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[2]+m2+c; z[2]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[3]+c; z[3]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[4]+c; z[4]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[5]+c; z[5]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[6]+c; z[6]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[7]+c; z[7]=(uint32_t)tv; }
    r[0]=z[0]|((uint64_t)z[1]<<32); r[1]=z[2]|((uint64_t)z[3]<<32);
    r[2]=z[4]|((uint64_t)z[5]<<32); r[3]=z[6]|((uint64_t)z[7]<<32);
    #undef MW
    (void)h0;(void)h1;(void)h2;(void)h3;(void)r0;(void)r1;(void)r2;(void)r3;(void)carry;(void)d;
#endif
}

#else
__device__ __forceinline__ void _ModSqr(uint64_t r[4], const uint64_t a[4]) {
#ifdef __CUDA_ARCH__
    uint64_t r0, r1, r2, r3;
    asm("{\n\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7;\n\t.reg .u64 e2,e4,e6,e8,e10,e12,o1,o3,o5,o7,o9,o11,o13,t;\n\t.reg .u32 ecy,ocy,e14,o15;\n\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n\t.reg .u32 y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n\t.reg .u64 d0,d1,d2,d3,d4,d5,d6,d7;\n\tmov.b64 {a0,a1}, %4;\n\tmov.b64 {a2,a3}, %5;\n\tmov.b64 {a4,a5}, %6;\n\tmov.b64 {a6,a7}, %7;\n\t.reg .u64 odd_t;\nmul.wide.u32 e6, a2, a4;\nmul.wide.u32 o7, a3, a4;\nmul.wide.u32 e8, a3, a5;\nmul.wide.u32 o5, a2, a3;\nmul.wide.u32 e4, a1, a3;\nmul.wide.u32 odd_t, a2, a5;\nmul.wide.u32 t, a1, a5;\nadd.cc.u64 o7, o7, odd_t;\nmul.wide.u32 odd_t, a4, a5;\naddc.u64 o9, odd_t, 0;\nadd.cc.u64 e6, e6, t;\nmul.wide.u32 t, a2, a6;\naddc.cc.u64 e8, e8, t;\nmul.wide.u32 t, a4, a6;\naddc.u64 e10, t, 0;\nmul.wide.u32 o3, a1, a2;\nmul.wide.u32 e2, a0, a2;\nmul.wide.u32 odd_t, a1, a4;\nmul.wide.u32 t, a0, a4;\nadd.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a1, a6;\naddc.cc.u64 o7, o7, odd_t;\nmul.wide.u32 odd_t, a3, a6;\naddc.cc.u64 o9, o9, odd_t;\nmul.wide.u32 odd_t, a5, a6;\naddc.u64 o11, odd_t, 0;\nadd.cc.u64 e4, e4, t;\nmul.wide.u32 t, a0, a6;\naddc.cc.u64 e6, e6, t;\nmul.wide.u32 t, a1, a7;\naddc.cc.u64 e8, e8, t;\nmul.wide.u32 t, a3, a7;\naddc.cc.u64 e10, e10, t;\nmul.wide.u32 t, a5, a7;\naddc.u64 e12, t, 0;\nmul.wide.u32 o1, a0, a1;\nmul.wide.u32 odd_t, a0, a3;\nadd.cc.u64 o3, o3, odd_t;\nmul.wide.u32 odd_t, a0, a5;\naddc.cc.u64 o5, o5, odd_t;\nmul.wide.u32 odd_t, a0, a7;\naddc.cc.u64 o7, o7, odd_t;\nmul.wide.u32 odd_t, a2, a7;\naddc.cc.u64 o9, o9, odd_t;\nmul.wide.u32 odd_t, a4, a7;\naddc.cc.u64 o11, o11, odd_t;\nmul.wide.u32 odd_t, a6, a7;\naddc.u64 o13, odd_t, 0;\nmov.u32 x0, 0;\n\tmov.b64 {x2,x3}, e2; mov.b64 {x4,x5}, e4; mov.b64 {x6,x7}, e6;\n\tmov.b64 {x8,x9}, e8; mov.b64 {x10,x11}, e10; mov.b64 {x12,x13}, e12;\n\tmov.u32 x14, 0; mov.u32 x15, 0;\n\tmov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n\tmov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n\tadd.cc.u32 x2, x2, y2; addc.cc.u32 x3, x3, y3;\n\taddc.cc.u32 x4, x4, y4; addc.cc.u32 x5, x5, y5; addc.cc.u32 x6, x6, y6;\n\taddc.cc.u32 x7, x7, y7; addc.cc.u32 x8, x8, y8; addc.cc.u32 x9, x9, y9;\n\taddc.cc.u32 x10, x10, y10; addc.cc.u32 x11, x11, y11; addc.cc.u32 x12, x12, y12;\n\taddc.cc.u32 x13, x13, y13; addc.cc.u32 x14, x14, y14;\naddc.u32 x15, x15, 0;\nshf.l.wrap.b32 x15, x14, x15, 1; shf.l.wrap.b32 x14, x13, x14, 1;\n\tshf.l.wrap.b32 x13, x12, x13, 1; shf.l.wrap.b32 x12, x11, x12, 1;\n\tshf.l.wrap.b32 x11, x10, x11, 1; shf.l.wrap.b32 x10, x9, x10, 1;\n\tshf.l.wrap.b32 x9, x8, x9, 1; shf.l.wrap.b32 x8, x7, x8, 1;\n\tshf.l.wrap.b32 x7, x6, x7, 1; shf.l.wrap.b32 x6, x5, x6, 1;\n\tshf.l.wrap.b32 x5, x4, x5, 1; shf.l.wrap.b32 x4, x3, x4, 1;\n\tshf.l.wrap.b32 x3, x2, x3, 1; shf.l.wrap.b32 x2, x1, x2, 1;\n\tshf.l.wrap.b32 x1, x0, x1, 1;\n\tmov.b64 d0, {x0,x1}; mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n\tmov.b64 d4, {x8,x9}; mov.b64 d5, {x10,x11}; mov.b64 d6, {x12,x13}; mov.b64 d7, {x14,x15};\n\tmul.wide.u32 t, a0, a0; add.cc.u64 d0, d0, t;\n\tmul.wide.u32 t, a1, a1; addc.cc.u64 d1, d1, t;\n\tmul.wide.u32 t, a2, a2; addc.cc.u64 d2, d2, t;\n\tmul.wide.u32 t, a3, a3; addc.cc.u64 d3, d3, t;\n\tmul.wide.u32 t, a4, a4; addc.cc.u64 d4, d4, t;\n\tmul.wide.u32 t, a5, a5; addc.cc.u64 d5, d5, t;\n\tmul.wide.u32 t, a6, a6; addc.cc.u64 d6, d6, t;\n\tmul.wide.u32 t, a7, a7; addc.u64 d7, d7, t;\n\tmov.b64 {x0,x1}, d0; mov.b64 {x2,x3}, d1; mov.b64 {x4,x5}, d2; mov.b64 {x6,x7}, d3;\n\tmov.b64 {x8,x9}, d4; mov.b64 {x10,x11}, d5; mov.b64 {x12,x13}, d6; mov.b64 {x14,x15}, d7;\n\t.reg .u64 fr0,fr1,fr2,fr3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n\tmov.b64 fr0, {x0,x1}; mov.b64 fr1, {x2,x3}; mov.b64 fr2, {x4,x5}; mov.b64 fr3, {x6,x7};\n\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, fr0, t;\n\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, fr1, t;\n\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, fr2, t;\n\t	mul.wide.u32 t, x14, 977; addc.cc.u64 f3, fr3, t;\n"
QSB_F8_CAP
"\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n"
"\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n"
"\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n"
"\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n"
QSB_SQR_G8
#if QSB_SQR_LANE64
/* fold merge in 64-bit lanes: the even fold is already there, the odd fold
 * enters one word up and w7 stays the ninth-word seed. */
"\t.reg .u32 zr;\n"
"\tmov.u32 zr, 0;\n"
"\tmov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n"
"\t.reg .u64 qg0,qg1,qg2,qg3;\n"
"\tmov.b64 qg0, {zr,w0}; mov.b64 qg1, {w1,w2}; mov.b64 qg2, {w3,w4}; mov.b64 qg3, {w5,w6};\n"
"\tadd.cc.u64 f0, f0, qg0; addc.cc.u64 f1, f1, qg1;\n"
"\taddc.cc.u64 f2, f2, qg2; addc.cc.u64 f3, f3, qg3;\n"
QSB_SQR_Z89
#else
"\tmov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1; mov.b64 {z4,z5}, f2; mov.b64 {z6,z7}, f3;\n"
"\tmov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n"
"\tadd.cc.u32 z1, z1, w0; addc.cc.u32 z2, z2, w1; addc.cc.u32 z3, z3, w2;\n"
"\taddc.cc.u32 z4, z4, w3; addc.cc.u32 z5, z5, w4; addc.cc.u32 z6, z6, w5;\n"
"\taddc.cc.u32 z7, z7, w6; " QSB_SQR_Z89
#endif
#if QSB_SQR_FOLD64
"\t{ .reg .u64 sft, sfz, sfw;\n"
"mul.wide.u32 sft, z8, 977;\n"
"mov.b64 sfz, {zr, z8};\n"
"add.cc.u64 sft, sft, sfz;\n"
"addc.u64 sfw, 0, 0;\n"
"add.cc.u64 f0, f0, sft;\n"
"addc.cc.u64 f1, f1, sfw;\n"
"addc.u64 f2, f2, 0; }\n"
"\tmov.u64 %0, f0; mov.u64 %1, f1; mov.u64 %2, f2; mov.u64 %3, f3;\n"
#else
#if QSB_SQR_LANE64
"\tmov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1;\n"
#endif
/* restored arm (QSB_SQR_FOLD64=0): the crown's 32-bit top fold and its
 * documented short carry tail, unchanged. Bound as on the crown: the pre-fold
 * accumulator is below 2^256 + K*(K+4) with K = 2^32+977, so the fold's
 * quotient limb is one word and the tail can only carry when the low limbs
 * are all-ones, the QSB_SHORT_CARRY / QSB_CARRY62 exposure stated above. */
"\t{ .reg .u64 sfz, sft; .reg .u32 sfc, sfq, sfl, sfh;\n"
QSB_SQR_SF_HEAD
"mov.b64 sfz, {z0, sfq};\n"
"mul.wide.u32 sft, z8, 977;\n"
"add.cc.u64 sft, sft, sfz;\n"
QSB_SQR_SFC
"mov.b64 {sfl, sfh}, sft;\n"
"mov.u32 z0, sfl;\n"
"add.cc.u32 z1, z1, sfh;\n"
"addc.cc.u32 z2, z2, sfc; }\n\n"
QSB_SECOND_FOLD_TAIL
#if QSB_SQR_LANE64
"\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.u64 %2, f2; mov.u64 %3, f3;\n"
#else
"\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n"
#endif
#endif
"\t}\n\t"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]));
    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
#else
    uint32_t A[8];
    for (int i=0;i<4;i++){ A[2*i]=(uint32_t)a[i]; A[2*i+1]=(uint32_t)(a[i]>>32); }
    #define PP(i,j) ((uint64_t)A[i]*A[j])
    __uint128_t t;
    /* even column words E_c (cols c,c+1), carry chain E_c -> E_{c+2} */
    uint64_t E2,E4,E6,E8,E10,E12,e14;
    t=(__uint128_t)PP(0,2);                          E2=(uint64_t)t;
    t=(t>>64)+PP(0,4)+PP(1,3);                       E4=(uint64_t)t;
    t=(t>>64)+PP(0,6)+PP(1,5)+PP(2,4);               E6=(uint64_t)t;
    t=(t>>64)+PP(1,7)+PP(2,6)+PP(3,5);               E8=(uint64_t)t;
    t=(t>>64)+PP(3,7)+PP(4,6);                       E10=(uint64_t)t;
    t=(t>>64)+PP(5,7);                               E12=(uint64_t)t;
    e14=(uint64_t)(t>>64);
    /* odd column words O_c */
    uint64_t O1,O3,O5,O7,O9,O11,O13,o15;
    t=(__uint128_t)PP(0,1);                          O1=(uint64_t)t;
    t=(t>>64)+PP(0,3)+PP(1,2);                       O3=(uint64_t)t;
    t=(t>>64)+PP(0,5)+PP(1,4)+PP(2,3);               O5=(uint64_t)t;
    t=(t>>64)+PP(0,7)+PP(1,6)+PP(2,5)+PP(3,4);       O7=(uint64_t)t;
    t=(t>>64)+PP(2,7)+PP(3,6)+PP(4,5);               O9=(uint64_t)t;
    t=(t>>64)+PP(4,7)+PP(5,6);                       O11=(uint64_t)t;
    t=(t>>64)+PP(6,7);                               O13=(uint64_t)t;
    o15=(uint64_t)(t>>64);
    #undef PP
    /* merge E (even cols) + O (odd cols) -> cross-sum X[0..15] (u32 limbs) */
    uint32_t X[16]; uint64_t cc;
    X[0]=0;
    cc=(uint64_t)(uint32_t)O1;                                    X[1]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E2 + (uint32_t)(O1>>32);              X[2]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E2>>32) + (uint32_t)O3;              X[3]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E4 + (uint32_t)(O3>>32);             X[4]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E4>>32) + (uint32_t)O5;             X[5]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E6 + (uint32_t)(O5>>32);             X[6]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E6>>32) + (uint32_t)O7;             X[7]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E8 + (uint32_t)(O7>>32);             X[8]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E8>>32) + (uint32_t)O9;             X[9]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E10 + (uint32_t)(O9>>32);            X[10]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E10>>32) + (uint32_t)O11;           X[11]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E12 + (uint32_t)(O11>>32);           X[12]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E12>>32) + (uint32_t)O13;           X[13]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)e14 + (uint32_t)(O13>>32);           X[14]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)o15;                                 X[15]=(uint32_t)cc;
    /* double: 2*cross */
    uint64_t d=0;
    for (int k=0;k<16;k++){ uint64_t v=((uint64_t)X[k]<<1)|d; X[k]=(uint32_t)v; d=v>>32; }
    /* add the 8 diagonal squares A[i]^2 at columns 2i */
    uint64_t carry=0;
    for (int i=0;i<8;i++){
        __uint128_t s=(__uint128_t)X[2*i] + ((uint64_t)X[2*i+1]<<32) + (uint64_t)A[i]*A[i] + carry;
        X[2*i]=(uint32_t)s; X[2*i+1]=(uint32_t)(s>>32); carry=(uint64_t)(s>>64);
    }
    /* secp256k1 double-fold (identical to the multiply's 6.2 reduction) */
    #define MW(x,y) ((uint64_t)(uint32_t)(x)*(uint32_t)(y))
    uint64_t r0=X[0]|((uint64_t)X[1]<<32), r1=X[2]|((uint64_t)X[3]<<32),
             r2=X[4]|((uint64_t)X[5]<<32), r3=X[6]|((uint64_t)X[7]<<32);
    uint64_t h0=X[8]|((uint64_t)X[9]<<32), h1=X[10]|((uint64_t)X[11]<<32),
             h2=X[12]|((uint64_t)X[13]<<32), h3=X[14]|((uint64_t)X[15]<<32);
    __uint128_t s;
    uint64_t f0,f1,f2,f3; uint32_t f8;
    s=(__uint128_t)r0+MW(X[8],977);  f0=(uint64_t)s;
    s=(s>>64)+r1+MW(X[10],977); f1=(uint64_t)s;
    s=(s>>64)+r2+MW(X[12],977); f2=(uint64_t)s;
    s=(s>>64)+r3+MW(X[14],977); f3=(uint64_t)s;
    f8=(uint32_t)(s>>64);
    uint64_t g0,g1,g2,g3; uint32_t g8;
    s=(__uint128_t)h0+MW(X[9],977);  g0=(uint64_t)s;
    s=(s>>64)+h1+MW(X[11],977); g1=(uint64_t)s;
    s=(s>>64)+h2+MW(X[13],977); g2=(uint64_t)s;
    s=(s>>64)+h3+MW(X[15],977); g3=(uint64_t)s;
    g8=(uint32_t)(s>>64);
    uint32_t z[10], w[8];
    z[0]=(uint32_t)f0;z[1]=(uint32_t)(f0>>32);z[2]=(uint32_t)f1;z[3]=(uint32_t)(f1>>32);
    z[4]=(uint32_t)f2;z[5]=(uint32_t)(f2>>32);z[6]=(uint32_t)f3;z[7]=(uint32_t)(f3>>32);
    w[0]=(uint32_t)g0;w[1]=(uint32_t)(g0>>32);w[2]=(uint32_t)g1;w[3]=(uint32_t)(g1>>32);
    w[4]=(uint32_t)g2;w[5]=(uint32_t)(g2>>32);w[6]=(uint32_t)g3;w[7]=(uint32_t)(g3>>32);
    { uint64_t c=0,tt;
      for(int k=0;k<7;k++){ tt=(uint64_t)z[k+1]+w[k]+c; z[k+1]=(uint32_t)tt; c=tt>>32; }
      tt=(uint64_t)f8+w[7]+c; z[8]=(uint32_t)tt; c=tt>>32; z[9]=(uint32_t)((uint64_t)g8+c); }
    uint64_t tt2=MW(z[8],977);
    uint32_t m0=(uint32_t)tt2,m1=(uint32_t)(tt2>>32),m2;
    m1=(uint32_t)(m1+(uint32_t)((uint64_t)z[9]*977));
    { uint64_t tv=(uint64_t)m1+z[8]; m1=(uint32_t)tv; uint64_t c=tv>>32; m2=(uint32_t)((uint64_t)z[9]+c); }
    { uint64_t c=0,tv;
      tv=(uint64_t)z[0]+m0+c; z[0]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[1]+m1+c; z[1]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[2]+m2+c; z[2]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[3]+c; z[3]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[4]+c; z[4]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[5]+c; z[5]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[6]+c; z[6]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[7]+c; z[7]=(uint32_t)tv; }
    r[0]=z[0]|((uint64_t)z[1]<<32); r[1]=z[2]|((uint64_t)z[3]<<32);
    r[2]=z[4]|((uint64_t)z[5]<<32); r[3]=z[6]|((uint64_t)z[7]<<32);
    #undef MW
    (void)h0;(void)h1;(void)h2;(void)h3;(void)r0;(void)r1;(void)r2;(void)r3;(void)carry;(void)d;
#endif
}

#endif
#if QSB_FUSE_SQRADDSUB2
// Experimental genuine fused r*r+e-2q reduction. The dedicated 8x32
// square schedule is unchanged; 3p+e-2q enters its first 320-bit fold before
// the remaining folds. All four pointers may alias, and inputs may be any
// 256-bit representatives. The result is congruent in [0, 2^256).
__device__ __forceinline__ void _ModSqrAddSub2(uint64_t out[4], const uint64_t a[4], const uint64_t e[4], const uint64_t q[4]) {
#ifdef __CUDA_ARCH__
    uint64_t r0, r1, r2, r3;
    asm(
        /* Final correction needs only three 32-bit limbs. With B=2^256
         * and K=2^32+977, the previous fold is below B+K*(K+4).
         * If it carries, low+K < K*(K+5) < 2^65; otherwise the
         * correction is zero. Upper five limbs remain unchanged. */
        "{\n"
        "\t.reg .u32 a0,a1,a2,a3,a4,a5,a6,a7;\n"
        "\t.reg .u64 e2,e4,e6,e8,e10,e12,o1,o3,o5,o7,o9,o11,o13,t;\n"
        "\t.reg .u32 ecy,ocy,e14,o15;\n"
        "\t.reg .u32 x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;\n"
        "\t.reg .u32 y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14;\n"
        "\t.reg .u64 d0,d1,d2,d3,d4,d5,d6,d7;\n"
        "\tmov.b64 {a0,a1}, %4;\n"
        "\tmov.b64 {a2,a3}, %5;\n"
        "\tmov.b64 {a4,a5}, %6;\n"
        "\tmov.b64 {a6,a7}, %7;\n"
        "\t.reg .u64 odd_t;\n"
        "mul.wide.u32 e6, a2, a4;\n"
        "mul.wide.u32 o7, a3, a4;\n"
        "mul.wide.u32 e8, a3, a5;\n"
        "mul.wide.u32 o5, a2, a3;\n"
        "mul.wide.u32 e4, a1, a3;\n"
        "mul.wide.u32 odd_t, a2, a5;\n"
        "mul.wide.u32 t, a1, a5;\n"
        "add.cc.u64 o7, o7, odd_t;\n"
        "mul.wide.u32 odd_t, a4, a5;\n"
        "addc.u64 o9, odd_t, 0;\n"
        "add.cc.u64 e6, e6, t;\n"
        "mul.wide.u32 t, a2, a6;\n"
        "addc.cc.u64 e8, e8, t;\n"
        "mul.wide.u32 t, a4, a6;\n"
        "addc.u64 e10, t, 0;\n"
        "mul.wide.u32 o3, a1, a2;\n"
        "mul.wide.u32 e2, a0, a2;\n"
        "mul.wide.u32 odd_t, a1, a4;\n"
        "mul.wide.u32 t, a0, a4;\n"
        "add.cc.u64 o5, o5, odd_t;\n"
        "mul.wide.u32 odd_t, a1, a6;\n"
        "addc.cc.u64 o7, o7, odd_t;\n"
        "mul.wide.u32 odd_t, a3, a6;\n"
        "addc.cc.u64 o9, o9, odd_t;\n"
        "mul.wide.u32 odd_t, a5, a6;\n"
        "addc.u64 o11, odd_t, 0;\n"
        "add.cc.u64 e4, e4, t;\n"
        "mul.wide.u32 t, a0, a6;\n"
        "addc.cc.u64 e6, e6, t;\n"
        "mul.wide.u32 t, a1, a7;\n"
        "addc.cc.u64 e8, e8, t;\n"
        "mul.wide.u32 t, a3, a7;\n"
        "addc.cc.u64 e10, e10, t;\n"
        "mul.wide.u32 t, a5, a7;\n"
        "addc.u64 e12, t, 0;\n"
        "mul.wide.u32 o1, a0, a1;\n"
        "mul.wide.u32 odd_t, a0, a3;\n"
        "add.cc.u64 o3, o3, odd_t;\n"
        "mul.wide.u32 odd_t, a0, a5;\n"
        "addc.cc.u64 o5, o5, odd_t;\n"
        "mul.wide.u32 odd_t, a0, a7;\n"
        "addc.cc.u64 o7, o7, odd_t;\n"
        "mul.wide.u32 odd_t, a2, a7;\n"
        "addc.cc.u64 o9, o9, odd_t;\n"
        "mul.wide.u32 odd_t, a4, a7;\n"
        "addc.cc.u64 o11, o11, odd_t;\n"
        "mul.wide.u32 odd_t, a6, a7;\n"
        "addc.u64 o13, odd_t, 0;\n"
        "mov.u32 x0, 0;\n"
#if QSB_SAS_XMERGE64
        /* lane-form cross-sum merge. The even columns already are the 64-bit
         * lanes e2..e12 (words 2..13) and the odd columns re-pair into lanes
         * one word up, so the fifteen carry-ordered 32-bit adds plus the
         * sixteen-word repack collapse into seven pairings and seven lane
         * adds. Exact: lane i is words 2i,2i+1, the carry between lanes is
         * exactly the carry out of word 2i+1, the top even column cannot
         * overflow its own lane (PP(5,7) plus one carry stays below
         * 2^64 - 2^33, which is why the word form set x14 = x15 = 0), so word
         * 14 holds the odd top y14 plus the chain carry and word 15 stays
         * zero -- bit for bit the cross sum the word chain produced. The
         * doubling is the same eight-lane self-add chain, and 2*cross plus
         * the diagonal is below 2^512, so its top carry is dropped exactly as
         * the funnel form dropped the bit shifted out of x15.
         * -DQSB_SAS_XMERGE64=0 restores the word merge byte for byte. */
        "\tmov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n"
        "\tmov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n"
        "\tmov.b64 d0, {x0,x1}; mov.b64 d1, {y2,y3}; mov.b64 d2, {y4,y5}; mov.b64 d3, {y6,y7};\n"
        "\tmov.b64 d4, {y8,y9}; mov.b64 d5, {y10,y11}; mov.b64 d6, {y12,y13}; mov.b64 d7, {y14,x0};\n"
        "\tadd.cc.u64 d1, d1, e2; addc.cc.u64 d2, d2, e4; addc.cc.u64 d3, d3, e6;\n"
        "\taddc.cc.u64 d4, d4, e8; addc.cc.u64 d5, d5, e10; addc.cc.u64 d6, d6, e12;\n"
        "\taddc.u64 d7, d7, 0;\n"
        "\tadd.cc.u64 d0, d0, d0; addc.cc.u64 d1, d1, d1;\n"
        "\taddc.cc.u64 d2, d2, d2; addc.cc.u64 d3, d3, d3;\n"
        "\taddc.cc.u64 d4, d4, d4; addc.cc.u64 d5, d5, d5;\n"
        "\taddc.cc.u64 d6, d6, d6; addc.u64 d7, d7, d7;\n"
#else
        "\tmov.b64 {x2,x3}, e2; mov.b64 {x4,x5}, e4; mov.b64 {x6,x7}, e6;\n"
        "\tmov.b64 {x8,x9}, e8; mov.b64 {x10,x11}, e10; mov.b64 {x12,x13}, e12;\n"
        "\tmov.u32 x14, 0; mov.u32 x15, 0;\n"
        "\tmov.b64 {x1,y2}, o1; mov.b64 {y3,y4}, o3; mov.b64 {y5,y6}, o5;\n"
        "\tmov.b64 {y7,y8}, o7; mov.b64 {y9,y10}, o9; mov.b64 {y11,y12}, o11; mov.b64 {y13,y14}, o13;\n"
        "\tadd.cc.u32 x2, x2, y2; addc.cc.u32 x3, x3, y3;\n"
        "\taddc.cc.u32 x4, x4, y4; addc.cc.u32 x5, x5, y5; addc.cc.u32 x6, x6, y6;\n"
        "\taddc.cc.u32 x7, x7, y7; addc.cc.u32 x8, x8, y8; addc.cc.u32 x9, x9, y9;\n"
        "\taddc.cc.u32 x10, x10, y10; addc.cc.u32 x11, x11, y11; addc.cc.u32 x12, x12, y12;\n"
        "\taddc.cc.u32 x13, x13, y13; addc.cc.u32 x14, x14, y14;\naddc.u32 x15, x15, 0;\n"
#if QSB_SAS_DBL64
        "\tmov.b64 d0, {x0,x1}; mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n"
        "\tmov.b64 d4, {x8,x9}; mov.b64 d5, {x10,x11}; mov.b64 d6, {x12,x13}; mov.b64 d7, {x14,x15};\n"
        "\tadd.cc.u64 d0, d0, d0; addc.cc.u64 d1, d1, d1;\n"
        "\taddc.cc.u64 d2, d2, d2; addc.cc.u64 d3, d3, d3;\n"
        "\taddc.cc.u64 d4, d4, d4; addc.cc.u64 d5, d5, d5;\n"
        "\taddc.cc.u64 d6, d6, d6; addc.u64 d7, d7, d7;\n"
#else
        "\tshf.l.wrap.b32 x15, x14, x15, 1; shf.l.wrap.b32 x14, x13, x14, 1;\n"
        "\tshf.l.wrap.b32 x13, x12, x13, 1; shf.l.wrap.b32 x12, x11, x12, 1;\n"
        "\tshf.l.wrap.b32 x11, x10, x11, 1; shf.l.wrap.b32 x10, x9, x10, 1;\n"
        "\tshf.l.wrap.b32 x9, x8, x9, 1; shf.l.wrap.b32 x8, x7, x8, 1;\n"
        "\tshf.l.wrap.b32 x7, x6, x7, 1; shf.l.wrap.b32 x6, x5, x6, 1;\n"
        "\tshf.l.wrap.b32 x5, x4, x5, 1; shf.l.wrap.b32 x4, x3, x4, 1;\n"
        "\tshf.l.wrap.b32 x3, x2, x3, 1; shf.l.wrap.b32 x2, x1, x2, 1;\n"
        "\tshf.l.wrap.b32 x1, x0, x1, 1;\n"
        "\tmov.b64 d0, {x0,x1}; mov.b64 d1, {x2,x3}; mov.b64 d2, {x4,x5}; mov.b64 d3, {x6,x7};\n"
        "\tmov.b64 d4, {x8,x9}; mov.b64 d5, {x10,x11}; mov.b64 d6, {x12,x13}; mov.b64 d7, {x14,x15};\n"
#endif
#endif
        "\tmul.wide.u32 t, a0, a0; add.cc.u64 d0, d0, t;\n"
        "\tmul.wide.u32 t, a1, a1; addc.cc.u64 d1, d1, t;\n"
        "\tmul.wide.u32 t, a2, a2; addc.cc.u64 d2, d2, t;\n"
        "\tmul.wide.u32 t, a3, a3; addc.cc.u64 d3, d3, t;\n"
        "\tmul.wide.u32 t, a4, a4; addc.cc.u64 d4, d4, t;\n"
        "\tmul.wide.u32 t, a5, a5; addc.cc.u64 d5, d5, t;\n"
        "\tmul.wide.u32 t, a6, a6; addc.cc.u64 d6, d6, t;\n"
        "\tmul.wide.u32 t, a7, a7; addc.u64 d7, d7, t;\n"
        "\t.reg .u32 u0,u1,u2,u3,u4,u5,u6,u7,v0,v1,v2,v3,v4,v5,v6,v7,cf,k0;\n"
        ".reg .u64 fr0,fr1,fr2,fr3,h0,h1,h2,h3,f0,f1,f2,f3,g0,g1,g2,g3;\n"
        "\t.reg .u32 f8,g8,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9,w0,w1,w2,w3,w4,w5,w6,w7,m0,m1,m2;\n"
#if QSB_SAS_FRD
        "\tmov.b64 {x8,x9}, d4; mov.b64 {x10,x11}, d5; mov.b64 {x12,x13}, d6; mov.b64 {x14,x15}, d7;\n"
        "\tmov.u64 fr0, d0; mov.u64 fr1, d1; mov.u64 fr2, d2; mov.u64 fr3, d3;\n"
        "\tmov.u64 h0, d4; mov.u64 h1, d5; mov.u64 h2, d6; mov.u64 h3, d7;\n"
#else
        "\tmov.b64 {x0,x1}, d0; mov.b64 {x2,x3}, d1; mov.b64 {x4,x5}, d2; mov.b64 {x6,x7}, d3;\n"
        "\tmov.b64 {x8,x9}, d4; mov.b64 {x10,x11}, d5; mov.b64 {x12,x13}, d6; mov.b64 {x14,x15}, d7;\n"
        "\tmov.b64 fr0, {x0,x1}; mov.b64 fr1, {x2,x3}; mov.b64 fr2, {x4,x5}; mov.b64 fr3, {x6,x7};\n"
        "\tmov.b64 h0, {x8,x9}; mov.b64 h1, {x10,x11}; mov.b64 h2, {x12,x13}; mov.b64 h3, {x14,x15};\n"
#endif
        /* Both 977 folds are exact: each column word is at most
         * (2^32-1)*977 < 2^42, the even chain is fr + 977*(x8,x10,x12,x14)
         * and the odd chain h + 977*(x9,x11,x13,x15), so neither exceeds
         * 2^288 and the ninth-word captures are the documented
         * QSB_RP_SQR / QSB_SHORT_CARRY reaps (switchable at those names).
         * The lane arms below only regroup these same sums. */
        "\tmul.wide.u32 t, x8, 977;  add.cc.u64  f0, fr0, t;\n"
        "\tmul.wide.u32 t, x10, 977; addc.cc.u64 f1, fr1, t;\n"
        "\tmul.wide.u32 t, x12, 977; addc.cc.u64 f2, fr2, t;\n"
        "\tmul.wide.u32 t, x14, 977; addc.cc.u64 f3, fr3, t;\n"
        QSB_F8_CAP
        "\tmul.wide.u32 t, x9, 977;  add.cc.u64  g0, h0, t;\n"
        "\tmul.wide.u32 t, x11, 977; addc.cc.u64 g1, h1, t;\n"
        "\tmul.wide.u32 t, x13, 977; addc.cc.u64 g2, h2, t;\n"
        "\tmul.wide.u32 t, x15, 977; addc.cc.u64 g3, h3, t;\n"
        QSB_SAS_G8
#if QSB_SAS_LANE64
        /* fold merge in 64-bit lanes: the even fold is already there, the odd
         * fold is re-paired one word up and w7 stays the ninth-word seed. */
        "\t.reg .u32 zr;\n"
        "\tmov.u32 zr, 0;\n"
        "\tmov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n"
        "\t.reg .u64 lg0,lg1,lg2,lg3;\n"
        "\tmov.b64 lg0, {zr,w0}; mov.b64 lg1, {w1,w2}; mov.b64 lg2, {w3,w4}; mov.b64 lg3, {w5,w6};\n"
        "\tadd.cc.u64 f0, f0, lg0; addc.cc.u64 f1, f1, lg1;\n"
        "\taddc.cc.u64 f2, f2, lg2; addc.cc.u64 f3, f3, lg3;\n"
        QSB_SAS_Z89
#else
        "\tmov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1; mov.b64 {z4,z5}, f2; mov.b64 {z6,z7}, f3;\n"
        "\tmov.b64 {w0,w1}, g0; mov.b64 {w2,w3}, g1; mov.b64 {w4,w5}, g2; mov.b64 {w6,w7}, g3;\n"
        "\tadd.cc.u32 z1, z1, w0; addc.cc.u32 z2, z2, w1; addc.cc.u32 z3, z3, w2;\n"
        "\taddc.cc.u32 z4, z4, w3; addc.cc.u32 z5, z5, w4; addc.cc.u32 z6, z6, w5;\n"
        "\taddc.cc.u32 z7, z7, w6; " QSB_SAS_Z89
#endif
#if !QSB_SAS_LANE64
        "\tmov.b64 {u0,u1}, %8; mov.b64 {u2,u3}, %9;\n"
        "\tmov.b64 {u4,u5}, %10; mov.b64 {u6,u7}, %11;\n"
#endif
#if !QSB_SAS_Q2LANE
        "\tmov.b64 {v0,v1}, %12; mov.b64 {v2,v3}, %13;\n"
        "\tmov.b64 {v4,v5}, %14; mov.b64 {v6,v7}, %15;\n"
#endif
#if QSB_SAS_LANE64
        /* S = e + 3*2^256 - 2q, built from the operand registers only: 2q as
         * four lane self-adds (QSB_SAS_Q2LANE) or nine funnel shifts, then four
         * 64-bit lanes and one top word for S. The accumulator takes it in four
         * carry-ordered lanes plus the top, and 3K is one 64-bit immediate whose
         * borrow dies in lane 1. */
#if QSB_SAS_Q2LANE
        "\t.reg .u32 nt,s8;\n"
        "\t.reg .u64 sn0,sn1,sn2,sn3,ls0,ls1,ls2,ls3;\n"
        "\tmov.u32 s8, 3;\n"
        "\tadd.cc.u64 sn0, %12, %12; addc.cc.u64 sn1, %13, %13;\n"
        "\taddc.cc.u64 sn2, %14, %14; addc.cc.u64 sn3, %15, %15;\n"
        "\taddc.u32 nt, 0, 0;\n"
#else
        "\t.reg .u32 n0,n1,n2,n3,n4,n5,n6,n7,nt,s8;\n"
        "\t.reg .u64 sn0,sn1,sn2,sn3,ls0,ls1,ls2,ls3;\n"
        "\tmov.u32 s8, 3;\n"
        "\tshr.u32 nt, v7, 31;\n"
        "\tshf.l.wrap.b32 n7, v6, v7, 1; shf.l.wrap.b32 n6, v5, v6, 1;\n"
        "\tshf.l.wrap.b32 n5, v4, v5, 1; shf.l.wrap.b32 n4, v3, v4, 1;\n"
        "\tshf.l.wrap.b32 n3, v2, v3, 1; shf.l.wrap.b32 n2, v1, v2, 1;\n"
        "\tshf.l.wrap.b32 n1, v0, v1, 1; shl.b32 n0, v0, 1;\n"
        "\tmov.b64 sn0, {n0,n1}; mov.b64 sn1, {n2,n3};\n"
        "\tmov.b64 sn2, {n4,n5}; mov.b64 sn3, {n6,n7};\n"
#endif
        "\tsub.cc.u64 ls0, %8, sn0; subc.cc.u64 ls1, %9, sn1;\n"
        "\tsubc.cc.u64 ls2, %10, sn2; subc.cc.u64 ls3, %11, sn3;\n"
        "\tsubc.u32 s8, s8, nt;\n"
        "\tadd.cc.u64 f0, f0, ls0; addc.cc.u64 f1, f1, ls1;\n"
        "\taddc.cc.u64 f2, f2, ls2; addc.cc.u64 f3, f3, ls3;\n"
        "\taddc.u32 z8, z8, s8;\n"
#if QSB_SAS_K3T1
        "\tsub.u64 f0, f0, 0x300000b73;\n"
#else
        "\tsub.cc.u64 f0, f0, 0x300000b73; subc.u64 f1, f1, 0;\n"
#endif
#elif QSB_SAS_SPRE
        /* the same single addend in 32-bit words: one nine-word add chain for
         * e, the split-3p offset and the doubled q together. */
        "\t.reg .u32 n0,n1,n2,n3,n4,n5,n6,n7,nt;\n"
        "\t.reg .u32 s0,s1,s2,s3,s4,s5,s6,s7,s8;\n"
        "\tmov.u32 s8, 3;\n"
        "\tshr.u32 nt, v7, 31;\n"
        "\tshf.l.wrap.b32 n7, v6, v7, 1; shf.l.wrap.b32 n6, v5, v6, 1;\n"
        "\tshf.l.wrap.b32 n5, v4, v5, 1; shf.l.wrap.b32 n4, v3, v4, 1;\n"
        "\tshf.l.wrap.b32 n3, v2, v3, 1; shf.l.wrap.b32 n2, v1, v2, 1;\n"
        "\tshf.l.wrap.b32 n1, v0, v1, 1; shl.b32 n0, v0, 1;\n"
        "\tsub.cc.u32 s0, u0, n0; subc.cc.u32 s1, u1, n1;\n"
        "\tsubc.cc.u32 s2, u2, n2; subc.cc.u32 s3, u3, n3;\n"
        "\tsubc.cc.u32 s4, u4, n4; subc.cc.u32 s5, u5, n5;\n"
        "\tsubc.cc.u32 s6, u6, n6; subc.cc.u32 s7, u7, n7;\n"
        "\tsubc.u32 s8, s8, nt;\n"
        "\tadd.cc.u32 z0, z0, s0; addc.cc.u32 z1, z1, s1;\n"
        "\taddc.cc.u32 z2, z2, s2; addc.cc.u32 z3, z3, s3;\n"
        "\taddc.cc.u32 z4, z4, s4; addc.cc.u32 z5, z5, s5;\n"
        "\taddc.cc.u32 z6, z6, s6; addc.cc.u32 z7, z7, s7;\n"
        "\taddc.u32 z8, z8, s8;\n"
#if QSB_C31 && QSB_SHORT_CARRY
        "\tsub.cc.u32 z0, z0, 0xb73; subc.u32 z1, z1, 3;\n"
#elif QSB_CARRY62 && QSB_SHORT_CARRY
        "\tsub.cc.u32 z0, z0, 0xb73; subc.cc.u32 z1, z1, 3; subc.u32 z2, z2, 0;\n"
#else
        "\tsub.cc.u32 z0, z0, 0xb73; subc.cc.u32 z1, z1, 3; subc.cc.u32 z2, z2, 0;\n"
        "\tsubc.cc.u32 z3, z3, 0; subc.u32 z4, z4, 0;\n"
#endif
#else
#if QSB_SAS_SPLIT3P
        /* 3p = 3*2^256 - 3K: add the 3*2^256 inside the e-chain's z8 limb (free) and subtract
         * 3K = 0x3_00000B73 after the two q subtractions (the value never drops below 2^256
         * before that). QSB_CARRY62 keeps the borrow through z2 and differs only when low96
         * < 3K (< 2^-62); its rollback keeps it through z4 (< 2^-126). */
        "\tadd.cc.u32 z0, z0, u0; addc.cc.u32 z1, z1, u1;\n"
        "\taddc.cc.u32 z2, z2, u2; addc.cc.u32 z3, z3, u3;\n"
        "\taddc.cc.u32 z4, z4, u4; addc.cc.u32 z5, z5, u5;\n"
        "\taddc.cc.u32 z6, z6, u6; addc.cc.u32 z7, z7, u7;\n"
        QSB_SAS_TOPADD3
#else
        /* restored arm (QSB_SAS_SPLIT3P=0): the crown's full-width form, byte
         * for byte -- 3p added as eight 32-bit immediates with its own ninth
         * and tenth words, then e, then the two q subtractions below. Exact
         * and carry-complete: the accumulator is a Z/2^320 register here, the
         * 3p addend is below 3*2^256 and e below 2^256, so the sum stays below
         * 2^258 above the fold's own bound B + K*(K+4) and cannot leave the
         * {z9,z8,z7..z0} width; every limb carry is propagated, so nothing is
         * truncated in this arm at all. */
        "\tadd.cc.u32 z0, z0, 0xfffff48d; addc.cc.u32 z1, z1, 0xfffffffc;\n"
        "\taddc.cc.u32 z2, z2, 0xffffffff; addc.cc.u32 z3, z3, 0xffffffff;\n"
        "\taddc.cc.u32 z4, z4, 0xffffffff; addc.cc.u32 z5, z5, 0xffffffff;\n"
        "\taddc.cc.u32 z6, z6, 0xffffffff; addc.cc.u32 z7, z7, 0xffffffff;\n"
        "\taddc.cc.u32 z8, z8, 2; addc.u32 z9, z9, 0;\n"
        "\tadd.cc.u32 z0, z0, u0; addc.cc.u32 z1, z1, u1;\n"
        "\taddc.cc.u32 z2, z2, u2; addc.cc.u32 z3, z3, u3;\n"
        "\taddc.cc.u32 z4, z4, u4; addc.cc.u32 z5, z5, u5;\n"
        "\taddc.cc.u32 z6, z6, u6; addc.cc.u32 z7, z7, u7;\n"
        "\taddc.cc.u32 z8, z8, 0; addc.u32 z9, z9, 0;\n"
#endif

#if QSB_SAS_Q2SHF
        "\t.reg .u32 n0,n1,n2,n3,n4,n5,n6,n7,nt;\n"
        "\tshr.u32 nt, v7, 31;\n"
        "\tshf.l.wrap.b32 n7, v6, v7, 1; shf.l.wrap.b32 n6, v5, v6, 1;\n"
        "\tshf.l.wrap.b32 n5, v4, v5, 1; shf.l.wrap.b32 n4, v3, v4, 1;\n"
        "\tshf.l.wrap.b32 n3, v2, v3, 1; shf.l.wrap.b32 n2, v1, v2, 1;\n"
        "\tshf.l.wrap.b32 n1, v0, v1, 1; shl.b32 n0, v0, 1;\n"
        "\tsub.cc.u32 z0, z0, n0; subc.cc.u32 z1, z1, n1;\n"
        "\tsubc.cc.u32 z2, z2, n2; subc.cc.u32 z3, z3, n3;\n"
        "\tsubc.cc.u32 z4, z4, n4; subc.cc.u32 z5, z5, n5;\n"
        "\tsubc.cc.u32 z6, z6, n6; subc.cc.u32 z7, z7, n7;\n"
        QSB_SAS_TOPSUB2
#else
        "\tsub.cc.u32 z0, z0, v0; subc.cc.u32 z1, z1, v1;\n"
        "\tsubc.cc.u32 z2, z2, v2; subc.cc.u32 z3, z3, v3;\n"
        "\tsubc.cc.u32 z4, z4, v4; subc.cc.u32 z5, z5, v5;\n"
        "\tsubc.cc.u32 z6, z6, v6; subc.cc.u32 z7, z7, v7;\n"
        QSB_SAS_TOPSUB
        "\tsub.cc.u32 z0, z0, v0; subc.cc.u32 z1, z1, v1;\n"
        "\tsubc.cc.u32 z2, z2, v2; subc.cc.u32 z3, z3, v3;\n"
        "\tsubc.cc.u32 z4, z4, v4; subc.cc.u32 z5, z5, v5;\n"
        "\tsubc.cc.u32 z6, z6, v6; subc.cc.u32 z7, z7, v7;\n"
        QSB_SAS_TOPSUB
#endif
#if QSB_SAS_SPLIT3P
#if QSB_C31 && QSB_SHORT_CARRY
        "\tsub.cc.u32 z0, z0, 0xb73; subc.u32 z1, z1, 3;\n"
#elif QSB_CARRY62 && QSB_SHORT_CARRY
        "\tsub.cc.u32 z0, z0, 0xb73; subc.cc.u32 z1, z1, 3; subc.u32 z2, z2, 0;\n"
#else
        "\tsub.cc.u32 z0, z0, 0xb73; subc.cc.u32 z1, z1, 3; subc.cc.u32 z2, z2, 0;\n"
        "\tsubc.cc.u32 z3, z3, 0; subc.u32 z4, z4, 0;\n"
#endif
#endif
#endif
#if QSB_SAS_FOLD64
        /* Exact: z8*(2^32+977) < 2^64 + 2^42, so its low 64 bits are
         * 977*z8 + (z8<<32) and its 65th bit is that add's carry (set only
         * for z8 >= 2^32-976). Lanes 0 and 1 take the pair, the residual
         * carry dies in lane 2 -- one lane further than the 32-bit tail in
         * the #else arm, so this form truncates strictly later.
         * -DQSB_SAS_FOLD64=0 restores that arm byte for byte. */
#if QSB_SAS_FOLD65
        "\t{ .reg .u64 sft; .reg .u32 sfl, sfh;\n"
        "mul.wide.u32 sft, z8, 977;\n"
        "mov.b64 {sfl, sfh}, sft;\n"
        "add.u32 sfh, sfh, z8;\n"
        "mov.b64 sft, {sfl, sfh};\n"
        "add.cc.u64 f0, f0, sft;\n"
#if QSB_SAS_FOLDT2
        "addc.u64 f1, f1, 0; }\n"
#else
        "addc.cc.u64 f1, f1, 0;\n"
        "addc.u64 f2, f2, 0; }\n"
#endif
#else
        "\t{ .reg .u64 sft, sfz, sfw;\n"
        "mul.wide.u32 sft, z8, 977;\n"
        "mov.b64 sfz, {zr, z8};\n"
        "add.cc.u64 sft, sft, sfz;\n"
        "addc.u64 sfw, 0, 0;\n"
        "add.cc.u64 f0, f0, sft;\n"
#if QSB_SAS_FOLDT2
        "addc.u64 f1, f1, sfw; }\n"
#else
        "addc.cc.u64 f1, f1, sfw;\n"
        "addc.u64 f2, f2, 0; }\n"
#endif
#endif
        "\tmov.u64 %0, f0; mov.u64 %1, f1; mov.u64 %2, f2; mov.u64 %3, f3;\n"
#else
#if QSB_SAS_LANE64
        "\tmov.b64 {z0,z1}, f0; mov.b64 {z2,z3}, f1;\n"
#endif
        QSB_SAS_FOLD
#if QSB_SHORT_CARRY
        QSB_SECOND_FOLD_TAIL
#else
        /* restored arm (QSB_SHORT_CARRY=0): the crown's full-width carry
         * terminator and its 977 re-fold, unchanged. Exact by construction --
         * every limb carry is propagated and the single re-fold addend
         * cf*977 < 2^10 cannot carry past z2. */
        "\taddc.cc.u32 z3, z3, 0; addc.cc.u32 z4, z4, 0; addc.cc.u32 z5, z5, 0;\n"
        "\taddc.cc.u32 z6, z6, 0; addc.cc.u32 z7, z7, 0;\n"
        "\taddc.u32 cf, 0, 0; mul.lo.u32 k0, cf, 977;\n"
        "\tadd.cc.u32 z0, z0, k0; addc.cc.u32 z1, z1, cf;\n"
        "addc.u32 z2, z2, 0;\n"
#endif
#if QSB_SAS_LANE64
        "\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.u64 %2, f2; mov.u64 %3, f3;\n"
#else
        "\tmov.b64 %0, {z0,z1}; mov.b64 %1, {z2,z3}; mov.b64 %2, {z4,z5}; mov.b64 %3, {z6,z7};\n"
#endif
#endif
        "\t}\n"
        "\t\n"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(e[0]), "l"(e[1]), "l"(e[2]), "l"(e[3]),
          "l"(q[0]), "l"(q[1]), "l"(q[2]), "l"(q[3]));
    out[0] = r0; out[1] = r1; out[2] = r2; out[3] = r3;
#else
    uint32_t A[8];
    for (int i=0;i<4;i++){ A[2*i]=(uint32_t)a[i]; A[2*i+1]=(uint32_t)(a[i]>>32); }
    #define PP(i,j) ((uint64_t)A[i]*A[j])
    __uint128_t t;
    /* even column words E_c (cols c,c+1), carry chain E_c -> E_{c+2} */
    uint64_t E2,E4,E6,E8,E10,E12,e14;
    t=(__uint128_t)PP(0,2);                          E2=(uint64_t)t;
    t=(t>>64)+PP(0,4)+PP(1,3);                       E4=(uint64_t)t;
    t=(t>>64)+PP(0,6)+PP(1,5)+PP(2,4);               E6=(uint64_t)t;
    t=(t>>64)+PP(1,7)+PP(2,6)+PP(3,5);               E8=(uint64_t)t;
    t=(t>>64)+PP(3,7)+PP(4,6);                       E10=(uint64_t)t;
    t=(t>>64)+PP(5,7);                               E12=(uint64_t)t;
    e14=(uint64_t)(t>>64);
    /* odd column words O_c */
    uint64_t O1,O3,O5,O7,O9,O11,O13,o15;
    t=(__uint128_t)PP(0,1);                          O1=(uint64_t)t;
    t=(t>>64)+PP(0,3)+PP(1,2);                       O3=(uint64_t)t;
    t=(t>>64)+PP(0,5)+PP(1,4)+PP(2,3);               O5=(uint64_t)t;
    t=(t>>64)+PP(0,7)+PP(1,6)+PP(2,5)+PP(3,4);       O7=(uint64_t)t;
    t=(t>>64)+PP(2,7)+PP(3,6)+PP(4,5);               O9=(uint64_t)t;
    t=(t>>64)+PP(4,7)+PP(5,6);                       O11=(uint64_t)t;
    t=(t>>64)+PP(6,7);                               O13=(uint64_t)t;
    o15=(uint64_t)(t>>64);
    #undef PP
    /* merge E (even cols) + O (odd cols) -> cross-sum X[0..15] (u32 limbs) */
    uint32_t X[16]; uint64_t cc;
    X[0]=0;
    cc=(uint64_t)(uint32_t)O1;                                    X[1]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E2 + (uint32_t)(O1>>32);              X[2]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E2>>32) + (uint32_t)O3;              X[3]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E4 + (uint32_t)(O3>>32);             X[4]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E4>>32) + (uint32_t)O5;             X[5]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E6 + (uint32_t)(O5>>32);             X[6]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E6>>32) + (uint32_t)O7;             X[7]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E8 + (uint32_t)(O7>>32);             X[8]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E8>>32) + (uint32_t)O9;             X[9]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E10 + (uint32_t)(O9>>32);            X[10]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E10>>32) + (uint32_t)O11;           X[11]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)E12 + (uint32_t)(O11>>32);           X[12]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)(E12>>32) + (uint32_t)O13;           X[13]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)e14 + (uint32_t)(O13>>32);           X[14]=(uint32_t)cc; cc>>=32;
    cc+=(uint64_t)(uint32_t)o15;                                 X[15]=(uint32_t)cc;
    /* double: 2*cross */
    uint64_t d=0;
    for (int k=0;k<16;k++){ uint64_t v=((uint64_t)X[k]<<1)|d; X[k]=(uint32_t)v; d=v>>32; }
    /* add the 8 diagonal squares A[i]^2 at columns 2i */
    uint64_t carry=0;
    for (int i=0;i<8;i++){
        __uint128_t s=(__uint128_t)X[2*i] + ((uint64_t)X[2*i+1]<<32) + (uint64_t)A[i]*A[i] + carry;
        X[2*i]=(uint32_t)s; X[2*i+1]=(uint32_t)(s>>32); carry=(uint64_t)(s>>64);
    }
    /* secp256k1 double-fold (identical to the multiply's 6.2 reduction) */
    #define MW(x,y) ((uint64_t)(uint32_t)(x)*(uint32_t)(y))
    uint64_t r0=X[0]|((uint64_t)X[1]<<32), r1=X[2]|((uint64_t)X[3]<<32),
             r2=X[4]|((uint64_t)X[5]<<32), r3=X[6]|((uint64_t)X[7]<<32);
    uint64_t h0=X[8]|((uint64_t)X[9]<<32), h1=X[10]|((uint64_t)X[11]<<32),
             h2=X[12]|((uint64_t)X[13]<<32), h3=X[14]|((uint64_t)X[15]<<32);
    __uint128_t s;
    uint64_t f0,f1,f2,f3; uint32_t f8;
    s=(__uint128_t)r0+MW(X[8],977);  f0=(uint64_t)s;
    s=(s>>64)+r1+MW(X[10],977); f1=(uint64_t)s;
    s=(s>>64)+r2+MW(X[12],977); f2=(uint64_t)s;
    s=(s>>64)+r3+MW(X[14],977); f3=(uint64_t)s;
    f8=(uint32_t)(s>>64);
    uint64_t g0,g1,g2,g3; uint32_t g8;
    s=(__uint128_t)h0+MW(X[9],977);  g0=(uint64_t)s;
    s=(s>>64)+h1+MW(X[11],977); g1=(uint64_t)s;
    s=(s>>64)+h2+MW(X[13],977); g2=(uint64_t)s;
    s=(s>>64)+h3+MW(X[15],977); g3=(uint64_t)s;
    g8=(uint32_t)(s>>64);
    uint32_t z[10], w[8];
    z[0]=(uint32_t)f0;z[1]=(uint32_t)(f0>>32);z[2]=(uint32_t)f1;z[3]=(uint32_t)(f1>>32);
    z[4]=(uint32_t)f2;z[5]=(uint32_t)(f2>>32);z[6]=(uint32_t)f3;z[7]=(uint32_t)(f3>>32);
    w[0]=(uint32_t)g0;w[1]=(uint32_t)(g0>>32);w[2]=(uint32_t)g1;w[3]=(uint32_t)(g1>>32);
    w[4]=(uint32_t)g2;w[5]=(uint32_t)(g2>>32);w[6]=(uint32_t)g3;w[7]=(uint32_t)(g3>>32);
    { uint64_t c=0,tt;
      for(int k=0;k<7;k++){ tt=(uint64_t)z[k+1]+w[k]+c; z[k+1]=(uint32_t)tt; c=tt>>32; }
      tt=(uint64_t)f8+w[7]+c; z[8]=(uint32_t)tt; c=tt>>32; z[9]=(uint32_t)((uint64_t)g8+c); }

    /* The positive first fold is L+kH+3p+e-2q. For all 256-bit
     * representatives it is in [0,(k+5)*B), so its high word fits 33 bits. */
    {
        const uint32_t bias[8] = {0xfffff48dU,0xfffffffcU,0xffffffffU,0xffffffffU,
                                  0xffffffffU,0xffffffffU,0xffffffffU,0xffffffffU};
        uint64_t carry=0;
        for(int i=0;i<8;i++){
            uint64_t tv=(uint64_t)z[i]+bias[i]+carry;
            z[i]=(uint32_t)tv; carry=tv>>32;
        }
        uint64_t tv=(uint64_t)z[8]+2+carry;
        z[8]=(uint32_t)tv; z[9]+=(uint32_t)(tv>>32);
        carry=0;
        for(int i=0;i<8;i++){
            uint64_t ei=(uint32_t)(e[i/2]>>(32*(i&1)));
            tv=(uint64_t)z[i]+ei+carry;
            z[i]=(uint32_t)tv; carry=tv>>32;
        }
        tv=(uint64_t)z[8]+carry;
        z[8]=(uint32_t)tv; z[9]+=(uint32_t)(tv>>32);
        for(int repeat=0;repeat<2;repeat++){
            uint64_t borrow=0;
            for(int i=0;i<8;i++){
                uint64_t qi=(uint32_t)(q[i/2]>>(32*(i&1)));
                uint64_t sub=qi+borrow;
                borrow=(uint64_t)z[i]<sub;
                z[i]=(uint32_t)((uint64_t)z[i]-sub);
            }
            tv=(uint64_t)z[8]-borrow;
            borrow=(uint64_t)z[8]<borrow;
            z[8]=(uint32_t)tv; z[9]-=(uint32_t)borrow;
        }
    }
    uint64_t tt2=MW(z[8],977);
    uint32_t m0=(uint32_t)tt2,m1=(uint32_t)(tt2>>32),m2;
    m1=(uint32_t)(m1+(uint32_t)((uint64_t)z[9]*977));
    { uint64_t tv=(uint64_t)m1+z[8]; m1=(uint32_t)tv; uint64_t c=tv>>32; m2=(uint32_t)((uint64_t)z[9]+c); }
    { uint64_t c=0,tv;
      tv=(uint64_t)z[0]+m0+c; z[0]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[1]+m1+c; z[1]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[2]+m2+c; z[2]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[3]+c; z[3]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[4]+c; z[4]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[5]+c; z[5]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[6]+c; z[6]=(uint32_t)tv; c=tv>>32;
      tv=(uint64_t)z[7]+c; z[7]=(uint32_t)tv; c=tv>>32;
      /* Fold the final 2^256 carry as k. A second carry is impossible. */
      tv=(uint64_t)z[0]+c*977; z[0]=(uint32_t)tv; c=(tv>>32)+c;
      tv=(uint64_t)z[1]+c; z[1]=(uint32_t)tv; c=tv>>32;
      for(int i=2;i<8;i++){ tv=(uint64_t)z[i]+c; z[i]=(uint32_t)tv; c=tv>>32; }
    }
    out[0]=z[0]|((uint64_t)z[1]<<32); out[1]=z[2]|((uint64_t)z[3]<<32);
    out[2]=z[4]|((uint64_t)z[5]<<32); out[3]=z[6]|((uint64_t)z[7]<<32);
    #undef MW
    (void)h0;(void)h1;(void)h2;(void)h3;(void)r0;(void)r1;(void)r2;(void)r3;(void)carry;(void)d;
#endif
}
#endif  /* QSB_FUSE_SQRADDSUB2 */
//Very efficient way of finding 8-byte target value in global memory buffer (Buffer must be ordered in ascending order)
//Each step it does fast division by half: mid = (hi + lo) >> 1; and checks resulting value
//Worst-case performance is O(log n), and we don't need to calculate any hashes by using this method.
__device__ int _BinarySearch(uint64_t *buffer, int hi, uint64_t target)
{
    int mid;
	int lo = 0;

	while (hi - lo > 1)
	{
		mid = (hi + lo) >> 1;
		if (buffer[mid] == target)
		{
			return mid;
		}
		else if (buffer[mid] < target)
		{
			lo = mid + 1;
		}
		else
		{
			hi = mid;
		}
	}

	if (buffer[lo] == target)
	{
		return lo;
	}
	else if (buffer[hi] == target)
	{
		return hi;
	}
	else
	{
		return -1;
	}
}

//Secp256k1 Point Addition implementation
__device__ void _PointAddSecp256k1(uint64_t *p1x, uint64_t *p1y, uint64_t *p1z, uint64_t *p2x, uint64_t *p2y)
{
  uint64_t u[4];
  uint64_t v[4];

  uint64_t us2[4];
  uint64_t vs2[4];
  uint64_t vs3[4];

  uint64_t a[4];

  uint64_t us2w[4];
  uint64_t vs2v2[4];
  uint64_t vs3u2[4];
  uint64_t _2vs2v2[4];

  _ModMult(u, p2y, p1z);
  _ModMult(v, p2x, p1z);

  _ModSub256(u, u, p1y);
  _ModSub256(v, v, p1x);

  _ModSqr(us2, u);
  _ModSqr(vs2, v);

  _ModMult(vs3, vs2, v);
  _ModMult(us2w, us2, p1z);
  _ModMult(vs2v2, vs2, p1x);

  _ModAdd256(_2vs2v2, vs2v2, vs2v2);

  _ModSub256(a, us2w, vs3);
  _ModSub256(a, _2vs2v2);

  _ModMult(p1x, v, a);
  _ModMult(vs3u2, vs3, p1y);

  _ModSub256(p1y, vs2v2, a);
  _ModMult(p1y, p1y, u);

  _ModSub256(p1y, vs3u2);
  _ModMult(p1z, vs3, p1z);
}

// ---------------------------------------------------------------------------------------
// XYZZ coordinates: x = X/ZZ, y = Y/ZZZ with the invariant ZZ^3 == ZZZ^2 (a = 0 plays no
// part in addition). Same limb convention as _ModMult: values in [0, 2^256), not
// necessarily < p. Outputs must not alias inputs.
//
// EFD "madd-2008-s" -- (X1,Y1,ZZ1,ZZZ1) += affine (X2,Y2) in place, 8M + 2S.
// This equivalent schedule anchors the final y formula at the affine addend:
//   U2 = X2*ZZ1, S2 = Y2*ZZZ1, P = U2-X1, R = S2-Y1, PP = P^2, PPP = P*PP,
//   V = U2*PP, X3 = R^2 + PPP - 2V,
//   Y3 = R*(V-X3) - Y2*ZZZ3, ZZ3 = ZZ1*PP, ZZZ3 = ZZZ1*PPP.
// Y1 may be an affine-anchor-deferred ordinate: Yactual=Y1-Yoff*ZZZ1.
// Adding Yoff to Y2 only for the slope restores the ordinary numerator. If
// defer_y is true, return Ycore=R*(V-X3), making Y2 the next affine anchor;
// otherwise subtract Y2*ZZZ3 and return the exact XYZZ ordinate.
// P == 0 (x1 == x2) gives ZZ3 == ZZZ3 == 0: the point at infinity for P1 == -P2 and, as
// with the homogeneous add this replaces, no valid answer for P1 == P2. Neither occurs in
// the fixed-base multiply, whose table entries are distinct non-opposite multiples of G.
// ---------------------------------------------------------------------------------------
template<bool DEFER_Y>
__device__ __forceinline__ void _PointAddXYZZT(
    uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
    const uint64_t *X2, const uint64_t *Y2, const uint64_t *Yoff);
#ifndef QSB_XYZZ_ZSPILL
#define QSB_XYZZ_ZSPILL 1 /* X3 and the deferred Y3 write their destinations directly, no T[4] */
#endif


__device__ void _PointAddXYZZ(uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
                              const uint64_t *X2, const uint64_t *Y2,
                              const uint64_t *Yoff, bool defer_y)
{
  uint64_t U2[4];
  uint64_t S2[4];
  uint64_t P[4];
  uint64_t R[4];
  uint64_t PP[4];
  uint64_t Q[4];
#if QSB_XYZZ_ZSPILL
  uint64_t *const X3d = X1;            /* X3 destination: X1 is dead after P = U2 - X1 */
#else
  uint64_t T[4];
  uint64_t *const X3d = T;
#endif

  _ModMult(U2, (uint64_t *)X2, ZZ1);   // U2 = X2*ZZ1
#if QSB_LAZY
  _ModAddLazy(S2, Y2, Yoff);
#else
  _ModAdd256(S2, (uint64_t *)Y2, (uint64_t *)Yoff);
#endif
  _ModMult(S2, ZZZ1);                  // S2 = (Y2+Yoff)*ZZZ1
  _ModSub256(P, U2, X1);               // P  = U2 - X1
  _ModSub256(R, S2, Y1);               // R  = S2 - Y1
  _ModSqr(PP, P);                      // PP = P^2
  _ModMult(PPP, PP, P);                // PPP = P*PP
  _ModMult(Q, U2, PP);                 // V  = U2*PP
  _ModMult(ZZ1, PP);                   // ZZ3; PP dies before the R^2/Y3 tail

#if QSB_FUSE_SQRADDSUB2
  /* xlib f297b0f9: one reduction for R^2 + PPP - 2V. */
  _ModSqrAddSub2(X3d, R, PPP, Q);      // X3 = R^2 + PPP - 2V
#else
  _ModSqr(X3d, R);                     // R^2
#if QSB_LAZY
  _ModX3Fused(X3d, X3d, PPP, Q);       // X3 = R^2 + PPP - 2V
#else
  _ModAdd256(X3d, X3d, PPP);
  _ModSub256(X3d, X3d, Q);
  _ModSub256(X3d, X3d, Q);             // X3 = R^2 + PPP - 2V
#endif
#endif

  _ModMult(ZZZ1, PPP);                 // ZZZ3
  _ModSub256(Q, Q, X3d);               // V - X3
#if QSB_XYZZ_ZSPILL
  if (defer_y) {
    _ModMult(Y1, Q, R);                // Direct write deferred Y3
  } else {
    _ModMult(Q, R);                    // R*(V - X3)
    _ModMult(S2, (uint64_t *)Y2, ZZZ1);// affine Y2*ZZZ3
    _ModSub256(Y1, Q, S2);             // exact Y3
  }
#else
  _ModMult(Q, R);                      // R*(V - X3)
  if (defer_y) {
    Load256(Y1, Q);                    // actual Y3 = Y1 - Y2*ZZZ3
  } else {
    _ModMult(S2, (uint64_t *)Y2, ZZZ1);// affine Y2*ZZZ3
    _ModSub256(Y1, Q, S2);             // exact Y3
  }

  Load256(X1, T);                      // X3
#endif
}

#ifndef QSB_XYZZ_ALIAS_U2
#define QSB_XYZZ_ALIAS_U2 1 /* the mixed addend abscissa reuses the slope numerator's storage */
#endif
#ifndef QSB_XYZZ_ALIAS_Q
#define QSB_XYZZ_ALIAS_Q 1  /* V = U2*PP is built over the dead difference P */
#endif
#ifndef QSB_XYZZ_ALIAS_PPP
#define QSB_XYZZ_ALIAS_PPP 1 /* the difference cube is built over the dead deferred ordinate */
#endif
#ifndef QSB_XYZZ_CONSTEXPR
#define QSB_XYZZ_CONSTEXPR 1 /* the resolving tail is not instantiated in the deferred calls */
#endif
// Compile-time twin of _PointAddXYZZ (delta C, jacklightChen e582bda4): the
// production chain calls <true> twelve times in its rolled loop and <false>
// once for the resolving final addition, so no defer_y branch is in the loop.
template<bool DEFER_Y>
__device__ __forceinline__ void _PointAddXYZZT(
    uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
    const uint64_t *X2, const uint64_t *Y2, const uint64_t *Yoff)
{
#if QSB_XYZZ_ALIAS_U2
  /* With DEFER_Y the last read of S2 is R = S2 - Y1, which is issued before
   * U2 = X2*ZZ1 is formed, and the affine Y2*ZZZ3 tail that re-uses S2 exists
   * only in the resolving <false> instantiation. In the twelve <true> calls
   * of the chain S2 is therefore dead at the U2 multiply and the two names
   * denote one 256-bit destination: the same limbs are written with the same
   * operands, and one fewer live four-word array is offered to the register
   * allocator in the body where PP, PPP, R and V are all live at once. */
  uint64_t S2[4];
  uint64_t U2s[DEFER_Y ? 1 : 4];
  uint64_t *const U2 = DEFER_Y ? S2 : U2s;
#else
  uint64_t U2s[4];
  uint64_t *const U2 = U2s;
  uint64_t S2[4];
#endif
  uint64_t P[4];
  uint64_t R[4];
  uint64_t PP[4];
#if QSB_XYZZ_ALIAS_PPP
  /* In the deferred instantiation the accumulator ordinate's last read is
   * R = S2 - Y1, two multiplies before PPP exists, and its next write is the
   * final R*(V - X3) of this same addition, one multiply after PPP's last
   * read (ZZZ3 = ZZZ1*PPP). The four words are therefore dead across exactly
   * the span PPP occupies, so the cube is formed in them: the same limbs are
   * written by the same multiply with the same operands, and the body's peak
   * offers the allocator one fewer live four-word array while PP, PPP, R and
   * V are simultaneously live. The resolving <false> tail reads Y1 nowhere
   * either, but it keeps its own array so its schedule is untouched. */
  uint64_t PPPs[DEFER_Y ? 1 : 4];
  uint64_t *const PPP = DEFER_Y ? Y1 : PPPs;
#else
  uint64_t PPP[4];
#endif
#if QSB_XYZZ_ALIAS_Q
  /* P's last read is PPP = PP*P, which precedes V = U2*PP; nothing below
   * reads P again (the abscissa fold consumes PPP and V, the ordinate tail
   * consumes V, R and ZZZ3). V may therefore be written over P. */
  uint64_t *const Q = P;
#else
  uint64_t Qs[4];
  uint64_t *const Q = Qs;
#endif

#if QSB_YOFF
  _ModAddLazyOff(S2, Y2, Yoff);        // offset ordinates: y2 + yoff (mod p)
#elif QSB_LAZY
  _ModAddLazy(S2, Y2, Yoff);
#else
  _ModAdd256(S2, (uint64_t *)Y2, (uint64_t *)Yoff);
#endif
  _ModMult(S2, ZZZ1);                  // S2 = (Y2+Yoff)*ZZZ1
  _ModSub256(R, S2, Y1);               // R  = S2 - Y1
  _ModMult(U2, (uint64_t *)X2, ZZ1);   // U2 = X2*ZZ1
  _ModSub256(P, U2, X1);               // P  = U2 - X1
  _ModSqr(PP, P);                      // PP = P^2
  _ModMult(PPP, PP, P);                // PPP = P*PP
  _ModMult(Q, U2, PP);                 // V  = U2*PP

#if QSB_FUSE_SQRADDSUB2
  /* xlib f297b0f9: one reduction for R^2 + PPP - 2V. Directly write into destination X1 */
  _ModSqrAddSub2(X1, R, PPP, Q);       // X3 = R^2 + PPP - 2V
#else
  _ModSqr(X1, R);                      // R^2
#if QSB_LAZY
  _ModX3Fused(X1, X1, PPP, Q);         // X3 = R^2 + PPP - 2V
#else
  _ModAdd256(X1, X1, PPP);
  _ModSub256(X1, X1, Q);
  _ModSub256(X1, X1, Q);               // X3 = R^2 + PPP - 2V
#endif
#endif

  _ModMult(ZZZ1, PPP);                 // ZZZ3
  _ModMult(ZZ1, PP);                   // ZZ3 (after ZZZ3: lets ptxas keep every multiply
                                       // on the paired-carry schedule without predicate spills)
  _ModSub256(Q, Q, X1);                // V - X3
#if QSB_XYZZ_CONSTEXPR
  /* DEFER_Y is a template argument, so exactly one arm of this test survives
   * in each instantiation; discarding the other at instantiation time rather
   * than after code generation keeps the resolving tail's two multiplies and
   * its second read of S2 out of the twelve chain calls entirely. That read
   * is the one place where the aliased U2 destination and S2 are distinct
   * names for the same storage, so the arm that assumes them separate is now
   * never formed in the instantiation that aliases them. The surviving arm is
   * the arm the run-time test selected: same operands, same order. */
  if constexpr (DEFER_Y) {
#else
  if (DEFER_Y) {
#endif
    _ModMult(Y1, Q, R);                // Direct write deferred Y3 into destination Y1
  } else {
    _ModMult(Q, R);                    // R*(V - X3)
    _ModMult(S2, (uint64_t *)Y2, ZZZ1);// affine Y2*ZZZ3
    _ModSub256(Y1, Q, S2);             // exact Y3
  }
}

#ifndef QSB_MM_FUSE
#define QSB_MM_FUSE 1   /* one reduction for the seed abscissa, re-anchored on X2*PP */
#endif
// Direct-three-affine prefix based on EFD "mmadd-2008-s", 3M + 2S. X3,
// ZZ3, and ZZZ3 are the ordinary coordinates of P1+P2, while Y3 deliberately
// holds only R*(Q-X3), omitting -Y1*ZZZ3. The caller adds Y1 only to the third
// affine point's slope input for one affine-anchored madd. Its slope numerator
// is then (Ythird+Y1)*ZZZ3-R*(Q-X3) = Ythird*ZZZ3-Y(P1+P2), while its final
// affine anchor remains Ythird. The combined seed is therefore exact and costs
// 11M+4S rather than 12M+4S.
__device__ void _PointAddXYZZ_mm(uint64_t *X3, uint64_t *Y3, uint64_t *ZZ3, uint64_t *ZZZ3,
                                 const uint64_t *X1, const uint64_t *Y1,
                                 const uint64_t *X2, const uint64_t *Y2)
{
  uint64_t P[4];
  uint64_t R[4];
  uint64_t Q[4];

  _ModSub256(P, (uint64_t *)X2, (uint64_t *)X1);   // P = X2 - X1
  _ModSub256(R, (uint64_t *)Y2, (uint64_t *)Y1);   // R = Y2 - Y1
  _ModSqr(ZZ3, P);                                 // ZZ3  = PP  = P^2
  _ModMult(ZZZ3, ZZ3, P);                          // ZZZ3 = PPP = P*PP
#if !QSB_MM_FUSE
  _ModMult(Q, (uint64_t *)X1, ZZ3);                // Q = X1*PP
#endif

#if QSB_MM_FUSE
  /* X2*PP == (X1+P)*PP == X1*PP + PPP exactly, so
   *   R^2 - PPP - 2*(X1*PP) == R^2 + PPP - 2*(X2*PP),
   * which is the shape _ModSqrAddSub2 already emits with a single reduction.
   * The abscissa costs one square-and-fold instead of a square and three
   * borrow-corrected subtractions, and the slope input X1*PP is recovered as
   * Q - PPP. Both multiplies keep the operand widths of the replaced ones and
   * X3 leaves in the congruent [0,2^256) form the chain's own fused abscissa
   * already hands to the next _ModSub256. */
  _ModMult(Q, (uint64_t *)X2, ZZ3);                // Q = X2*PP = X1*PP + PPP
  _ModSqrAddSub2(X3, R, ZZZ3, Q);                  // X3 = R^2 + PPP - 2*X2*PP
  _ModSub256(Q, Q, ZZZ3);                          // X1*PP
  _ModSub256(Q, Q, X3);                            // (X1*PP) - X3
#else
  _ModSqr(X3, R);                                  // R^2
  _ModSub256(X3, X3, ZZZ3);
  _ModSub256(X3, X3, Q);
  _ModSub256(X3, X3, Q);                           // X3 = R^2 - PPP - 2Q

  _ModSub256(Q, Q, X3);                            // Q - X3
#endif
  _ModMult(Y3, Q, R);                              // deferred R*(Q-X3)
}
