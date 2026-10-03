#pragma once
/* Host-CPU co-grinder for the subset track. The field arithmetic, the windowed host table and the
 * batch-affine additions derive from Ryun1's pinning CpuGrind.h (public submission 7a75fa50, GPL-3);
 * the candidate enumeration, preimage hashing and hit publication are subset's. The table is sized at
 * run time: signed digits in mixed-width windows (12 windows of 20-22 bits, 1.06 GiB, 11 additions per
 * candidate where memory allows; at worst 15 windows, 68 MiB), on 2 MiB pages, each window's rows
 * prefetched during the previous window's backward pass.
 *
 * Candidates are disjoint from the GPU's: the GPU grinds every epoch (6 early omissions below
 * the cut) with its 128 window-omission patterns (h_win3); the CPU grinds epochs t, t+T, t+2T, ...
 * (T threads) with the other 158 of the C(13,3)=286 window patterns. Every CPU hit passes the same
 * exact OpenSSL gate as the GPU's tentatives (qsb_hv_check) before it is appended to
 * results/digest_hit_cpu.txt, which the harness collects with the GPU's hit file. Workers run at
 * SCHED_IDLE, so they never delay the GPU host thread; QSB_CPU_GRIND=0 compiles it out.
 *
 * 2026-09-26 (GLV12xl_x_wt_nx): the batch inversions invert one scalar (Bernstein-Yang safegcd on the
 * integer pipes, the 8 lanes combined and split with 6 permuted multiplications) instead of 255 vector
 * squarings; the window's backward pass runs one group at a time in registers; x3 and y3 subtract from
 * the folded product columns and carry once; multiplication-only elements skip the limb masks (IFMA reads
 * bits 51:0); products accumulate low and high partial products separately (shorter chains); the
 * word-message SHA-256 is VEX-encoded and starts from the IV; ec8_final stores its outputs with vector
 * transposes; the table conversion uses VBMI2 funnel shifts. Same candidates, gate and records.
 *
 * 2026-09-26 (GLV12xl_x_wt_nx_r3): the backward pass prefetches the next window's rows four at a time between
 * the multiplications instead of 32 in a burst per four groups (the burst stalled on outstanding misses);
 * batches of 1,024 candidates keep both SMT threads' EC state and prefetched rows in the core's L2; the
 * point C is folded into the top window's table (built once), so the windows end at Q_0 = z*A + C and the
 * final step is one addition of -2C for Q_1 instead of two sharing a denominator. Same candidates, gate and records.
 *
 * 2026-09-26 (polish): column 9 of every product is a single high partial product (< 2^52), so its fold needs no split:
 * hi(c9*R) rides in column 5's low-product chain and lo(c9*R) goes into column 4, whose three fold terms are summed on a
 * side chain (QSB_CPU_FOLD3: 14 IFMA per reduction instead of 17, same critical path); the fused subtractions start
 * columns 0..4 at 4p (QSB_CPU_PSEED); the key hashes are computed h0-only from message words built in registers, with a
 * 4-lane vector prefilter; the hashing loop runs over an epoch's patterns without per-candidate epoch checks, the lanes
 * point at double-buffered group states, the first digests go straight into the second SHA-256's message words and the
 * second into z; the epoch prefix is rebuilt from its first changed push. Same candidates, gate and records.
 *
 * 2026-09-26 (r7): the rows of the first two windows are prefetched from the hashing phase, 8 candidates at a time as their z
 * words are computed (QSB_CPU_HPF): the first window's loop used to issue 16 table misses per group into a loop of ~170
 * instructions and ran memory-bound (lookups of window 0 confined to an L2-resident corner: +3.0 % per CPU-second); it now finds
 * them cached. The window step keeps D = TX - X (normalized) instead of the table x, x3 = lam^2 - D - 2X (QSB_CPU_NOTX, after
 * Meganpark's 9745ce9b / 296e5e53 and ercumentyildirim's 9f8a33d8, which do the same): one fe8 array (40 KB per thread) and
 * 10 vector memory operations per group fewer. The backward pass's two mul-only differences skip the limb-4 fold
 * (QSB_CPU_NF, after the same two lanes' fe8_sub_nf / fe8_carry_m). Same candidates, gate and records.
 *
 * 2026-09-26 (w10): a 10-window table (17,408 MiB, signed digits of 26/25/24 bits) when the process sees >= 79.5 GiB (the
 * 11-window rule: table <= 1/4 of avail - 11.5 GiB) and >= 95 % of it gets transparent huge pages; else 11, else 12 (an mmap
 * failure or a failed table check at 10 also goes to 11), as Meganpark's 296e5e53 takes a 10-lookup table (at avail/4 >= 19 GiB)
 * built on its 8-lane path. Here the table's chunk steps and C fold run through the lane's own window step on blocks of 1,024
 * points (QSB_CPU_VBUILD): the same bytes as the scalar build, ~5x faster. The diagnostic walk start sets bit 28 for 10 windows.
 *
 * 2026-09-26 (r8): the batch-affine steps keep a weighted prefix (QSB_CPU_WPRE): the forward pass stores W = (product of the chain's
 * earlier denominators) * (sign * ty - Y) instead of the bare prefix, and the backward pass takes the slope as one product,
 * lam = (running inverse) * W, instead of dinv = inverse * prefix, lam = dinv * (sign * ty - Y). An exact reassociation: the same six
 * multiplications per step, one of them moved off the backward pass's dependency chain onto the forward pass, and the table y is no longer
 * stored between the passes (after jacklightChen's 55757d4d, which does the same; implemented here from its description). Also in the
 * final addition of -2C. The rows prefetched a pass ahead -- the next window's during the backward pass and windows 0/1's from the
 * hashing phase -- now go to L2 (prefetcht1) instead of L1 (QSB_CPU_PFNX): 64 KB of rows per window and thread do not fit the 32 KB L1
 * and evicted the backward pass's own lines; the forward pass still pulls its rows into L1 QSB_CPU_PFD groups ahead. Same candidates,
 * gate and records.
 *
 * 2026-09-27 (package y2d): this engine (terrapinelf's 2d1631b0, with jacklightChen's weighted-prefix idea from 55757d4d) at the
 * thread footprint of Meganpark980320's 296e5e53 co-grinder: ncpu - QSB_CPU_RESERVE workers, none on the GPU host thread's core
 * or its SMT sibling, the tree's own host producers unchanged. 2d1631b0's footprint items are switches below, each off by default
 * (QSB_CPU_HP_SHARE, QSB_CPU_NTH_RAISE, QSB_CPU_RSV_CORE=0); QSB_CPU_DIAG_EPOCH (its walk-start diagnostic) is off as well, so the
 * walk starts at epoch 0. Same candidates, gate and records.
 *
 * Three host-only items at the same thread footprint, each a switch whose off value restores the
 * previous file's behaviour. QSB_CPU_PFD 3 instead of 8 (the forward pass's row-prefetch distance, as in the promoted 521075fe).
 * QSB_CPU_TRY9 (from 789aed1b): a 9-window table (114,688 MiB, signed digits of 29/28/27 bits: 8 additions per candidate instead
 * of 9) when the 10-window rule holds and the table fits in half of what the process sees after the reserve (>= ~235.5 GiB), with
 * the same huge-page guard; an mmap failure, poor backing or a failed table check at 9 goes to 10. The r9 SHA-NI trims (from
 * c13302f3, without its host-core sharing): the first SHA-256's tail blocks (every block after the per-pattern one holds the same
 * bytes for all candidates) load each W+K pair once for the four lanes (QSB_CPU_X4PS: 2 loads per round pair instead of 8); the
 * key hashes and the second SHA-256 skip the message-schedule steps that are identities for their padding (W9..W14 = 0: step
 * r = 4 has no W[t-7] term, step r = 6's sha256msg1 returns its first operand), and the second SHA-256 takes its padding words as
 * constants (QSB_CPU_SHC): fewer sha256msg1 ops, which on Zen 4 issue to the pipes the lane's IFMA multiplications use. With
 * QSB_CPU_PFD 8 and QSB_CPU_TRY9, QSB_CPU_X4PS, QSB_CPU_SHC all 0 the file behaves as 8f99a3e9's. Same candidates, gate and records.
 *
 * The heavy host footprint of the queued 789aed1b / a141df2b lineage (4da17ebc, 3ff68d21, 57065b7d,
 * 2f690f1a), host-only and behind one switch, QSB_CPU_ALLCPU (default 1; 0 restores the the base footprint above byte for byte in
 * behaviour). Under QSB_HOST_BLOCKING (tree.cu: the GPU host thread sleeps in blocking event waits, so its core idles between
 * launches) the workers run on every CPU of the process's pre-main() set, the host thread's core included, and the worker count
 * rises to one per CPU within the cgroup quota (32 of 32 on the ranked host; the workers are SCHED_IDLE and yield at once to the
 * host thread and to the producers). QSB_CPU_RSV_CORE therefore defaults to 0 under QSB_CPU_ALLCPU (rsv_core_mask would take the
 * host core back). The producers are the same lineage's v3 file (tests/gpu_epochs/host_producers.h, QSB_HP_V3). Same candidates,
 * gate and records: only the worker count and placement change.
 *
 * The second SHA-256 and the key hashes run as four interleaved SHA-NI lanes per call instead of two calls or
 * passes of two lanes (QSB_CPU_SHA4; 0 = the 2-lane code above). Two lanes issue one sha256rnds2 per 2 cycles (a dependent pair per
 * lane, latency 4); four issue one per cycle. sha256rnds2 addresses only xmm0..15, so the four lanes' message rings stay in memory
 * (qsha_rounds4m) and only the eight chaining registers stay in registers. The first SHA-256's blocks (qsha_x4p) were 4-lane already.
 * Same hash inputs and digests, same prefilter, gate and records; host-only, no image knob.
 *
 * Two host-only co-grinder items from queued rival trees, each behind a switch whose 0 restores
 * the previous file's behaviour. (1) QSB_CPU_BATCH_AUTO (the rule of 86c643ae's QSB_CPU_BATCH_SOLO, core_sharing and batch_choose,
 * taken as written): the candidates per batch are chosen in start() from the thread_siblings_list of the workers' CPUs, 4,096
 * (QSB_CPU_BATCH_SOLO) when no two workers can share a physical core (at most one worker per CPU and no two of their CPUs SMT
 * siblings; also with one worker), QSB_CPU_BATCH (1,024) whenever siblings are among the workers' CPUs, the topology is unreadable or
 * there are more workers than CPUs; QSB_CPU_BATCH_RT=<n> (dev) overrides. A larger batch spreads each window step's shared inversion
 * over more candidates; 1,024 was sized for two SMT threads' batch state in one 1 MB L2. The batch only regroups the same candidates
 * (the walk order and the hit set are unchanged). (2) QSB_CPU_KH16, QSB_CPU_MRG and QSB_CPU_AINL (a33e04c3's three items, taken as
 * written): the key hashes of a group's 16 keys (8 candidates x 2 recids) take their message schedule from AVX-512 (16 lanes, the
 * zero padding words folded at compile time) instead of sha256msg1/msg2 and run their rounds with SHA-NI four keys at a time from
 * the stored W + K pairs (ec8_final_cf_kh writes the message words straight from the canonical limbs; run-time guarded: AVX-512VL +
 * SHA-NI + the C fold, else qsha_keyhash4_h0 as before, so with the guard true QSB_CPU_SHA4's key-hash routine is bypassed and SHA4
 * keeps only the second SHA-256); every 8-lane product accumulates each column's low and high partial products in one register (the
 * same column sums, 18 vector operations fewer per product); the small field operations are always inlined. Same candidates, same
 * hash inputs and digests, same prefilter, gate and records; host-only, no image knob.
 *
 * The CPU walks 100 of its 158 window patterns (QSB_CPU_PREFIX100, default 1; 0 restores the
 * 158-pattern walk of the base byte for byte in behaviour), b67487a1's hash_plan_cpu_patterns taken as written (the family selection
 * after 4a197f06). hash_plan groups the patterns by their first message block (block 0 of the first SHA-256 after the epoch's
 * state); with the 158/77 shape of this problem, 20 of those groups hold 5 patterns each. The walk keeps exactly those 100 patterns,
 * in their original order, so each block-0 compression serves 5 candidates instead of about 2 (158 / 77), and 100 is a multiple of
 * the four SHA-NI lanes. Any other shape (pattern count, group count, number of 5-pattern groups) keeps all 158. The kept patterns
 * are a subset of the 158, so the CPU's candidates stay disjoint from the GPU's 128 patterns; each worker still walks epochs
 * t, t + T, ... in order, now over 100 patterns per epoch. Same hash inputs, prefilter, gate and records for every candidate
 * walked; host-only, no image knob. */
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <utility>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <new>
#ifndef QSB_CPU_VEC
#define QSB_CPU_VEC 1
#endif
#if QSB_CPU_VEC && defined(__x86_64__) && !defined(__CUDA_ARCH__)
#define QCPU_VEC 1
#include <immintrin.h>
#else
#define QCPU_VEC 0
#endif
#ifndef QSB_CPU_SHANI
#define QSB_CPU_SHANI 1
#endif
#if QSB_CPU_SHANI && defined(__x86_64__) && !defined(__CUDA_ARCH__)
#define QCPU_SHANI 1
#include <immintrin.h>
#include <cpuid.h>
#else
#define QCPU_SHANI 0
#endif

#ifndef QSB_CPU_RESERVE
#define QSB_CPU_RESERVE 2
#endif
#ifndef QSB_CPU_FOLD2
#define QSB_CPU_FOLD2 1
#endif
#ifndef QSB_CPU_FOLD3
#define QSB_CPU_FOLD3 QSB_CPU_FOLD2

#endif
#if QSB_CPU_FOLD3 && !QSB_CPU_FOLD2
#error "QSB_CPU_FOLD3 extends the split-column fold (QSB_CPU_FOLD2)"
#endif
#ifndef QSB_CPU_PSEED
#define QSB_CPU_PSEED 1
#endif








#ifndef QSB_CPU_FOLD4
#define QSB_CPU_FOLD4 1
#endif
#if QSB_CPU_FOLD4 && !QSB_CPU_FOLD3
#error "QSB_CPU_FOLD4 extends QSB_CPU_FOLD3 (column 9's upper fold term pre-added to column 5)"
#endif
#ifndef QSB_CPU_BATCH
#define QSB_CPU_BATCH 1024
#endif













#ifndef QSB_CPU_NW_MIN
#define QSB_CPU_NW_MIN 12
#endif
#ifndef QSB_CPU_TRY11
#define QSB_CPU_TRY11 1
#endif
#ifndef QSB_CPU_TRY10
#define QSB_CPU_TRY10 1
#endif




#ifndef QSB_CPU_TRY9
#define QSB_CPU_TRY9 1
#endif
#ifndef QSB_CPU_TAB9_FRAC
#define QSB_CPU_TAB9_FRAC 0.5
#endif
#ifndef QSB_CPU_TAB10_CAP_MB
#define QSB_CPU_TAB10_CAP_MB 20480
#endif
#ifndef QSB_CPU_FALL13
#define QSB_CPU_FALL13 0
#endif
#ifndef QSB_CPU_TAB_RESERVE_MB
#define QSB_CPU_TAB_RESERVE_MB 11776
#endif
#ifndef QSB_CPU_HP_MIN
#define QSB_CPU_HP_MIN 0.95
#endif
#ifndef QSB_CPU_TAB_CAP_MB
#define QSB_CPU_TAB_CAP_MB 4096
#endif
#ifndef QSB_CPU_TAB_FRAC
#define QSB_CPU_TAB_FRAC 0.25
#endif
#ifndef QSB_CPU_VBUILD
#define QSB_CPU_VBUILD 1
#endif
#ifndef QSB_CPU_CFOLD
#define QSB_CPU_CFOLD 1
#endif
#ifndef QSB_CPU_PFD1
#define QSB_CPU_PFD1 8
#endif
#ifndef QSB_CPU_PFD
#define QSB_CPU_PFD 3
#endif
#ifndef QSB_CPU_HPF
#define QSB_CPU_HPF 2
#endif
#ifndef QSB_CPU_NOTX
#define QSB_CPU_NOTX 1
#endif
#ifndef QSB_CPU_NF
#define QSB_CPU_NF 1
#endif
#ifndef QSB_CPU_DNF
#define QSB_CPU_DNF 1
#endif
#if QSB_CPU_DNF && !(QSB_CPU_NOTX && QSB_CPU_PSEED)
#error "QSB_CPU_DNF is written for the QSB_CPU_NOTX window step with QSB_CPU_PSEED (fe8_sqr_subdx's seeded columns)"
#endif
#ifndef QSB_CPU_WPRE
#define QSB_CPU_WPRE 1
#endif
#if QSB_CPU_WPRE && !QSB_CPU_NOTX
#error "QSB_CPU_WPRE is written for the QSB_CPU_NOTX window step"
#endif
#ifndef QSB_CPU_PFNX
#define QSB_CPU_PFNX 1
#endif
#define QCPU_NXT_HINT (QSB_CPU_PFNX ? _MM_HINT_T1 : _MM_HINT_T0)
#ifndef QSB_CPU_X4PS
#define QSB_CPU_X4PS 1
#endif
#ifndef QSB_CPU_SHC
#define QSB_CPU_SHC 1
#endif
#ifndef QSB_CPU_SHA4
#define QSB_CPU_SHA4 1

#endif








#ifndef QSB_CPU_BATCH_AUTO
#define QSB_CPU_BATCH_AUTO 1
#endif
#ifndef QSB_CPU_BATCH_SOLO
#define QSB_CPU_BATCH_SOLO 4096
#endif
static_assert(QSB_CPU_BATCH % 32 == 0 && QSB_CPU_BATCH >= 32 && QSB_CPU_BATCH <= 8192, "QSB_CPU_BATCH: a multiple of 32, 32..8192");
static_assert(QSB_CPU_BATCH_SOLO % 32 == 0 && QSB_CPU_BATCH_SOLO >= 32 && QSB_CPU_BATCH_SOLO <= 8192, "QSB_CPU_BATCH_SOLO: a multiple of 32, 32..8192");





#ifndef QSB_CPU_KH16
#define QSB_CPU_KH16 1
#endif
#ifndef QSB_CPU_MRG
#define QSB_CPU_MRG 1
#endif
#ifndef QSB_CPU_AINL
#define QSB_CPU_AINL 1
#endif




#ifndef QSB_CPU_RECODE_NG
#define QSB_CPU_RECODE_NG 1
#endif


















#ifndef QSB_CPU_ILP2
#define QSB_CPU_ILP2 1
#endif











#ifndef QSB_CPU_INV_PEEL
#define QSB_CPU_INV_PEEL 0
#endif
#if QSB_CPU_INV_PEEL != 0 && QSB_CPU_INV_PEEL != 1
#error "QSB_CPU_INV_PEEL must be 0 or 1"
#endif
#if QSB_CPU_INV_PEEL && (QSB_CPU_ILP2 != 1 || !QSB_CPU_WPRE || !QSB_CPU_NOTX || !QSB_CPU_KH16)
#error "QSB_CPU_INV_PEEL requires ILP2=1, WPRE, NOTX, KH16"
#endif
#ifndef QSB_CPU_GWK_CACHE
#define QSB_CPU_GWK_CACHE 0
#endif
#if QSB_CPU_GWK_CACHE != 0 && QSB_CPU_GWK_CACHE != 1
#error "QSB_CPU_GWK_CACHE must be 0 or 1"
#endif
#ifndef QSB_CPU_JL_INV_FIRST
#define QSB_CPU_JL_INV_FIRST 0
#endif
#ifndef QSB_CPU_JL_INV_LAST
#define QSB_CPU_JL_INV_LAST 0
#endif
#ifndef QSB_CPU_JL_INV_LANE0
#define QSB_CPU_JL_INV_LANE0 0
#endif
#ifndef QSB_CPU_NCH
#define QSB_CPU_NCH 2
#endif
#ifndef QSB_CPU_PFSPREAD
#define QSB_CPU_PFSPREAD 3
#endif
#ifndef QSB_CPU_MRGS
#define QSB_CPU_MRGS 0
#endif
#define QCPU_PFQ ((QSB_CPU_PFSPREAD & 1) && QCPU_VEC && QCPU_SHANI)
#if QSB_CPU_NCH != 4 && QSB_CPU_NCH != 2
#error "QSB_CPU_NCH: 4 (the code before) or 2"
#endif
#if (QSB_CPU_ILP2 || QSB_CPU_NCH != 4) && !(QSB_CPU_WPRE && QSB_CPU_NOTX)
#error "QSB_CPU_ILP2 / QSB_CPU_NCH are written for the QSB_CPU_WPRE + QSB_CPU_NOTX window step"
#endif
#if QSB_CPU_AINL
#define QCPU_AI always_inline,
#define QCPU_AIF __attribute__((always_inline))
#else
#define QCPU_AI
#define QCPU_AIF
#endif
#ifndef QSB_CPU_F1N
#define QSB_CPU_F1N (QSB_CPU_HPF < 2)
#endif








#ifndef QSB_CPU_ALLCPU
#define QSB_CPU_ALLCPU 1
#endif
#ifndef QSB_CPU_HP_SHARE
#define QSB_CPU_HP_SHARE 0


#endif
#ifndef QSB_CPU_NTH_RAISE
#define QSB_CPU_NTH_RAISE 0


#endif
#ifndef QSB_CPU_RSV_CORE
#if QSB_CPU_ALLCPU
#define QSB_CPU_RSV_CORE 0
#else
#define QSB_CPU_RSV_CORE 1


#endif
#endif
#if QSB_CPU_ALLCPU && QSB_CPU_RSV_CORE
#error "QSB_CPU_ALLCPU 1 needs QSB_CPU_RSV_CORE 0 (rsv_core_mask would take the GPU host thread's core back from the workers)"
#endif
#if QSB_CPU_ALLCPU && QSB_CPU_HP_SHARE
#error "QSB_CPU_ALLCPU 1 needs QSB_CPU_HP_SHARE 0 (the host thread's CPU is already a worker CPU)"
#endif
#ifndef QSB_CPU_DIAG_EPOCH
#define QSB_CPU_DIAG_EPOCH 0

#endif








#ifndef QSB_CPU_TOUCH_GUARD
#define QSB_CPU_TOUCH_GUARD 1
#endif
#ifndef QSB_CPU_TOUCH9_MAX_S
#define QSB_CPU_TOUCH9_MAX_S 20.0
#endif
#ifndef QSB_CPU_FOLD_PAR
#define QSB_CPU_FOLD_PAR 1
#endif

namespace qcpu {
static const int NWMAX = 16;
#if QCPU_PFQ


struct QPfRing { const char *a[64]; unsigned head = 0, tail = 0; };
static inline __attribute__((always_inline)) void qpf_drain(QPfRing &q, unsigned n) {
while (n-- && q.head != q.tail) { _mm_prefetch(q.a[q.head & 63], QSB_CPU_PFNX ? _MM_HINT_T1 : _MM_HINT_T0); q.head++; }
}
#endif
typedef unsigned __int128 u128;

template <class T> struct qalloc64 {
typedef T value_type;
qalloc64() = default;
template <class U> qalloc64(const qalloc64<U> &) {}
T *allocate(size_t n) { void *q = nullptr; if (posix_memalign(&q, 64, n * sizeof(T) + 64)) throw std::bad_alloc(); return (T *)q; }
void deallocate(T *q, size_t) { free(q); }
template <class U> bool operator==(const qalloc64<U> &) const { return true; }
template <class U> bool operator!=(const qalloc64<U> &) const { return false; }
};
struct fe { uint64_t v[4]; };
static const uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL, PK = 0x1000003D1ULL;

static inline bool fe_is_zero(const fe &a) { return !(a.v[0] | a.v[1] | a.v[2] | a.v[3]); }
static inline bool fe_eq(const fe &a, const fe &b) {
return !((a.v[0] ^ b.v[0]) | (a.v[1] ^ b.v[1]) | (a.v[2] ^ b.v[2]) | (a.v[3] ^ b.v[3]));
}
static inline bool fe_ge_p(const uint64_t v[4]) {
return v[3] == ~0ULL && v[2] == ~0ULL && v[1] == ~0ULL && v[0] >= P0;
}
static inline void fe_sub_p(uint64_t v[4]) {
u128 c = (u128)v[0] + PK; v[0] = (uint64_t)c; c >>= 64;
for (int i = 1; i < 4; i++) { c += v[i]; v[i] = (uint64_t)c; c >>= 64; }
}
static inline void fe_add(fe &r, const fe &a, const fe &b) {
u128 c = 0; uint64_t t[4];
for (int i = 0; i < 4; i++) { c += (u128)a.v[i] + b.v[i]; t[i] = (uint64_t)c; c >>= 64; }
if (c || fe_ge_p(t)) fe_sub_p(t);
memcpy(r.v, t, 32);
}
static inline void fe_sub(fe &r, const fe &a, const fe &b) {
uint64_t t[4]; unsigned borrow = 0;
for (int i = 0; i < 4; i++) {
u128 d = (u128)a.v[i] - b.v[i] - borrow;
t[i] = (uint64_t)d; borrow = (unsigned)((d >> 64) & 1);
}
if (borrow) {
u128 d = (u128)t[0] - PK; t[0] = (uint64_t)d; unsigned b = (unsigned)((d >> 64) & 1);
for (int i = 1; i < 4; i++) { d = (u128)t[i] - b; t[i] = (uint64_t)d; b = (unsigned)((d >> 64) & 1); }
}
memcpy(r.v, t, 32);
}
static inline void fe_mul(fe &r, const fe &a, const fe &b) {
uint64_t l[8] = {0};
for (int i = 0; i < 4; i++) {
uint64_t carry = 0;
for (int j = 0; j < 4; j++) {
u128 acc = (u128)a.v[i] * b.v[j] + l[i + j] + carry;
l[i + j] = (uint64_t)acc; carry = (uint64_t)(acc >> 64);
}
l[i + 4] = carry;
}

uint64_t m[4]; u128 c = 0;
for (int i = 0; i < 4; i++) { c += (u128)l[i] + (u128)l[i + 4] * PK; m[i] = (uint64_t)c; c >>= 64; }
uint64_t top = (uint64_t)c;
c = (u128)m[0] + (u128)top * PK; m[0] = (uint64_t)c; c >>= 64;
for (int i = 1; i < 4; i++) { c += m[i]; m[i] = (uint64_t)c; c >>= 64; }
if (c) fe_sub_p(m);
if (fe_ge_p(m)) fe_sub_p(m);
memcpy(r.v, m, 32);
}
static inline void fe_sqr(fe &r, const fe &a) { fe_mul(r, a, a); }
static void fe_inv(fe &r, const fe &a) {
static const uint64_t e[4] = {0xFFFFFFFEFFFFFC2DULL, ~0ULL, ~0ULL, ~0ULL};
fe x = a, acc = {{1, 0, 0, 0}};
for (int i = 0; i < 256; i++) {
if ((e[i >> 6] >> (i & 63)) & 1) fe_mul(acc, acc, x);
fe_sqr(x, x);
}
r = acc;
}
static void fe_from_le32(fe &r, const uint8_t b[32]) {
for (int i = 0; i < 4; i++) { uint64_t w = 0; for (int k = 7; k >= 0; k--) w = (w << 8) | b[i * 8 + k]; r.v[i] = w; }
}
static void fe_from_bn(fe &r, const BIGNUM *bn) {
uint8_t be[32] = {0}; int n = BN_num_bytes(bn); BN_bn2bin(bn, be + 32 - n);
for (int i = 0; i < 4; i++) { uint64_t w = 0; for (int k = 0; k < 8; k++) w = (w << 8) | be[(3 - i) * 8 + k]; r.v[i] = w; }
}
struct pt { fe x, y; };
static pt pt_double(const pt &q) {
fe x2, num, den, inv, lam, x3, y3, t;
fe_sqr(x2, q.x); fe_add(num, x2, x2); fe_add(num, num, x2);
fe_add(den, q.y, q.y); fe_inv(inv, den); fe_mul(lam, num, inv);
fe_sqr(x3, lam); fe_sub(x3, x3, q.x); fe_sub(x3, x3, q.x);
fe_sub(t, q.x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, q.y);
return {x3, y3};
}




static void batch_add(pt *acc, const pt *const *tp, uint8_t *inf, uint8_t *bad, int n,
fe *d, fe *pre, const uint8_t *neg = nullptr) {
fe run = {{1, 0, 0, 0}};
for (int k = 0; k < n; k++) {
if (k + 16 < n && tp[k + 16]) __builtin_prefetch(tp[k + 16]);
if (bad[k] || !tp[k]) { pre[k] = run; continue; }
if (inf[k]) { pre[k] = run; continue; }
fe_sub(d[k], tp[k]->x, acc[k].x);
if (fe_is_zero(d[k])) { bad[k] = 1; pre[k] = run; continue; }
pre[k] = run; fe_mul(run, run, d[k]);
}
fe inv; fe_inv(inv, run);
for (int k = n - 1; k >= 0; k--) {
if (bad[k] || !tp[k]) continue;
const bool ng = neg && neg[k];
if (inf[k]) { acc[k] = *tp[k]; if (ng) { const fe z0 = {{0, 0, 0, 0}}; fe_sub(acc[k].y, z0, acc[k].y); } inf[k] = 0; continue; }
fe dinv; fe_mul(dinv, inv, pre[k]); fe_mul(inv, inv, d[k]);
fe lam, t, x3, y3;
if (ng) { fe_add(t, tp[k]->y, acc[k].y); const fe z0 = {{0, 0, 0, 0}}; fe_sub(t, z0, t); }
else fe_sub(t, tp[k]->y, acc[k].y);
fe_mul(lam, t, dinv);
fe_sqr(x3, lam); fe_sub(x3, x3, acc[k].x); fe_sub(x3, x3, tp[k]->x);
fe_sub(t, acc[k].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[k].y);
acc[k].x = x3; acc[k].y = y3;
}
}






struct Geo { int nw = 0; bool sgn = false; int off[NWMAX] = {0}, wid[NWMAX] = {0}; uint32_t ent[NWMAX] = {0}; size_t base[NWMAX] = {0}; size_t total = 0; };
static Geo geo_make(int nw, bool sgn) {
Geo g; if (nw < 9) nw = 9; if (nw > NWMAX) nw = NWMAX;
g.nw = nw; g.sgn = sgn;
if (!sgn) {
const int w = 256 / nw, r = 256 - w * nw;
for (int i = 0; i < nw; i++) g.wid[i] = w + (i < r);
} else {
double best = -1; int bb = 1;
for (int b = 1; b <= 30; b++) {
const int k = nw - 1, rem = 256 - b, w = rem / k, r = rem - w * k;
if (w + (r > 0) > 30) continue;
const double cost = r * ldexp(1.0, w) + (k - r) * ldexp(1.0, w - 1) + ldexp(1.0, b);
if (best < 0 || cost < best) { best = cost; bb = b; }
}
const int k = nw - 1, rem = 256 - bb, w = rem / k, r = rem - w * k;
for (int i = 0; i < k; i++) g.wid[i] = w + (i < r);
g.wid[k] = bb;
}
size_t o = 0; int off = 0;
for (int i = 0; i < nw; i++) {
g.off[i] = off; off += g.wid[i];
g.ent[i] = !sgn ? (uint32_t)((1u << g.wid[i]) - 1) : (i < nw - 1 ? 1u << (g.wid[i] - 1) : 1u << g.wid[i]);
g.base[i] = o; o += g.ent[i];
}
g.total = o;
return g;
}

static void recode_scalar(const Geo &g, const uint32_t *zb, uint32_t *ds, int B) {
for (int k = 0; k < B; k++) {
const uint32_t *h = zb + (size_t)k * 8;
uint64_t zl[5];
for (int i = 0; i < 4; i++) zl[i] = ((uint64_t)h[6 - 2 * i] << 32) | h[7 - 2 * i];
zl[4] = 0;
uint32_t cy = 0;
for (int i = 0; i < g.nw; i++) {
const int li = g.off[i] >> 6, sh = g.off[i] & 63, w = g.wid[i];
uint64_t v = zl[li] >> sh;
if (sh + w > 64) v |= zl[li + 1] << (64 - sh);
uint32_t u = (uint32_t)(v & ((1ULL << w) - 1)), ng = 0;
if (g.sgn) {
u += cy;
if (i < g.nw - 1) { ng = u > (1u << (w - 1)); if (ng) u = (1u << w) - u; cy = ng; }
}
ds[(size_t)i * B + k] = u | (ng << 31);
}
}
}

#if QCPU_VEC








struct fe8 { __m512i l[5]; };
#define F8_M52 _mm512_set1_epi64(0xFFFFFFFFFFFFFULL)
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_carry(fe8 &r) {


const __m512i M = F8_M52, M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), K = _mm512_set1_epi64(0x1000003D1ULL);
__m512i c;
c = _mm512_srli_epi64(r.l[4], 48); r.l[4] = _mm512_and_si512(r.l[4], M48);
r.l[0] = _mm512_madd52lo_epu64(r.l[0], c, K);
c = _mm512_srli_epi64(r.l[0], 52); r.l[0] = _mm512_and_si512(r.l[0], M); r.l[1] = _mm512_add_epi64(r.l[1], c);
c = _mm512_srli_epi64(r.l[1], 52); r.l[1] = _mm512_and_si512(r.l[1], M); r.l[2] = _mm512_add_epi64(r.l[2], c);
c = _mm512_srli_epi64(r.l[2], 52); r.l[2] = _mm512_and_si512(r.l[2], M); r.l[3] = _mm512_add_epi64(r.l[3], c);
c = _mm512_srli_epi64(r.l[3], 52); r.l[3] = _mm512_and_si512(r.l[3], M); r.l[4] = _mm512_add_epi64(r.l[4], c);
}






static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_carry_lz(fe8 &r) {
const __m512i M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), K = _mm512_set1_epi64(0x1000003D1ULL);
__m512i c;
c = _mm512_srli_epi64(r.l[4], 48); r.l[4] = _mm512_and_si512(r.l[4], M48);
r.l[0] = _mm512_madd52lo_epu64(r.l[0], c, K);
c = _mm512_srli_epi64(r.l[0], 52); r.l[1] = _mm512_add_epi64(r.l[1], c);
c = _mm512_srli_epi64(r.l[1], 52); r.l[2] = _mm512_add_epi64(r.l[2], c);
c = _mm512_srli_epi64(r.l[2], 52); r.l[3] = _mm512_add_epi64(r.l[3], c);
c = _mm512_srli_epi64(r.l[3], 52); r.l[4] = _mm512_add_epi64(r.l[4], c);
}






static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_fold(__m512i *o, __m512i c0, __m512i c1, __m512i c2, __m512i c3, __m512i c4,
__m512i c5, __m512i c6, __m512i c7, __m512i c8, __m512i c9) {
const __m512i Z = _mm512_setzero_si512();
#define LO(acc, x, y) acc = _mm512_madd52lo_epu64(acc, x, y)
#define HI(acc, x, y) acc = _mm512_madd52hi_epu64(acc, x, y)
#if QSB_CPU_FOLD3






#if QSB_CPU_FOLD4







const __m512i R = _mm512_set1_epi64(0x1000003D10ULL);
const __m512i h5 = _mm512_srli_epi64(c5, 52);
c6 = _mm512_add_epi64(c6, h5); const __m512i h6 = _mm512_srli_epi64(c6, 52);
c7 = _mm512_add_epi64(c7, h6); const __m512i h7 = _mm512_srli_epi64(c7, 52);
c8 = _mm512_add_epi64(c8, h7); const __m512i h8 = _mm512_srli_epi64(c8, 52);
__m512i f4 = Z; LO(f4,c9,R); LO(f4,h8,R);
LO(c0,c5,R); HI(c1,c5,R);
LO(c1,c6,R); HI(c2,c6,R);
LO(c2,c7,R); HI(c3,c7,R);
LO(c3,c8,R); HI(c4,c8,R);
c4 = _mm512_add_epi64(c4, f4);
#else
const __m512i M = F8_M52, R = _mm512_set1_epi64(0x1000003D10ULL);
const __m512i l5 = _mm512_and_si512(c5, M), h5 = _mm512_srli_epi64(c5, 52);
const __m512i l6 = _mm512_and_si512(c6, M), h6 = _mm512_srli_epi64(c6, 52);
const __m512i l7 = _mm512_and_si512(c7, M), h7 = _mm512_srli_epi64(c7, 52);
const __m512i l8 = _mm512_and_si512(c8, M), h8 = _mm512_srli_epi64(c8, 52);
__m512i f4 = Z; LO(f4,c9,R); HI(f4,l8,R); LO(f4,h8,R);
LO(c3,l8,R); HI(c3,l7,R); LO(c3,h7,R);
LO(c2,l7,R); HI(c2,l6,R); LO(c2,h6,R);
LO(c1,l6,R); HI(c1,l5,R); LO(c1,h5,R);
LO(c0,l5,R);
c4 = _mm512_add_epi64(c4, f4);
#endif
#elif QSB_CPU_FOLD2






const __m512i M = F8_M52, R = _mm512_set1_epi64(0x1000003D10ULL);
const __m512i l5 = _mm512_and_si512(c5, M), h5 = _mm512_srli_epi64(c5, 52);
const __m512i l6 = _mm512_and_si512(c6, M), h6 = _mm512_srli_epi64(c6, 52);
const __m512i l7 = _mm512_and_si512(c7, M), h7 = _mm512_srli_epi64(c7, 52);
const __m512i l8 = _mm512_and_si512(c8, M), h8 = _mm512_srli_epi64(c8, 52);
const __m512i l9 = _mm512_and_si512(c9, M), h9 = _mm512_srli_epi64(c9, 52);
LO(c0,l5,R); HI(c1,l5,R); LO(c1,h5,R);
LO(c1,l6,R); HI(c2,l6,R); LO(c2,h6,R);
LO(c2,l7,R); HI(c3,l7,R); LO(c3,h7,R);
LO(c3,l8,R); HI(c4,l8,R); LO(c4,h8,R);
LO(c4,l9,R); __m512i t9 = Z; HI(t9,l9,R); LO(t9,h9,R);
c0 = _mm512_madd52lo_epu64(c0, t9, R);
c1 = _mm512_madd52hi_epu64(c1, t9, R);
#else
const __m512i M = F8_M52;

__m512i t;
t = _mm512_srli_epi64(c5, 52); c5 = _mm512_and_si512(c5, M); c6 = _mm512_add_epi64(c6, t);
t = _mm512_srli_epi64(c6, 52); c6 = _mm512_and_si512(c6, M); c7 = _mm512_add_epi64(c7, t);
t = _mm512_srli_epi64(c7, 52); c7 = _mm512_and_si512(c7, M); c8 = _mm512_add_epi64(c8, t);
t = _mm512_srli_epi64(c8, 52); c8 = _mm512_and_si512(c8, M); c9 = _mm512_add_epi64(c9, t);
__m512i c10 = _mm512_srli_epi64(c9, 52); c9 = _mm512_and_si512(c9, M);

const __m512i R = _mm512_set1_epi64(0x1000003D10ULL);
LO(c0,c5,R); HI(c1,c5,R);
LO(c1,c6,R); HI(c2,c6,R);
LO(c2,c7,R); HI(c3,c7,R);
LO(c3,c8,R); HI(c4,c8,R);
LO(c4,c9,R); __m512i c5b = Z; HI(c5b,c9,R);

c5b = _mm512_madd52lo_epu64(c5b, c10, R);

c0 = _mm512_madd52lo_epu64(c0, c5b, R);
c1 = _mm512_madd52hi_epu64(c1, c5b, R);
#endif
o[0] = c0; o[1] = c1; o[2] = c2; o[3] = c3; o[4] = c4;
}
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_red(fe8 &r, __m512i c0, __m512i c1, __m512i c2, __m512i c3, __m512i c4,
__m512i c5, __m512i c6, __m512i c7, __m512i c8, __m512i c9) {
fe8 o; fe8_fold(o.l, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9);
fe8_carry(o);
r = o;
}



template <int S>
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_mul_cols(__m512i *c, const fe8 &a, const fe8 &b) {
const __m512i Z = _mm512_setzero_si512();
#if QSB_CPU_MRG && QSB_CPU_FOLD3 && QSB_CPU_MRGS






__m512i c0 = S ? _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4) : Z, c1 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
c2 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z, c3 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
c4 = S ? _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4) : Z;
__m512i c5 = Z, c6 = Z, c7 = Z, c8 = Z, d9 = Z, e3 = Z, e4 = Z, e5 = Z, e6 = Z;
const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
const __m512i b0 = b.l[0], b1 = b.l[1], b2 = b.l[2], b3 = b.l[3], b4 = b.l[4];
HI(d9,a4,b4);
LO(c4,a0,b4); LO(c5,a1,b4); LO(c3,a0,b3); LO(c6,a2,b4);
LO(c4,a1,b3); LO(c5,a2,b3); LO(c3,a1,b2); LO(c6,a3,b3);
LO(c4,a2,b2); LO(c5,a3,b2); LO(c3,a2,b1); LO(c6,a4,b2);
LO(c4,a3,b1); LO(c5,a4,b1); LO(c3,a3,b0); HI(e6,a1,b4);
LO(c4,a4,b0); HI(e5,a0,b4); HI(e3,a0,b2); HI(e6,a2,b3);
HI(e4,a0,b3); HI(e5,a1,b3); HI(e3,a1,b1); HI(e6,a3,b2);
HI(e4,a1,b2); HI(e5,a2,b2); HI(e3,a2,b0); HI(e6,a4,b1);
HI(e4,a2,b1); HI(e5,a3,b1);
HI(e4,a3,b0); HI(e5,a4,b0);
LO(c2,a0,b2); LO(c7,a3,b4); LO(c1,a0,b1); LO(c8,a4,b4);
LO(c2,a1,b1); LO(c7,a4,b3); LO(c1,a1,b0); HI(c8,a3,b4);
LO(c2,a2,b0); HI(c7,a2,b4); HI(c1,a0,b0); HI(c8,a4,b3);
HI(c2,a0,b1); HI(c7,a3,b3);
HI(c2,a1,b0); HI(c7,a4,b2);
LO(c0,a0,b0);
c5 = _mm512_madd52hi_epu64(c5, d9, _mm512_set1_epi64(0x1000003D10ULL));
c3 = _mm512_add_epi64(c3, e3); c4 = _mm512_add_epi64(c4, e4); c5 = _mm512_add_epi64(c5, e5); c6 = _mm512_add_epi64(c6, e6);
c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3; c[4] = c4; c[5] = c5; c[6] = c6; c[7] = c7; c[8] = c8; c[9] = d9;
#elif QSB_CPU_MRG && QSB_CPU_FOLD3



__m512i c0 = S ? _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4) : Z, c1 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
c2 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z, c3 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
c4 = S ? _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4) : Z;
__m512i c5 = Z, c6 = Z, c7 = Z, c8 = Z, d9 = Z;
const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
const __m512i b0 = b.l[0], b1 = b.l[1], b2 = b.l[2], b3 = b.l[3], b4 = b.l[4];
HI(d9,a4,b4);
LO(c4,a0,b4); LO(c5,a1,b4); LO(c3,a0,b3); LO(c6,a2,b4);
LO(c4,a1,b3); LO(c5,a2,b3); LO(c3,a1,b2); LO(c6,a3,b3);
LO(c4,a2,b2); LO(c5,a3,b2); LO(c3,a2,b1); LO(c6,a4,b2);
LO(c4,a3,b1); LO(c5,a4,b1); LO(c3,a3,b0); HI(c6,a1,b4);
LO(c4,a4,b0); HI(c5,a0,b4); HI(c3,a0,b2); HI(c6,a2,b3);
HI(c4,a0,b3); HI(c5,a1,b3); HI(c3,a1,b1); HI(c6,a3,b2);
HI(c4,a1,b2); HI(c5,a2,b2); HI(c3,a2,b0); HI(c6,a4,b1);
HI(c4,a2,b1); HI(c5,a3,b1);
HI(c4,a3,b0); HI(c5,a4,b0);
LO(c2,a0,b2); LO(c7,a3,b4); LO(c1,a0,b1); LO(c8,a4,b4);
LO(c2,a1,b1); LO(c7,a4,b3); LO(c1,a1,b0); HI(c8,a3,b4);
LO(c2,a2,b0); HI(c7,a2,b4); HI(c1,a0,b0); HI(c8,a4,b3);
HI(c2,a0,b1); HI(c7,a3,b3);
HI(c2,a1,b0); HI(c7,a4,b2);
LO(c0,a0,b0);
c5 = _mm512_madd52hi_epu64(c5, d9, _mm512_set1_epi64(0x1000003D10ULL));
c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3; c[4] = c4; c[5] = c5; c[6] = c6; c[7] = c7; c[8] = c8; c[9] = d9;
#else

__m512i c0 = S ? _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4) : Z, c1 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
c2 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z, c3 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
c4 = S ? _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4) : Z;
__m512i c5 = Z, c6 = Z, c7 = Z, c8 = Z, d1 = Z, d2 = Z, d3 = Z, d4 = Z, d5 = Z, d6 = Z, d7 = Z, d8 = Z, d9 = Z;
const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
const __m512i b0 = b.l[0], b1 = b.l[1], b2 = b.l[2], b3 = b.l[3], b4 = b.l[4];
LO(c0,a0,b0); HI(d1,a0,b0);
LO(c1,a0,b1); LO(c1,a1,b0); HI(d2,a0,b1); HI(d2,a1,b0);
LO(c2,a0,b2); LO(c2,a1,b1); LO(c2,a2,b0); HI(d3,a0,b2); HI(d3,a1,b1); HI(d3,a2,b0);
LO(c3,a0,b3); LO(c3,a1,b2); LO(c3,a2,b1); LO(c3,a3,b0); HI(d4,a0,b3); HI(d4,a1,b2); HI(d4,a2,b1); HI(d4,a3,b0);
LO(c4,a0,b4); LO(c4,a1,b3); LO(c4,a2,b2); LO(c4,a3,b1); LO(c4,a4,b0);
HI(d5,a0,b4); HI(d5,a1,b3); HI(d5,a2,b2); HI(d5,a3,b1); HI(d5,a4,b0);
#if QSB_CPU_FOLD3
HI(d9,a4,b4);
LO(c5,a1,b4); LO(c5,a2,b3); LO(c5,a3,b2); LO(c5,a4,b1); c5 = _mm512_madd52hi_epu64(c5, d9, _mm512_set1_epi64(0x1000003D10ULL));
HI(d6,a1,b4); HI(d6,a2,b3); HI(d6,a3,b2); HI(d6,a4,b1);
LO(c6,a2,b4); LO(c6,a3,b3); LO(c6,a4,b2); HI(d7,a2,b4); HI(d7,a3,b3); HI(d7,a4,b2);
LO(c7,a3,b4); LO(c7,a4,b3); HI(d8,a3,b4); HI(d8,a4,b3);
LO(c8,a4,b4);
#else
LO(c5,a1,b4); LO(c5,a2,b3); LO(c5,a3,b2); LO(c5,a4,b1); HI(d6,a1,b4); HI(d6,a2,b3); HI(d6,a3,b2); HI(d6,a4,b1);
LO(c6,a2,b4); LO(c6,a3,b3); LO(c6,a4,b2); HI(d7,a2,b4); HI(d7,a3,b3); HI(d7,a4,b2);
LO(c7,a3,b4); LO(c7,a4,b3); HI(d8,a3,b4); HI(d8,a4,b3);
LO(c8,a4,b4); HI(d9,a4,b4);
#endif
c[0] = c0; c[1] = _mm512_add_epi64(c1, d1); c[2] = _mm512_add_epi64(c2, d2); c[3] = _mm512_add_epi64(c3, d3); c[4] = _mm512_add_epi64(c4, d4);
c[5] = _mm512_add_epi64(c5, d5); c[6] = _mm512_add_epi64(c6, d6); c[7] = _mm512_add_epi64(c7, d7); c[8] = _mm512_add_epi64(c8, d8); c[9] = d9;
#endif
}
template <int S>
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_sqr_cols(__m512i *c, const fe8 &a) {
const __m512i Z = _mm512_setzero_si512();


const uint64_t SM = S == 2 ? 8 : 2;
__m512i x1 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * SM) : Z, x2 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * SM) : Z,
x3 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * SM) : Z, x4 = S ? _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * SM) : Z;
__m512i x5 = Z, x6 = Z, x7 = Z, x8 = Z;
const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
LO(x1,a0,a1); HI(x2,a0,a1);
LO(x2,a0,a2); HI(x3,a0,a2);
LO(x3,a0,a3); LO(x3,a1,a2); HI(x4,a0,a3); HI(x4,a1,a2);
LO(x4,a0,a4); LO(x4,a1,a3); HI(x5,a0,a4); HI(x5,a1,a3);
LO(x5,a1,a4); LO(x5,a2,a3); HI(x6,a1,a4); HI(x6,a2,a3);
LO(x6,a2,a4); HI(x7,a2,a4);
LO(x7,a3,a4); HI(x8,a3,a4);
__m512i c0 = S ? _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 2 * SM) : Z, c1 = _mm512_slli_epi64(x1, 1), c2 = _mm512_slli_epi64(x2, 1), c3 = _mm512_slli_epi64(x3, 1),
c4 = _mm512_slli_epi64(x4, 1), c5 = _mm512_slli_epi64(x5, 1), c6 = _mm512_slli_epi64(x6, 1),
c7 = _mm512_slli_epi64(x7, 1), c8 = _mm512_slli_epi64(x8, 1), c9 = Z;
#if QSB_CPU_FOLD3
HI(c9,a4,a4);
{ __m512i s5 = Z; HI(s5,a2,a2); s5 = _mm512_madd52hi_epu64(s5, c9, _mm512_set1_epi64(0x1000003D10ULL)); c5 = _mm512_add_epi64(c5, s5); }
LO(c0,a0,a0); HI(c1,a0,a0);
LO(c2,a1,a1); HI(c3,a1,a1);
LO(c4,a2,a2);
LO(c6,a3,a3); HI(c7,a3,a3);
LO(c8,a4,a4);
#else
LO(c0,a0,a0); HI(c1,a0,a0);
LO(c2,a1,a1); HI(c3,a1,a1);
LO(c4,a2,a2); HI(c5,a2,a2);
LO(c6,a3,a3); HI(c7,a3,a3);
LO(c8,a4,a4); HI(c9,a4,a4);
#endif
c[0] = c0; c[1] = c1; c[2] = c2; c[3] = c3; c[4] = c4; c[5] = c5; c[6] = c6; c[7] = c7; c[8] = c8; c[9] = c9;
}
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_mul(fe8 &r, const fe8 &a, const fe8 &b) {
__m512i c[10]; fe8_mul_cols<0>(c, a, b);
fe8_red(r, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
}
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_mul_lz(fe8 &r, const fe8 &a, const fe8 &b) {
__m512i c[10]; fe8_mul_cols<0>(c, a, b);
fe8 o; fe8_fold(o.l, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
fe8_carry_lz(o); r = o;
}
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sqr(fe8 &r, const fe8 &a) {
__m512i c[10]; fe8_sqr_cols<0>(c, a);
fe8_red(r, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);
}






static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sqr_sub2(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
#if QSB_CPU_PSEED
(void)P0; (void)P1; (void)P4;
__m512i k[10]; fe8_sqr_cols<1>(k, a);
fe8 o; fe8_fold(o.l, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]);
for (int i = 0; i < 5; i++) o.l[i] = _mm512_sub_epi64(_mm512_sub_epi64(o.l[i], b.l[i]), c.l[i]);
#else
__m512i k[10]; fe8_sqr_cols<0>(k, a);
fe8 o; fe8_fold(o.l, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]);
o.l[0] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[0], P0), b.l[0]), c.l[0]);
o.l[1] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[1], P1), b.l[1]), c.l[1]);
o.l[2] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[2], P1), b.l[2]), c.l[2]);
o.l[3] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[3], P1), b.l[3]), c.l[3]);
o.l[4] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(o.l[4], P4), b.l[4]), c.l[4]);
#endif
fe8_carry(o);
r = o;
}
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_mul_sub(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
#if QSB_CPU_PSEED
(void)P0; (void)P1; (void)P4;
__m512i k[10]; fe8_mul_cols<1>(k, a, b);
fe8 o; fe8_fold(o.l, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]);
for (int i = 0; i < 5; i++) o.l[i] = _mm512_sub_epi64(o.l[i], c.l[i]);
#else
__m512i k[10]; fe8_mul_cols<0>(k, a, b);
fe8 o; fe8_fold(o.l, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]);
o.l[0] = _mm512_sub_epi64(_mm512_add_epi64(o.l[0], P0), c.l[0]);
o.l[1] = _mm512_sub_epi64(_mm512_add_epi64(o.l[1], P1), c.l[1]);
o.l[2] = _mm512_sub_epi64(_mm512_add_epi64(o.l[2], P1), c.l[2]);
o.l[3] = _mm512_sub_epi64(_mm512_add_epi64(o.l[3], P1), c.l[3]);
o.l[4] = _mm512_sub_epi64(_mm512_add_epi64(o.l[4], P4), c.l[4]);
#endif
fe8_carry(o);
r = o;
}



static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sqr_subdx(fe8 &r, const fe8 &a, const fe8 &d, const fe8 &x) {
#if QSB_CPU_PSEED



__m512i k[10]; fe8_sqr_cols<QSB_CPU_DNF ? 2 : 1>(k, a);
fe8 o; fe8_fold(o.l, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]);
for (int i = 0; i < 5; i++) o.l[i] = _mm512_sub_epi64(o.l[i], _mm512_add_epi64(d.l[i], _mm512_add_epi64(x.l[i], x.l[i])));
#else
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
const __m512i Pl[5] = {P0, P1, P1, P1, P4};
__m512i k[10]; fe8_sqr_cols<0>(k, a);
fe8 o; fe8_fold(o.l, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9]);
for (int i = 0; i < 5; i++) o.l[i] = _mm512_sub_epi64(_mm512_add_epi64(o.l[i], Pl[i]), _mm512_add_epi64(d.l[i], _mm512_add_epi64(x.l[i], x.l[i])));
#endif
fe8_carry(o);
r = o;
}
#undef LO
#undef HI

static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_add(fe8 &r, const fe8 &a, const fe8 &b) {
for (int i = 0; i < 5; i++) r.l[i] = _mm512_add_epi64(a.l[i], b.l[i]);
fe8_carry(r);
}

static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sub(fe8 &r, const fe8 &a, const fe8 &b) {
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]);
r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]);
r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]);
r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]);
r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]);
fe8_carry(r);
}



static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sub_sgn(fe8 &r, const fe8 &a, const fe8 &b, __mmask8 m) {
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4), Z = _mm512_setzero_si512();
r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[0], m, Z, a.l[0]), P0), b.l[0]);
r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[1], m, Z, a.l[1]), P1), b.l[1]);
r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[2], m, Z, a.l[2]), P1), b.l[2]);
r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[3], m, Z, a.l[3]), P1), b.l[3]);
r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[4], m, Z, a.l[4]), P4), b.l[4]);
fe8_carry(r);
}

static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sub_lz(fe8 &r, const fe8 &a, const fe8 &b) {
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
fe8 o;
o.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]);
o.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]);
o.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]);
o.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]);
o.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]);
fe8_carry_lz(o); r = o;
}
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sub_sgn_lz(fe8 &r, const fe8 &a, const fe8 &b, __mmask8 m) {
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4), Z = _mm512_setzero_si512();
fe8 o;
o.l[0] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[0], m, Z, a.l[0]), P0), b.l[0]);
o.l[1] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[1], m, Z, a.l[1]), P1), b.l[1]);
o.l[2] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[2], m, Z, a.l[2]), P1), b.l[2]);
o.l[3] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[3], m, Z, a.l[3]), P1), b.l[3]);
o.l[4] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[4], m, Z, a.l[4]), P4), b.l[4]);
fe8_carry_lz(o); r = o;
}




static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_carry_nf(fe8 &r) {
__m512i c;
c = _mm512_srli_epi64(r.l[0], 52); r.l[1] = _mm512_add_epi64(r.l[1], c);
c = _mm512_srli_epi64(r.l[1], 52); r.l[2] = _mm512_add_epi64(r.l[2], c);
c = _mm512_srli_epi64(r.l[2], 52); r.l[3] = _mm512_add_epi64(r.l[3], c);
c = _mm512_srli_epi64(r.l[3], 52); r.l[4] = _mm512_add_epi64(r.l[4], c);
}




static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_carry_m(fe8 &r) {
const __m512i M = F8_M52;
__m512i c;
c = _mm512_srli_epi64(r.l[0], 52); r.l[0] = _mm512_and_si512(r.l[0], M); r.l[1] = _mm512_add_epi64(r.l[1], c);
c = _mm512_srli_epi64(r.l[1], 52); r.l[1] = _mm512_and_si512(r.l[1], M); r.l[2] = _mm512_add_epi64(r.l[2], c);
c = _mm512_srli_epi64(r.l[2], 52); r.l[2] = _mm512_and_si512(r.l[2], M); r.l[3] = _mm512_add_epi64(r.l[3], c);
c = _mm512_srli_epi64(r.l[3], 52); r.l[3] = _mm512_and_si512(r.l[3], M); r.l[4] = _mm512_add_epi64(r.l[4], c);
}
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sub_m(fe8 &r, const fe8 &a, const fe8 &b) {
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
fe8 o;
o.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]);
o.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]);
o.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]);
o.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]);
o.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]);
fe8_carry_m(o); r = o;
}

static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sub_d(fe8 &r, const fe8 &a, const fe8 &b) {
#if QSB_CPU_DNF
fe8_sub_m(r, a, b);
#else
fe8_sub(r, a, b);
#endif
}
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sub_nf(fe8 &r, const fe8 &a, const fe8 &b) {
#if QSB_CPU_NF
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
fe8 o;
o.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]);
o.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]);
o.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]);
o.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]);
o.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]);
fe8_carry_nf(o); r = o;
#else
fe8_sub_lz(r, a, b);
#endif
}
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sub_sgn_nf(fe8 &r, const fe8 &a, const fe8 &b, __mmask8 m) {
#if QSB_CPU_NF
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4), Z = _mm512_setzero_si512();
fe8 o;
o.l[0] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[0], m, Z, a.l[0]), P0), b.l[0]);
o.l[1] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[1], m, Z, a.l[1]), P1), b.l[1]);
o.l[2] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[2], m, Z, a.l[2]), P1), b.l[2]);
o.l[3] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[3], m, Z, a.l[3]), P1), b.l[3]);
o.l[4] = _mm512_sub_epi64(_mm512_add_epi64(_mm512_mask_sub_epi64(a.l[4], m, Z, a.l[4]), P4), b.l[4]);
fe8_carry_nf(o); r = o;
#else
fe8_sub_sgn_lz(r, a, b, m);
#endif
}


static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sub2(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 8), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 8),
P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 8);
r.l[0] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[0], P0), b.l[0]), c.l[0]);
r.l[1] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[1], P1), b.l[1]), c.l[1]);
r.l[2] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[2], P1), b.l[2]), c.l[2]);
r.l[3] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[3], P1), b.l[3]), c.l[3]);
r.l[4] = _mm512_sub_epi64(_mm512_sub_epi64(_mm512_add_epi64(a.l[4], P4), b.l[4]), c.l[4]);
fe8_carry(r);
}


#define Q8T __attribute__((target("avx512f,avx512ifma")))


Q8T static inline QCPU_AIF void fe8_cp(fe8 &r, const fe8 &a) {
for (int i = 0; i < 5; i++) _mm512_store_si512(&r.l[i], _mm512_load_si512(&a.l[i]));
}
Q8T static inline QCPU_AIF void fe8_set1(fe8 &r) {
r.l[0] = _mm512_set1_epi64(1);
for (int i = 1; i < 5; i++) r.l[i] = _mm512_setzero_si512();
}
Q8T static inline QCPU_AIF void fe8_bcast(fe8 &r, const fe &a) {
const uint64_t M = 0xFFFFFFFFFFFFFULL;
r.l[0] = _mm512_set1_epi64((long long)(a.v[0] & M));
r.l[1] = _mm512_set1_epi64((long long)((a.v[0] >> 52 | a.v[1] << 12) & M));
r.l[2] = _mm512_set1_epi64((long long)((a.v[1] >> 40 | a.v[2] << 24) & M));
r.l[3] = _mm512_set1_epi64((long long)((a.v[2] >> 28 | a.v[3] << 36) & M));
r.l[4] = _mm512_set1_epi64((long long)(a.v[3] >> 16));
}
#ifndef QSB_CPU_VBMI2
#define QSB_CPU_VBMI2 1




#endif
#if QSB_CPU_VBMI2
#define Q8TX __attribute__((target("avx512f,avx512ifma,avx512vbmi2")))
Q8TX static inline QCPU_AIF void fe8_from64(fe8 &r, __m512i a0, __m512i a1, __m512i a2, __m512i a3) {
const __m512i M = F8_M52;
r.l[0] = _mm512_and_si512(a0, M);
r.l[1] = _mm512_and_si512(_mm512_shrdi_epi64(a0, a1, 52), M);
r.l[2] = _mm512_and_si512(_mm512_shrdi_epi64(a1, a2, 40), M);
r.l[3] = _mm512_and_si512(_mm512_shrdi_epi64(a2, a3, 28), M);
r.l[4] = _mm512_srli_epi64(a3, 16);
}
#else
#define Q8TX Q8T
Q8T static inline QCPU_AIF void fe8_from64(fe8 &r, __m512i a0, __m512i a1, __m512i a2, __m512i a3) {
const __m512i M = F8_M52;
r.l[0] = _mm512_and_si512(a0, M);
r.l[1] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a0, 52), _mm512_slli_epi64(a1, 12)), M);
r.l[2] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a1, 40), _mm512_slli_epi64(a2, 24)), M);
r.l[3] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(a2, 28), _mm512_slli_epi64(a3, 36)), M);
r.l[4] = _mm512_srli_epi64(a3, 16);
}
#endif

Q8TX static inline QCPU_AIF void pt8_load(fe8 &x, fe8 &y, const pt *const *rows) {
const __m512i r0 = _mm512_loadu_si512(rows[0]), r1 = _mm512_loadu_si512(rows[1]),
r2 = _mm512_loadu_si512(rows[2]), r3 = _mm512_loadu_si512(rows[3]),
r4 = _mm512_loadu_si512(rows[4]), r5 = _mm512_loadu_si512(rows[5]),
r6 = _mm512_loadu_si512(rows[6]), r7 = _mm512_loadu_si512(rows[7]);
const __m512i a0 = _mm512_unpacklo_epi64(r0, r1), a1 = _mm512_unpackhi_epi64(r0, r1),
a2 = _mm512_unpacklo_epi64(r2, r3), a3 = _mm512_unpackhi_epi64(r2, r3),
a4 = _mm512_unpacklo_epi64(r4, r5), a5 = _mm512_unpackhi_epi64(r4, r5),
a6 = _mm512_unpacklo_epi64(r6, r7), a7 = _mm512_unpackhi_epi64(r6, r7);
const __m512i iL = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0), iH = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
const __m512i b0 = _mm512_permutex2var_epi64(a0, iL, a2), b2 = _mm512_permutex2var_epi64(a0, iH, a2),
b1 = _mm512_permutex2var_epi64(a1, iL, a3), b3 = _mm512_permutex2var_epi64(a1, iH, a3),
b4 = _mm512_permutex2var_epi64(a4, iL, a6), b6 = _mm512_permutex2var_epi64(a4, iH, a6),
b5 = _mm512_permutex2var_epi64(a5, iL, a7), b7 = _mm512_permutex2var_epi64(a5, iH, a7);
const __m512i c0 = _mm512_shuffle_i64x2(b0, b4, 0x44), c4 = _mm512_shuffle_i64x2(b0, b4, 0xEE),
c1 = _mm512_shuffle_i64x2(b1, b5, 0x44), c5 = _mm512_shuffle_i64x2(b1, b5, 0xEE),
c2 = _mm512_shuffle_i64x2(b2, b6, 0x44), c6 = _mm512_shuffle_i64x2(b2, b6, 0xEE),
c3 = _mm512_shuffle_i64x2(b3, b7, 0x44), c7 = _mm512_shuffle_i64x2(b3, b7, 0xEE);
fe8_from64(x, c0, c1, c2, c3);
fe8_from64(y, c4, c5, c6, c7);
}

static inline void fe8_lane_canon(fe &r, const uint64_t l[5]) {
typedef unsigned __int128 q128;
q128 c = (q128)l[0] + ((q128)l[1] << 52);
uint64_t w0 = (uint64_t)c; c >>= 64;
c += (q128)l[2] << 40; uint64_t w1 = (uint64_t)c; c >>= 64;
c += (q128)l[3] << 28; uint64_t w2 = (uint64_t)c; c >>= 64;
c += (q128)l[4] << 16; uint64_t w3 = (uint64_t)c; c >>= 64;
uint64_t top = (uint64_t)c;
while (top) {
q128 d = (q128)w0 + (q128)top * 0x1000003D1ULL; w0 = (uint64_t)d; d >>= 64;
d += w1; w1 = (uint64_t)d; d >>= 64; d += w2; w2 = (uint64_t)d; d >>= 64;
d += w3; w3 = (uint64_t)d; d >>= 64; top = (uint64_t)d;
}
if (w3 == ~0ULL && w2 == ~0ULL && w1 == ~0ULL && w0 >= 0xFFFFFFFEFFFFFC2FULL) {
q128 d = (q128)w0 + 0x1000003D1ULL; w0 = (uint64_t)d; d >>= 64;
d += w1; w1 = (uint64_t)d; d >>= 64; d += w2; w2 = (uint64_t)d; d >>= 64; w3 += (uint64_t)d;
}
r.v[0] = w0; r.v[1] = w1; r.v[2] = w2; r.v[3] = w3;
}
Q8T static inline void fe8_store_canon(fe out[8], const fe8 &a) {
alignas(64) uint64_t L[5][8];
for (int i = 0; i < 5; i++) _mm512_store_si512(L[i], a.l[i]);
for (int j = 0; j < 8; j++) { uint64_t l[5] = {L[0][j], L[1][j], L[2][j], L[3][j], L[4][j]}; fe8_lane_canon(out[j], l); }
}


Q8T static inline QCPU_AIF __mmask8 fe8_ge_p(const fe8 &a, fe8 *u) {
const __m512i M = F8_M52, K = _mm512_set1_epi64(0x1000003D1ULL);
__m512i u0 = _mm512_add_epi64(a.l[0], K), c;
c = _mm512_srli_epi64(u0, 52); u0 = _mm512_and_si512(u0, M); __m512i u1 = _mm512_add_epi64(a.l[1], c);
c = _mm512_srli_epi64(u1, 52); u1 = _mm512_and_si512(u1, M); __m512i u2 = _mm512_add_epi64(a.l[2], c);
c = _mm512_srli_epi64(u2, 52); u2 = _mm512_and_si512(u2, M); __m512i u3 = _mm512_add_epi64(a.l[3], c);
c = _mm512_srli_epi64(u3, 52); u3 = _mm512_and_si512(u3, M); __m512i u4 = _mm512_add_epi64(a.l[4], c);
const __mmask8 ge = _mm512_test_epi64_mask(u4, _mm512_set1_epi64((long long)~0x0FFFFFFFFFFFFULL));
if (u) { u->l[0] = u0; u->l[1] = u1; u->l[2] = u2; u->l[3] = u3; u->l[4] = _mm512_and_si512(u4, _mm512_set1_epi64(0x0FFFFFFFFFFFFULL)); }
return ge;
}

Q8T static inline QCPU_AIF void fe8_canon64(uint64_t w[4][8], const fe8 &a) {
fe8 u; const __mmask8 ge = fe8_ge_p(a, &u);
__m512i r[5];
for (int i = 0; i < 5; i++) r[i] = _mm512_mask_blend_epi64(ge, a.l[i], u.l[i]);
_mm512_store_si512(w[0], _mm512_or_si512(r[0], _mm512_slli_epi64(r[1], 52)));
_mm512_store_si512(w[1], _mm512_or_si512(_mm512_srli_epi64(r[1], 12), _mm512_slli_epi64(r[2], 40)));
_mm512_store_si512(w[2], _mm512_or_si512(_mm512_srli_epi64(r[2], 24), _mm512_slli_epi64(r[3], 28)));
_mm512_store_si512(w[3], _mm512_or_si512(_mm512_srli_epi64(r[3], 36), _mm512_slli_epi64(r[4], 16)));
}

Q8T static inline QCPU_AIF __mmask8 fe8_parity(const fe8 &a) {
return (__mmask8)(_mm512_test_epi64_mask(a.l[0], _mm512_set1_epi64(1)) ^ fe8_ge_p(a, nullptr));
}


Q8T static void fe8_inv1(fe8 &x) {
fe8 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
#define SQN(dst, src, n) do { dst = src; for (int i = 0; i < (n); i++) fe8_sqr(dst, dst); } while (0)
SQN(x2, x, 1); fe8_mul(x2, x2, x);
SQN(x3, x2, 1); fe8_mul(x3, x3, x);
SQN(x6, x3, 3); fe8_mul(x6, x6, x3);
SQN(x9, x6, 3); fe8_mul(x9, x9, x3);
SQN(x11, x9, 2); fe8_mul(x11, x11, x2);
SQN(x22, x11, 11); fe8_mul(x22, x22, x11);
SQN(x44, x22, 22); fe8_mul(x44, x44, x22);
SQN(x88, x44, 44); fe8_mul(x88, x88, x44);
SQN(x176, x88, 88); fe8_mul(x176, x176, x88);
SQN(x220, x176, 44); fe8_mul(x220, x220, x44);
SQN(x223, x220, 3); fe8_mul(x223, x223, x3);
SQN(t, x223, 23); fe8_mul(t, t, x22);
SQN(t, t, 5); fe8_mul(t, t, x);
SQN(t, t, 3); fe8_mul(t, t, x2);
SQN(t, t, 2); fe8_mul(x, t, x);
#undef SQN
}
#ifndef QSB_CPU_SCALAR_INV
#define QSB_CPU_SCALAR_INV 1
#endif
#if QSB_CPU_SCALAR_INV



struct s62 { int64_t v[5]; };
struct trans2x2 { int64_t u, v, q, r; };
static const s62 S62_P = {{-0x1000003D1LL, 0, 0, 0, 256}};
static const uint64_t S62_PINV = 0x27C7F6E22DDACACFULL;

static inline int64_t divsteps_62_var(int64_t eta, uint64_t f0, uint64_t g0, trans2x2 *t) {
uint64_t u = 1, v = 0, q = 0, r = 1, f = f0, g = g0, m; uint32_t w; int i = 62, limit, zeros;
for (;;) {
zeros = __builtin_ctzll(g | (~0ULL << i));
g >>= zeros; u <<= zeros; v <<= zeros; eta -= zeros; i -= zeros;
if (i == 0) break;
if (eta < 0) {
uint64_t tmp;
eta = -eta;
tmp = f; f = g; g = -tmp;
tmp = u; u = q; q = -tmp;
tmp = v; v = r; r = -tmp;
limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
m = (~0ULL >> (64 - limit)) & 63U;
w = (uint32_t)((f * g * (f * f - 2)) & m);
} else {
limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
m = (~0ULL >> (64 - limit)) & 15U;
w = (uint32_t)(f + (((f + 1) & 4) << 1));
w = (uint32_t)((-(uint64_t)w * g) & m);
}
g += f * w; q += u * w; r += v * w;
}
t->u = (int64_t)u; t->v = (int64_t)v; t->q = (int64_t)q; t->r = (int64_t)r;
return eta;
}

static inline void update_de_62(s62 *d, s62 *e, const trans2x2 *t) {
const uint64_t M62 = ~0ULL >> 2;
const int64_t d0 = d->v[0], d1 = d->v[1], d2 = d->v[2], d3 = d->v[3], d4 = d->v[4];
const int64_t e0 = e->v[0], e1 = e->v[1], e2 = e->v[2], e3 = e->v[3], e4 = e->v[4];
const int64_t u = t->u, v = t->v, q = t->q, r = t->r;
int64_t md, me, sd, se; __int128 cd, ce;
sd = d4 >> 63; se = e4 >> 63;
md = (u & sd) + (v & se);
me = (q & sd) + (r & se);
cd = (__int128)u * d0 + (__int128)v * e0;
ce = (__int128)q * d0 + (__int128)r * e0;
md -= (int64_t)((S62_PINV * (uint64_t)cd + (uint64_t)md) & M62);
me -= (int64_t)((S62_PINV * (uint64_t)ce + (uint64_t)me) & M62);
cd += (__int128)S62_P.v[0] * md;
ce += (__int128)S62_P.v[0] * me;
cd >>= 62; ce >>= 62;
cd += (__int128)u * d1 + (__int128)v * e1;
ce += (__int128)q * d1 + (__int128)r * e1;
d->v[0] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
e->v[0] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
cd += (__int128)u * d2 + (__int128)v * e2;
ce += (__int128)q * d2 + (__int128)r * e2;
d->v[1] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
e->v[1] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
cd += (__int128)u * d3 + (__int128)v * e3;
ce += (__int128)q * d3 + (__int128)r * e3;
d->v[2] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
e->v[2] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
cd += (__int128)u * d4 + (__int128)v * e4;
ce += (__int128)q * d4 + (__int128)r * e4;
cd += (__int128)S62_P.v[4] * md;
ce += (__int128)S62_P.v[4] * me;
d->v[3] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
e->v[3] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
d->v[4] = (int64_t)cd;
e->v[4] = (int64_t)ce;
}

static inline void update_fg_62_var(int len, s62 *f, s62 *g, const trans2x2 *t) {
const uint64_t M62 = ~0ULL >> 2;
const int64_t u = t->u, v = t->v, q = t->q, r = t->r;
int64_t fi = f->v[0], gi = g->v[0];
__int128 cf = (__int128)u * fi + (__int128)v * gi;
__int128 cg = (__int128)q * fi + (__int128)r * gi;
cf >>= 62; cg >>= 62;
for (int i = 1; i < len; ++i) {
fi = f->v[i]; gi = g->v[i];
cf += (__int128)u * fi + (__int128)v * gi;
cg += (__int128)q * fi + (__int128)r * gi;
f->v[i - 1] = (int64_t)((uint64_t)(int64_t)cf & M62); cf >>= 62;
g->v[i - 1] = (int64_t)((uint64_t)(int64_t)cg & M62); cg >>= 62;
}
f->v[len - 1] = (int64_t)cf;
g->v[len - 1] = (int64_t)cg;
}

static inline void normalize_62(s62 *r, int64_t sign) {
const int64_t M62 = (int64_t)(~0ULL >> 2);
int64_t r0 = r->v[0], r1 = r->v[1], r2 = r->v[2], r3 = r->v[3], r4 = r->v[4];
int64_t cond_add = r4 >> 63;
r0 += S62_P.v[0] & cond_add; r4 += S62_P.v[4] & cond_add;
const int64_t cond_negate = sign >> 63;
r0 = (r0 ^ cond_negate) - cond_negate; r1 = (r1 ^ cond_negate) - cond_negate; r2 = (r2 ^ cond_negate) - cond_negate;
r3 = (r3 ^ cond_negate) - cond_negate; r4 = (r4 ^ cond_negate) - cond_negate;
r1 += r0 >> 62; r0 &= M62; r2 += r1 >> 62; r1 &= M62; r3 += r2 >> 62; r2 &= M62; r4 += r3 >> 62; r3 &= M62;
cond_add = r4 >> 63;
r0 += S62_P.v[0] & cond_add; r4 += S62_P.v[4] & cond_add;
r1 += r0 >> 62; r0 &= M62; r2 += r1 >> 62; r1 &= M62; r3 += r2 >> 62; r2 &= M62; r4 += r3 >> 62; r3 &= M62;
r->v[0] = r0; r->v[1] = r1; r->v[2] = r2; r->v[3] = r3; r->v[4] = r4;
}

static bool modinv_var(s62 *x) {
s62 d = {{0, 0, 0, 0, 0}}, e = {{1, 0, 0, 0, 0}}, f = S62_P, g = *x;
int len = 5; int64_t eta = -1;
for (int it = 0; it < 24; it++) {
trans2x2 t;
eta = divsteps_62_var(eta, (uint64_t)f.v[0], (uint64_t)g.v[0], &t);
update_de_62(&d, &e, &t);
update_fg_62_var(len, &f, &g, &t);
if (g.v[0] == 0) {
int64_t cond = 0; for (int j = 1; j < len; ++j) cond |= g.v[j];
if (cond == 0) { normalize_62(&d, f.v[len - 1]); *x = d; return true; }
}
const int64_t fn = f.v[len - 1], gn = g.v[len - 1];
int64_t cond = ((int64_t)len - 2) >> 63; cond |= fn ^ (fn >> 63); cond |= gn ^ (gn >> 63);
if (cond == 0) { f.v[len - 2] |= (int64_t)((uint64_t)fn << 62); g.v[len - 2] |= (int64_t)((uint64_t)gn << 62); --len; }
}
return false;
}

static void fe_inv_var(fe &r, const fe &a) {
const uint64_t M = ~0ULL >> 2; s62 x;
x.v[0] = (int64_t)(a.v[0] & M); x.v[1] = (int64_t)((a.v[0] >> 62 | a.v[1] << 2) & M);
x.v[2] = (int64_t)((a.v[1] >> 60 | a.v[2] << 4) & M); x.v[3] = (int64_t)((a.v[2] >> 58 | a.v[3] << 6) & M);
x.v[4] = (int64_t)(a.v[3] >> 56);
if (!modinv_var(&x)) { fe_inv(r, a); return; }
r.v[0] = (uint64_t)x.v[0] | (uint64_t)x.v[1] << 62; r.v[1] = (uint64_t)x.v[1] >> 2 | (uint64_t)x.v[2] << 60;
r.v[2] = (uint64_t)x.v[2] >> 4 | (uint64_t)x.v[3] << 58; r.v[3] = (uint64_t)x.v[3] >> 6 | (uint64_t)x.v[4] << 56;
}



Q8T static inline void fe8_swap1(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_permutex_epi64(a.l[i], 0xB1); }
Q8T static inline void fe8_swap2(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_permutex_epi64(a.l[i], 0x4E); }
Q8T static inline void fe8_swap4(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_shuffle_i64x2(a.l[i], a.l[i], 0x4E); }
Q8T static void fe8_inv_lanes(fe8 &x) {
fe8 a1, p1, p2, t, i2;
fe8_swap1(a1, x); fe8_mul(p1, x, a1);
fe8_swap2(t, p1); fe8_mul(p2, p1, t);
fe8_swap4(t, p2); fe8_mul(t, p2, t);
fe pr, pi;
#if QSB_CPU_JL_INV_LANE0


uint64_t l0[5];
for (int i = 0; i < 5; i++) l0[i] = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(t.l[i]));
fe8_lane_canon(pr, l0);
#else
alignas(64) uint64_t w[4][8]; fe8_canon64(w, t);
pr = fe{{w[0][0], w[1][0], w[2][0], w[3][0]}};
#endif
fe_inv_var(pi, pr);
fe8 I; fe8_bcast(I, pi);
fe8_swap4(t, p2); fe8_mul(i2, I, t);
fe8_swap2(t, p1); fe8_mul(i2, i2, t);
fe8_mul(x, i2, a1);
}
#endif


Q8T static void fe8_inv4(fe8 *x) {
fe8 a01, a23, a, i01, i23;
fe8_mul(a01, x[0], x[1]); fe8_mul(a23, x[2], x[3]); fe8_mul(a, a01, a23);
#if QSB_CPU_SCALAR_INV
fe8_inv_lanes(a);
#else
fe8_inv1(a);
#endif
fe8_mul(i01, a, a23); fe8_mul(i23, a, a01);
const fe8 x0 = x[0], x2 = x[2];
fe8_mul(x[0], i01, x[1]); fe8_mul(x[1], i01, x0);
fe8_mul(x[2], i23, x[3]); fe8_mul(x[3], i23, x2);
}
#if QSB_CPU_NCH == 2


Q8T static void fe8_inv2(fe8 *x) {
fe8 a;
fe8_mul(a, x[0], x[1]);
#if QSB_CPU_SCALAR_INV
fe8_inv_lanes(a);
#else
fe8_inv1(a);
#endif
const fe8 x0 = x[0];
fe8_mul(x[0], a, x[1]); fe8_mul(x[1], a, x0);
}
#define QCPU_INVC(r) fe8_inv2(r)
#else
#define QCPU_INVC(r) fe8_inv4(r)
#endif

static inline QCPU_AIF void qcpu_pf_rows(const pt *const *pr, int j0, int j1) {
if (pr) for (int j = j0; j < j1; j++) _mm_prefetch((const char *)pr[j], QCPU_NXT_HINT);
}
#include "cpu_inverse_peel_window.h"





template <class RowFn>
Q8TX static void ec8_window(fe8 *X, fe8 *Y, fe8 *D, fe8 *PRE, fe8 *TX, fe8 *TY, int G, const pt *const *rp, const __mmask8 *ng,
const pt **rpn, __mmask8 *ngn, const RowFn *nxt) {
#if QSB_CPU_INV_PEEL
ec8_window_peeled(X, Y, D, PRE, G, rp, ng, rpn, ngn, nxt);
(void)TX; (void)TY;
#else
const int PF = QSB_CPU_PFD;
const int NC = QSB_CPU_NCH;
fe8 run[4]; for (int c = 0; c < NC; c++) fe8_set1(run[c]);
#if QSB_CPU_ILP2 & 2


for (int g = 0; g < G; g += NC) {
for (int c = 0; c < NC; c += 2) {
const int hA = g + c, hB = hA + 1;
if (hA + PF < G) { const pt *const *pr = rp + (size_t)(hA + PF) * 8; for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
if (hB + PF < G) { const pt *const *pr = rp + (size_t)(hB + PF) * 8; for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
fe8 txA, tyA, tA, txB, tyB, tB;
pt8_load(txA, tyA, rp + (size_t)hA * 8); pt8_load(txB, tyB, rp + (size_t)hB * 8);
fe8_sub_d(D[hA], txA, X[hA]); fe8_sub_d(D[hB], txB, X[hB]);
fe8_sub_sgn_nf(tA, tyA, Y[hA], ng[hA]); fe8_sub_sgn_nf(tB, tyB, Y[hB], ng[hB]);
if (QSB_CPU_JL_INV_FIRST && g == 0) {
fe8_cp(PRE[hA], tA); fe8_cp(PRE[hB], tB);
fe8_cp(run[c], D[hA]); fe8_cp(run[c + 1], D[hB]);
} else {
fe8_mul_lz(PRE[hA], run[c], tA); fe8_mul_lz(PRE[hB], run[c + 1], tB);
fe8_mul_lz(run[c], run[c], D[hA]); fe8_mul_lz(run[c + 1], run[c + 1], D[hB]);
}
}
}
(void)TX; (void)TY;
#else
for (int g = 0; g < G; g += NC) {
for (int c = 0; c < NC; c++) {
const int h = g + c;
if (h + PF < G) { const pt *const *pr = rp + (size_t)(h + PF) * 8; for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
#if QSB_CPU_WPRE


{ fe8 tx, ty, t; pt8_load(tx, ty, rp + (size_t)h * 8); fe8_sub_d(D[h], tx, X[h]);
fe8_sub_sgn_nf(t, ty, Y[h], ng[h]);
if (QSB_CPU_JL_INV_FIRST >= 2 && g == 0) fe8_cp(PRE[h], t);
else fe8_mul_lz(PRE[h], run[c], t); }
#elif QSB_CPU_NOTX
{ fe8 tx; pt8_load(tx, TY[h], rp + (size_t)h * 8); fe8_sub_d(D[h], tx, X[h]); }
fe8_cp(PRE[h], run[c]);
#else
pt8_load(TX[h], TY[h], rp + (size_t)h * 8);
fe8_sub_lz(D[h], TX[h], X[h]);
fe8_cp(PRE[h], run[c]);
#endif
if (QSB_CPU_JL_INV_FIRST >= 2 && QSB_CPU_WPRE && g == 0) fe8_cp(run[c], D[h]);
else fe8_mul_lz(run[c], run[c], D[h]);
}
}
#endif
QCPU_INVC(run);
#if QSB_CPU_ILP2 & 1


for (int g = G - NC; g >= 0; g -= NC) {
for (int c = NC - 1; c >= 1; c -= 2) {
const int hA = g + c, hB = hA - 1;
const pt **pA = nullptr, **pB = nullptr;
if (nxt) { const int nA = G - 1 - hA, nB = nA + 1; pA = rpn + (size_t)nA * 8; ngn[nA] = (*nxt)(nA, pA);
pB = rpn + (size_t)nB * 8; ngn[nB] = (*nxt)(nB, pB); }
qcpu_pf_rows(pA, 0, 4);
fe8 lamA, lamB, x3A, x3B, tA, tB;
fe8_mul_lz(lamA, run[c], PRE[hA]); fe8_mul_lz(lamB, run[c - 1], PRE[hB]);
qcpu_pf_rows(pA, 4, 8);
if (!QSB_CPU_JL_INV_LAST || g > 0) {
fe8_mul_lz(run[c], run[c], D[hA]); fe8_mul_lz(run[c - 1], run[c - 1], D[hB]);
}
qcpu_pf_rows(pB, 0, 4);
fe8_sqr_subdx(x3A, lamA, D[hA], X[hA]); fe8_sqr_subdx(x3B, lamB, D[hB], X[hB]);
qcpu_pf_rows(pB, 4, 8);
fe8_sub_nf(tA, X[hA], x3A); fe8_sub_nf(tB, X[hB], x3B);
fe8_mul_sub(Y[hA], lamA, tA, Y[hA]); fe8_mul_sub(Y[hB], lamB, tB, Y[hB]);
fe8_cp(X[hA], x3A); fe8_cp(X[hB], x3B);
}
}
#else
for (int g = G - NC; g >= 0; g -= NC) {
for (int c = NC - 1; c >= 0; c--) {
const int h = g + c;
const pt **pr = nullptr;
#if QSB_CPU_PFSPREAD & 2

if (nxt) { const int hn = G - 1 - h; pr = rpn + (size_t)hn * 8; ngn[hn] = (*nxt)(hn, pr); }
qcpu_pf_rows(pr, 0, 2);
fe8 t, lam, x3;
fe8_mul_lz(lam, run[c], PRE[h]); qcpu_pf_rows(pr, 2, 4);
fe8_mul_lz(run[c], run[c], D[h]); qcpu_pf_rows(pr, 4, 6);
fe8_sqr_subdx(x3, lam, D[h], X[h]); qcpu_pf_rows(pr, 6, 8);
fe8_sub_nf(t, X[h], x3); fe8_mul_sub(Y[h], lam, t, Y[h]); fe8_cp(X[h], x3);
#else
if (nxt) { const int hn = G - 1 - h; pr = rpn + (size_t)hn * 8; ngn[hn] = (*nxt)(hn, pr);
for (int j = 0; j < 4; j++) _mm_prefetch((const char *)pr[j], QCPU_NXT_HINT); }
fe8 t, lam, x3;
#if QSB_CPU_WPRE
fe8_mul_lz(lam, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);
if (pr) for (int j = 4; j < 8; j++) _mm_prefetch((const char *)pr[j], QCPU_NXT_HINT);
#else
fe8 dinv;
fe8_mul_lz(dinv, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);
if (pr) for (int j = 4; j < 8; j++) _mm_prefetch((const char *)pr[j], QCPU_NXT_HINT);
fe8_sub_sgn_nf(t, TY[h], Y[h], ng[h]); fe8_mul_lz(lam, t, dinv);
#endif
#if QSB_CPU_NOTX
fe8_sqr_subdx(x3, lam, D[h], X[h]);
#else
fe8_sqr_sub2(x3, lam, X[h], TX[h]);
#endif
fe8_sub_nf(t, X[h], x3); fe8_mul_sub(Y[h], lam, t, Y[h]); fe8_cp(X[h], x3);
#endif
}
}
#endif
#endif
}



Q8T static inline QCPU_AIF void fe8_store_x(fe *o, const fe8 &a) {
fe8 u; const __mmask8 ge = fe8_ge_p(a, &u);
__m512i r[5];
for (int i = 0; i < 5; i++) r[i] = _mm512_mask_blend_epi64(ge, a.l[i], u.l[i]);
const __m512i w0 = _mm512_or_si512(r[0], _mm512_slli_epi64(r[1], 52)),
w1 = _mm512_or_si512(_mm512_srli_epi64(r[1], 12), _mm512_slli_epi64(r[2], 40)),
w2 = _mm512_or_si512(_mm512_srli_epi64(r[2], 24), _mm512_slli_epi64(r[3], 28)),
w3 = _mm512_or_si512(_mm512_srli_epi64(r[3], 36), _mm512_slli_epi64(r[4], 16));
const __m512i a0 = _mm512_unpacklo_epi64(w0, w1), a1 = _mm512_unpackhi_epi64(w0, w1),
a2 = _mm512_unpacklo_epi64(w2, w3), a3 = _mm512_unpackhi_epi64(w2, w3);
const __m512i iL = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), iH = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
const __m512i b0 = _mm512_permutex2var_epi64(a0, iL, a2),
b1 = _mm512_permutex2var_epi64(a1, iL, a3),
b2 = _mm512_permutex2var_epi64(a0, iH, a2),
b3 = _mm512_permutex2var_epi64(a1, iH, a3);
_mm256_storeu_si256((__m256i *)(o + 0), _mm512_castsi512_si256(b0)); _mm256_storeu_si256((__m256i *)(o + 4), _mm512_extracti64x4_epi64(b0, 1));
_mm256_storeu_si256((__m256i *)(o + 2), _mm512_castsi512_si256(b1)); _mm256_storeu_si256((__m256i *)(o + 6), _mm512_extracti64x4_epi64(b1, 1));
_mm256_storeu_si256((__m256i *)(o + 8), _mm512_castsi512_si256(b2)); _mm256_storeu_si256((__m256i *)(o + 12), _mm512_extracti64x4_epi64(b2, 1));
_mm256_storeu_si256((__m256i *)(o + 10), _mm512_castsi512_si256(b3)); _mm256_storeu_si256((__m256i *)(o + 14), _mm512_extracti64x4_epi64(b3, 1));
}

static inline QCPU_AIF uint64_t spread8(unsigned x) {
const uint64_t v = ((uint64_t)x * 0x0101010101010101ULL) & 0x8040201008040201ULL;
return ((v + 0x7F7F7F7F7F7F7F7FULL) & 0x8080808080808080ULL) >> 7;
}


Q8T static void ec8_final(const fe8 *X, const fe8 *Y, fe8 *D, fe8 *PRE, int G, const fe &cx, const fe &cy,
fe *qx , uint8_t *qp) {
fe8 CX, CY0, CY1, Z; fe8_bcast(CX, cx); fe8_bcast(CY0, cy);
for (int i = 0; i < 5; i++) Z.l[i] = _mm512_setzero_si512();
fe8_sub(CY1, Z, CY0);
fe8 run[4]; for (int c = 0; c < 4; c++) fe8_set1(run[c]);
for (int g = 0; g < G; g += 4)
for (int c = 0; c < 4; c++) { const int h = g + c; fe8_sub_lz(D[h], CX, X[h]); fe8_cp(PRE[h], run[c]); fe8_mul_lz(run[c], run[c], D[h]); }
fe8_inv4(run);
for (int g = G - 4; g >= 0; g -= 4) {
for (int c = 3; c >= 0; c--) {
const int h = g + c;
fe8 dinv; fe8_mul_lz(dinv, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);
unsigned par[2];
for (int ri = 0; ri < 2; ri++) {
fe8 t, lam, x3, y3;
fe8_sub_lz(t, ri ? CY1 : CY0, Y[h]); fe8_mul_lz(lam, t, dinv);
fe8_sqr_sub2(x3, lam, X[h], CX);
fe8_sub_lz(t, X[h], x3); fe8_mul_sub(y3, lam, t, Y[h]);
fe8_store_x(qx + (size_t)h * 16 + ri, x3);
par[ri] = fe8_parity(y3);
}

_mm_storeu_si128((__m128i *)(qp + (size_t)h * 16),
_mm_unpacklo_epi8(_mm_cvtsi64_si128((long long)spread8(par[0])), _mm_cvtsi64_si128((long long)spread8(par[1]))));
}
}
}



Q8T static void ec8_final_cf(const fe8 *X, const fe8 *Y, fe8 *D, fe8 *PRE, int G, const fe &mx, const fe &my,
fe *qx , uint8_t *qp) {
fe8 MX, MY; fe8_bcast(MX, mx); fe8_bcast(MY, my);
fe8 run[4]; for (int c = 0; c < 4; c++) fe8_set1(run[c]);
for (int g = 0; g < G; g += 4)
for (int c = 0; c < 4; c++) {
const int h = g + c; fe8_sub_lz(D[h], MX, X[h]);
#if QSB_CPU_WPRE
{ fe8 t; fe8_sub_lz(t, MY, Y[h]); fe8_mul_lz(PRE[h], run[c], t); }
#else
fe8_cp(PRE[h], run[c]);
#endif
fe8_mul_lz(run[c], run[c], D[h]);
}
fe8_inv4(run);
for (int g = G - 4; g >= 0; g -= 4) {
for (int c = 3; c >= 0; c--) {
const int h = g + c;
fe8 t, lam, x3, y3;
#if QSB_CPU_WPRE
fe8_mul_lz(lam, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);
#else
fe8 dinv;
fe8_mul_lz(dinv, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);
fe8_sub_lz(t, MY, Y[h]); fe8_mul_lz(lam, t, dinv);
#endif
fe8_sqr_sub2(x3, lam, X[h], MX);
fe8_sub_lz(t, X[h], x3); fe8_mul_sub(y3, lam, t, Y[h]);
fe8_store_x(qx + (size_t)h * 16 + 0, X[h]);
fe8_store_x(qx + (size_t)h * 16 + 1, x3);
const unsigned p0 = fe8_parity(Y[h]), p1 = fe8_parity(y3);
_mm_storeu_si128((__m128i *)(qp + (size_t)h * 16),
_mm_unpacklo_epi8(_mm_cvtsi64_si128((long long)spread8(p0)), _mm_cvtsi64_si128((long long)spread8(p1))));
}
}
}
#if QSB_CPU_KH16 && QCPU_SHANI






Q8T static inline QCPU_AIF void kh16_canon(__m512i w[4], const fe8 &a) {
fe8 u; const __mmask8 ge = fe8_ge_p(a, &u);
__m512i r[5];
for (int i = 0; i < 5; i++) r[i] = _mm512_mask_blend_epi64(ge, a.l[i], u.l[i]);
w[0] = _mm512_or_si512(r[0], _mm512_slli_epi64(r[1], 52));
w[1] = _mm512_or_si512(_mm512_srli_epi64(r[1], 12), _mm512_slli_epi64(r[2], 40));
w[2] = _mm512_or_si512(_mm512_srli_epi64(r[2], 24), _mm512_slli_epi64(r[3], 28));
w[3] = _mm512_or_si512(_mm512_srli_epi64(r[3], 36), _mm512_slli_epi64(r[4], 16));
}


Q8T static inline QCPU_AIF void kh16_words8(__m512i W[9], const __m512i w[4], __mmask8 par) {
W[0] = _mm512_mask_or_epi64(_mm512_or_si512(_mm512_srli_epi64(w[3], 40), _mm512_set1_epi64(0x02000000)), par,
_mm512_or_si512(_mm512_srli_epi64(w[3], 40), _mm512_set1_epi64(0x02000000)), _mm512_set1_epi64(0x01000000));
W[1] = _mm512_srli_epi64(w[3], 8);
W[2] = _mm512_or_si512(_mm512_slli_epi64(w[3], 24), _mm512_srli_epi64(w[2], 40));
W[3] = _mm512_srli_epi64(w[2], 8);
W[4] = _mm512_or_si512(_mm512_slli_epi64(w[2], 24), _mm512_srli_epi64(w[1], 40));
W[5] = _mm512_srli_epi64(w[1], 8);
W[6] = _mm512_or_si512(_mm512_slli_epi64(w[1], 24), _mm512_srli_epi64(w[0], 40));
W[7] = _mm512_srli_epi64(w[0], 8);
W[8] = _mm512_or_si512(_mm512_slli_epi64(w[0], 24), _mm512_set1_epi64(0x00800000));
}

Q8T static inline QCPU_AIF void kh16_store(uint32_t *m, const fe8 &x0, __mmask8 p0, const fe8 &x1, __mmask8 p1) {
__m512i w[4], W0[9], W1[9];
kh16_canon(w, x0); kh16_words8(W0, w, p0);
kh16_canon(w, x1); kh16_words8(W1, w, p1);
for (int i = 0; i < 9; i++)
_mm512_store_si512((void *)(m + 16 * i), _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi64_epi32(W0[i])), _mm512_cvtepi64_epi32(W1[i]), 1));
}
#include "cpu_inverse_peel_final.h"


Q8T static void ec8_final_cf_kh(const fe8 *X, const fe8 *Y, fe8 *D, fe8 *PRE, int G, const fe &mx, const fe &my, uint32_t *m16) {
#if QSB_CPU_INV_PEEL
ec8_final_cf_kh_peeled(X, Y, D, PRE, G, mx, my, m16);
#else
fe8 MX, MY; fe8_bcast(MX, mx); fe8_bcast(MY, my);
const int NC = QSB_CPU_NCH;
fe8 run[4]; for (int c = 0; c < NC; c++) fe8_set1(run[c]);
for (int g = 0; g < G; g += NC)
for (int c = 0; c < NC; c++) {
const int h = g + c; fe8_sub_lz(D[h], MX, X[h]);
#if QSB_CPU_WPRE
{ fe8 t; fe8_sub_lz(t, MY, Y[h]);
if (QSB_CPU_JL_INV_FIRST && g == 0) fe8_cp(PRE[h], t);
else fe8_mul_lz(PRE[h], run[c], t); }
#else
fe8_cp(PRE[h], run[c]);
#endif
if (QSB_CPU_JL_INV_FIRST && g == 0) fe8_cp(run[c], D[h]);
else fe8_mul_lz(run[c], run[c], D[h]);
}
QCPU_INVC(run);
#if QSB_CPU_ILP2 & 1

for (int g = G - NC; g >= 0; g -= NC) {
for (int c = NC - 1; c >= 1; c -= 2) {
const int hA = g + c, hB = hA - 1;
fe8 tA, tB, lamA, lamB, x3A, x3B, y3A, y3B;
fe8_mul_lz(lamA, run[c], PRE[hA]); fe8_mul_lz(lamB, run[c - 1], PRE[hB]);
if (!QSB_CPU_JL_INV_LAST || g > 0) {
fe8_mul_lz(run[c], run[c], D[hA]); fe8_mul_lz(run[c - 1], run[c - 1], D[hB]);
}
fe8_sqr_sub2(x3A, lamA, X[hA], MX); fe8_sqr_sub2(x3B, lamB, X[hB], MX);
fe8_sub_lz(tA, X[hA], x3A); fe8_sub_lz(tB, X[hB], x3B);
fe8_mul_sub(y3A, lamA, tA, Y[hA]); fe8_mul_sub(y3B, lamB, tB, Y[hB]);
kh16_store(m16 + (size_t)hA * 144, X[hA], fe8_parity(Y[hA]), x3A, fe8_parity(y3A));
kh16_store(m16 + (size_t)hB * 144, X[hB], fe8_parity(Y[hB]), x3B, fe8_parity(y3B));
}
}
#else
for (int g = G - NC; g >= 0; g -= NC) {
for (int c = NC - 1; c >= 0; c--) {
const int h = g + c;
fe8 t, lam, x3, y3;
#if QSB_CPU_WPRE
fe8_mul_lz(lam, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);
#else
fe8 dinv;
fe8_mul_lz(dinv, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);
fe8_sub_lz(t, MY, Y[h]); fe8_mul_lz(lam, t, dinv);
#endif
fe8_sqr_sub2(x3, lam, X[h], MX);
fe8_sub_lz(t, X[h], x3); fe8_mul_sub(y3, lam, t, Y[h]);
kh16_store(m16 + (size_t)h * 144, X[h], fe8_parity(Y[h]), x3, fe8_parity(y3));
}
}
#endif
#endif
}
#endif


template <class RowFn>
Q8TX static void ec8_first(fe8 *X, fe8 *Y, int G, RowFn rowfn, RowFn nxt, const pt **rpn, __mmask8 *ngn) {
alignas(64) const pt *rows[8];
fe8 Z; for (int i = 0; i < 5; i++) Z.l[i] = _mm512_setzero_si512();
const int PF = QSB_CPU_PFD1;
for (int h = 0; h < PF && h < G; h++) { alignas(64) const pt *pr[8]; rowfn(h, pr); for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
for (int h = 0; h < G; h++) {
if (PF > 0 && h + PF < G) { alignas(64) const pt *pr[8]; rowfn(h + PF, pr); for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
{ const pt **pn = rpn + (size_t)h * 8; ngn[h] = nxt(h, pn); if (QSB_CPU_F1N) for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pn[j], _MM_HINT_T0); }
const __mmask8 ng = rowfn(h, rows);
fe8 ty; pt8_load(X[h], ty, rows);
if (ng) { fe8 n; fe8_sub(n, Z, ty); for (int i = 0; i < 5; i++) Y[h].l[i] = _mm512_mask_blend_epi64(ng, ty.l[i], n.l[i]); }
else fe8_cp(Y[h], ty);
}
}


Q8T static void recode16(const Geo &g, const uint32_t *zb, uint32_t *ds, uint8_t *bad, int B) {
#if !QSB_CPU_RECODE_NG
const __m512i vidx = _mm512_set_epi32(120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0);
#endif
const __m512i Z = _mm512_setzero_si512(), SB = _mm512_set1_epi32((int)0x80000000u), ONE = _mm512_set1_epi32(1);
const int nw = g.nw, ns = g.sgn ? nw - 1 : 0;
__m128i shr[NWMAX], shl[NWMAX]; __m512i msk[NWMAX], half[NWMAX], full[NWMAX]; int wq[NWMAX]; bool two[NWMAX];
for (int i = 0; i < nw; i++) {
const int s = g.off[i] & 31, w = g.wid[i];
wq[i] = g.off[i] >> 5; two[i] = s + w > 32;
shr[i] = _mm_cvtsi32_si128(s); shl[i] = _mm_cvtsi32_si128(32 - s);
msk[i] = _mm512_set1_epi32((int)((1u << w) - 1)); half[i] = _mm512_set1_epi32((int)(1u << (w - 1))); full[i] = _mm512_set1_epi32((int)(1u << w));
}
for (int k0 = 0; k0 < B; k0 += 16) {
__m512i zw[9];
#if QSB_CPU_RECODE_NG
{

const __m512i I00 = _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0);
const __m512i I01 = _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 23, 22, 21, 20, 7, 6, 5, 4);
const __m512i I10 = _mm512_set_epi32(29, 28, 13, 12, 25, 24, 9, 8, 21, 20, 5, 4, 17, 16, 1, 0);
const __m512i I11 = _mm512_set_epi32(31, 30, 15, 14, 27, 26, 11, 10, 23, 22, 7, 6, 19, 18, 3, 2);
const __m512i I20 = _mm512_set_epi32(30, 22, 14, 6, 28, 20, 12, 4, 26, 18, 10, 2, 24, 16, 8, 0);
const __m512i I21 = _mm512_set_epi32(31, 23, 15, 7, 29, 21, 13, 5, 27, 19, 11, 3, 25, 17, 9, 1);
const uint32_t *zk = zb + (size_t)k0 * 8;
__m512i r[8], s[8];
for (int m = 0; m < 8; m++) r[m] = _mm512_loadu_si512((const void *)(zk + 16 * m));
for (int v = 0; v < 4; v++) { s[v] = _mm512_permutex2var_epi32(r[v], I00, r[v | 4]); s[v | 4] = _mm512_permutex2var_epi32(r[v], I01, r[v | 4]); }
for (int v : {0, 1, 4, 5}) { r[v] = _mm512_permutex2var_epi32(s[v], I10, s[v | 2]); r[v | 2] = _mm512_permutex2var_epi32(s[v], I11, s[v | 2]); }
for (int v : {0, 2, 4, 6}) { s[v] = _mm512_permutex2var_epi32(r[v], I20, r[v | 1]); s[v | 1] = _mm512_permutex2var_epi32(r[v], I21, r[v | 1]); }
for (int j = 0; j < 8; j++) zw[j] = s[7 - j];
}
#else
for (int j = 0; j < 8; j++) zw[j] = _mm512_i32gather_epi32(vidx, (const void *)(zb + (size_t)k0 * 8 + (7 - j)), 4);
#endif
zw[8] = Z;
__m512i cy = Z; __mmask16 zero = 0;
for (int i = 0; i < nw; i++) {
__m512i v = _mm512_srl_epi32(zw[wq[i]], shr[i]);
if (two[i]) v = _mm512_or_si512(v, _mm512_sll_epi32(zw[wq[i] + 1], shl[i]));
v = _mm512_add_epi32(_mm512_and_si512(v, msk[i]), cy);
if (i < ns) {
const __mmask16 m = _mm512_cmpgt_epu32_mask(v, half[i]);
v = _mm512_mask_sub_epi32(v, m, full[i], v);
cy = _mm512_maskz_mov_epi32(m, ONE);
zero |= _mm512_testn_epi32_mask(v, v);
v = _mm512_mask_or_epi32(v, m, v, SB);
} else zero |= _mm512_testn_epi32_mask(v, v);
_mm512_storeu_si512((void *)(ds + (size_t)i * B + k0), v);
}
_mm_storeu_si128((__m128i *)(bad + k0), _mm512_cvtepi32_epi8(_mm512_maskz_mov_epi32(zero, ONE)));
}
}



static const uint32_t VB8 = 1024;
struct Build8 { fe8 *X = nullptr, *Y = nullptr, *D = nullptr, *P = nullptr, *TX = nullptr, *TY = nullptr; const pt **rp = nullptr; __mmask8 *ng = nullptr; };
static void build8_free(Build8 &b) {
free(b.X); free(b.Y); free(b.D); free(b.P); free(b.TX); free(b.TY); free((void *)b.rp); free(b.ng);
b = Build8();
}
static bool build8_alloc(Build8 &b) {
const int G = VB8 / 8; void *q[8] = {nullptr};
const size_t sz[8] = {sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G,
sizeof(const pt *) * VB8, (size_t)G};
for (int i = 0; i < 8; i++) if (posix_memalign(&q[i], 64, sz[i])) { for (int j = 0; j < i; j++) free(q[j]); return false; }
b.X = (fe8 *)q[0]; b.Y = (fe8 *)q[1]; b.D = (fe8 *)q[2]; b.P = (fe8 *)q[3]; b.TX = (fe8 *)q[4]; b.TY = (fe8 *)q[5];
b.rp = (const pt **)q[6]; b.ng = (__mmask8 *)q[7];
memset(b.ng, 0, (size_t)G);
return true;
}
struct RowNone { __mmask8 operator()(int, const pt **) const { return 0; } };
Q8TX static bool build8_block(pt *dst, const pt *src, const pt &Q, Build8 &b) {
for (uint32_t k = 0; k < VB8; k++) if (fe_eq(src[k].x, Q.x)) return false;
const int G = VB8 / 8;
for (uint32_t k = 0; k < VB8; k++) b.rp[k] = &Q;
for (int h = 0; h < G; h++) { const pt *rows[8]; for (int j = 0; j < 8; j++) rows[j] = &src[8 * h + j]; pt8_load(b.X[h], b.Y[h], rows); }
ec8_window<RowNone>(b.X, b.Y, b.D, b.P, b.TX, b.TY, G, b.rp, b.ng, nullptr, nullptr, nullptr);
for (int h = 0; h < G; h++) {
fe xs[8], ys[8]; fe8_store_canon(xs, b.X[h]); fe8_store_canon(ys, b.Y[h]);
for (int j = 0; j < 8; j++) { dst[8 * h + j].x = xs[j]; dst[8 * h + j].y = ys[j]; }
}
return true;
}
#endif
#if !QCPU_VEC
struct Build8 {};
#endif

#if QCPU_SHANI


#define QSHA __attribute__((target("sha,sse4.1,ssse3,avx")))
alignas(16) static const uint32_t qsha_k[64] = {
0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

QSHA static void qsha_x4(uint32_t (*st)[8], const uint8_t *const *blk) {
const __m128i BSWAP = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
__m128i S0[4], S1[4], I0[4], I1[4], M[4][4];
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) {
__m128i t = _mm_loadu_si128((const __m128i *)&st[l][0]);
__m128i u = _mm_loadu_si128((const __m128i *)&st[l][4]);
t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
S0[l] = _mm_alignr_epi8(t, u, 8);
S1[l] = _mm_blend_epi16(u, t, 0xF0);
I0[l] = S0[l]; I1[l] = S1[l];
#pragma GCC unroll 4
for (int j = 0; j < 4; j++) M[l][j] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(blk[l] + 16 * j)), BSWAP);
}
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) {
if (r >= 4) {
__m128i t = _mm_sha256msg1_epu32(M[l][r & 3], M[l][(r + 1) & 3]);
t = _mm_add_epi32(t, _mm_alignr_epi8(M[l][(r + 3) & 3], M[l][(r + 2) & 3], 4));
M[l][r & 3] = _mm_sha256msg2_epu32(t, M[l][(r + 3) & 3]);
}
__m128i m = _mm_add_epi32(M[l][r & 3], K);
S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], m);
m = _mm_shuffle_epi32(m, 0x0E);
S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], m);
}
}
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) {
__m128i a = _mm_add_epi32(S0[l], I0[l]), b = _mm_add_epi32(S1[l], I1[l]);
__m128i t = _mm_shuffle_epi32(a, 0x1B);
b = _mm_shuffle_epi32(b, 0xB1);
_mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0));
_mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));
}
}
static const uint32_t qsha_iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
static bool qsha_supported() {
unsigned a, b, cc, d;
if (!__get_cpuid_count(7, 0, &a, &b, &cc, &d)) return false;
__builtin_cpu_init();
return ((b >> 29) & 1) && __builtin_cpu_supports("avx");
}





QSHA static void qsha_x4p(uint32_t (*st)[8], const uint32_t *const *const *rows, int nblk, uint32_t *out = nullptr, int ostride = 8,
const uint32_t *const *in = nullptr
#if QCPU_PFQ
, QPfRing *pq = nullptr
#endif
) {
__m128i S0[4], S1[4];
if (!out) out = &st[0][0];
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) {
const uint32_t *s = in ? in[l] : st[l];
__m128i t = _mm_loadu_si128((const __m128i *)&s[0]);
__m128i u = _mm_loadu_si128((const __m128i *)&s[4]);
t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
S0[l] = _mm_alignr_epi8(t, u, 8); S1[l] = _mm_blend_epi16(u, t, 0xF0);
}
for (int b = 0; b < nblk; b++) {
#if QCPU_PFQ
if (pq) qpf_drain(*pq, 2);
#endif
const uint32_t *const w[4] = {rows[0][b], rows[1][b], rows[2][b], rows[3][b]};
__m128i I0[4], I1[4];
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) { I0[l] = S0[l]; I1[l] = S1[l]; }
if (QSB_CPU_X4PS && w[0] == w[1] && w[0] == w[2] && w[0] == w[3]) {

const uint32_t *ws = w[0];
__asm__("" : "+r"(ws));
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
const __m128i m0 = _mm_loadl_epi64((const __m128i *)(ws + 4 * r));
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], m0);
const __m128i m1 = _mm_loadl_epi64((const __m128i *)(ws + 4 * r + 2));
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], m1);
}
} else {
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) {
S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], _mm_loadl_epi64((const __m128i *)(w[l] + 4 * r)));
S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], _mm_loadl_epi64((const __m128i *)(w[l] + 4 * r + 2)));
}
}
}
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) { S0[l] = _mm_add_epi32(S0[l], I0[l]); S1[l] = _mm_add_epi32(S1[l], I1[l]); }
}
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) {
__m128i t = _mm_shuffle_epi32(S0[l], 0x1B), b = _mm_shuffle_epi32(S1[l], 0xB1);
_mm_storeu_si128((__m128i *)(out + l * ostride), _mm_blend_epi16(t, b, 0xF0));
_mm_storeu_si128((__m128i *)(out + l * ostride + 4), _mm_alignr_epi8(b, t, 8));
}
}
#if QSB_CPU_SHA4













template <bool SHC>
QSHA static inline void qsha_rounds4m(__m128i *mr, __m128i *out0, __m128i *out1) {
const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);
const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);
__m128i a0 = IV0, a1 = IV0, a2 = IV0, a3 = IV0, b0 = IV1, b1 = IV1, b2 = IV1, b3 = IV1;
__m128i p0 = IV0, p1 = IV0, p2 = IV0, p3 = IV0;
__asm__("" : "+r"(mr));
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
#define QSHA4_LANE(L, A, B, P) do { \
__m128i *const M = mr + 4 * (L); __m128i w; \
if (r >= 4) { \
__m128i t = (SHC && r == 6) ? _mm_load_si128(M + 2) \
: _mm_sha256msg1_epu32(_mm_load_si128(M + (r & 3)), _mm_load_si128(M + ((r + 1) & 3))); \
if (!(SHC && r == 4)) t = _mm_add_epi32(t, _mm_alignr_epi8(P, _mm_load_si128(M + ((r + 2) & 3)), 4)); \
w = _mm_sha256msg2_epu32(t, P); \
if (r < 14) _mm_store_si128(M + (r & 3), w); \
} else w = _mm_load_si128(M + (r & 3)); \
P = w; \
__m128i m = _mm_add_epi32(w, K); \
B = _mm_sha256rnds2_epu32(B, A, m); m = _mm_shuffle_epi32(m, 0x0E); A = _mm_sha256rnds2_epu32(A, B, m); } while (0)
QSHA4_LANE(0, a0, b0, p0); QSHA4_LANE(1, a1, b1, p1); QSHA4_LANE(2, a2, b2, p2); QSHA4_LANE(3, a3, b3, p3);
#undef QSHA4_LANE
__asm__ volatile("" ::: "memory");
}
out0[0] = _mm_add_epi32(a0, IV0); out0[1] = _mm_add_epi32(a1, IV0); out0[2] = _mm_add_epi32(a2, IV0); out0[3] = _mm_add_epi32(a3, IV0);
out1[0] = _mm_add_epi32(b0, IV1); out1[1] = _mm_add_epi32(b1, IV1); out1[2] = _mm_add_epi32(b2, IV1); out1[3] = _mm_add_epi32(b3, IV1);
}

QSHA static inline void qsha_st8(uint32_t *st, __m128i a, __m128i b) {
const __m128i t = _mm_shuffle_epi32(a, 0x1B);
b = _mm_shuffle_epi32(b, 0xB1);
_mm_storeu_si128((__m128i *)&st[0], _mm_blend_epi16(t, b, 0xF0));
_mm_storeu_si128((__m128i *)&st[4], _mm_alignr_epi8(b, t, 8));
}
#endif





template <int L>
QSHA static inline void qsha_xw(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
__m128i S0[L], S1[L], I0[L], I1[L], M[L][4];
#pragma GCC unroll 4
for (int l = 0; l < L; l++) {
__m128i t = _mm_loadu_si128((const __m128i *)&st[l][0]);
__m128i u = _mm_loadu_si128((const __m128i *)&st[l][4]);
t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
S0[l] = _mm_alignr_epi8(t, u, 8); S1[l] = _mm_blend_epi16(u, t, 0xF0);
I0[l] = S0[l]; I1[l] = S1[l];
#pragma GCC unroll 4
for (int j = 0; j < 4; j++) M[l][j] = _mm_loadu_si128((const __m128i *)(wd[l] + 4 * j));
}
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
#pragma GCC unroll 4
for (int l = 0; l < L; l++) {
if (r >= 4) {
__m128i t = _mm_sha256msg1_epu32(M[l][r & 3], M[l][(r + 1) & 3]);
t = _mm_add_epi32(t, _mm_alignr_epi8(M[l][(r + 3) & 3], M[l][(r + 2) & 3], 4));
M[l][r & 3] = _mm_sha256msg2_epu32(t, M[l][(r + 3) & 3]);
}
__m128i m = _mm_add_epi32(M[l][r & 3], K);
S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], m);
m = _mm_shuffle_epi32(m, 0x0E);
S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], m);
}
}
#pragma GCC unroll 4
for (int l = 0; l < L; l++) {
__m128i a = _mm_add_epi32(S0[l], I0[l]), b = _mm_add_epi32(S1[l], I1[l]);
__m128i t = _mm_shuffle_epi32(a, 0x1B);
b = _mm_shuffle_epi32(b, 0xB1);
_mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0));
_mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));
}
}
QSHA static void qsha_x4w(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
qsha_xw<2>(st, wd); qsha_xw<2>(st + 2, wd + 2);
}


template <int L>
QSHA static inline void qsha_xw_iv(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);
const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);
__m128i S0[L], S1[L], M[L][4];
#pragma GCC unroll 4
for (int l = 0; l < L; l++) {
S0[l] = IV0; S1[l] = IV1;
#pragma GCC unroll 4
for (int j = 0; j < 4; j++) M[l][j] = _mm_loadu_si128((const __m128i *)(wd[l] + 4 * j));
}
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
#pragma GCC unroll 4
for (int l = 0; l < L; l++) {
if (r >= 4) {
__m128i t = _mm_sha256msg1_epu32(M[l][r & 3], M[l][(r + 1) & 3]);
t = _mm_add_epi32(t, _mm_alignr_epi8(M[l][(r + 3) & 3], M[l][(r + 2) & 3], 4));
M[l][r & 3] = _mm_sha256msg2_epu32(t, M[l][(r + 3) & 3]);
}
__m128i m = _mm_add_epi32(M[l][r & 3], K);
S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], m);
m = _mm_shuffle_epi32(m, 0x0E);
S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], m);
}
}
#pragma GCC unroll 4
for (int l = 0; l < L; l++) {
__m128i a = _mm_add_epi32(S0[l], IV0), b = _mm_add_epi32(S1[l], IV1);
__m128i t = _mm_shuffle_epi32(a, 0x1B);
b = _mm_shuffle_epi32(b, 0xB1);
_mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0));
_mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));
}
}
QSHA static void qsha_x4w_iv(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
#if QSB_CPU_SHA4
__m128i mr[16], a[4], b[4];
#pragma GCC unroll 4
for (int l = 0; l < 4; l++)
#pragma GCC unroll 4
for (int j = 0; j < 4; j++) _mm_store_si128(mr + 4 * l + j, _mm_loadu_si128((const __m128i *)(wd[l] + 4 * j)));
qsha_rounds4m<false>(mr, a, b);
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) qsha_st8(st[l], a[l], b[l]);
#else
qsha_xw_iv<2>(st, wd); qsha_xw_iv<2>(st + 2, wd + 2);
#endif
}
#if QSB_CPU_SHC



template <int L>
QSHA static inline void qsha_xw_iv32(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);
const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);
const __m128i P2 = _mm_set_epi32(0, 0, 0, (int)0x80000000u), P3 = _mm_set_epi32(256, 0, 0, 0);
__m128i S0[L], S1[L], M[L][4];
#pragma GCC unroll 4
for (int l = 0; l < L; l++) {
S0[l] = IV0; S1[l] = IV1;
M[l][0] = _mm_loadu_si128((const __m128i *)(wd[l] + 0)); M[l][1] = _mm_loadu_si128((const __m128i *)(wd[l] + 4));
M[l][2] = P2; M[l][3] = P3;
}
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
#pragma GCC unroll 4
for (int l = 0; l < L; l++) {
if (r >= 4) {
__m128i t = r == 6 ? M[l][2] : _mm_sha256msg1_epu32(M[l][r & 3], M[l][(r + 1) & 3]);
if (r != 4) t = _mm_add_epi32(t, _mm_alignr_epi8(M[l][(r + 3) & 3], M[l][(r + 2) & 3], 4));
M[l][r & 3] = _mm_sha256msg2_epu32(t, M[l][(r + 3) & 3]);
}
__m128i m = _mm_add_epi32(M[l][r & 3], K);
S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], m);
m = _mm_shuffle_epi32(m, 0x0E);
S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], m);
}
}
#pragma GCC unroll 4
for (int l = 0; l < L; l++) {
__m128i a = _mm_add_epi32(S0[l], IV0), b = _mm_add_epi32(S1[l], IV1);
__m128i t = _mm_shuffle_epi32(a, 0x1B);
b = _mm_shuffle_epi32(b, 0xB1);
_mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0));
_mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));
}
}
QSHA static void qsha_x4w_iv32(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
#if QSB_CPU_SHA4
const __m128i P2 = _mm_set_epi32(0, 0, 0, (int)0x80000000u), P3 = _mm_set_epi32(256, 0, 0, 0);
__m128i mr[16], a[4], b[4];
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) {
_mm_store_si128(mr + 4 * l + 0, _mm_loadu_si128((const __m128i *)(wd[l] + 0)));
_mm_store_si128(mr + 4 * l + 1, _mm_loadu_si128((const __m128i *)(wd[l] + 4)));
_mm_store_si128(mr + 4 * l + 2, P2); _mm_store_si128(mr + 4 * l + 3, P3);
}
qsha_rounds4m<true>(mr, a, b);
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) qsha_st8(st[l], a[l], b[l]);
#else
qsha_xw_iv32<2>(st, wd); qsha_xw_iv32<2>(st + 2, wd + 2);
#endif
}
#endif




QSHA static __m128i qsha_keyhash4_h0(const fe *qx, const uint8_t *qp) {
const __m128i I03 = _mm_set_epi8(4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, -128, 15, 14, 13);
const __m128i I47 = _mm_set_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
__m128i a[4];
#if QSB_CPU_SHA4
__m128i mr[16], b[4];
#pragma GCC unroll 4
for (int l = 0; l < 4; l++) {
const fe &x = qx[l];
const __m128i lo = _mm_loadu_si128((const __m128i *)&x.v[0]), hi = _mm_loadu_si128((const __m128i *)&x.v[2]);
_mm_store_si128(mr + 4 * l + 0, _mm_or_si128(_mm_shuffle_epi8(hi, I03), _mm_cvtsi32_si128((int)((0x02u | (qp[l] & 1)) << 24))));
_mm_store_si128(mr + 4 * l + 1, _mm_shuffle_epi8(_mm_alignr_epi8(hi, lo, 1), I47));
_mm_store_si128(mr + 4 * l + 2, _mm_cvtsi32_si128((int)(((uint32_t)(x.v[0] & 0xff) << 24) | 0x00800000u)));
_mm_store_si128(mr + 4 * l + 3, _mm_set_epi32(264, 0, 0, 0));
}
qsha_rounds4m<(QSB_CPU_SHC != 0)>(mr, a, b);
#else
const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);
const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);
#pragma GCC unroll 2
for (int h = 0; h < 2; h++) {
__m128i S0[2], S1[2], M[2][4];
#pragma GCC unroll 2
for (int l = 0; l < 2; l++) {
const fe &x = qx[2 * h + l];
const __m128i lo = _mm_loadu_si128((const __m128i *)&x.v[0]), hi = _mm_loadu_si128((const __m128i *)&x.v[2]);
M[l][0] = _mm_or_si128(_mm_shuffle_epi8(hi, I03), _mm_cvtsi32_si128((int)((0x02u | (qp[2 * h + l] & 1)) << 24)));
M[l][1] = _mm_shuffle_epi8(_mm_alignr_epi8(hi, lo, 1), I47);
M[l][2] = _mm_cvtsi32_si128((int)(((uint32_t)(x.v[0] & 0xff) << 24) | 0x00800000u));
M[l][3] = _mm_set_epi32(264, 0, 0, 0);
S0[l] = IV0; S1[l] = IV1;
}
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
#pragma GCC unroll 2
for (int l = 0; l < 2; l++) {
if (r >= 4) {

__m128i t = (QSB_CPU_SHC && r == 6) ? M[l][2] : _mm_sha256msg1_epu32(M[l][r & 3], M[l][(r + 1) & 3]);
if (!(QSB_CPU_SHC && r == 4)) t = _mm_add_epi32(t, _mm_alignr_epi8(M[l][(r + 3) & 3], M[l][(r + 2) & 3], 4));
M[l][r & 3] = _mm_sha256msg2_epu32(t, M[l][(r + 3) & 3]);
}
__m128i m = _mm_add_epi32(M[l][r & 3], K);
S1[l] = _mm_sha256rnds2_epu32(S1[l], S0[l], m);
m = _mm_shuffle_epi32(m, 0x0E);
S0[l] = _mm_sha256rnds2_epu32(S0[l], S1[l], m);
}
}
a[2 * h] = _mm_add_epi32(S0[0], IV0); a[2 * h + 1] = _mm_add_epi32(S0[1], IV0);
}
#endif
return _mm_unpackhi_epi64(_mm_unpackhi_epi32(a[0], a[1]), _mm_unpackhi_epi32(a[2], a[3]));
}
#if QSB_CPU_KH16 && QCPU_VEC
#define QSHA16 __attribute__((target("sha,sse4.1,ssse3,avx,avx2,avx512f,avx512vl")))



QSHA16 static unsigned kh16_pass(const uint32_t *m, uint32_t *wk
#ifdef QSB_CPU_DEVBENCH
, uint32_t *h0_out = nullptr
#endif
) {
#define R16(x, n) _mm512_ror_epi32((x), (n))
#define S0_16(x) _mm512_ternarylogic_epi32(R16(x, 7), R16(x, 18), _mm512_srli_epi32(x, 3), 0x96)
#define S1_16(x) _mm512_ternarylogic_epi32(R16(x, 17), R16(x, 19), _mm512_srli_epi32(x, 10), 0x96)
__m512i W[16];
#pragma GCC unroll 9
for (int i = 0; i < 9; i++) W[i] = _mm512_load_si512((const void *)(m + 16 * i));
for (int i = 9; i < 15; i++) W[i] = _mm512_setzero_si512();
W[15] = _mm512_set1_epi32(264);
__m512i prev = _mm512_setzero_si512();
#pragma GCC unroll 64
for (int t = 0; t < 64; t++) {
__m512i wt;
if (t < 16) wt = W[t];
else {

const int a2 = (t - 2) & 15, a7 = (t - 7) & 15, a15 = (t - 15) & 15, a16 = t & 15;
const bool z2 = (t - 2) >= 9 && (t - 2) <= 14, z7 = (t - 7) >= 9 && (t - 7) <= 14, z15 = (t - 15) >= 9 && (t - 15) <= 14,
z16 = (t - 16) >= 9 && (t - 16) <= 14;
wt = z16 ? _mm512_setzero_si512() : W[a16];
if (!z15) wt = _mm512_add_epi32(wt, S0_16(W[a15]));
if (!z7) wt = _mm512_add_epi32(wt, W[a7]);
if (!z2) wt = _mm512_add_epi32(wt, S1_16(W[a2]));
W[a16] = wt;
}
const __m512i wkt = _mm512_add_epi32(wt, _mm512_set1_epi32((int)qsha_k[t]));
if (t & 1) {
_mm512_store_si512((void *)(wk + 32 * (t >> 1)), _mm512_unpacklo_epi32(prev, wkt));
_mm512_store_si512((void *)(wk + 32 * (t >> 1) + 16), _mm512_unpackhi_epi32(prev, wkt));
} else prev = wkt;
}
#undef S0_16
#undef S1_16
#undef R16

const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);
const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);
alignas(64) uint32_t h0[16];
#pragma GCC unroll 1
for (int L = 0; L < 4; L++) {
__m128i S0[4], S1[4];
const uint32_t *base[4] = {wk + 4 * L, wk + 4 * L + 2, wk + 16 + 4 * L, wk + 16 + 4 * L + 2};
#pragma GCC unroll 4
for (int e = 0; e < 4; e++) { S0[e] = IV0; S1[e] = IV1; }
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
#pragma GCC unroll 4
for (int e = 0; e < 4; e++) S1[e] = _mm_sha256rnds2_epu32(S1[e], S0[e], _mm_loadl_epi64((const __m128i *)(base[e] + 64 * r)));
#pragma GCC unroll 4
for (int e = 0; e < 4; e++) S0[e] = _mm_sha256rnds2_epu32(S0[e], S1[e], _mm_loadl_epi64((const __m128i *)(base[e] + 64 * r + 32)));
}
#pragma GCC unroll 4
for (int e = 0; e < 4; e++) h0[4 * L + e] = (uint32_t)_mm_extract_epi32(_mm_add_epi32(S0[e], IV0), 3);
}
#ifdef QSB_CPU_DEVBENCH
if (h0_out) memcpy(h0_out, h0, sizeof h0);
#endif
const __m512i hv = _mm512_load_si512((const void *)h0);
return (unsigned)_mm512_cmpeq_epi32_mask(_mm512_srli_epi32(hv, 32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32)), _mm512_setzero_si512());
}
#endif

QSHA static void qsha_schedule(uint32_t wk[64], const uint8_t *blk) {
const __m128i BSWAP = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
__m128i M[4];
for (int j = 0; j < 4; j++) M[j] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(blk + 16 * j)), BSWAP);
#pragma GCC unroll 16
for (int r = 0; r < 16; r++) {
if (r >= 4) {
__m128i t = _mm_sha256msg1_epu32(M[r & 3], M[(r + 1) & 3]);
t = _mm_add_epi32(t, _mm_alignr_epi8(M[(r + 3) & 3], M[(r + 2) & 3], 4));
M[r & 3] = _mm_sha256msg2_epu32(t, M[(r + 3) & 3]);
}
_mm_storeu_si128((__m128i *)(wk + 4 * r), _mm_add_epi32(M[r & 3], _mm_load_si128((const __m128i *)&qsha_k[4 * r])));
}
}
#endif
struct Ctx {
const digest_params_t *dp;
Geo g;
pt *table = nullptr;
void *table_map = nullptr; size_t table_map_bytes = 0;
const pt *tw[NWMAX] = {nullptr};
fe cx, cy;
bool cfold = false;
fe mx, my;
uint8_t cwin[286][3];
int ncwin = 0;
int cut = 137, early = 6;
uint64_t mid_bytes = 0;
uint64_t n_epochs = 0;
uint64_t epoch_base = 0;
std::atomic<uint64_t> cand{0};
std::atomic<uint32_t> hits{0};
std::mutex io;
qsb_hv_t hv;
FILE *out = nullptr;
int nthreads = 0;
int batch = QSB_CPU_BATCH;
char batch_why[48] = "default";
#ifdef QSB_CPU_DEVBENCH
uint64_t dev_limit = 0;
std::atomic<int> dev_live{0};
#endif
bool vec = false;
bool shani = false;
bool kh16 = false;
std::atomic<int> live{0}, stop{0}, ready{0};
#if QCPU_SHANI



bool hplan = false;
int h_nb = 0, h_ng = 0;
uint8_t h_g0[286];
const uint32_t *h_wkp[286][16];
uint64_t h_binom[256][8];
std::vector<uint8_t> h_gblk;
std::vector<uint32_t, qalloc64<uint32_t> > h_wk;
#endif
};



static bool table_alloc(Ctx &c) {
const size_t H = (size_t)2 << 20, bytes = (c.g.total * sizeof(pt) + H - 1) / H * H;
void *m = mmap(nullptr, bytes + H, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
if (m == MAP_FAILED) return false;
const uintptr_t a = ((uintptr_t)m + H - 1) & ~(uintptr_t)(H - 1);
#ifdef MADV_HUGEPAGE
if (!getenv("QSB_CPU_NOTHP")) madvise((void *)a, bytes, MADV_HUGEPAGE);
#endif
c.table_map = m; c.table_map_bytes = bytes + H; c.table = (pt *)a;
for (int i = 0; i < c.g.nw; i++) c.tw[i] = c.table + c.g.base[i];
return true;
}
static void table_free(Ctx &c) {
if (c.table_map) munmap(c.table_map, c.table_map_bytes);
c.table_map = nullptr; c.table = nullptr;
}


static double thp_bytes(const void *p, size_t n) {
FILE *f = fopen("/proc/self/smaps", "r"); if (!f) return -1;
const unsigned long a = (unsigned long)p, e = a + n;
double huge = 0, ov = 0; bool seen = false; char line[512];
while (fgets(line, sizeof line, f)) {
unsigned long s, t; double kb;
if (sscanf(line, "%lx-%lx ", &s, &t) == 2) {
if (s >= e) break;
ov = t > a ? (double)((t < e ? t : e) - (s > a ? s : a)) : 0; if (ov < 0) ov = 0;
if (ov > 0) seen = true;
} else if (ov > 0 && sscanf(line, "AnonHugePages: %lf kB", &kb) == 1) huge += kb * 1024.0 < ov ? kb * 1024.0 : ov;
}
fclose(f);
return seen ? huge : -1;
}


static double table_touch(Ctx &c, int nth, size_t step, double deadline = 0) {
const size_t H = (size_t)2 << 20, n = (c.g.total * sizeof(pt) + H - 1) / H, m = (n + step - 1) / step;
char *p = (char *)c.table;
std::atomic<size_t> next{0};
std::atomic<int> late{0};
auto work = [&]() {
for (size_t i; (i = next.fetch_add(1)) < m;) {
if (deadline > 0) {
struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
if (t.tv_sec + 1e-9 * t.tv_nsec > deadline) { late = 1; next = m; break; }
}
*(volatile char *)(p + i * step * H) = 0;
}
};
std::vector<std::thread> ts;
const int k = (size_t)nth < m ? nth : (int)m;
for (int t = 1; t < k; t++) ts.emplace_back(work);
work();
for (auto &t : ts) t.join();
if (late) return -3;
const double h = thp_bytes(p, n * H);
return h < 0 ? -1 : h / (double)(m * H);
}
static pt pt_add_aff(const pt &a, const pt &b) {
fe d, inv, t, lam, x3, y3;
fe_sub(d, b.x, a.x); fe_inv(inv, d); fe_sub(t, b.y, a.y); fe_mul(lam, t, inv);
fe_sqr(x3, lam); fe_sub(x3, x3, a.x); fe_sub(x3, x3, b.x);
fe_sub(t, a.x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, a.y);
return {x3, y3};
}
static pt pt_mul_small(const pt &q, uint64_t k) {
pt r = q;
for (int b = 62 - __builtin_clzll(k | 1); b >= 0; b--) { r = pt_double(r); if ((k >> b) & 1) r = pt_add_aff(r, q); }
return r;
}


static void build_table(Ctx &c, const fe &ax, const fe &ay, int nth) {
const Geo &g = c.g;
const uint32_t S = 1u << 16;
std::vector<pt> base(g.nw);
{ pt q = {ax, ay}; int at = 0; for (int i = 0; i < g.nw; i++) { while (at < g.off[i]) { q = pt_double(q); at++; } base[i] = q; } }
auto chunk0 = [&](int i) {
pt *T = c.table + g.base[i];
const uint32_t n0 = g.ent[i] < S ? g.ent[i] : S;
T[0] = base[i];
if (n0 < 2) return;
T[1] = pt_double(base[i]);
uint32_t have = 2;
std::vector<fe> d(n0), pre(n0); std::vector<uint8_t> inf(n0), bad(n0); std::vector<const pt *> tp(n0);
while (have < n0) {
uint32_t n = have; if (have + n > n0) n = n0 - have;

for (uint32_t k = 0; k < n; k++) { T[have + k] = T[k]; tp[k] = &T[have - 1]; inf[k] = 0; bad[k] = 0; }
batch_add(&T[have], tp.data(), inf.data(), bad.data(), (int)n, d.data(), pre.data());

for (uint32_t k = 0; k < n; k++) if (bad[k]) T[have + k] = pt_double(T[have - 1]);
have += n;
}
};
auto chunks = [&](pt *T, uint32_t o, uint32_t k0, uint32_t n, const pt &Q) {
std::vector<fe> d(n), pre(n); std::vector<uint8_t> inf(n, 0), bad(n, 0); std::vector<const pt *> tp(n, &Q);
for (uint32_t k = 0; k < n; k++) T[o + k0 + k] = T[k0 + k];
batch_add(&T[o + k0], tp.data(), inf.data(), bad.data(), (int)n, d.data(), pre.data());
for (uint32_t k = 0; k < n; k++) if (bad[k]) T[o + k0 + k] = pt_double(Q);
};
auto chunkc = [&](int i, uint32_t cc, Build8 *b8) {
pt *T = c.table + g.base[i];
const uint32_t o = cc * S, n = g.ent[i] - o < S ? g.ent[i] - o : S;
const pt Q = pt_mul_small(T[S - 1], cc);
#if QCPU_VEC
if (b8) {
uint32_t k = 0;
for (; k + VB8 <= n; k += VB8) if (!build8_block(&T[o + k], &T[k], Q, *b8)) chunks(T, o, k, VB8, Q);
if (k < n) chunks(T, o, k, n - k, Q);
return;
}
#else
(void)b8;
#endif
chunks(T, o, 0, n, Q);
};
const bool vb = QSB_CPU_VBUILD && c.vec && !getenv("QSB_CPU_NOVBUILD");
std::vector<std::pair<int, uint32_t> > tasks;
for (int i = 0; i < g.nw; i++) for (uint32_t cc = 1; (uint64_t)cc * S < g.ent[i]; cc++) tasks.push_back({i, cc});
for (int phase = 0; phase < 2; phase++) {
std::atomic<size_t> next{0};
const size_t nt = phase ? tasks.size() : (size_t)g.nw;
auto work = [&]() {
Build8 b8; bool okb = false;
#if QCPU_VEC
okb = vb && phase && build8_alloc(b8);
#endif
for (size_t t; (t = next.fetch_add(1)) < nt;) { if (phase) chunkc(tasks[t].first, tasks[t].second, okb ? &b8 : nullptr); else chunk0((int)t); }
#if QCPU_VEC
if (okb) build8_free(b8);
#endif
};
std::vector<std::thread> ts;
const int m = (size_t)nth < nt ? nth : (int)nt;
for (int t = 0; t < m; t++) ts.emplace_back(work);
for (auto &t : ts) t.join();
}
c.cfold = false;
if (QSB_CPU_CFOLD && c.vec) {

const int top = g.nw - 1; pt *T = c.table + g.base[top]; const uint32_t n = g.ent[top], CH = 1u << 16;
const pt C = {c.cx, c.cy};
std::atomic<uint32_t> next{0}; std::atomic<int> anybad{0};
std::vector<pt> tmpv; pt *tmp = nullptr;
#if QSB_CPU_FOLD_PAR
const size_t tH = (size_t)2 << 20, tb = ((size_t)n * sizeof(pt) + tH - 1) / tH * tH;
void *tmap = mmap(nullptr, tb + tH, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
if (tmap == MAP_FAILED) tmap = nullptr;
else { tmp = (pt *)(((uintptr_t)tmap + tH - 1) & ~(uintptr_t)(tH - 1));
#ifdef MADV_HUGEPAGE
madvise((void *)tmp, tb, MADV_HUGEPAGE);
#endif
}
#endif
if (!tmp) { tmpv.resize(n); tmp = tmpv.data(); }
auto work = [&]() {
std::vector<fe> d(CH), pre(CH); std::vector<uint8_t> inf(CH), bad(CH); std::vector<const pt *> tp(CH, &C);
Build8 b8; bool okb = false;
#if QCPU_VEC
okb = vb && build8_alloc(b8);
#endif
for (uint32_t o; (o = next.fetch_add(CH)) < n;) {
uint32_t m = n - o < CH ? n - o : CH;
#if QCPU_VEC
if (okb) {
uint32_t k = 0;
for (; k + VB8 <= m; k += VB8) if (!build8_block(&tmp[o + k], &T[o + k], C, b8)) anybad = 1;
if (k == m) continue;
o += k; m -= k;
}
#endif
for (uint32_t k = 0; k < m; k++) { tmp[o + k] = T[o + k]; inf[k] = 0; bad[k] = 0; }
batch_add(&tmp[o], tp.data(), inf.data(), bad.data(), (int)m, d.data(), pre.data());
for (uint32_t k = 0; k < m; k++) if (bad[k] || inf[k]) anybad = 1;
}
#if QCPU_VEC
if (okb) build8_free(b8);
#endif
};
std::vector<std::thread> ts;
for (int t = 0; t < nth; t++) ts.emplace_back(work);
for (auto &t : ts) t.join();
if (!anybad) {
#if QSB_CPU_FOLD_PAR
{
std::atomic<uint32_t> nc{0};
auto cp = [&]() { for (uint32_t o; (o = nc.fetch_add(CH)) < n;) memcpy(T + o, tmp + o, (size_t)(n - o < CH ? n - o : CH) * sizeof(pt)); };
std::vector<std::thread> tc;
for (int t = 1; t < nth; t++) tc.emplace_back(cp);
cp();
for (auto &t : tc) t.join();
}
#else
memcpy(T, tmp, (size_t)n * sizeof(pt));
#endif
const pt C2 = pt_double(C);
c.mx = C2.x; fe z0 = {{0, 0, 0, 0}}; fe_sub(c.my, z0, C2.y);
c.cfold = true;
}
#if QSB_CPU_FOLD_PAR
if (tmap) munmap(tmap, tb + tH);
#endif
}
}


static bool table_check(const Ctx &c) {
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *bx = BN_CTX_new(); BIGNUM *nri = BN_new(), *k = BN_new(), *ord = BN_new(), *xx = BN_new(), *yy = BN_new();
EC_POINT *P = grp ? EC_POINT_new(grp) : nullptr, *Cp = grp ? EC_POINT_new(grp) : nullptr;
bool ok = grp && bx && nri && k && ord && xx && yy && P && Cp && EC_GROUP_get_order(grp, ord, bx) &&
BN_lebin2bn(c.dp->neg_r_inv, 32, nri);
if (ok && c.cfold) {
BIGNUM *cxb = BN_new(), *cyb = BN_new();
ok = cxb && cyb && BN_lebin2bn(c.dp->u2r_x, 32, cxb) && BN_lebin2bn(c.dp->u2r_y, 32, cyb) &&
EC_POINT_set_affine_coordinates_GFp(grp, Cp, cxb, cyb, bx);
BN_free(cxb); BN_free(cyb);
}
for (int i = 0; ok && i < c.g.nw; i++) {
const uint32_t e = c.g.ent[i], nc = (e + 65535u) >> 16, cm = nc / 2 << 16, cl = (nc - 1) << 16;
const uint32_t js[9] = {0, e / 2, e - 1, e > 65536 ? 65535u : 0u, e > 65536 ? 65536u : 0u,
cm ? cm - 1 : 0u, cm, cl ? cl - 1 : 0u, cl};
for (int q = 0; ok && q < 9; q++) {
const uint32_t j = js[q];
ok = BN_set_word(k, (BN_ULONG)j + 1) && BN_lshift(k, k, c.g.off[i]) && BN_mod_mul(k, k, nri, ord, bx) &&
EC_POINT_mul(grp, P, k, NULL, NULL, bx) && (!c.cfold || i != c.g.nw - 1 || EC_POINT_add(grp, P, P, Cp, bx)) &&
EC_POINT_get_affine_coordinates_GFp(grp, P, xx, yy, bx);
if (ok) { fe fx, fy; fe_from_bn(fx, xx); fe_from_bn(fy, yy); const pt &t = c.table[c.g.base[i] + j]; ok = fe_eq(fx, t.x) && fe_eq(fy, t.y); }
}
}
if (P) EC_POINT_free(P);
if (Cp) EC_POINT_free(Cp);
BN_free(nri); BN_free(k); BN_free(ord); BN_free(xx); BN_free(yy); if (bx) BN_CTX_free(bx); if (grp) EC_GROUP_free(grp);
return ok;
}

static double mem_avail() {
double avail = -1;
if (FILE *f = fopen("/proc/meminfo", "r")) {
char line[256]; long long kb;
while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) { avail = (double)kb * 1024.0; break; }
fclose(f);
}
auto rd = [](const char *path, double &v) -> bool {
FILE *f = fopen(path, "r"); if (!f) return false;
char b[64] = {0}; const bool ok = fscanf(f, "%63s", b) == 1; fclose(f);
if (!ok || !strcmp(b, "max")) return false;
v = atof(b); return v > 0 && v < 1e18;
};
auto cap = [&](const char *dir, bool v2) {
char p[640]; double lim, cur;
for (int k = 0; k < (v2 ? 2 : 1); k++) {
snprintf(p, sizeof p, "%s/%s", dir, v2 ? (k ? "memory.high" : "memory.max") : "memory.limit_in_bytes");
if (!rd(p, lim)) continue;
snprintf(p, sizeof p, "%s/%s", dir, v2 ? "memory.current" : "memory.usage_in_bytes");
if (!rd(p, cur)) cur = 0;
if (avail < 0 || lim - cur < avail) avail = lim - cur;
}
};
cap("/sys/fs/cgroup", true); cap("/sys/fs/cgroup/memory", false);
if (FILE *f = fopen("/proc/self/cgroup", "r")) {
char line[512];
while (fgets(line, sizeof line, f)) {
char *c1 = strchr(line, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : nullptr; if (!c2) continue;
line[strcspn(line, "\n")] = 0; *c1 = 0; *c2 = 0;
const bool v2 = !strcmp(line, "0") && c1[1] == 0;
bool v1 = false;
for (char *q = c1 + 1; *q;) { const size_t n = strcspn(q, ","); if (n == 6 && !strncmp(q, "memory", 6)) v1 = true; q += n; if (*q) q++; }
if (!v2 && !v1) continue;
char dir[640]; snprintf(dir, sizeof dir, "%s%s", v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/memory", c2 + 1);
for (size_t n = strlen(dir), r = strlen(v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/memory"); n > r; ) {
cap(dir, v2);
while (n > r && dir[n - 1] != '/') n--;
if (n > r) n--;
dir[n] = 0;
}
}
fclose(f);
}
if (const char *e = getenv("QSB_CPU_MEM_MB")) avail = atof(e) * 1048576.0;
return avail;
}


static Geo geo_choose(bool try11 = false, bool try10 = false) {
const double avail = mem_avail(), capb = (double)QSB_CPU_TAB_CAP_MB * 1048576.0;
double budget = avail * QSB_CPU_TAB_FRAC;
if (budget > capb) budget = capb;
int nw = 15;
for (int n = QSB_CPU_NW_MIN; n < 15; n++) if ((double)geo_make(n, true).total * sizeof(pt) <= budget) { nw = n; break; }
if (try11 && nw > 11) {
double b11 = (avail - (double)QSB_CPU_TAB_RESERVE_MB * 1048576.0) * QSB_CPU_TAB_FRAC;
if (b11 > capb) b11 = capb;
if ((double)geo_make(11, true).total * sizeof(pt) <= b11) nw = 11;
}
if (try10 && nw == 11) {
double b10 = (avail - (double)QSB_CPU_TAB_RESERVE_MB * 1048576.0) * QSB_CPU_TAB_FRAC;
if (b10 > (double)QSB_CPU_TAB10_CAP_MB * 1048576.0) b10 = (double)QSB_CPU_TAB10_CAP_MB * 1048576.0;
if ((double)geo_make(10, true).total * sizeof(pt) <= b10) nw = 10;
}
if (QSB_CPU_TRY9 && try10 && nw == 10) {
const double b9 = (avail - (double)QSB_CPU_TAB_RESERVE_MB * 1048576.0) * QSB_CPU_TAB9_FRAC;
if ((double)geo_make(9, true).total * sizeof(pt) <= b9) nw = 9;
}
if (const char *e = getenv("QSB_CPU_NW")) nw = atoi(e);
return geo_make(nw, !getenv("QSB_CPU_UNSIGNED"));
}





static bool table_setup(Ctx &c, int nth, double &hp, char *note, size_t nn, int nw_floor = 0) {
note[0] = 0; hp = -1;
bool thp = !getenv("QSB_CPU_NOTHP");
if (FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
char b[128] = {0}; if (fgets(b, sizeof b, f) && strstr(b, "[never]")) thp = false; fclose(f);
}
const bool forced = getenv("QSB_CPU_NW") != nullptr;
c.g = geo_choose(QSB_CPU_TRY11 && thp && !forced, QSB_CPU_TRY10 && thp && !forced);
if (!forced && c.g.nw < nw_floor) c.g = geo_make(nw_floor, true);
const char *fail = getenv("QSB_CPU_HP_FAIL");
auto failed = [&](int nw) {
if (!fail) return false;
char k[8]; const size_t kl = (size_t)snprintf(k, sizeof k, "%d", nw);
for (const char *q = fail; *q;) { const size_t n = strcspn(q, ","); if (n == kl && !strncmp(q, k, n)) return true; q += n; if (*q) q++; }
return false;
};
double touch_dl = 0;
auto measure = [&](size_t step) { const double f = table_touch(c, nth, step, c.g.nw == 9 ? touch_dl : 0); return failed(c.g.nw) && f > -2.5 ? 0.0 : f; };
size_t at = 0;
auto add = [&](int nw, double f) {
if (at >= nn) return;
if (f < -2.5) at += snprintf(note + at, nn - at, "; %d windows: first touch over %g s", nw, getenv("QSB_CPU_TOUCH9_MAX_S") ? atof(getenv("QSB_CPU_TOUCH9_MAX_S")) : (double)QSB_CPU_TOUCH9_MAX_S);
else if (f < 0) at += snprintf(note + at, nn - at, "; %d windows: %s", nw, f < -1.5 ? "no memory" : "no smaps");
else at += snprintf(note + at, nn - at, "; %d windows: %.1f%% huge pages", nw, 100.0 * f);
};
if (c.g.nw == 9 && c.g.sgn && !forced) {
if (table_alloc(c)) {
#if QSB_CPU_TOUCH_GUARD
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
const char *e = getenv("QSB_CPU_TOUCH9_MAX_S");
touch_dl = t.tv_sec + 1e-9 * t.tv_nsec + (e ? atof(e) : (double)QSB_CPU_TOUCH9_MAX_S); }
#endif
double f = measure(32);
if (f >= QSB_CPU_HP_MIN) f = measure(1);
if (f >= QSB_CPU_HP_MIN) { hp = f; return true; }
add(9, f);
table_free(c);
} else add(9, -2);
c.g = geo_make(10, true);
}
if (c.g.nw == 10 && c.g.sgn && !forced) {
if (table_alloc(c)) {
double f = measure(32);
if (f >= QSB_CPU_HP_MIN) f = measure(1);
if (f >= QSB_CPU_HP_MIN) { hp = f; return true; }
add(10, f);
table_free(c);
} else add(10, -2);
c.g = geo_make(11, true);
}
if (c.g.nw == 11 && c.g.sgn && !forced) {
if (table_alloc(c)) {
double f = measure(32);
if (f >= QSB_CPU_HP_MIN) f = measure(1);
if (f >= QSB_CPU_HP_MIN) { hp = f; return true; }
add(11, f);
table_free(c);
} else add(11, -2);
c.g = geo_make(12, true);
}
if (!table_alloc(c)) { c.g = geo_make(15, true); if (!table_alloc(c)) return false; }
if (forced || !thp || c.g.nw != 12 || !c.g.sgn || !QSB_CPU_FALL13) { hp = measure(1); return true; }
const double f = measure(16);
if (f >= QSB_CPU_HP_MIN || f < 0) { hp = f >= QSB_CPU_HP_MIN ? measure(1) : f; if (hp >= QSB_CPU_HP_MIN || hp < 0) return true; }
else hp = f;

const Geo g12 = c.g; pt *const t12 = c.table; void *const m12 = c.table_map; const size_t b12 = c.table_map_bytes;
c.table_map = nullptr; c.g = geo_make(13, true);
if (table_alloc(c)) {
const double f13 = measure(1);
if (f13 >= QSB_CPU_HP_MIN || f13 > hp + 0.25) { add(12, hp); munmap(m12, b12); hp = f13; return true; }
add(13, f13);
table_free(c);
}
c.g = g12; c.table = t12; c.table_map = m12; c.table_map_bytes = b12;
for (int i = 0; i < c.g.nw; i++) c.tw[i] = c.table + c.g.base[i];
hp = measure(1);
return true;
}

#if QCPU_SHANI


static void hash_plan(Ctx &c) {
const digest_params_t *dp = c.dp;
const size_t prl = dp->prefix_remainder_len, pl = (size_t)(c.cut - c.early) * SIG_PUSH_SIZE,
wlen = (size_t)(dp->n - c.cut - 3) * SIG_PUSH_SIZE, tl = dp->tail_section_len, sl = dp->tx_suffix_len;
const size_t remlen = (prl + pl) % 64;
const int nb = (int)((remlen + wlen + tl + sl + 9 + 63) / 64);
if (nb < 2 || nb > 16 || c.ncwin < 4 || c.ncwin > 286) return;
const uint64_t tbits = (c.mid_bytes + prl + pl + wlen + tl + sl) * 8;
std::vector<uint8_t> blocks, m((size_t)nb * 64);
std::vector<uint16_t> sidx((size_t)c.ncwin * 16);
int ng = 0; c.h_gblk.clear();
for (int wi = 0; wi < c.ncwin; wi++) {
const uint8_t *w3 = c.cwin[wi];
size_t o = remlen; memset(m.data(), 0, m.size());
for (int i = c.cut; i < (int)dp->n; i++) {
if (i == w3[0] || i == w3[1] || i == w3[2]) continue;
memcpy(&m[o], dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); o += SIG_PUSH_SIZE;
}
memcpy(&m[o], dp->tail_section, tl); o += tl;
memcpy(&m[o], dp->tx_suffix, sl); o += sl;
m[o] = 0x80;
for (int b = 0; b < 8; b++) m[(size_t)nb * 64 - 1 - b] = (uint8_t)(tbits >> (8 * b));
int g = 0; while (g < ng && memcmp(&c.h_gblk[(size_t)g * 64], m.data(), 64)) g++;
if (g == ng) { c.h_gblk.insert(c.h_gblk.end(), m.begin(), m.begin() + 64); ng++; }
c.h_g0[wi] = (uint8_t)g;
for (int b = 1; b < nb; b++) {
const size_t nbk = blocks.size() / 64; size_t q = 0;
while (q < nbk && memcmp(&blocks[q * 64], &m[(size_t)b * 64], 64)) q++;
if (q == nbk) blocks.insert(blocks.end(), m.begin() + (size_t)b * 64, m.begin() + (size_t)b * 64 + 64);
sidx[(size_t)wi * 16 + b] = (uint16_t)q;
}
}
c.h_wk.assign(blocks.size(), 0);
for (size_t q = 0; q < blocks.size() / 64; q++) qsha_schedule(&c.h_wk[q * 64], &blocks[q * 64]);
for (int wi = 0; wi < c.ncwin; wi++) for (int b = 1; b < nb; b++) c.h_wkp[wi][b - 1] = &c.h_wk[(size_t)sidx[(size_t)wi * 16 + b] * 64];
for (int n = 0; n < 256; n++) for (int k = 0; k < 8; k++) c.h_binom[n][k] = binom_u64(n, k);
if (c.cut > 255 || c.early > 7) return;
c.h_nb = nb; c.h_ng = ng; c.hplan = true;
}

#ifndef QSB_CPU_PREFIX100
#define QSB_CPU_PREFIX100 1
#endif



static void hash_plan_cpu_patterns(Ctx &c) {
hash_plan(c);
#if QSB_CPU_PREFIX100
if (!c.hplan || c.ncwin != 158 || c.h_ng != 77) return;
int size[286] = {0}, order[20], count = 0;
bool seen[286] = {false};
for (int i = 0; i < c.ncwin; i++) size[c.h_g0[i]]++;
for (int i = 0; i < c.ncwin; i++) {
const int g = c.h_g0[i];
if (!seen[g] && size[g] == 5) {
seen[g] = true;
if (count == 20) return;
order[count++] = g;
}
}
if (count != 20) return;
bool take[286] = {false};
for (int i = 0; i < 20; i++) take[order[i]] = true;
uint8_t keep[100][3]; int n = 0;
for (int i = 0; i < c.ncwin; i++) if (take[c.h_g0[i]]) {
if (n == 100) return;
memcpy(keep[n++], c.cwin[i], 3);
}
if (n != 100) return;
memcpy(c.cwin, keep, sizeof keep); c.ncwin = n;
hash_plan(c);
#endif
}
#endif

static inline void pk_block(uint8_t *pk, const fe &x3, unsigned ypar) {
pk[0] = (uint8_t)(0x02 | (ypar & 1));
for (int b = 0; b < 32; b++) pk[1 + b] = (uint8_t)(x3.v[3 - (b >> 3)] >> (8 * (7 - (b & 7))));
}
static inline bool pk_prefilter(uint32_t h0) { return (h0 >> (32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32))) == 0; }
static bool gate_publish_exact(Ctx *c, const uint8_t *sk, int ri);
static bool gate_publish(Ctx *c, uint8_t *pk, const uint8_t *sk, int ri, const fe &x3, unsigned ypar) {
pk_block(pk, x3, ypar);
SHA256_CTX s3; SHA256_Init(&s3); SHA256_Transform(&s3, pk);
if (!pk_prefilter(s3.h[0])) return false;
return gate_publish_exact(c, sk, ri);
}

static bool gate_publish_exact(Ctx *c, const uint8_t *sk, int ri) {
std::lock_guard<std::mutex> g(c->io);
if (!qsb_hv_check(&c->hv, sk, ri)) return false;
if (!c->out) { mkdir("results", 0755); c->out = fopen("results/digest_hit_cpu.txt", "a"); }
if (c->out) {
fprintf(c->out, "indices=%d,%d,%d,%d,%d,%d,%d,%d,%d recid=%d\n",
sk[0], sk[1], sk[2], sk[3], sk[4], sk[5], sk[6], sk[7], sk[8], ri);
fflush(c->out);
}
c->hits++;
return true;
}

#if QCPU_VEC


Q8T static void hpf_rows8(const uint32_t *zb, int k0, const pt *t0, const pt *t1, int w0, int w1, bool sgn) {
const __m512i one = _mm512_set1_epi64(1);
const __m512i vi = _mm512_set_epi64(28, 24, 20, 16, 12, 8, 4, 0);
const __m512i v = _mm512_i64gather_epi64(vi, (const void *)(zb + (size_t)k0 * 8 + 6), 8);
const __m512i zl = _mm512_ror_epi64(v, 32);
__m512i u = _mm512_and_si512(zl, _mm512_set1_epi64((long long)((1ULL << w0) - 1)));
const __mmask8 n0 = sgn ? _mm512_cmpgt_epu64_mask(u, _mm512_set1_epi64(1LL << (w0 - 1))) : 0;
u = _mm512_mask_sub_epi64(u, n0, _mm512_set1_epi64(1LL << w0), u);
alignas(64) uint64_t a[16];
_mm512_store_si512(a, _mm512_add_epi64(_mm512_set1_epi64((long long)(uintptr_t)t0), _mm512_slli_epi64(_mm512_sub_epi64(_mm512_max_epu64(u, one), one), 6)));
if (QSB_CPU_HPF > 1) {
__m512i u1 = _mm512_and_si512(_mm512_srl_epi64(zl, _mm_cvtsi32_si128(w0)), _mm512_set1_epi64((long long)((1ULL << w1) - 1)));
u1 = _mm512_mask_add_epi64(u1, n0, u1, one);
const __mmask8 n1 = sgn ? _mm512_cmpgt_epu64_mask(u1, _mm512_set1_epi64(1LL << (w1 - 1))) : 0;
u1 = _mm512_mask_sub_epi64(u1, n1, _mm512_set1_epi64(1LL << w1), u1);
_mm512_store_si512(a + 8, _mm512_add_epi64(_mm512_set1_epi64((long long)(uintptr_t)t1), _mm512_slli_epi64(_mm512_sub_epi64(_mm512_max_epu64(u1, one), one), 6)));
}
for (int j = 0; j < (QSB_CPU_HPF > 1 ? 16 : 8); j++) _mm_prefetch((const char *)a[j], QCPU_NXT_HINT);
}
#if QCPU_PFQ

Q8T static void hpf_rows8q(const uint32_t *zb, int k0, const pt *t0, const pt *t1, int w0, int w1, bool sgn, QPfRing &q) {
const __m512i one = _mm512_set1_epi64(1);
const __m512i vi = _mm512_set_epi64(28, 24, 20, 16, 12, 8, 4, 0);
const __m512i v = _mm512_i64gather_epi64(vi, (const void *)(zb + (size_t)k0 * 8 + 6), 8);
const __m512i zl = _mm512_ror_epi64(v, 32);
__m512i u = _mm512_and_si512(zl, _mm512_set1_epi64((long long)((1ULL << w0) - 1)));
const __mmask8 n0 = sgn ? _mm512_cmpgt_epu64_mask(u, _mm512_set1_epi64(1LL << (w0 - 1))) : 0;
u = _mm512_mask_sub_epi64(u, n0, _mm512_set1_epi64(1LL << w0), u);
alignas(64) uint64_t a[16];
_mm512_store_si512(a, _mm512_add_epi64(_mm512_set1_epi64((long long)(uintptr_t)t0), _mm512_slli_epi64(_mm512_sub_epi64(_mm512_max_epu64(u, one), one), 6)));
if (QSB_CPU_HPF > 1) {
__m512i u1 = _mm512_and_si512(_mm512_srl_epi64(zl, _mm_cvtsi32_si128(w0)), _mm512_set1_epi64((long long)((1ULL << w1) - 1)));
u1 = _mm512_mask_add_epi64(u1, n0, u1, one);
const __mmask8 n1 = sgn ? _mm512_cmpgt_epu64_mask(u1, _mm512_set1_epi64(1LL << (w1 - 1))) : 0;
u1 = _mm512_mask_sub_epi64(u1, n1, _mm512_set1_epi64(1LL << w1), u1);
_mm512_store_si512(a + 8, _mm512_add_epi64(_mm512_set1_epi64((long long)(uintptr_t)t1), _mm512_slli_epi64(_mm512_sub_epi64(_mm512_max_epu64(u1, one), one), 6)));
}
for (int j = 0; j < (QSB_CPU_HPF > 1 ? 16 : 8); j++) { if (q.tail - q.head >= 64) qpf_drain(q, 1); q.a[q.tail & 63] = (const char *)a[j]; q.tail++; }
}
#endif
struct VecBuf { fe8 *X = nullptr, *Y = nullptr, *D = nullptr, *P = nullptr, *TX = nullptr, *TY = nullptr; fe *qx = nullptr; uint8_t *qp = nullptr, *bad = nullptr; uint32_t *m16 = nullptr, *wk16 = nullptr; };
static bool vecbuf_alloc(VecBuf &v, int B) {
const int G = B / 8; void *q[9] = {nullptr};
const size_t sz[9] = {sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G, sizeof(fe8) * G,
sizeof(fe) * 2 * (size_t)B, 2 * (size_t)B, (size_t)B};
for (int i = 0; i < 9; i++) if (posix_memalign(&q[i], 64, sz[i])) { for (int j = 0; j < i; j++) free(q[j]); return false; }
v.X = (fe8 *)q[0]; v.Y = (fe8 *)q[1]; v.D = (fe8 *)q[2]; v.P = (fe8 *)q[3]; v.TX = (fe8 *)q[4]; v.TY = (fe8 *)q[5];
v.qx = (fe *)q[6]; v.qp = (uint8_t *)q[7]; v.bad = (uint8_t *)q[8];
#if QSB_CPU_KH16 && QCPU_SHANI
void *m = nullptr, *w = nullptr;
if (posix_memalign(&m, 64, (size_t)G * 144 * 4) || posix_memalign(&w, 64, 32 * 32 * 4)) { free(m); for (int j = 0; j < 9; j++) free(q[j]); return false; }
v.m16 = (uint32_t *)m; v.wk16 = (uint32_t *)w;
#endif
return true;
}

struct RowSgn {
const pt *T; const uint32_t *di;
Q8T __mmask8 operator()(int h, const pt **rows) const {
const __m512i one = _mm512_set1_epi64(1);
const __m512i e = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i *)(di + (size_t)h * 8)));
__m512i ix = _mm512_and_si512(e, _mm512_set1_epi64(0x7FFFFFFF));
ix = _mm512_sub_epi64(_mm512_max_epu64(ix, one), one);
_mm512_storeu_si512((void *)rows, _mm512_add_epi64(_mm512_set1_epi64((long long)(uintptr_t)T), _mm512_slli_epi64(ix, 6)));
return _mm512_test_epi64_mask(e, _mm512_set1_epi64(0x80000000LL));
}
};


Q8TX static void vec_batch(const Ctx *c, const uint32_t *zb, uint32_t *ds, int B, VecBuf &v) {
const int G = B / 8;
const Geo &g = c->g;
recode16(g, zb, ds, v.bad, B);

static thread_local std::vector<const pt *> rpa, rpb;
static thread_local std::vector<__mmask8> nga, ngb;
if ((int)rpa.size() < G * 8) { rpa.resize((size_t)G * 8); rpb.resize((size_t)G * 8); nga.resize((size_t)G); ngb.resize((size_t)G); }
const pt **rp = rpa.data(), **rpn = rpb.data(); __mmask8 *ng = nga.data(), *ngn = ngb.data();
ec8_first(v.X, v.Y, G, RowSgn{c->tw[0], ds}, RowSgn{c->tw[1], ds + B}, rp, ng);
for (int i = 1; i < g.nw; i++) {
if (i + 1 < g.nw) { const RowSgn rn{c->tw[i + 1], ds + (size_t)(i + 1) * B}; ec8_window(v.X, v.Y, v.D, v.P, v.TX, v.TY, G, rp, ng, rpn, ngn, &rn); }
else ec8_window<RowSgn>(v.X, v.Y, v.D, v.P, v.TX, v.TY, G, rp, ng, nullptr, nullptr, nullptr);
std::swap(rp, rpn); std::swap(ng, ngn);
}
#if QSB_CPU_KH16 && QCPU_SHANI
if (c->cfold && c->kh16) ec8_final_cf_kh(v.X, v.Y, v.D, v.P, G, c->mx, c->my, v.m16); else
#endif
if (c->cfold) ec8_final_cf(v.X, v.Y, v.D, v.P, G, c->mx, c->my, v.qx, v.qp);
else ec8_final(v.X, v.Y, v.D, v.P, G, c->cx, c->cy, v.qx, v.qp);
}
#endif

#ifndef QSB_CPU_EPOCH_CONTIG
#define QSB_CPU_EPOCH_CONTIG 1
#endif
#ifndef QSB_CPU_EPOCH_CAP
#define QSB_CPU_EPOCH_CAP (~0ull)
#endif
static void worker(Ctx *c, int tid) {
struct LiveGuard { std::atomic<int> &n; ~LiveGuard() { n--; } } live_guard{c->live};
#ifdef SCHED_IDLE
struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
const digest_params_t *dp = c->dp;
const int B = QSB_CPU_BATCH_AUTO ? c->batch : QSB_CPU_BATCH;

std::vector<pt> acc(B); std::vector<fe> d(2 * B), pre(2 * B);
std::vector<uint8_t> inf(B), bad(B); std::vector<const pt *> tp(B);
std::vector<uint32_t, qalloc64<uint32_t> > zb((size_t)B * 8), ds((size_t)NWMAX * B);
std::vector<uint8_t> ngs(B);
std::vector<uint8_t> skips((size_t)B * 9 + 8);
uint8_t pk[64]; memset(pk, 0, 64); pk[33] = 0x80; pk[62] = 0x01; pk[63] = 0x08;
uint8_t blk2[64]; memset(blk2, 0, 64); blk2[32] = 0x80; blk2[62] = 0x01;
#if QSB_CPU_EPOCH_CONTIG







const uint64_t eavail = c->n_epochs > c->epoch_base ? c->n_epochs - c->epoch_base : 0;
const uint64_t espan = (eavail < (uint64_t)QSB_CPU_EPOCH_CAP ? eavail : (uint64_t)QSB_CPU_EPOCH_CAP) / (uint64_t)(c->nthreads > 0 ? c->nthreads : 1);
uint64_t epoch = c->epoch_base + (uint64_t)tid * espan;
const uint64_t epoch_end = epoch + espan, estep = 1;
#else
uint64_t epoch = c->epoch_base + (uint64_t)tid;
const uint64_t epoch_end = ~0ull, estep = (uint64_t)c->nthreads;
#endif
int wi = c->ncwin;
SHA256_CTX ectx; uint8_t early[16];
std::vector<uint8_t> pbuf((size_t)dp->n * SIG_PUSH_SIZE + 64);
auto put_digits = [&](int kk, const uint32_t *h) { memcpy(&zb[(size_t)kk * 8], h, 32); };
#if QCPU_SHANI


const size_t tl = dp->tail_section_len, sl = dp->tx_suffix_len;
alignas(16) uint32_t lst[4][8]; alignas(16) uint8_t lmsg[4][16 * 64]; const uint8_t *lp[4];
uint32_t est[8]; uint8_t erem[64]; size_t remlen = 0; int nb = 0; uint64_t tbits = 0;
bool shani = c->shani;

const bool hplan = shani && c->hplan;
std::vector<uint8_t> lgblk(c->h_gblk);


std::vector<uint32_t, qalloc64<uint32_t> > gwk((size_t)(c->h_ng + 3) * 64);
uint8_t gwk_erem[64]; bool gwk_ok = false;
#if QSB_CPU_GWK_CACHE
const size_t gwk_n = (size_t)(c->h_ng + 3) * 64;
std::vector<uint32_t, qalloc64<uint32_t> > gwkc(gwk_n * 8);
uint8_t gwkc_key[8][64]; int gwkc_used = 0;
const uint32_t *gwkp = nullptr;
#endif


const size_t pfx_max = dp->prefix_remainder_len + (size_t)c->cut * SIG_PUSH_SIZE;
std::vector<uint8_t> pfx(pfx_max + 64);
if (dp->prefix_remainder_len) memcpy(pfx.data(), dp->prefix_remainder, dp->prefix_remainder_len);
std::vector<uint32_t> pst((pfx_max / 64 + 2) * 8);
memcpy(pst.data(), dp->midstate, 32);
uint8_t pv_early[16]; bool pv_ok = false; uint64_t pv_epoch = 0;
alignas(16) uint32_t gstb[2][286 + 3][8]; int gpar = 0;
uint32_t (*gst)[8] = gstb[0];
const uint32_t *lin[4];
const uint32_t *const *lrow[4];
alignas(16) uint32_t w2[4][16]; memset(w2, 0, sizeof w2);
for (int l = 0; l < 4; l++) { w2[l][8] = 0x80000000u; w2[l][15] = 256; }
uint32_t cw4[286];
for (int i = 0; i < c->ncwin; i++) cw4[i] = (uint32_t)c->cwin[i][0] | (uint32_t)c->cwin[i][1] << 8 | (uint32_t)c->cwin[i][2] << 16;
#endif
#if QCPU_PFQ
QPfRing pfq;
#endif
#if QCPU_VEC
VecBuf vb;
if (c->vec && !vecbuf_alloc(vb, B)) vb = VecBuf();
if (c->cfold && !vb.X) return;
const bool hpf = QSB_CPU_HPF > 0 && vb.X != nullptr && c->g.nw >= 3 && c->g.wid[0] + c->g.wid[1] <= 64;
#else
const bool hpf = false;
#endif


const int hw0 = c->g.wid[0], hw1 = c->g.wid[1]; const bool hsg = c->g.sgn;
const pt *const ht0 = c->tw[0], *const ht1 = c->tw[1];
auto hpf_rows = [&](int kk) {
const uint32_t *h = &zb[(size_t)kk * 8];
const uint64_t zl = (uint64_t)h[6] << 32 | h[7];
uint32_t u = (uint32_t)(zl & ((1ULL << hw0) - 1)), cy = 0;
if (hsg && u > (1u << (hw0 - 1))) { u = (1u << hw0) - u; cy = 1; }
__builtin_prefetch(ht0 + (u ? u - 1 : 0), 0, QSB_CPU_PFNX ? 2 : 3);
if (QSB_CPU_HPF > 1) {
uint32_t u1 = (uint32_t)((zl >> hw0) & ((1ULL << hw1) - 1)) + cy;
if (hsg && u1 > (1u << (hw1 - 1))) u1 = (1u << hw1) - u1;
__builtin_prefetch(ht1 + (u1 ? u1 - 1 : 0), 0, QSB_CPU_PFNX ? 2 : 3);
}
};
auto hpf_after = [&](int kl, int cnt) {
if (hpf) for (int q = kl - cnt + 1; q <= kl; q++) hpf_rows(q);
};
for (;;) {
if (c->stop.load(std::memory_order_relaxed)) return;
#ifdef QSB_CPU_DEVBENCH
if (c->dev_limit && c->cand.load() >= c->dev_limit) { c->dev_live--; return; }
#endif
int k = 0;
while (k < B) {
if (wi == c->ncwin) {
if (epoch >= c->n_epochs || epoch >= epoch_end) return;
#if QCPU_SHANI
if (shani && hplan) {
int su = -1;
if (QSB_CPU_EPOCH_CONTIG && pv_ok && epoch == pv_epoch + 1) {
su = c->early - 1; while (su >= 0 && pv_early[su] == c->cut - c->early + su) su--;
if (su >= 0) {
memcpy(early, pv_early, (size_t)c->early); early[su]++;
for (int i = su + 1; i < c->early; i++) early[i] = (uint8_t)(early[i - 1] + 1);
}
}
if (su < 0) {
uint64_t rank = epoch; int lo = 0;
for (int i = 0; i < c->early; i++) {
int cc = lo;
for (;;) { const uint64_t cnt = c->h_binom[c->cut - cc - 1][c->early - i - 1]; if (rank < cnt) break; rank -= cnt; cc++; }
early[i] = (uint8_t)cc; lo = cc + 1;
}
}
pv_epoch = epoch;
const size_t prl = dp->prefix_remainder_len;
size_t from = 0, pl = 0; int e2 = 0, i0 = 0;
if (pv_ok) {
int e = 0; while (e < c->early && early[e] == pv_early[e]) e++;
const int lo = e < c->early ? (early[e] < pv_early[e] ? early[e] : pv_early[e]) : c->cut;
from = prl + (size_t)(lo - e) * SIG_PUSH_SIZE;
i0 = lo; e2 = e; pl = (size_t)(lo - e) * SIG_PUSH_SIZE;
}
for (int i = i0; i < c->cut; i++) {
if (e2 < c->early && early[e2] == i) { e2++; continue; }
memcpy(&pfx[prl + pl], dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); pl += SIG_PUSH_SIZE;
}
const size_t lp = prl + pl, nfull = lp / 64;
SHA256_CTX pc;
for (size_t b = from / 64; b < nfull; b++) {
for (int i = 0; i < 8; i++) pc.h[i] = pst[b * 8 + i];
SHA256_Transform(&pc, &pfx[b * 64]);
for (int i = 0; i < 8; i++) pst[(b + 1) * 8 + i] = (uint32_t)pc.h[i];
}
memcpy(est, &pst[nfull * 8], 32);
remlen = lp % 64; memcpy(erem, &pfx[nfull * 64], remlen);
nb = c->h_nb;
memcpy(pv_early, early, (size_t)c->early); pv_ok = true;
const int ng = c->h_ng;
if (!gwk_ok || memcmp(gwk_erem, erem, remlen)) {
#if QSB_CPU_GWK_CACHE
int u = 0;
while (u < gwkc_used && memcmp(gwkc_key[u], erem, remlen)) u++;
if (u == gwkc_used) {
if (gwkc_used < 8) gwkc_used++; else u = 7;
uint32_t *gw = &gwkc[(size_t)u * gwk_n];
for (int q = 0; q < ng; q++) { memcpy(&lgblk[(size_t)q * 64], erem, remlen); qsha_schedule(&gw[(size_t)q * 64], &lgblk[(size_t)q * 64]); }
for (int q = ng; q < ng + 3; q++) memcpy(&gw[(size_t)q * 64], &gw[0], 64 * sizeof(uint32_t));
memcpy(gwkc_key[u], erem, remlen);
}
gwkp = &gwkc[(size_t)u * gwk_n];
#else
for (int q = 0; q < ng; q++) { memcpy(&lgblk[(size_t)q * 64], erem, remlen); qsha_schedule(&gwk[(size_t)q * 64], &lgblk[(size_t)q * 64]); }
for (int q = ng; q < ng + 3; q++) memcpy(&gwk[(size_t)q * 64], &gwk[0], 64 * sizeof(uint32_t));
#endif
memcpy(gwk_erem, erem, remlen); gwk_ok = true;
}
gpar ^= 1; gst = gstb[gpar];
for (int g = 0; g < ng; g += 4) {
const uint32_t *gp[4]; const uint32_t *const *gr[4] = {&gp[0], &gp[1], &gp[2], &gp[3]};
#if QSB_CPU_GWK_CACHE
for (int l = 0; l < 4; l++) { memcpy(gst[g + l], est, 32); gp[l] = &gwkp[(size_t)(g + l) * 64]; }
#else
for (int l = 0; l < 4; l++) { memcpy(gst[g + l], est, 32); gp[l] = &gwk[(size_t)(g + l) * 64]; }
#endif
qsha_x4p(&gst[g], gr, 1);
}
epoch += estep; wi = 0;
goto have_epoch;
}
#endif
qsb_host_unrank(epoch, c->cut, c->early, early);
SHA256_Init(&ectx);
for (int i = 0; i < 8; i++) ectx.h[i] = dp->midstate[i];
const uint64_t bits = c->mid_bytes * 8;
ectx.Nl = (SHA_LONG)bits; ectx.Nh = (SHA_LONG)(bits >> 32); ectx.num = 0;
if (dp->prefix_remainder_len) SHA256_Update(&ectx, dp->prefix_remainder, dp->prefix_remainder_len);
size_t pl = 0; int e = 0;
for (int i = 0; i < c->cut; i++) {
if (e < c->early && early[e] == i) { e++; continue; }
memcpy(pbuf.data() + pl, dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); pl += SIG_PUSH_SIZE;
}
SHA256_Update(&ectx, pbuf.data(), pl);
#if QCPU_SHANI
if (shani) {
const size_t prl = dp->prefix_remainder_len, wlen = (size_t)(dp->n - c->cut - 3) * SIG_PUSH_SIZE;
remlen = (prl + pl) % 64;
for (size_t q = 0; q < remlen; q++) { const size_t pos = prl + pl - remlen + q; erem[q] = pos < prl ? dp->prefix_remainder[pos] : pbuf[pos - prl]; }
for (int i = 0; i < 8; i++) est[i] = (uint32_t)ectx.h[i];
tbits = (c->mid_bytes + prl + pl + wlen + tl + sl) * 8;
nb = (int)((remlen + wlen + tl + sl + 9 + 63) / 64);
if (ectx.num != remlen || nb > 16) shani = false;
}
#endif
epoch += estep; wi = 0;
}
#if QCPU_SHANI
have_epoch:
if (shani && hplan) {
const int n = B - k < c->ncwin - wi ? B - k : c->ncwin - wi;
uint64_t e8; memcpy(&e8, early, 8);
for (int t = 0; t < n; t++) {
const int pi = wi + t, kq = k + t, j = kq & 3;
lin[j] = gst[c->h_g0[pi]];
lrow[j] = c->h_wkp[pi];
uint8_t *sk = &skips[(size_t)kq * 9];
memcpy(sk, &e8, 8); memcpy(sk + 6, &cw4[pi], 4);
if (j == 3) {

#if QCPU_PFQ
qsha_x4p(nullptr, lrow, nb - 1, &w2[0][0], 16, lin, hpf ? &pfq : nullptr);
#else
qsha_x4p(nullptr, lrow, nb - 1, &w2[0][0], 16, lin);
#endif
#if QSB_CPU_SHC
qsha_x4w_iv32((uint32_t (*)[8])&zb[(size_t)(kq - 3) * 8], w2);
#else
qsha_x4w_iv((uint32_t (*)[8])&zb[(size_t)(kq - 3) * 8], w2);
#endif
#if QCPU_VEC
#if QCPU_PFQ
if (hpf && (kq & 7) == 7) hpf_rows8q(zb.data(), kq - 7, ht0, ht1, hw0, hw1, hsg, pfq);
#else
if (hpf && (kq & 7) == 7) hpf_rows8(zb.data(), kq - 7, ht0, ht1, hw0, hw1, hsg);
#endif
#endif
}
}
k += n; wi += n;
continue;
}
#endif
const int pi = wi++;
const uint8_t *w3 = c->cwin[pi];
#if QCPU_SHANI
if (shani) {
const int j = k & 3;
uint8_t *m = lmsg[j]; size_t o = remlen;
memcpy(m, erem, remlen);
for (int i = c->cut; i < (int)dp->n; i++) {
if (i == w3[0] || i == w3[1] || i == w3[2]) continue;
memcpy(m + o, dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); o += SIG_PUSH_SIZE;
}
memcpy(m + o, dp->tail_section, tl); o += tl;
memcpy(m + o, dp->tx_suffix, sl); o += sl;
m[o++] = 0x80; memset(m + o, 0, (size_t)nb * 64 - o);
for (int b = 0; b < 8; b++) m[(size_t)nb * 64 - 1 - b] = (uint8_t)(tbits >> (8 * b));
memcpy(lst[j], est, 32);
uint8_t *sk = &skips[(size_t)k * 9];
for (int q = 0; q < 6; q++) sk[q] = early[q];
sk[6] = w3[0]; sk[7] = w3[1]; sk[8] = w3[2];
if (j == 3) {
for (int bl = 0; bl < nb; bl++) { for (int l = 0; l < 4; l++) lp[l] = lmsg[l] + (size_t)bl * 64; qsha_x4(lst, lp); }
alignas(16) uint8_t b2[4][64]; alignas(16) uint32_t s2[4][8];
for (int l = 0; l < 4; l++) {
memcpy(b2[l], blk2, 64);
for (int w = 0; w < 8; w++) { b2[l][4 * w] = (uint8_t)(lst[l][w] >> 24); b2[l][4 * w + 1] = (uint8_t)(lst[l][w] >> 16); b2[l][4 * w + 2] = (uint8_t)(lst[l][w] >> 8); b2[l][4 * w + 3] = (uint8_t)lst[l][w]; }
memcpy(s2[l], qsha_iv, 32); lp[l] = b2[l];
}
qsha_x4(s2, lp);
for (int l = 0; l < 4; l++) put_digits(k - 3 + l, s2[l]);
hpf_after(k, 4);
}
k++;
continue;
}
#endif
SHA256_CTX s = ectx;
uint8_t wbuf[16 * SIG_PUSH_SIZE]; size_t wl = 0;
for (int i = c->cut; i < (int)dp->n; i++) {
if (i == w3[0] || i == w3[1] || i == w3[2]) continue;
memcpy(wbuf + wl, dp->dummy_sigs + (size_t)i * SIG_PUSH_SIZE, SIG_PUSH_SIZE); wl += SIG_PUSH_SIZE;
}
SHA256_Update(&s, wbuf, wl);
SHA256_Update(&s, dp->tail_section, dp->tail_section_len);
SHA256_Update(&s, dp->tx_suffix, dp->tx_suffix_len);
SHA256_Final(blk2, &s);
SHA256_CTX s2; SHA256_Init(&s2); SHA256_Transform(&s2, blk2);
{ uint32_t hw[8]; for (int i = 0; i < 8; i++) hw[i] = (uint32_t)s2.h[i]; put_digits(k, hw); }
hpf_after(k, 1);
uint8_t *sk = &skips[(size_t)k * 9];
for (int j = 0; j < 6; j++) sk[j] = early[j];
sk[6] = w3[0]; sk[7] = w3[1]; sk[8] = w3[2];
k++;
}
#if QCPU_PFQ
qpf_drain(pfq, 64);
#endif
#if QCPU_VEC
if (vb.X) {
vec_batch(c, zb.data(), ds.data(), B, vb);
#if QCPU_SHANI
#if QSB_CPU_KH16
if (c->kh16 && c->cfold) {
for (int h = 0; h < B / 8; h++) {
const unsigned pass = kh16_pass(vb.m16 + (size_t)h * 144, vb.wk16);
if (!pass) continue;
for (int j = 0; j < 8; j++) {
const int q = h * 8 + j;
if (vb.bad[q]) continue;
for (int ri = 0; ri < 2; ri++)
if (((pass >> (8 * ri + j)) & 1) && gate_publish_exact(c, &skips[(size_t)q * 9], ri)) break;
}
}
} else
#endif
if (c->shani) {
const __m128i zero = _mm_setzero_si128();
for (int kk = 0; kk < B; kk += 2) {
const __m128i h0 = qsha_keyhash4_h0(&vb.qx[(size_t)kk * 2], &vb.qp[(size_t)kk * 2]);
const int pass = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(_mm_srli_epi32(h0, 32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32)), zero)));
if (!pass) continue;
for (int q2 = 0; q2 < 2; q2++) {
const int q = kk + q2;
if (vb.bad[q]) continue;
for (int ri = 0; ri < 2; ri++)
if (((pass >> (2 * q2 + ri)) & 1) && gate_publish_exact(c, &skips[(size_t)q * 9], ri)) break;
}
}
} else
#endif
for (int kk = 0; kk < B; kk++) {
if (vb.bad[kk]) continue;
const uint8_t *sk = &skips[(size_t)kk * 9];
for (int ri = 0; ri < 2; ri++)
if (gate_publish(c, pk, sk, ri, vb.qx[(size_t)kk * 2 + ri], vb.qp[(size_t)kk * 2 + ri])) break;
}
c->cand += B;
continue;
}
#endif
memset(inf.data(), 1, (size_t)B); memset(bad.data(), 0, (size_t)B);
recode_scalar(c->g, zb.data(), ds.data(), B);
for (int i = 0; i < c->g.nw; i++) {
const pt *T = c->tw[i]; const uint32_t *di = &ds[(size_t)i * B];
for (int kk = 0; kk < B; kk++) { const uint32_t e = di[kk], m = e & 0x7FFFFFFFu; tp[kk] = m ? &T[m - 1] : nullptr; ngs[kk] = (uint8_t)(e >> 31); }
batch_add(acc.data(), tp.data(), inf.data(), bad.data(), B, d.data(), pre.data(), ngs.data());
}

fe run = {{1, 0, 0, 0}};
for (int kk = 0; kk < B; kk++) {
if (bad[kk] || inf[kk]) { pre[kk] = run; continue; }
fe_sub(d[kk], c->cx, acc[kk].x);
if (fe_is_zero(d[kk])) { bad[kk] = 1; pre[kk] = run; continue; }
pre[kk] = run; fe_mul(run, run, d[kk]);
}
fe inv; fe_inv(inv, run);
for (int kk = B - 1; kk >= 0; kk--) {
if (bad[kk] || inf[kk]) continue;
fe dinv; fe_mul(dinv, inv, pre[kk]); fe_mul(inv, inv, d[kk]);
for (int ri = 0; ri < 2; ri++) {
fe cy = c->cy; if (ri) { fe z0 = {{0, 0, 0, 0}}; fe_sub(cy, z0, cy); }
fe lam, t, x3, y3;
fe_sub(t, cy, acc[kk].y); fe_mul(lam, t, dinv);
fe_sqr(x3, lam); fe_sub(x3, x3, acc[kk].x); fe_sub(x3, x3, c->cx);
fe_sub(t, acc[kk].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[kk].y);
if (gate_publish(c, pk, &skips[(size_t)kk * 9], ri, x3, (unsigned)(y3.v[0] & 1))) break;
}
}
c->cand += B;
}
}

static Ctx *g_ctx = nullptr;



#ifdef CPU_COUNT
static cpu_set_t g_initial_cpus;
static int g_initial_ok = [] { CPU_ZERO(&g_initial_cpus); return sched_getaffinity(0, sizeof g_initial_cpus, &g_initial_cpus) == 0 ? 1 : 0; }();
#endif
#if defined(CPU_COUNT) && QSB_CPU_RSV_CORE




static bool rsv_core_mask(bool plan_cores, cpu_set_t *out) {
cpu_set_t cs; CPU_ZERO(&cs);
if (g_initial_ok) cs = g_initial_cpus; else if (sched_getaffinity(0, sizeof cs, &cs) != 0) return false;
cpu_set_t host; CPU_ZERO(&host);
const bool host_pinned = sched_getaffinity(0, sizeof host, &host) == 0 && CPU_COUNT(&host) < CPU_COUNT(&cs);
std::vector<std::vector<int> > cores; std::vector<uint8_t> seen(CPU_SETSIZE, 0);
bool topo = plan_cores;
for (int cpu = 0; cpu < CPU_SETSIZE && topo; cpu++) {
if (!CPU_ISSET(cpu, &cs) || seen[cpu]) continue;
char path[128]; snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
FILE *f = fopen(path, "r"); if (!f) { topo = false; break; }
char buf[256] = {0}; const bool ok = fgets(buf, sizeof buf, f) != nullptr; fclose(f); if (!ok) { topo = false; break; }
std::vector<int> core;
for (char *q = buf; *q;) {
char *e; long a = strtol(q, &e, 10); if (e == q) break; long b = a;
if (*e == '-') { q = e + 1; b = strtol(q, &e, 10); }
for (long x = a; x <= b && x < CPU_SETSIZE; x++) if (x >= 0 && CPU_ISSET(x, &cs) && !seen[x]) { core.push_back((int)x); seen[x] = 1; }
q = e; if (*q == ',') q++; else break;
}
if (core.empty()) { core.push_back(cpu); seen[cpu] = 1; }
cores.push_back(core);
}
int rsv = -1;
if (topo) {
if (host_pinned)
for (int i = 0; i < (int)cores.size() && rsv < 0; i++) for (int x : cores[i]) if (CPU_ISSET(x, &host)) { rsv = i; break; }
for (int i = (int)cores.size() - 1; i >= 0 && rsv < 0; i--) if (cores[i].size() >= 2) rsv = i;
if (cores.size() < 2) rsv = -1;
}
*out = cs;
if (rsv >= 0) { for (int x : cores[rsv]) CPU_CLR(x, out); return CPU_COUNT(out) > 0; }
if (!host_pinned) return false;
for (int x = 0; x < CPU_SETSIZE; x++) if (CPU_ISSET(x, &host)) CPU_CLR(x, out);
return CPU_COUNT(out) > 0;
}
#endif
#if QSB_CPU_BATCH_AUTO


#ifdef CPU_COUNT

static bool cpu_siblings(int cpu, std::vector<int> &out) {
out.clear();
char p[96]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
FILE *f = fopen(p, "r"); if (!f) return false;
char buf[512]; const bool ok = fgets(buf, sizeof buf, f) != nullptr; fclose(f);
if (!ok) return false;
for (char *q = buf; *q; ) {
char *e; const long a = strtol(q, &e, 10); if (e == q) break;
long b = a; q = e;
if (*q == '-') { b = strtol(q + 1, &e, 10); if (e == q + 1) break; q = e; }
for (long v = a; v <= b && v - a < 4096; v++) if (v >= 0 && v < CPU_SETSIZE) out.push_back((int)v);
if (*q == ',') q++; else break;
}
return !out.empty();
}


static void core_groups(const cpu_set_t &m, std::vector<std::vector<int> > &cores, bool *all_read = nullptr) {
cores.clear(); std::vector<char> seen(CPU_SETSIZE, 0); std::vector<int> sib;
if (all_read) *all_read = true;
for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
if (!CPU_ISSET(cpu, &m) || seen[cpu]) continue;
std::vector<int> core; core.push_back(cpu); seen[cpu] = 1;
if (cpu_siblings(cpu, sib)) { for (int s : sib) if (CPU_ISSET(s, &m) && !seen[s]) { core.push_back(s); seen[s] = 1; } }
else if (all_read) *all_read = false;
cores.push_back(core);
}
}


static const char *core_sharing(const cpu_set_t &wm, int nw) {
if (CPU_COUNT(&wm) < 1) return "CPU set unknown";
if (nw > CPU_COUNT(&wm)) return "more workers than CPUs";
std::vector<std::vector<int> > cores; bool all = true; core_groups(wm, cores, &all);
if (!all) return "topology unreadable";
for (const std::vector<int> &k : cores) if (k.size() > 1) return "SMT siblings among the workers' CPUs";
return nullptr;
}
#endif


static int batch_choose(int nw, const char *shared, char *why, size_t nwhy) {
if (const char *e = getenv("QSB_CPU_BATCH_RT")) {
char *end = nullptr; const long v = strtol(e, &end, 10);
if (end != e && *end == 0 && v >= 32 && v <= 8192 && v % 32 == 0) { snprintf(why, nwhy, "QSB_CPU_BATCH_RT"); return (int)v; }
printf("  CPU co-grind: QSB_CPU_BATCH_RT=%s ignored (a multiple of 32, 32..8192)\n", e); fflush(stdout);
}
if (nw <= 1) { snprintf(why, nwhy, "one worker"); return QSB_CPU_BATCH_SOLO; }
if (shared) { snprintf(why, nwhy, "%s", shared); return QSB_CPU_BATCH; }
snprintf(why, nwhy, "no shared cores"); return QSB_CPU_BATCH_SOLO;
}
#endif
static void start(const digest_params_t *dp, const uint8_t win3[][3], int nwin, int cut, int early) {
long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
#ifdef CPU_COUNT
cpu_set_t work_cpus; CPU_ZERO(&work_cpus); bool work_mask = false;
{
cpu_set_t cs; CPU_ZERO(&cs);
if (g_initial_ok) cs = g_initial_cpus; else if (sched_getaffinity(0, sizeof cs, &cs) != 0) CPU_ZERO(&cs);
if (CPU_COUNT(&cs) > 0) ncpu = CPU_COUNT(&cs);
cpu_set_t now; CPU_ZERO(&now);
if (g_initial_ok && sched_getaffinity(0, sizeof now, &now) == 0 && CPU_COUNT(&now) < CPU_COUNT(&cs)) {
for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &cs) && !CPU_ISSET(c, &now)) CPU_SET(c, &work_cpus);
#if defined(QSB_HP_ON) && QSB_CPU_HP_SHARE

if (qhp::g_share_cpu >= 0 && CPU_ISSET(qhp::g_share_cpu, &cs)) CPU_SET(qhp::g_share_cpu, &work_cpus);
#endif
#if QSB_CPU_ALLCPU && defined(QSB_HOST_BLOCKING) && QSB_HOST_BLOCKING



for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &now)) CPU_SET(c, &work_cpus);
#endif
work_mask = CPU_COUNT(&work_cpus) >= 1;
}
}
#endif
if (FILE *q = fopen("/sys/fs/cgroup/cpu.max", "r")) {
char quota[32] = {0}; long period = 0;
if (fscanf(q, "%31s %ld", quota, &period) == 2 && strcmp(quota, "max") != 0 && period > 0) {
long lim = (atol(quota) + period - 1) / period; if (lim > 0 && lim < ncpu) ncpu = lim;
}
fclose(q);
}
if (FILE *q = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
long quota = -1, period = 0; if (fscanf(q, "%ld", &quota) != 1) quota = -1; fclose(q);
if (FILE *pf = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) { if (fscanf(pf, "%ld", &period) != 1) period = 0; fclose(pf); }
if (quota > 0 && period > 0) { long lim = (quota + period - 1) / period; if (lim > 0 && lim < ncpu) ncpu = lim; }
}
#ifdef QSB_CPU_THREADS
int nth = QSB_CPU_THREADS;
#else
int nth = (int)ncpu - QSB_CPU_RESERVE;
#if defined(CPU_COUNT) && (QSB_CPU_NTH_RAISE || QSB_CPU_ALLCPU)



if (work_mask && CPU_COUNT(&work_cpus) > nth) nth = CPU_COUNT(&work_cpus) < (int)ncpu ? CPU_COUNT(&work_cpus) : (int)ncpu;
#endif
#endif
if (const char *e = getenv("QSB_CPU_THREADS_ENV")) nth = atoi(e);
if (nth < 1 || dp->n != 150 || cut != 137 || early != 6) { printf("  CPU co-grind: off (%d threads)\n", nth); return; }
Ctx *c = new Ctx(); c->dp = dp; c->nthreads = nth; c->cut = cut; c->early = early;
#if QCPU_VEC
__builtin_cpu_init();
c->vec = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512ifma") && !getenv("QSB_CPU_NOVEC");
#if QSB_CPU_VBMI2
c->vec = c->vec && __builtin_cpu_supports("avx512vbmi2");
#endif
#endif
#if QCPU_SHANI
c->shani = qsha_supported() && !getenv("QSB_CPU_NOSHANI");
#if QSB_CPU_KH16 && QCPU_VEC
c->kh16 = c->shani && c->vec && __builtin_cpu_supports("avx512vl") && !getenv("QSB_CPU_NOKH16");
#endif
#endif
#if defined(CPU_COUNT) && QSB_CPU_RSV_CORE
{
cpu_set_t m;
if (rsv_core_mask(c->vec && !getenv("QSB_CPU_NOPIN"), &m)) { work_cpus = m; work_mask = true; }
}
#endif
#ifdef QSB_CPU_DEVBENCH
if (const char *e = getenv("QSB_CPU_DEVCAND")) c->dev_limit = strtoull(e, nullptr, 10);
c->dev_live = nth;
#endif
#if QSB_CPU_BATCH_AUTO
{

const char *shared = "CPU set unknown";
#ifdef CPU_COUNT
cpu_set_t wcs; CPU_ZERO(&wcs);
if (work_mask) wcs = work_cpus; else if (sched_getaffinity(0, sizeof wcs, &wcs) != 0) CPU_ZERO(&wcs);
shared = core_sharing(wcs, c->nthreads);
#endif
c->batch = batch_choose(c->nthreads, shared, c->batch_why, sizeof c->batch_why);
}
#endif

for (int a = cut; a < (int)dp->n; a++) for (int b = a + 1; b < (int)dp->n; b++) for (int d3 = b + 1; d3 < (int)dp->n; d3++) {
int used = 0;
for (int i = 0; i < nwin; i++) if (win3[i][0] == a && win3[i][1] == b && win3[i][2] == d3) { used = 1; break; }
if (!used) { c->cwin[c->ncwin][0] = (uint8_t)a; c->cwin[c->ncwin][1] = (uint8_t)b; c->cwin[c->ncwin][2] = (uint8_t)d3; c->ncwin++; }
}
const uint64_t unpadded = (uint64_t)dp->prefix_remainder_len + (uint64_t)(dp->n - dp->t) * SIG_PUSH_SIZE +
dp->tail_section_len + dp->tx_suffix_len;
if (dp->t != 9 || dp->total_preimage_len < unpadded || ((dp->total_preimage_len - unpadded) % 64) != 0 || c->ncwin < 1) {
printf("  CPU co-grind: off (unexpected problem shape)\n"); delete c; return;
}
c->mid_bytes = dp->total_preimage_len - unpadded;
c->n_epochs = binom_u64(cut, early);
#if QCPU_SHANI
if (c->shani) hash_plan_cpu_patterns(*c);
#endif
if (!qsb_hv_init(&c->hv, dp, (const uint8_t (*)[QSB_SE_TWIN])win3, cut, early)) { printf("  CPU co-grind: off (gate)\n"); delete c; return; }
EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
BN_CTX *bctx = BN_CTX_new(); BIGNUM *nri = BN_new(), *ax = BN_new(), *ay = BN_new();
EC_POINT *A = EC_POINT_new(grp);
BN_lebin2bn(dp->neg_r_inv, 32, nri);
if (!EC_POINT_mul(grp, A, nri, NULL, NULL, bctx) ||
!EC_POINT_get_affine_coordinates_GFp(grp, A, ax, ay, bctx)) { printf("  CPU co-grind: off (A)\n"); delete c; return; }
fe fax, fay; fe_from_bn(fax, ax); fe_from_bn(fay, ay);
fe_from_le32(c->cx, dp->u2r_x); fe_from_le32(c->cy, dp->u2r_y);
EC_POINT_free(A); BN_free(nri); BN_free(ax); BN_free(ay); BN_CTX_free(bctx); EC_GROUP_free(grp);
#ifdef CPU_COUNT
std::thread([c, fax, fay, nth, work_mask, work_cpus, ncpu, nwin]() {
if (work_mask) sched_setaffinity(0, sizeof work_cpus, &work_cpus);
#else
std::thread([c, fax, fay, nth, ncpu, nwin]() {
#endif
#ifdef SCHED_IDLE
struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
double hp; char note[160];
if (!table_setup(*c, nth, hp, note, sizeof note)) { printf("  CPU co-grind: off (table memory)\n"); fflush(stdout); return; }
clock_gettime(CLOCK_MONOTONIC, &t1);
char hps[16]; if (hp < 0) snprintf(hps, sizeof hps, "n/a"); else snprintf(hps, sizeof hps, "%.1f%%", 100.0 * hp);
printf("  CPU co-grind: %d threads (of %ld CPUs), %s, %s%s, %d window patterns per epoch disjoint from the GPU's %d, batch %d (%s); "
"table %d %s windows of %d..%d bits, %.0f MiB, huge pages %s (%.2f s%s)\n",
nth, ncpu, c->vec ? "8-lane IFMA" : "scalar", c->shani ? "4-lane SHA-NI" : "OpenSSL SHA-256", c->kh16 ? " (16-lane key hashes)" : "",
c->ncwin, nwin, c->batch, c->batch_why,
c->g.nw, c->g.sgn ? "signed" : "unsigned", c->g.wid[c->g.nw - 1], c->g.wid[0], c->g.total * sizeof(pt) / 1048576.0,
hps, (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec), note);
fflush(stdout);
build_table(*c, fax, fay, nth);
bool tab_ok = table_check(*c);
if (!tab_ok && c->g.nw == 9 && !getenv("QSB_CPU_NW")) {
table_free(*c);
if (!table_setup(*c, nth, hp, note, sizeof note, 10)) { printf("  CPU co-grind: off (table memory)\n"); fflush(stdout); return; }
printf("  CPU co-grind: 9-window table check failed; table %d windows, %.0f MiB, huge pages %.1f%%%s\n",
c->g.nw, c->g.total * sizeof(pt) / 1048576.0, 100.0 * hp, note);
fflush(stdout);
build_table(*c, fax, fay, nth);
tab_ok = table_check(*c);
}
if (!tab_ok && c->g.nw == 10 && !getenv("QSB_CPU_NW")) {
table_free(*c);
if (!table_setup(*c, nth, hp, note, sizeof note, 11)) { printf("  CPU co-grind: off (table memory)\n"); fflush(stdout); return; }
printf("  CPU co-grind: 10-window table check failed; table %d windows, %.0f MiB, huge pages %.1f%%%s\n",
c->g.nw, c->g.total * sizeof(pt) / 1048576.0, 100.0 * hp, note);
fflush(stdout);
build_table(*c, fax, fay, nth);
tab_ok = table_check(*c);
}
if (!tab_ok) { printf("  CPU co-grind: off (table check failed)\n"); fflush(stdout); table_free(*c); return; }
clock_gettime(CLOCK_MONOTONIC, &t1);
#if QSB_CPU_DIAG_EPOCH
{






const int cg = c->g.nw == 11 ? 1 : c->g.nw == 12 ? 2 : 3;
const int ch = (hp >= 0.95) ? 0 : 1;
const int ct = nth >= 28 ? 0 : 1;
uint64_t base = (uint64_t)(cg + 3 * ch + 6 * ct) << 29 | (uint64_t)(c->g.nw == 10 || c->g.nw == 9) << 28 | (uint64_t)(c->g.nw == 9) << 19;
{

double av = mem_avail(); if (av >= 0) av += (double)c->g.total * sizeof(pt);
long gib = av < 0 ? 0 : (long)(av / 1073741824.0); if (gib > 255) gib = 255;
base += (uint64_t)gib << 20;
}
if (base + (3ull << 29) <= c->n_epochs) c->epoch_base = base;
printf("  CPU co-grind: epoch walk starts at %llu (diagnostic code %d%s, memory %d GiB; table ready in %.2f s)\n", (unsigned long long)c->epoch_base,
(int)(c->epoch_base >> 29), (c->epoch_base >> 28) & 1 ? ", 10 windows" : "", (int)((c->epoch_base >> 20) & 255),
(t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
fflush(stdout);
}
#endif
c->live += nth; c->ready = 1;
for (int t = 0; t < nth; t++) std::thread(worker, c, t).detach();
}).detach();
g_ctx = c;
#ifdef QSB_CPU_DEVBENCH
if (c->dev_limit) {
while (c->cand.load() == 0 || c->dev_live.load() > 0) usleep(10000);
{ std::lock_guard<std::mutex> g(c->io); if (c->out) fflush(c->out); }
printf("DEVCAND %llu candidates, %u hits\n", (unsigned long long)c->cand.load(), c->hits.load());
fflush(stdout); _exit(0);
}
#endif
}
static uint64_t candidates() { return g_ctx ? g_ctx->cand.load() : 0; }
static uint32_t hits() { return g_ctx ? g_ctx->hits.load() : 0; }





static double stop_unmap() {
Ctx *c = g_ctx; if (!c) return -1;
struct timespec t0, t; clock_gettime(CLOCK_MONOTONIC, &t0);
auto el = [&]() { clock_gettime(CLOCK_MONOTONIC, &t); return (t.tv_sec - t0.tv_sec) + 1e-9 * (t.tv_nsec - t0.tv_nsec); };
c->stop = 1;
if (!c->ready.load()) return -1;
while (c->live.load() > 0) { if (el() > 0.25) return -1; usleep(200); }
if (c->table_map) {
char *a = (char *)c->table_map; const size_t n = c->table_map_bytes, S = (size_t)1 << 30;
for (size_t o = 0; o < n; o += S) madvise(a + o, n - o < S ? n - o : S, MADV_DONTNEED);
table_free(*c);
}
return el();
}
}
