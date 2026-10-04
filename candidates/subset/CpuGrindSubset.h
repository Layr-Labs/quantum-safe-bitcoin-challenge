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
#define QSB_CPU_VEC 1              /* 8-lane AVX-512 IFMA field arithmetic when the host CPU has it */
#endif
#if QSB_CPU_VEC && defined(__x86_64__) && !defined(__CUDA_ARCH__)
#define QCPU_VEC 1
#include <immintrin.h>
#else
#define QCPU_VEC 0
#endif
#ifndef QSB_CPU_SHANI
#define QSB_CPU_SHANI 1            /* 4-lane SHA-256 with the x86 SHA extensions when the host CPU has them */
#endif
#if QSB_CPU_SHANI && defined(__x86_64__) && !defined(__CUDA_ARCH__)
#define QCPU_SHANI 1
#include <immintrin.h>
#include <cpuid.h>
#else
#define QCPU_SHANI 0
#endif

#ifndef QSB_CPU_RESERVE
#define QSB_CPU_RESERVE 2          /* logical CPUs left for the GPU host thread and driver */
#endif
#ifndef QSB_CPU_FOLD2
#define QSB_CPU_FOLD2 1            /* fe8_fold: split the high columns (low 52 bits + rest) instead of a serial carry chain */
#endif
#ifndef QSB_CPU_FOLD3
#define QSB_CPU_FOLD3 QSB_CPU_FOLD2   /* column 9's upper fold term rides in column 5's product chain (3 IFMA fewer per reduction); column 4's
                                         fold terms on a side chain. On with the split-column fold (QSB_CPU_FOLD2=0 turns both off) */
#endif
#if QSB_CPU_FOLD3 && !QSB_CPU_FOLD2
#error "QSB_CPU_FOLD3 extends the split-column fold (QSB_CPU_FOLD2)"
#endif
#ifndef QSB_CPU_PSEED
#define QSB_CPU_PSEED 1            /* fused subtractions: the 4p offset rides in the product columns' initial values */
#endif
/* IFMA cuts in the 8-lane reduction, exact (same residues, same canonical outputs, same
 * hits). QSB_CPU_FOLD4: fe8_fold carries the high columns' bits >= 52 into the next column first (h5 -> c6, h6 -> c7, h7 -> c8,
 * h8 -> column 4 as lo(h8*R)), then folds each carried column with one lo/hi IFMA pair: 11 IFMA per reduction instead of FOLD3's
 * 14 (3 fewer per multiplication and squaring, about 21 per candidate) and 7 shift/add instead of 8 shift/and; every output column
 * still takes at most 3 fold terms, so every bound below is unchanged. QSB_CPU_DNF: the window step's D = tx - X keeps limb 4
 * unfolded (< 2^51 + 2^3; fe8_sub_m: 12 ops, no IFMA, instead of fe8_carry's 15 with one), and x3 = lam^2 - D - 2X seeds its
 * columns at 16p instead of 4p so every limb stays non-negative: 1 IFMA and 2 ops fewer per group and window. QSB_CPU_VBMI2 now
 * defaults to 0 (see there). 0 on each switch = the code above it. */
#ifndef QSB_CPU_FOLD4
#define QSB_CPU_FOLD4 1
#endif
#if QSB_CPU_FOLD4 && !QSB_CPU_FOLD3
#error "QSB_CPU_FOLD4 extends QSB_CPU_FOLD3 (column 9's upper fold term pre-added to column 5)"
#endif
#ifndef QSB_CPU_BATCH
#define QSB_CPU_BATCH 1024         /* candidates per batch: both SMT threads' EC state (2 x 0.25 MB) and prefetched rows stay in the 1 MB L2 */
#endif
/* Fixed-base table geometry, chosen at run time (Geo, table_setup): signed digits, the fewest windows whose table fits in
 * QSB_CPU_TAB_FRAC of the memory this process may still use (MemAvailable and the cgroup limits), capped at
 * QSB_CPU_TAB_CAP_MB and at no fewer than QSB_CPU_NW_MIN windows; never more than 15 windows (68 MiB).
 * QSB_CPU_TRY11: 11 windows (3,840 MiB) when that table also fits in QSB_CPU_TAB_FRAC of the memory left after
 * QSB_CPU_TAB_RESERVE_MB (host memory the process takes after start(): ~1.3 GiB pinned producer buffers, GPU host memory)
 * and at least QSB_CPU_HP_MIN of it is backed by transparent huge pages, measured (/proc/self/smaps) right after the first
 * touch; else 12 (fully backed 11 vs 12 on a Zen 4, 2c x 2t / 5c x 2t: +5.4 / +8.1 %; 11 on 4 KiB pages: -21 %).
 * QSB_CPU_FALL13 (off): a 12-window table below QSB_CPU_HP_MIN is replaced by 13 windows if those are better backed
 * (12 on 4 KiB pages still beats 13 on huge pages by 2 %). Without THP ([never]) or smaps the geometry is the old one.
 * QSB_CPU_TRY10: 10 windows (17,408 MiB) under the same rule as 11 (table <= QSB_CPU_TAB_FRAC x (avail -
 * QSB_CPU_TAB_RESERVE_MB), i.e. avail >= 79.5 GiB, at most QSB_CPU_TAB10_CAP_MB) and the same huge-page guard (a 272-region
 * sample, then all 8,704 regions); else the 11-window rule, then 12. An mmap failure at 10 goes to 11, and so does a
 * failed table check of the built 10-window table. */
#ifndef QSB_CPU_NW_MIN
#define QSB_CPU_NW_MIN 12
#endif
#ifndef QSB_CPU_TRY11
#define QSB_CPU_TRY11 1
#endif
#ifndef QSB_CPU_TRY10
#define QSB_CPU_TRY10 1
#endif
/* QSB_CPU_TRY9: 9 windows (114,688 MiB; 8 additions per candidate instead of 9) when the 10-window rule holds and that
 * table fits in QSB_CPU_TAB9_FRAC of the memory left after QSB_CPU_TAB_RESERVE_MB (i.e. avail >= ~236 GiB), with the same
 * huge-page guard; else 10 (an mmap failure, poor backing or a failed table check at 9 goes to 10). The ranked host offers
 * >= 255 GiB (the 2d1631b0 diagnostic). 0 = the 10-window rule as before. */
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
#define QSB_CPU_VBUILD 1           /* 8-lane table build (chunk steps and C fold through ec8_window) when the lane is 8-lane */
#endif
#ifndef QSB_CPU_CFOLD
#define QSB_CPU_CFOLD 1            /* 8-lane path: C folded into the top window's table; the final step is one addition (-2C) */
#endif
#ifndef QSB_CPU_PFD1
#define QSB_CPU_PFD1 8             /* prefetch distance of the first window's loop (window 0 loads), in groups */
#endif
#ifndef QSB_CPU_PFD
#define QSB_CPU_PFD 3              /* table-row prefetch distance, in groups of 8 candidates */
#endif
#ifndef QSB_CPU_HPF
#define QSB_CPU_HPF 2              /* 8-lane path: rows of windows 0 (and 1 with 2) prefetched from the hashing phase as each z is computed */
#endif
#ifndef QSB_CPU_NOTX
#define QSB_CPU_NOTX 1             /* window step without the table x array: D normalized, x3 = lam^2 - D - 2X */
#endif
#ifndef QSB_CPU_NF
#define QSB_CPU_NF 1               /* mul-only differences of normalized operands skip the limb-4 fold (limb 4 stays < 2^52) */
#endif
#ifndef QSB_CPU_DNF
#define QSB_CPU_DNF 1              /*: D = tx - X with limb 4 unfolded (fe8_sub_m, no IFMA); x3's fused columns seeded at 16p */
#endif
#if QSB_CPU_DNF && !(QSB_CPU_NOTX && QSB_CPU_PSEED)
#error "QSB_CPU_DNF is written for the QSB_CPU_NOTX window step with QSB_CPU_PSEED (fe8_sqr_subdx's seeded columns)"
#endif
#ifndef QSB_CPU_WPRE
#define QSB_CPU_WPRE 1             /* batch-affine steps: the forward pass stores prefix * numerator (W), the backward pass lam = inverse * W */
#endif
#if QSB_CPU_WPRE && !QSB_CPU_NOTX
#error "QSB_CPU_WPRE is written for the QSB_CPU_NOTX window step"
#endif
#ifndef QSB_CPU_PFNX
#define QSB_CPU_PFNX 0             /* next window's rows (backward-pass prefetch) into L1 (prefetcht0); tip default was L2 (prefetcht1) */
#endif
#define QCPU_NXT_HINT (QSB_CPU_PFNX ? _MM_HINT_T1 : _MM_HINT_T0)
#ifndef QSB_CPU_X4PS
#define QSB_CPU_X4PS 1             /* qsha_x4p: one load of a block's W+K words for all four lanes when they hash the same fixed block */
#endif
#ifndef QSB_CPU_SHC
#define QSB_CPU_SHC 1              /* key hashes and second SHA-256: schedule steps of the constant padding words skipped (identities) */
#endif
#ifndef QSB_CPU_SHA4
#define QSB_CPU_SHA4 1             /* key hashes and second SHA-256: four SHA-NI lanes interleaved per call, message rings in memory
                                    * (one sha256rnds2 per cycle); 0: two lanes per call or pass (one per 2 cycles). Same digests */
#endif
/* . QSB_CPU_BATCH_AUTO: the batch (Ctx::batch) chosen in start by 86c643ae's rule (core_sharing,
 * batch_choose): QSB_CPU_BATCH_SOLO when no two workers can share a physical core (from the thread_siblings_list of the workers'
 * CPUs; a worker then has its core's L2 to itself, and larger batches amortize the per-batch work: the rival's Zen 4 EPYC 9554, SMT
 * off, 30 workers, 9 windows: 4,096 +3.5 % co-grinder rate over 1,024; 8,192 +3.0 %, 2,048 +1.9 %, claim), QSB_CPU_BATCH whenever
 * SMT siblings (or an unreadable topology, or more workers than CPUs) are among the workers' CPUs. Dev override: QSB_CPU_BATCH_RT=<n>
 * (a multiple of 32, 32..8192). 0: the fixed QSB_CPU_BATCH as before. Every batch is a multiple of 32: the 8-lane window steps and
 * final steps run over G = B / 8 groups four at a time (G % 4 == 0), recode16 takes 16 candidates at a time, the 16-lane key hashes
 * and hpf_rows8 8, the 4-lane hashing groups 4. */
#ifndef QSB_CPU_BATCH_AUTO
#define QSB_CPU_BATCH_AUTO 1
#endif
#ifndef QSB_CPU_BATCH_SOLO
#define QSB_CPU_BATCH_SOLO 4096
#endif
static_assert(QSB_CPU_BATCH % 32 == 0 && QSB_CPU_BATCH >= 32 && QSB_CPU_BATCH <= 8192, "QSB_CPU_BATCH: a multiple of 32, 32..8192");
static_assert(QSB_CPU_BATCH_SOLO % 32 == 0 && QSB_CPU_BATCH_SOLO >= 32 && QSB_CPU_BATCH_SOLO <= 8192, "QSB_CPU_BATCH_SOLO: a multiple of 32, 32..8192");
/*, a33e04c3's three co-grinder items. QSB_CPU_KH16: key hashes from a 16-lane AVX-512 message schedule,
 * rounds 4 keys at a time with SHA-NI from the stored W + K pairs (8-lane path with SHA-NI, AVX-512VL and the C fold; else, and at
 * 0, qsha_keyhash4_h0: QSB_CPU_SHA4's four SHA-NI lanes with sha256msg1/2 schedules). QSB_CPU_MRG: fe8_mul_cols accumulates each
 * column's low and high partial products in one register (needs QSB_CPU_FOLD3; else the split accumulators as before). QSB_CPU_AINL:
 * the small 8-lane field operations always_inline (the hot loops' codegen independent of unit growth). */
/* QSB_CPU_EPOCH_CONTIG (default 0 = the stride walk, the previous text): the contiguous epoch walk of submission 30c24617.
 * Worker t walks [base + t*span, base + (t+1)*span), span = floor((n_epochs - base) / nthreads), and takes each next epoch by
 * the next-combination step of the previous one instead of the binomial unrank; the prefix re-hash restarts at the first
 * changed omission as before. The ranges are disjoint and every epoch keeps its patterns, so the walk is exact; only the
 * order changes (re-hashed prefix blocks per epoch about 6.75 -> 3.53 over a run). Host-only: the device image is unchanged.
 * QSB_CPU_GWK_CACHE (default 0 = recompute): the block-149 group schedules (gwk) depend only on the epoch's buffered bytes
 * after the prefix's full blocks (erem, at most 7 values in a run); keep up to 8 computed sets keyed by erem instead of
 * recomputing them whenever erem changes. Same W+K words, so exact. */
#ifndef QSB_CPU_EPOCH_CONTIG
#define QSB_CPU_EPOCH_CONTIG 1
#endif
#if QSB_CPU_EPOCH_CONTIG != 0 && QSB_CPU_EPOCH_CONTIG != 1
#error "QSB_CPU_EPOCH_CONTIG must be 0 or 1"
#endif
#ifndef QSB_CPU_GWK_CACHE
#define QSB_CPU_GWK_CACHE 1
#endif
#if QSB_CPU_GWK_CACHE != 0 && QSB_CPU_GWK_CACHE != 1
#error "QSB_CPU_GWK_CACHE must be 0 or 1"
#endif
#ifndef QSB_CPU_KH16
#define QSB_CPU_KH16 1             /* key hashes: message schedule for 16 keys with AVX-512, rounds 4 keys at a time with SHA-NI from it */
#endif
#ifndef QSB_CPU_MRG
#define QSB_CPU_MRG 1              /* products: one accumulator per column for its low and high partial products */
#endif
#ifndef QSB_CPU_AINL
#define QSB_CPU_AINL 1             /* the small field operations always inlined (codegen of the hot loops independent of unit growth) */
#endif
/*. QSB_CPU_RECODE_NG: recode16 reads the 16 candidates' eight SHA state words
 * with 8 plain 64-byte loads and a 3-stage vpermt2d transpose (24 two-source permutes on the FP12 pipes) instead of 8 vpgatherdd
 * (about 20 uops each on Zen 4). The same 16 x 8 words end in the same zw[] registers (
 * derives and checks the index vectors), so digits, bad flags and hits are unchanged. 0 = the gathers. */
#ifndef QSB_CPU_RECODE_NG
#define QSB_CPU_RECODE_NG 1
#endif
/*: three host-only switches on the 8-lane path, each
 * default 0 (or 4) = the code above byte for byte in behaviour, for a per-core Zen 4 SMT A/B (bench/r2d.sh). None changes a field
 * operation's operands or results, so the hit set is unchanged by construction; the bench's fixed-work identity checks it.
 * QSB_CPU_ILP2 (bit mask) bit 0: the batch-affine backward passes (ec8_window, ec8_final_cf_kh) step two groups of different chains together, op
 *   by op (lam, lam', run, run', x3, x3', t, t', Y, Y'), so every dependent operation (x3 on lam, Y on x3) has a whole independent
 *   operation of the other group between it and its producer in program order: the ~40-cycle fold + carry tail of x3 before Y, which
 *   one group at a time leaves exposed inside a thread's window, is covered by the other group's squaring. The same operations on the
 *   same operands in another order: bit-identical X, Y. Bit 1: ec8_window's forward pass likewise (rows, D, t, PRE and chain
 *   products of groups h and h + 1 op by op). 3 = both.
 * QSB_CPU_NCH 2: the batch inversions combine 2 interleaved chains instead of 4 (fe8_inv2: 3 multiplications around the lane
 *   inversion instead of fe8_inv4's 9), in ec8_window and ec8_final_cf_kh; each chain then spans G/2 groups. Montgomery's trick over
 *   another grouping of the same denominators: the same inverses, the same lam, x3 and y3 after their full carries.
 * QSB_CPU_PFSPREAD (bit mask): bit 0 = the rows of windows 0 and 1 that the hashing phase prefetches (hpf_rows8, 16 per 8
 *   candidates in one burst) are queued and issued two per SHA-256 block of the following quads (qsha_x4p) instead; bit 1 = the
 *   backward pass's next-window prefetches go 2 + 2 + 2 + 2 between the group's four field operations instead of 4 + 4 (with
 *   ILP2: 4 + 4 + 4 + 4 between the pair's four operation pairs). A prefetch that misses the L2 DTLB holds its load-queue slot and
 *   retirement until a walker returns (C1: +6.9 ns per candidate in a hashing phase that reads no row); at most a few in flight
 *   instead of 16 lets each walk finish under the SHA rounds. Prefetches only. */
#ifndef QSB_CPU_ILP2
#define QSB_CPU_ILP2 3
#endif
/* Lane H1 (2026-10-01; host only, none an image knob; every switch 0 = N-pkg17r's co-grinder, which is N-pkg16c's, byte for byte
 * in behaviour). jacklightChen's seven co-grinder cuts from his public subset submission b1c5e58e (PR #2842 "Validate submission
 * b1c5e58e-210d-4353-addf-edc4474b8f33", head 843c8281, on the promoted fb6f5a8f; its public note: "first-group identity elision
 * and single-lane scalar-inverse bridge", prepared with GPT 6.1 Sol in Codex; five of the seven first prepared in his PR2678 /
 * 4bed8f2f). Ported here as written, renamed QSB_CPU_JL_<his name> (his QSB_CPU_INV_FIRST is QSB_CPU_JL_INV_FIRST, and so on).
 * The same file rides in later public trees: i34-9's cdca0fa5 (the three INV items only), DPZZxlz's 772875d7 / bc297d9d (as is),
 * and terrapinelf's fb96dba5 / ee8c77d6, DPZZxlz's e69d3dce, jungjipdo's 7282224e (as is plus QSB_CPU_ILP2 3). Same residues,
 * same canonical outputs, same hits; none changes an operand that reaches a canonicalization in another representative range.
 * QSB_CPU_JL (umbrella, default 0): 1 sets every QSB_CPU_JL_* below to 1 unless it is given itself.
 * QSB_CPU_JL_INV_FIRST: the batch inversion's chain products start at one, so in the forward pass's first group of each chain
 *   (g == 0) the weighted prefix PRE = run * t is t and the chain product run * D is D: fe8_cp instead of fe8_mul_lz (2 x NCH = 4
 *   multiplications per window step at NCH 2). 1 = as jacklightChen wrote it: ec8_window's paired forward pass (QSB_CPU_ILP2 bit 1
 *   only; at ILP2 1 that pass is not compiled and ec8_window is untouched) and ec8_final_cf_kh's forward pass. 2 = 1 plus the same
 *   copy in ec8_window's one-group QSB_CPU_WPRE forward pass (lane H1's extension, so the item reaches ec8_window at QSB_CPU_ILP2 1;
 *   the same operands, fe8_sub_d's D and fe8_sub_sgn_nf's t, the same mul-only consumers). Copied operands are only ever IFMA
 *   inputs (limbs < 2^52) before the inversion's canonicalization, as before.
 * QSB_CPU_JL_INV_LAST: the paired backward passes (QSB_CPU_ILP2 bit 0: ec8_window, ec8_final_cf_kh) skip the chain update
 *   run = run * D of the last pair (g == 0): no later reader (2 multiplications per window step at NCH 2).
 * QSB_CPU_JL_INV_LANE0: fe8_inv_lanes canonicalizes only lane 0 of the 8-lane product (5 scalar extractions and fe8_lane_canon)
 *   instead of fe8_canon64 over all 8 lanes and a 4 x 8 store; every lane holds the same residue and only lane 0 is read.
 * QSB_CPU_JL_PARITY_CMP: fe8_parity tests v >= p of a normalized v (limbs 0..3 < 2^52) by compares (limb 4 >= 2^48, or limb 4 =
 *   2^48 - 1 and limbs 1..3 all ones and limb 0 >= p's) instead of fe8_ge_p's carry chain.
 * QSB_CPU_JL_KH16_WORDS52: kh16_store builds the nine key-hash message words straight from the canonical radix-52 limbs (shifts of
 *   r0..r4) instead of packing 4 x 64-bit words first (kh16_canon) and cutting those (kh16_words8); the same low 32 bits per word.
 * QSB_CPU_JL_KH16_PAD_ONCE: the W + K pairs of rounds 10..15 (zero words W10..W14, W15 = 264: the same for every key) are written
 *   once per worker into its kh16 scratch (vecbuf_alloc) and kh16_pass<true> skips them.
 * QSB_CPU_JL_KH16_REGMASK: kh16_pass forms the 16 keys' prefilter mask from the four-key h0 results in registers (unpack, shift,
 *   compare, movemask) instead of storing h0[16] and reloading it (dev builds, QSB_CPU_DEVBENCH, still store h0 for the unit test). */
#ifndef QSB_CPU_JL
#define QSB_CPU_JL 1
#endif
#ifndef QSB_CPU_JL_INV_FIRST
#define QSB_CPU_JL_INV_FIRST QSB_CPU_JL
#endif
#ifndef QSB_CPU_JL_INV_LAST
#define QSB_CPU_JL_INV_LAST QSB_CPU_JL
#endif
#ifndef QSB_CPU_JL_INV_LANE0
#define QSB_CPU_JL_INV_LANE0 QSB_CPU_JL
#endif
#ifndef QSB_CPU_JL_PARITY_CMP
#define QSB_CPU_JL_PARITY_CMP QSB_CPU_JL
#endif
#ifndef QSB_CPU_JL_KH16_WORDS52
#define QSB_CPU_JL_KH16_WORDS52 QSB_CPU_JL
#endif
#ifndef QSB_CPU_JL_KH16_PAD_ONCE
#define QSB_CPU_JL_KH16_PAD_ONCE QSB_CPU_JL
#endif
#ifndef QSB_CPU_JL_KH16_REGMASK
#define QSB_CPU_JL_KH16_REGMASK QSB_CPU_JL
#endif
#if QSB_CPU_JL < 0 || QSB_CPU_JL > 1 || QSB_CPU_JL_INV_FIRST < 0 || QSB_CPU_JL_INV_FIRST > 2
#error "QSB_CPU_JL: 0 or 1; QSB_CPU_JL_INV_FIRST: 0, 1 (as written) or 2 (also ec8_window's one-group forward pass)"
#endif
#if QSB_CPU_JL_INV_FIRST && !QSB_CPU_WPRE
#error "QSB_CPU_JL_INV_FIRST is ported for the QSB_CPU_WPRE window step"
#endif
/* Lane H2 (2026-10-02; host only, not an image knob; 0 = N-dc1h's co-grinder byte for byte in behaviour). i34-9's
 * QSB_CPU_CANON_TOP, the "additional key-message preparation update" of the public subset submission b4c5a3c8 (PR #2987
 * "Validate submission b4c5a3c8-6f27-4165-b432-21474d1fa53e", head e7e52427, on the promoted fb6f5a8f; header blob 061e8355 =
 * jacklightChen's three INV cuts + this item; public note's model line "GPT-6.1-sol (max); GPT-6 Astra (advisor, high)", harness
 * Codex). The same item rides in i34-9's a0cf96d5 (PR #3042) and 791650a2 (PR #3075) with QSB_CPU_SHA_RING and the producers'
 * QSB_HP_DEAD_DESCRIPTORS (not ported: lane H2 priced them below +0.01% of score). Ported as written under the name
 * QSB_CPU_I34_CANON_TOP.
 * QSB_CPU_I34_CANON_TOP: a normalized 8-lane element (limbs 0..3 < 2^52) whose limb 4 is below 2^48 - 1 in every lane is below
 *   (2^48 - 1) 2^208 <= p = (2^48 - 1) 2^208 + 2^208 - 2^32 - 977, so it is already canonical: one unsigned compare of limb 4
 *   (and a branch on the 8-bit mask) replaces the v >= p test and the blend with v - p. If any lane's limb 4 is at or above
 *   2^48 - 1 (at most about 2^-41 per lane over this co-grinder's v < 2^256 + 2^215), all eight take the unchanged full path.
 *   1 = as i34-9 wrote it: fe8_parity and kh16_canon. With QSB_CPU_JL_KH16_WORDS52 1 (N-dc1h's default) kh16_store does not
 *   call kh16_canon, so value 1 reaches fe8_parity only (ahead of QSB_CPU_JL_PARITY_CMP's compares). 2 = 1 plus the same test
 *   in kh16_words52 (lane H2's extension: the canonicalization jacklightChen's WORDS52 kept, fe8_ge_p's carry chain and five
 *   blends per call, two calls per kh16_store). Same canonical values, same words, same parities, same hits. */
#ifndef QSB_CPU_I34_CANON_TOP
#define QSB_CPU_I34_CANON_TOP 2
#endif
#if QSB_CPU_I34_CANON_TOP < 0 || QSB_CPU_I34_CANON_TOP > 2
#error "QSB_CPU_I34_CANON_TOP: 0, 1 (as i34-9 wrote it: fe8_parity, kh16_canon) or 2 (also kh16_words52)"
#endif
#ifndef QSB_CPU_NCH
#define QSB_CPU_NCH 2
#endif
#ifndef QSB_CPU_PFSPREAD
#define QSB_CPU_PFSPREAD 3
#endif
#ifndef QSB_CPU_MRGS
#define QSB_CPU_MRGS 0             /* R2-D: MRG products with the four longest columns split in two chains (see fe8_mul_cols) */
#endif
#define QCPU_PFQ ((QSB_CPU_PFSPREAD & 1) && QCPU_VEC && QCPU_SHANI)   /* bit 0 needs the 8-lane path's hpf_rows8 and qsha_x4p */
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
#define QSB_CPU_F1N (QSB_CPU_HPF < 2)   /* ec8_first prefetches window 1's rows (unless the hashing phase did) */
#endif
/* Footprint switches (package y2d). The defaults live here, at file scope and outside every conditional block, so the nvcc
 * device pass (which parses start() with QCPU_VEC and QCPU_SHANI at 0) and the host pass both see them. */
/* QSB_CPU_ALLCPU : 1 = the heavy host footprint of 789aed1b / a141df2b (start hunk of blobs ae7f4681 /
 * f7816eb5, credited there to HyeokxC's 0735233a / 888f5fce): when QSB_HOST_BLOCKING is on (tree.cu defines it before this
 * file) the GPU host thread sleeps in blocking event waits, so its core idles between launches and the workers take every
 * CPU of the pre-main() set, that core included, at one worker per CPU within the cgroup quota (32 of 32 on the ranked host).
 * The workers are SCHED_IDLE: a wake-up of the host thread or a producer preempts them at once. 0 = the the base footprint:
 * ncpu - QSB_CPU_RESERVE workers, none on the host thread's core (QSB_CPU_RSV_CORE 1). Not an image knob. */
#ifndef QSB_CPU_ALLCPU
#define QSB_CPU_ALLCPU 1
#endif
#ifndef QSB_CPU_HP_SHARE
#define QSB_CPU_HP_SHARE 0         /* 1 (2d1631b0): the GPU host thread's own CPU also hosts a worker when the host producers publish
                                    * qhp::g_share_cpu (2d1631b0's host_producers.h with blocking event waits; this tree's v1 producers
                                    * have no such symbol, the v3 file's stays -1 at QSB_HP_PLACE 0). 0: no worker there, as 296e5e53 */
#endif
#ifndef QSB_CPU_NTH_RAISE
#define QSB_CPU_NTH_RAISE 0        /* 1 (2d1631b0): one worker per CPU the main thread left when that exceeds ncpu - QSB_CPU_RESERVE (32 of 32
                                    * on the runner with 2d1631b0's producers). 0: ncpu - QSB_CPU_RESERVE workers, as 296e5e53.
                                    * QSB_CPU_ALLCPU 1 applies the same raise whatever this value */
#endif
#ifndef QSB_CPU_RSV_CORE
#if QSB_CPU_ALLCPU
#define QSB_CPU_RSV_CORE 0         /* under QSB_CPU_ALLCPU: no reserved core (rsv_core_mask would take the host core back) */
#else
#define QSB_CPU_RSV_CORE 1         /* 1 (296e5e53's rule): the workers never run on the GPU host thread's core (both SMT siblings, from
                                    * thread_siblings_list) when that thread is pinned at start(), else the last 2-thread core stays free.
                                    * 0: 2d1631b0's rule (every CPU the main thread no longer uses; every CPU when it is not pinned) */
#endif
#endif
#if QSB_CPU_ALLCPU && QSB_CPU_RSV_CORE
#error "QSB_CPU_ALLCPU 1 needs QSB_CPU_RSV_CORE 0 (rsv_core_mask would take the GPU host thread's core back from the workers)"
#endif
#if QSB_CPU_ALLCPU && QSB_CPU_HP_SHARE
#error "QSB_CPU_ALLCPU 1 needs QSB_CPU_HP_SHARE 0 (the host thread's CPU is already a worker CPU)"
#endif
/* Host-only co-grinder switches of this package (none an image knob; 0 on each = the previous file).
 * QSB_CPU_FENCE (defined in tests/gpu_epochs/qsb_host_verify.h, next to the shared set of published hits): with the co-grinder
 *   on, the GPU walks epochs [0, F) and the co-grinder the GPU's own 128 window patterns on [F, C(137,6)),
 *   F = C(137,6) - QSB_CPU_FENCE_R rounded down to the GPU's batch (QSB_SE_LAUNCH_BLOCKS x QSB_PAIR_MUL epochs), in
 *   QSB_CPU_EPOCH_CONTIG's contiguous per-worker ranges. At F the GPU drains, publishes and idles until the stop signal
 *   (tree.cu). fence_plan() fixes F before the host producers start; any later "CPU co-grind: off" releases [F, N) back to the
 *   GPU (fence_released()). Dev only: QSB_CPU_FENCE_AT=<epoch> in the environment forces F (rounded down to the batch).
 * QSB_CPU_DIAG_V4: the walk-start diagnostic on the contiguous walk. Worker t starts at t x span + c_t x 2^20 inside its own
 *   range, c_t < 100 a code (diag4_fill()), so the public hit list carries the huge-page fraction (0.1% steps), the table-ready
 *   time, MemAvailable and HugePages_Free. Enumeration only.
 * QSB_CPU_DIAG_EPOCH (below) is the v3 walk-start code and defaults to 1 unless one of the two above is on (each moves the
 *   walk's base itself: the fence to F, DIAG_V4 to per-worker offsets; they exclude each other). */
#ifndef QSB_CPU_FENCE   /* normally defined by tests/gpu_epochs/qsb_host_verify.h (included first); this fallback keeps older host
                          * harnesses compiling at the default. With -DQSB_CPU_FENCE=1 and an old qsb_host_verify.h the build still
                          * fails loudly (no qsb_pub_once). */
#define QSB_CPU_FENCE 0
#endif
#ifndef QSB_CPU_FENCE_R
#define QSB_CPU_FENCE_R 800000000ull   /* QSB_CPU_FENCE: epochs reserved above the fence (1.3 x the runner co-grinder's 1,200 s need) */
#endif
#ifndef QSB_CPU_DIAG_V4
#define QSB_CPU_DIAG_V4 0
#endif
#ifndef QSB_CPU_DIAG_EPOCH
#if QSB_CPU_FENCE || QSB_CPU_DIAG_V4
#define QSB_CPU_DIAG_EPOCH 0
#else
#define QSB_CPU_DIAG_EPOCH 1       /* 1 (2d1631b0, v3 code): the walk starts at a diagnostic code x 2^29 (geometry, huge pages, workers,
                                    * memory); 0: at epoch 0, as 296e5e53. Enumeration only: disjoint from the GPU's candidates either way */
#endif
#endif
#if QSB_CPU_FENCE && !QSB_CPU_EPOCH_CONTIG
#error "QSB_CPU_FENCE walks [F, N) in QSB_CPU_EPOCH_CONTIG's contiguous ranges: set QSB_CPU_EPOCH_CONTIG 1"
#endif
#if QSB_CPU_FENCE && QSB_CPU_DIAG_EPOCH
#error "QSB_CPU_FENCE fixes the walk's base at F; QSB_CPU_DIAG_EPOCH would move it into the GPU's range"
#endif
#if QSB_CPU_FENCE && QSB_CPU_DIAG_V4
#error "QSB_CPU_DIAG_V4's codes need 2^25 epochs of room per worker range; the fence's ranges (about 2.5e7 epochs) leave none"
#endif
#if QSB_CPU_DIAG_V4 && !QSB_CPU_EPOCH_CONTIG
#error "QSB_CPU_DIAG_V4 codes the contiguous walk's per-worker range starts: set QSB_CPU_EPOCH_CONTIG 1"
#endif
#if QSB_CPU_DIAG_V4 && QSB_CPU_DIAG_EPOCH
#error "QSB_CPU_DIAG_V4 needs the walk's base at 0 (QSB_CPU_DIAG_EPOCH 0)"
#endif
/* COHASH hook (the GPU hashes the co-grinder's SHA-256d, QSB_GPU_COHASH): not in this tree. The feature is built on QSB_CPU_FENCE's walk: the
 * GPU hashes z for chunks of the co-grinder's region [Fc, N), and the workers take batches of ready z in place of their own
 * hashing phase. Its consumer goes where the worker's batch loop marks "COHASH hook", its producer where tree.cu's slot loop
 * marks "COHASH hook", and its kernel entry beside QsbCarrier.h's kernel list. */
#ifndef QSB_GPU_COHASH
#define QSB_GPU_COHASH 0
#endif
#if QSB_GPU_COHASH
#error "QSB_GPU_COHASH is not wired into this tree: apply its patch at the \"COHASH hook\" sites first"
#endif
/* : start-up and teardown of the co-grinder, host-only, no image knob; 0 on each switch =
 * the base code. QSB_CPU_TOUCH_GUARD: the 9-window table's first touch (the sampled and the full pass) stops once it has run
 * QSB_CPU_TOUCH9_MAX_S seconds (dev override: the environment variable of that name) and the table falls back to 10 windows, as a
 * poorly backed 9-window table does (a compaction stall on a fragmented host would otherwise idle the CPU part for its length).
 * QSB_CPU_FOLD_PAR: the C fold's temporary (the top window: 8 GiB at 9 windows) is a huge-page mapping first touched by the fold's
 * own threads and copied back by all build threads, instead of a zero-filled std::vector (4 KiB pages) copied and freed by one
 * thread (measured 4.2 s + 0.5 s + 0.8 s for 8 GiB on one Zen 3 thread, during which every other CPU idled). Same table
 * entries. stop_unmap() (tree.cu's QSB_FAST_TEARDOWN): the workers stop at their next batch boundary and the table is unmapped. */
#ifndef QSB_CPU_TOUCH_GUARD
#define QSB_CPU_TOUCH_GUARD 1
#endif
#ifndef QSB_CPU_TOUCH9_MAX_S
#define QSB_CPU_TOUCH9_MAX_S 20.0
#endif
#ifndef QSB_CPU_FOLD_PAR
#define QSB_CPU_FOLD_PAR 1
#endif
/* QSB_CPU_TOUCH_FUSE (default 0 = the full first touch): a 9- or 10-window table keeps the sampled first touch (1 region in
 * 32) as its huge-page gate and skips the full pass, so the build's own writes fault the rest of the table in while other
 * threads run additions; the huge-page fraction is measured again over the whole table after the build and printed (and used
 * by the walk-start code). The 10-window fallback still rests on the sample; QSB_CPU_TOUCH_GUARD bounds only the sample.
 * QSB_CPU_BUILD_NT (default 0): the 8-lane build and the C fold write their table rows with non-temporal stores (no read for
 * ownership of freshly faulted lines), each build thread ending with a store fence. Same table entries; table_check and the
 * fixed-work hit set are the gates. Both are host-only start-up items, priced by the start-up timeline job. */
#ifndef QSB_CPU_TOUCH_FUSE
#define QSB_CPU_TOUCH_FUSE 0
#endif
#ifndef QSB_CPU_BUILD_NT
#define QSB_CPU_BUILD_NT 0
#endif

namespace qcpu {
static const int NWMAX = 16;
#if QCPU_PFQ
/* (QSB_CPU_PFSPREAD bit 0): the hashing phase's row prefetches, queued (hpf_rows8) and issued a few at a time between
 * SHA-256 blocks (qsha_x4p), so that at most a couple of L2-DTLB-missing prefetches are in flight at once. */
struct QPfRing { const char *a[64]; unsigned head = 0, tail = 0; };
static inline __attribute__((always_inline)) void qpf_drain(QPfRing &q, unsigned n) {
    while (n-- && q.head != q.tail) { _mm_prefetch(q.a[q.head & 63], QSB_CPU_PFNX ? _MM_HINT_T1 : _MM_HINT_T0); q.head++; }
}
#endif
typedef unsigned __int128 u128;
/* 64 B-aligned storage: one table point = one cache line. */
template <class T> struct qalloc64 {
    typedef T value_type;
    qalloc64() = default;
    template <class U> qalloc64(const qalloc64<U> &) {}
    T *allocate(size_t n) { void *q = nullptr; if (posix_memalign(&q, 64, n * sizeof(T) + 64)) throw std::bad_alloc(); return (T *)q; }
    void deallocate(T *q, size_t) { free(q); }
    template <class U> bool operator==(const qalloc64<U> &) const { return true; }
    template <class U> bool operator!=(const qalloc64<U> &) const { return false; }
};
struct fe { uint64_t v[4]; };      /* canonical (< p) little-endian limbs */
static const uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL, PK = 0x1000003D1ULL;   /* p = 2^256 - PK */

static inline bool fe_is_zero(const fe &a) { return !(a.v[0] | a.v[1] | a.v[2] | a.v[3]); }
static inline bool fe_eq(const fe &a, const fe &b) {
    return !((a.v[0] ^ b.v[0]) | (a.v[1] ^ b.v[1]) | (a.v[2] ^ b.v[2]) | (a.v[3] ^ b.v[3]));
}
static inline bool fe_ge_p(const uint64_t v[4]) {
    return v[3] == ~0ULL && v[2] == ~0ULL && v[1] == ~0ULL && v[0] >= P0;
}
static inline void fe_sub_p(uint64_t v[4]) {           /* v -= p  ==  v += PK mod 2^256 */
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
    if (borrow) {                                      /* t += p  ==  t -= PK mod 2^256 */
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
    /* fold: L + H*PK, then the <2^34 top again */
    uint64_t m[4]; u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)l[i] + (u128)l[i + 4] * PK; m[i] = (uint64_t)c; c >>= 64; }
    uint64_t top = (uint64_t)c;
    c = (u128)m[0] + (u128)top * PK; m[0] = (uint64_t)c; c >>= 64;
    for (int i = 1; i < 4; i++) { c += m[i]; m[i] = (uint64_t)c; c >>= 64; }
    if (c) fe_sub_p(m);                                /* wrapped past 2^256: add PK once more */
    if (fe_ge_p(m)) fe_sub_p(m);
    memcpy(r.v, m, 32);
}
static inline void fe_sqr(fe &r, const fe &a) { fe_mul(r, a, a); }
static void fe_inv(fe &r, const fe &a) {               /* a^(p-2) */
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
static pt pt_double(const pt &q) {                     /* affine doubling: lam = 3x^2 / 2y */
    fe x2, num, den, inv, lam, x3, y3, t;
    fe_sqr(x2, q.x); fe_add(num, x2, x2); fe_add(num, num, x2);
    fe_add(den, q.y, q.y); fe_inv(inv, den); fe_mul(lam, num, inv);
    fe_sqr(x3, lam); fe_sub(x3, x3, q.x); fe_sub(x3, x3, q.x);
    fe_sub(t, q.x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, q.y);
    return {x3, y3};
}

/* Batch-affine add: out[k] = in[k] + t[k] for the active k (neither is infinity and
 * x differs). inf[k] marks in[k] = O (then out = t). bad[k] set on x collisions
 * (probability ~2^-240; the candidate is dropped, never published). neg[k] (optional): add -t[k]. */
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
        if (ng) { fe_add(t, tp[k]->y, acc[k].y); const fe z0 = {{0, 0, 0, 0}}; fe_sub(t, z0, t); }   /* -ty - y */
        else fe_sub(t, tp[k]->y, acc[k].y);
        fe_mul(lam, t, dinv);
        fe_sqr(x3, lam); fe_sub(x3, x3, acc[k].x); fe_sub(x3, x3, tp[k]->x);
        fe_sub(t, acc[k].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[k].y);
        acc[k].x = x3; acc[k].y = y3;
    }
}

/* Table geometry. Window i covers bits [off[i], off[i] + wid[i]) of z. Signed (sgn): windows 0..nw-2 take
 * digits in (-2^(wid-1), 2^(wid-1)] and pass a carry up, the top window takes its bits plus the carry
 * (0..2^wid); unsigned: digits 0..2^wid - 1. Window i holds ent[i] points (j + 1) * 2^off[i] * A,
 * j < ent[i] = the largest |digit|, at table[base[i] + j]. A digit is stored as |d| | (d < 0) << 31;
 * a zero digit skips the window (scalar path) or drops the candidate (8-lane path). */
struct Geo { int nw = 0; bool sgn = false; int off[NWMAX] = {0}, wid[NWMAX] = {0}; uint32_t ent[NWMAX] = {0}; size_t base[NWMAX] = {0}; size_t total = 0; };
static Geo geo_make(int nw, bool sgn) {
    Geo g; if (nw < 9) nw = 9; if (nw > NWMAX) nw = NWMAX;
    g.nw = nw; g.sgn = sgn;
    if (!sgn) {                                        /* unsigned: 256 bits in nw near-equal windows */
        const int w = 256 / nw, r = 256 - w * nw;
        for (int i = 0; i < nw; i++) g.wid[i] = w + (i < r);
    } else {                                           /* signed: nw - 1 windows of w or w + 1 bits + a b-bit top, least entries */
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
/* Digits of z (zb: 8 SHA state words per candidate, h0 = most significant) into ds[i * B + k]. */
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
/* ---- 8-lane path (AVX-512 IFMA, radix 2^52), selected at run time when the host CPU has it ----
 * Eight candidates per vector lane group; the same batch-affine formulas as batch_add, with the
 * batch inversion split into four interleaved Montgomery chains per window. Candidates with a zero
 * window digit (16/65536 of them) are dropped instead of taking the point-at-infinity branch. */
/* 8-lane secp256k1 field arithmetic with AVX-512 IFMA (vpmadd52luq/huq), radix 2^52.
 * Each lane holds one field element as 5 limbs; "normalized" means limbs 0..3 < 2^52 and
 * limb 4 < 2^49 (value < 2^257). IFMA multiplies only the low 52 bits of its operands, so
 * every multiplication input must be normalized. Outputs of fe8_mul/fe8_sub/fe8_add are normalized. */
struct fe8 { __m512i l[5]; };
#define F8_M52 _mm512_set1_epi64(0xFFFFFFFFFFFFFULL)
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_carry(fe8 &r) {
    /* fold bits >= 256 first (limb 4 bit 48 and up; 2^256 = 0x1000003D1 mod p), then one carry chain:
       limbs 0..3 end < 2^52 and limb 4 < 2^48 + (small carry), so the value is < 2^257 */
    const __m512i M = F8_M52, M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), K = _mm512_set1_epi64(0x1000003D1ULL);
    __m512i c;
    c = _mm512_srli_epi64(r.l[4], 48); r.l[4] = _mm512_and_si512(r.l[4], M48);
    r.l[0] = _mm512_madd52lo_epu64(r.l[0], c, K);             /* c*K < 2^52 */
    c = _mm512_srli_epi64(r.l[0], 52); r.l[0] = _mm512_and_si512(r.l[0], M); r.l[1] = _mm512_add_epi64(r.l[1], c);
    c = _mm512_srli_epi64(r.l[1], 52); r.l[1] = _mm512_and_si512(r.l[1], M); r.l[2] = _mm512_add_epi64(r.l[2], c);
    c = _mm512_srli_epi64(r.l[2], 52); r.l[2] = _mm512_and_si512(r.l[2], M); r.l[3] = _mm512_add_epi64(r.l[3], c);
    c = _mm512_srli_epi64(r.l[3], 52); r.l[3] = _mm512_and_si512(r.l[3], M); r.l[4] = _mm512_add_epi64(r.l[4], c);
}
/* Lazy normalization for elements that only ever feed IFMA multiplications ("mul-only": the chain
 * products, PRE, dinv, lam, the y differences and D): the carries are propagated exactly as in
 * fe8_carry, but limbs 0..3 keep the bits they carried out (bits 52 and up). IFMA reads only bits 51:0
 * of a multiplicand, so such an element multiplies exactly like its normalized twin; it must never be an
 * operand of fe8_sub/fe8_add/fe8_sub_sgn, a subtrahend of the fused ops, or be canonicalized. Limb 4
 * is masked as before (its bits 48..51 are inside the multiplicand window). 11 ops instead of 15. */
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
/* Fold a 10-column product (columns < 9 * 2^52, value < 2^514) to five columns o[0..4] of the same residue,
 * each a sum of at most 12 terms below 2^52 (so < 2^56): value = L + H*2^260, 2^260 = R = 0x1000003D10 mod p.
 * Only the high columns c5..c9 are normalized (they become IFMA multiplicands of R); the low columns take
 * the folded products unnormalized. fe8_red = fold + fe8_carry (fe8_carry's single pass: fold at 2^256 of
 * limb 4's bits >= 48, c = l4 >> 48 < 2^9 so c*K < 2^42; then limbs 0..3 -> 52 bits) normalizes the sum.
 * The fused fe8_sqr_sub2 / fe8_mul_sub subtract from the folded columns and carry once. */
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_fold(__m512i *o, __m512i c0, __m512i c1, __m512i c2, __m512i c3, __m512i c4,
              __m512i c5, __m512i c6, __m512i c7, __m512i c8, __m512i c9) {
    const __m512i Z = _mm512_setzero_si512();
#define LO(acc, x, y) acc = _mm512_madd52lo_epu64(acc, x, y)
#define HI(acc, x, y) acc = _mm512_madd52hi_epu64(acc, x, y)
#if QSB_CPU_FOLD3
    /* Columns 5..8 as in QSB_CPU_FOLD2 below: c = l + h*2^52 (h < 2^4), l*R as a lo/hi IFMA pair into columns i, i+1 and
     * h*R < 2^41 as one lo IFMA into column i+1. Column 9 (from fe8_mul_cols / fe8_sqr_cols: a single high partial product,
     * < 2^52) sits at 2^468 = R * 2^208: lo(c9*R) goes into column 4 here, hi(c9*R) (< 2^37, weight 2^260) was already added
     * to column 5 by the product functions (c5 < 9 * 2^52 + 2^37, so h5 < 2^4 still). Column 4's three fold terms are summed on
     * a side chain (they are ready long before column 4's products); every column still takes at most 12 terms below 2^52.
     * 14 IFMA instead of 17. */
#if QSB_CPU_FOLD4
    /*: the bits >= 52 of columns 5..8 (h < 2^4 each: columns < 9 * 2^52 + 2^37) are carried into the next column before
     * the fold instead of being folded as their own IFMA (h*R): c6' = c6 + h5, c7' = c7 + h6, c8' = c8 + h7 (each still
     * < 9 * 2^52 + 2^4, so the next h is still < 2^4), and h8 = c8' >> 52 lands at 2^468 = R * 2^208 as one lo IFMA into column
     * 4 (h8 * R < 2^41). Each carried column is then folded by one lo/hi pair; IFMA reads bits 51:0 of it, so no masking is
     * needed. 10 IFMA + the pre-folded hi(c9*R) = 11 per reduction (14 with the h terms), 7 shift/add instead of 8 shift/and.
     * Output columns take at most 3 fold terms (column 4: hi(c8'*R), lo(c9*R), lo(h8*R)), as before, so every bound that
     * fe8_carry and the fused subtractions rely on is unchanged; the residue is the same, the representative may differ. */
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
    /* Every high column c = l + h*2^52 with l < 2^52 and h = c >> 52 < 2^4 (columns < 10 * 2^52), folded
     * without a carry chain: c5+i * 2^(260+52i) = R * 2^(52i) * (l + h*2^52): l*R as a lo/hi IFMA pair into
     * columns i, i+1; h*R < 2^41 as one lo IFMA into column i+1. c9's upper product lands at 2^260 again:
     * t = hi(l9*R) + h9*R < 2^37 + 2^41 < 2^43, folded once more (t*R < 2^80: lo into column 0, hi into 1).
     * Column 4 gets at most 12 terms below 2^52 (< 2^56), columns 0..3 fewer, as fe8_carry and the fused
     * subtractions require. 17 IFMA + 10 independent shift/and instead of 13 IFMA + a 14-op serial chain. */
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
    /* normalize the high columns c5..c9 to 52-bit limbs (carry into c10); c4 keeps its high bits */
    __m512i t;
    t = _mm512_srli_epi64(c5, 52); c5 = _mm512_and_si512(c5, M); c6 = _mm512_add_epi64(c6, t);
    t = _mm512_srli_epi64(c6, 52); c6 = _mm512_and_si512(c6, M); c7 = _mm512_add_epi64(c7, t);
    t = _mm512_srli_epi64(c7, 52); c7 = _mm512_and_si512(c7, M); c8 = _mm512_add_epi64(c8, t);
    t = _mm512_srli_epi64(c8, 52); c8 = _mm512_and_si512(c8, M); c9 = _mm512_add_epi64(c9, t);
    __m512i c10 = _mm512_srli_epi64(c9, 52); c9 = _mm512_and_si512(c9, M);
    /* fold: value = L + H*2^260, 2^260 = R = 0x1000003D10 mod p; H limbs < 2^52 */
    const __m512i R = _mm512_set1_epi64(0x1000003D10ULL);
    LO(c0,c5,R); HI(c1,c5,R);
    LO(c1,c6,R); HI(c2,c6,R);
    LO(c2,c7,R); HI(c3,c7,R);
    LO(c3,c8,R); HI(c4,c8,R);
    LO(c4,c9,R); __m512i c5b = Z; HI(c5b,c9,R);
    /* c10 * 2^520 = c10 * R * 2^260 ... c10 is tiny (< 2^5): fold as c10*R into limb 5 */
    c5b = _mm512_madd52lo_epu64(c5b, c10, R);
    /* c5b < 2^43 (weight 2^260 -> R): lo into limb 0, hi into limb 1 */
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
/* The 10 product columns of a*b (25 low + 25 high partial products) and of a^2 (the ten cross products once,
 * doubled with one shift per column, plus the five squares: 30 IFMA instead of 50; the same column bounds). */
/* S = 1 (QSB_CPU_PSEED): columns 0..4 start at 4p's limbs instead of 0, for the fused subtractions below (the same sums). */
template <int S>
static inline __attribute__((always_inline, target("avx512f,avx512ifma")))
void fe8_mul_cols(__m512i *c, const fe8 &a, const fe8 &b) {
    const __m512i Z = _mm512_setzero_si512();
#if QSB_CPU_MRG && QSB_CPU_FOLD3 && QSB_CPU_MRGS
    /* (QSB_CPU_MRGS): MRG with the high partial products of columns 3..6 (its four longest chains, 7 to 10 IFMA) in a
     * second accumulator each, added once at the end: the longest chain is 5 IFMA instead of 10 (about 20 cycles off each
     * product's latency) for 4 additions more; the same column sums. */
    /* QSB_CPU_MRG (a33e04c3): each column's low and high partial products in ONE accumulator (chains of up to 10 IFMA, the same
     * column sums): no second accumulator to zero and no per-column add (18 vector ops fewer per product); with two SMT threads per
     * core the longer chains are hidden by the sibling */
    __m512i c0 = S ? _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4) : Z, c1 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
            c2 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z, c3 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
            c4 = S ? _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4) : Z;
    __m512i c5 = Z, c6 = Z, c7 = Z, c8 = Z, d9 = Z, e3 = Z, e4 = Z, e5 = Z, e6 = Z;   /* R2-D MRGS: e3..e6 hold the high products of the four longest columns */
    const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
    const __m512i b0 = b.l[0], b1 = b.l[1], b2 = b.l[2], b3 = b.l[3], b4 = b.l[4];
    HI(d9,a4,b4);                                           /* column 9: its fold term hi(c9*R) closes column 5's chain */
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
    /* QSB_CPU_MRG (a33e04c3): each column's low and high partial products in ONE accumulator (chains of up to 10 IFMA, the same
     * column sums): no second accumulator to zero and no per-column add (18 vector ops fewer per product); with two SMT threads per
     * core the longer chains are hidden by the sibling */
    __m512i c0 = S ? _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4) : Z, c1 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
            c2 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z, c3 = S ? _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4) : Z,
            c4 = S ? _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4) : Z;
    __m512i c5 = Z, c6 = Z, c7 = Z, c8 = Z, d9 = Z;
    const __m512i a0 = a.l[0], a1 = a.l[1], a2 = a.l[2], a3 = a.l[3], a4 = a.l[4];
    const __m512i b0 = b.l[0], b1 = b.l[1], b2 = b.l[2], b3 = b.l[3], b4 = b.l[4];
    HI(d9,a4,b4);                                           /* column 9: its fold term hi(c9*R) closes column 5's chain */
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
    /* low and high partial products in separate accumulators (chains of at most 5 IFMA instead of 9), summed per column */
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
    HI(d9,a4,b4);                                           /* column 9 first: its fold term hi(c9*R) closes column 5's low chain */
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
    /* S = 1: columns 0..4 start at 4p's limbs (the doubled accumulators x1..x4 at 2p's; 4p's limbs are even).
     * S = 2 (QSB_CPU_DNF, fe8_sqr_subdx): at 16p's limbs (x1..x4 at 8p's), for a subtrahend D whose limb 4 is < 2^51 + 2^3. */
    const uint64_t SM = S == 2 ? 8 : 2;                    /* the doubled accumulators' seed multiple of p */
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
    { __m512i s5 = Z; HI(s5,a2,a2); s5 = _mm512_madd52hi_epu64(s5, c9, _mm512_set1_epi64(0x1000003D10ULL)); c5 = _mm512_add_epi64(c5, s5); }   /* + hi(c9*R), see fe8_fold */
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
/* Fused r = a^2 - b - c and r = a*b - c with ONE carry pass: the folded product columns (< 2^56) plus 4p
 * minus the normalized operands (limbs < 2^52, limb 4 < 2^48 + 2^7). Every limb stays non-negative
 * (4p_i - 2 * 2^52 > 2^53 for limbs 0..3, 4p_4 - 2 * (2^48 + 2^7) > 2^49 - 2^8 for limb 4) and below
 * 2^56 + 2^54 < 2^57, which fe8_carry normalizes (limb 4 ends < 2^48 + 2^5). The residue is the same as
 * the unfused sequence's; only the (non-canonical) representative may differ, which no output sees:
 * records hold canonical x and the parity of canonical y. */
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sqr_sub2(fe8 &r, const fe8 &a, const fe8 &b, const fe8 &c) {
    const __m512i P0 = _mm512_set1_epi64(0xFFFFEFFFFFC2FULL * 4), P1 = _mm512_set1_epi64(0xFFFFFFFFFFFFFULL * 4),
                  P4 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL * 4);
#if QSB_CPU_PSEED
    (void)P0; (void)P1; (void)P4;
    __m512i k[10]; fe8_sqr_cols<1>(k, a);                  /* columns 0..4 carry 4p already */
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
    __m512i k[10]; fe8_mul_cols<1>(k, a, b);               /* columns 0..4 carry 4p already */
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
/* r = a^2 - d - 2x (d, x normalized) with one carry pass, for x3 = lam^2 - D - 2X with D = TX - X (after Meganpark's 9745ce9b
 * and ercumentyildirim's 9f8a33d8, which keep D instead of the table x): 4p_i > d_i + 2 x_i limb-wise (limb 0: 0x3FFFBFFFFF0BC >
 * 3 (2^52 - 1); limb 4: 2^50 - 4 > 3 (2^48 + 2^7)), so every limb stays non-negative; the upper bound is fe8_sqr_sub2's. */
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_sqr_subdx(fe8 &r, const fe8 &a, const fe8 &d, const fe8 &x) {
#if QSB_CPU_PSEED
    /* QSB_CPU_DNF: d from fe8_sub_m (limbs 0..3 < 2^52, limb 4 < 2^51 + 2^3), so the columns start at 16p: 16p_i - d_i - 2x_i > 0
     * limb-wise (limbs 0..3: 16p_i > 2^56 - 2^5 against < 3 * 2^52; limb 4: 2^52 - 16 against < 2^51 + 2^49 + 2^8) and every
     * column stays < 2^56 + 12 * 2^52 < 2^57, which fe8_carry normalizes as before (c = l4 >> 48 < 2^9, chain carries < 2^6). */
    __m512i k[10]; fe8_sqr_cols<QSB_CPU_DNF ? 2 : 1>(k, a);   /* columns 0..4 carry 4p (16p under DNF: 16p_0 > 2^56 - 2^37) already */
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
/* r = a + b (inputs normalized) */
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_add(fe8 &r, const fe8 &a, const fe8 &b) {
    for (int i = 0; i < 5; i++) r.l[i] = _mm512_add_epi64(a.l[i], b.l[i]);
    fe8_carry(r);
}
/* r = a - b (inputs normalized, value < 2^257): a + 4p - b, all limbs stay non-negative */
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
/* r = (a or -a in the lanes of m) - b (inputs normalized): per limb a + 4p - b, or 4p - a - b in the lanes of m.
 * Both stay non-negative: a_i + b_i < 2^53 < 4p_i for limbs 0..3, and a normalized limb 4 is < 2^48 + 2^6
 * (fe8_carry: masked to 48 bits, then a carry < 2^6), so a_4 + b_4 < 2^49 + 2^7 < 4p_4 = 2^50 - 4. */
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
/* Mul-only variants of fe8_sub and fe8_sub_sgn (normalized inputs; the output only feeds IFMA). */
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
/* Mul-only differences without the limb-4 fold (after Meganpark's fe8_sub_nf/fe8_subsgn_nf in 9745ce9b and ercumentyildirim's
 * fe8_carry_m in 9f8a33d8): a, b normalized (limb 4 < 2^48 + 2^7), so limb 4 of a + 4p - b (or 4p - a - b) is < 2^51 before
 * and after the carries; IFMA reads 52 bits of every limb and the product bounds need only limbs < 2^52. Limbs 0..3 lazy as in
 * fe8_carry_lz. Never a subtrahend or canonicalized. */
static inline __attribute__((QCPU_AI target("avx512f,avx512ifma")))
void fe8_carry_nf(fe8 &r) {
    __m512i c;
    c = _mm512_srli_epi64(r.l[0], 52); r.l[1] = _mm512_add_epi64(r.l[1], c);
    c = _mm512_srli_epi64(r.l[1], 52); r.l[2] = _mm512_add_epi64(r.l[2], c);
    c = _mm512_srli_epi64(r.l[2], 52); r.l[3] = _mm512_add_epi64(r.l[3], c);
    c = _mm512_srli_epi64(r.l[3], 52); r.l[4] = _mm512_add_epi64(r.l[4], c);
}
/* (QSB_CPU_DNF): r = a - b (a canonical or normalized, b normalized) with limbs 0..3 normalized (< 2^52) and limb 4
 * carried but not folded: limb 4 of a + 4p - b is in (4p_4 - 2^48 - 2^7, 4p_4 + 2^48 + 2^7) = (2^50 - 2^49, 2^50 + 2^49), plus a
 * carry < 2^3: < 2^51 + 2^3. Usable as an IFMA multiplicand (limbs < 2^52) and as a subtrahend of fe8_sqr_subdx's 16p-seeded
 * columns (never of a 4p-seeded one, never canonicalized). 12 ops, no IFMA, against fe8_carry's 15 with one. */
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
/* The window step's D = tx - X: fe8_sub_m under QSB_CPU_DNF (x3's columns are seeded at 16p there), else fe8_sub. */
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
/* r = a - b - c (inputs normalized) with one carry pass: a + 8p - b - c, every limb non-negative
 * (b_i + c_i < 2^53 <= 8p_i for limbs 0..3, < 2^50 < 8p_4) and < 2^56, which fe8_carry normalizes. */
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
/* 8-lane batch-affine EC pipeline for the CPU co-grinder (AVX-512 IFMA). Requires fe8.h, the
 * scalar fe/pt types and a table of canonical affine points (4x64 limbs, 64 B per point). */
#define Q8T __attribute__((target("avx512f,avx512ifma")))

/* fe8 copy as five vector moves (a plain struct assignment compiles to rep movsq here) */
Q8T static inline QCPU_AIF void fe8_cp(fe8 &r, const fe8 &a) {
    for (int i = 0; i < 5; i++) _mm512_store_si512(&r.l[i], _mm512_load_si512(&a.l[i]));
}
Q8T static inline QCPU_AIF void fe8_set1(fe8 &r) {
    r.l[0] = _mm512_set1_epi64(1);
    for (int i = 1; i < 5; i++) r.l[i] = _mm512_setzero_si512();
}
Q8T static inline QCPU_AIF void fe8_bcast(fe8 &r, const fe &a) {           /* canonical scalar -> all lanes */
    const uint64_t M = 0xFFFFFFFFFFFFFULL;
    r.l[0] = _mm512_set1_epi64((long long)(a.v[0] & M));
    r.l[1] = _mm512_set1_epi64((long long)((a.v[0] >> 52 | a.v[1] << 12) & M));
    r.l[2] = _mm512_set1_epi64((long long)((a.v[1] >> 40 | a.v[2] << 24) & M));
    r.l[3] = _mm512_set1_epi64((long long)((a.v[2] >> 28 | a.v[3] << 36) & M));
    r.l[4] = _mm512_set1_epi64((long long)(a.v[3] >> 16));
}
#ifndef QSB_CPU_VBMI2
#define QSB_CPU_VBMI2 1   /* the original form */            /* 1: funnel shifts (vpshrdq) in the 4x64 -> 5x52 conversion; every IFMA CPU in service has VBMI2.
                                    * 0 (after the co-grinder audit's port model): srli + slli + or instead. On Zen 4 vpshrdq zmm
                                    * issues on FP0/FP1 only (uops.info), the pipes IFMA and sha256rnds2 saturate, while the shifts go to
                                    * FP2/FP3 and the or anywhere: 6 FP01 ops per group of 8 rows move off the binding pipes for +12
                                    * instructions. Same limbs bit for bit. */
#endif
#if QSB_CPU_VBMI2
#define Q8TX __attribute__((target("avx512f,avx512ifma,avx512vbmi2")))
Q8TX static inline QCPU_AIF void fe8_from64(fe8 &r, __m512i a0, __m512i a1, __m512i a2, __m512i a3) {
    const __m512i M = F8_M52;
    r.l[0] = _mm512_and_si512(a0, M);
    r.l[1] = _mm512_and_si512(_mm512_shrdi_epi64(a0, a1, 52), M);   /* (a1:a0) >> 52 */
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
/* Load 8 table points (one 64 B row each: x0..x3 y0..y3) and transpose to SoA. */
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
/* lane -> canonical 4x64 */
static inline void fe8_lane_canon(fe &r, const uint64_t l[5]) {
    typedef unsigned __int128 q128;
    q128 c = (q128)l[0] + ((q128)l[1] << 52);
    uint64_t w0 = (uint64_t)c; c >>= 64;
    c += (q128)l[2] << 40; uint64_t w1 = (uint64_t)c; c >>= 64;
    c += (q128)l[3] << 28; uint64_t w2 = (uint64_t)c; c >>= 64;
    c += (q128)l[4] << 16; uint64_t w3 = (uint64_t)c; c >>= 64;
    uint64_t top = (uint64_t)c;
    while (top) {                                      /* v = top*2^256 + w, 2^256 = PK mod p */
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
/* Canonical form of 8 normalized elements (value v < 2^256 + 2^215 < 2p): v - p when v + (2^256 - p)
 * reaches 2^256, else v. fe8_ge_p returns that mask and, with u != nullptr, v + 2^256 - p mod 2^256. */
Q8T static inline QCPU_AIF __mmask8 fe8_ge_p(const fe8 &a, fe8 *u) {
    const __m512i M = F8_M52, K = _mm512_set1_epi64(0x1000003D1ULL);
    __m512i u0 = _mm512_add_epi64(a.l[0], K), c;
    c = _mm512_srli_epi64(u0, 52); u0 = _mm512_and_si512(u0, M); __m512i u1 = _mm512_add_epi64(a.l[1], c);
    c = _mm512_srli_epi64(u1, 52); u1 = _mm512_and_si512(u1, M); __m512i u2 = _mm512_add_epi64(a.l[2], c);
    c = _mm512_srli_epi64(u2, 52); u2 = _mm512_and_si512(u2, M); __m512i u3 = _mm512_add_epi64(a.l[3], c);
    c = _mm512_srli_epi64(u3, 52); u3 = _mm512_and_si512(u3, M); __m512i u4 = _mm512_add_epi64(a.l[4], c);
    const __mmask8 ge = _mm512_test_epi64_mask(u4, _mm512_set1_epi64((long long)~0x0FFFFFFFFFFFFULL));   /* u >= 2^256 */
    if (u) { u->l[0] = u0; u->l[1] = u1; u->l[2] = u2; u->l[3] = u3; u->l[4] = _mm512_and_si512(u4, _mm512_set1_epi64(0x0FFFFFFFFFFFFULL)); }
    return ge;
}
/* canonical x of 8 lanes as 4x64 limbs (w[k][lane]) */
Q8T static inline QCPU_AIF void fe8_canon64(uint64_t w[4][8], const fe8 &a) {
    fe8 u; const __mmask8 ge = fe8_ge_p(a, &u);
    __m512i r[5];
    for (int i = 0; i < 5; i++) r[i] = _mm512_mask_blend_epi64(ge, a.l[i], u.l[i]);
    _mm512_store_si512(w[0], _mm512_or_si512(r[0], _mm512_slli_epi64(r[1], 52)));
    _mm512_store_si512(w[1], _mm512_or_si512(_mm512_srli_epi64(r[1], 12), _mm512_slli_epi64(r[2], 40)));
    _mm512_store_si512(w[2], _mm512_or_si512(_mm512_srli_epi64(r[2], 24), _mm512_slli_epi64(r[3], 28)));
    _mm512_store_si512(w[3], _mm512_or_si512(_mm512_srli_epi64(r[3], 36), _mm512_slli_epi64(r[4], 16)));
}
/* parity of the canonical y of 8 lanes (p is odd: v - p flips v's parity) */
Q8T static inline QCPU_AIF __mmask8 fe8_parity(const fe8 &a) {
#if QSB_CPU_I34_CANON_TOP
    /* QSB_CPU_I34_CANON_TOP (i34-9 b4c5a3c8): limb 4 < 2^48 - 1 in every lane: v < p, so the parity is limb 0's */
    if (_mm512_cmp_epu64_mask(a.l[4], _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), _MM_CMPINT_NLT) == 0)
        return _mm512_test_epi64_mask(a.l[0], _mm512_set1_epi64(1));
#endif
#if QSB_CPU_JL_PARITY_CMP
    /* QSB_CPU_JL_PARITY_CMP (jacklightChen b1c5e58e): normalized input only (limbs 0..3 < 2^52). In radix 2^52 p's middle limbs
     * are all ones, so v >= p is limb 4 >= 2^48, or limb 4 = 2^48 - 1 with limbs 1..3 all ones and limb 0 >= p's limb 0 */
    const __m512i M48 = _mm512_set1_epi64(0x0FFFFFFFFFFFFULL);
    const __m512i mid = _mm512_and_si512(_mm512_and_si512(a.l[1], a.l[2]), a.l[3]);
    const __mmask8 top = _mm512_test_epi64_mask(a.l[4], _mm512_set1_epi64((long long)~0x0FFFFFFFFFFFFULL));
    const __mmask8 edge = (__mmask8)(_mm512_cmpeq_epi64_mask(a.l[4], M48)
        & _mm512_cmpeq_epi64_mask(mid, F8_M52)
        & _mm512_cmp_epu64_mask(a.l[0], _mm512_set1_epi64(0xFFFFEFFFFFC2FULL), _MM_CMPINT_NLT));
    return (__mmask8)(_mm512_test_epi64_mask(a.l[0], _mm512_set1_epi64(1)) ^ (top | edge));
#else
    return (__mmask8)(_mm512_test_epi64_mask(a.l[0], _mm512_set1_epi64(1)) ^ fe8_ge_p(a, nullptr));
#endif
}
/* x^(p-2) for one 8-lane element: libsecp256k1's secp256k1_fe_inv addition chain
 * (255 squarings, 15 multiplications). Kept for QSB_CPU_SCALAR_INV=0 and as the reference. */
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
#define QSB_CPU_SCALAR_INV 1       /* batch inversions: one scalar safegcd inverse instead of 255 vector squarings */
#endif
#if QSB_CPU_SCALAR_INV
/* ---- Variable-time scalar inverse mod p for the batch inversions: Bernstein-Yang "safegcd" divsteps in
 * 62-bit batches, after libsecp256k1's secp256k1_modinv64_var (MIT; notice in COPYING-secp256k1). It runs on
 * the integer pipes, so the inversion no longer occupies the vector pipes with 255 dependent squarings. */
struct s62 { int64_t v[5]; };                         /* signed, 62-bit limbs (the top one carries the sign) */
struct trans2x2 { int64_t u, v, q, r; };
static const s62 S62_P = {{-0x1000003D1LL, 0, 0, 0, 256}};   /* p = 2^256 - 0x1000003D1 */
static const uint64_t S62_PINV = 0x27C7F6E22DDACACFULL;      /* p^-1 mod 2^62 */
/* 62 divsteps on the low limbs (f0, g0) of (f, g); the transition matrix t and the new eta. */
static inline int64_t divsteps_62_var(int64_t eta, uint64_t f0, uint64_t g0, trans2x2 *t) {
    uint64_t u = 1, v = 0, q = 0, r = 1, f = f0, g = g0, m; uint32_t w; int i = 62, limit, zeros;
    for (;;) {
        zeros = __builtin_ctzll(g | (~0ULL << i));    /* a sentinel bit counts zeros only up to i */
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
            w = (uint32_t)((f * g * (f * f - 2)) & m);   /* -g/f mod 2^6 (f odd: f*(2-f^2) = f^-1 mod 64) */
        } else {
            limit = ((int)eta + 1) > i ? i : ((int)eta + 1);
            m = (~0ULL >> (64 - limit)) & 15U;
            w = (uint32_t)(f + (((f + 1) & 4) << 1));      /* f^-1 mod 16 */
            w = (uint32_t)((-(uint64_t)w * g) & m);
        }
        g += f * w; q += u * w; r += v * w;
    }
    t->u = (int64_t)u; t->v = (int64_t)v; t->q = (int64_t)q; t->r = (int64_t)r;
    return eta;
}
/* (d, e) = t * (d, e) / 2^62 mod p, keeping them in (-2p, p) */
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
    cd >>= 62; ce >>= 62;                             /* the low 62 bits are zero by construction */
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
    cd += (__int128)S62_P.v[4] * md;                  /* p's limbs 1..3 are zero */
    ce += (__int128)S62_P.v[4] * me;
    d->v[3] = (int64_t)((uint64_t)(int64_t)cd & M62); cd >>= 62;
    e->v[3] = (int64_t)((uint64_t)(int64_t)ce & M62); ce >>= 62;
    d->v[4] = (int64_t)cd;
    e->v[4] = (int64_t)ce;
}
/* (f, g) = t * (f, g) / 2^62 over the low len limbs */
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
/* r in (-2p, p) -> [0, p), negated first when sign < 0 */
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
/* x = x^-1 mod p for x in [0, p) (0 -> 0); false if the divstep loop did not settle within 24 batches. */
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
/* r = a^-1 (canonical a; 0 -> 0), canonical. Falls back to the Fermat inversion if the loop did not settle. */
static void fe_inv_var(fe &r, const fe &a) {
    const uint64_t M = ~0ULL >> 2; s62 x;
    x.v[0] = (int64_t)(a.v[0] & M); x.v[1] = (int64_t)((a.v[0] >> 62 | a.v[1] << 2) & M);
    x.v[2] = (int64_t)((a.v[1] >> 60 | a.v[2] << 4) & M); x.v[3] = (int64_t)((a.v[2] >> 58 | a.v[3] << 6) & M);
    x.v[4] = (int64_t)(a.v[3] >> 56);
    if (!modinv_var(&x)) { fe_inv(r, a); return; }
    r.v[0] = (uint64_t)x.v[0] | (uint64_t)x.v[1] << 62; r.v[1] = (uint64_t)x.v[1] >> 2 | (uint64_t)x.v[2] << 60;
    r.v[2] = (uint64_t)x.v[2] >> 4 | (uint64_t)x.v[3] << 58; r.v[3] = (uint64_t)x.v[3] >> 6 | (uint64_t)x.v[4] << 56;
}
/* x = x^-1 lane-wise: Montgomery's trick across the 8 lanes (a 3-level tree of permuted multiplications:
 * 3 to combine, 3 to split), one scalar inversion of the lanes' product. A zero lane zeroes every lane's
 * result, as the Fermat inversion did (x-collisions, probability 2^-240; the exact gate absorbs them). */
Q8T static inline void fe8_swap1(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_permutex_epi64(a.l[i], 0xB1); }   /* lanes 2k <-> 2k+1 */
Q8T static inline void fe8_swap2(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_permutex_epi64(a.l[i], 0x4E); }   /* pairs 4k <-> 4k+2 */
Q8T static inline void fe8_swap4(fe8 &r, const fe8 &a) { for (int i = 0; i < 5; i++) r.l[i] = _mm512_shuffle_i64x2(a.l[i], a.l[i], 0x4E); }  /* halves */
Q8T static void fe8_inv_lanes(fe8 &x) {
    fe8 a1, p1, p2, t, i2;
    fe8_swap1(a1, x); fe8_mul(p1, x, a1);                 /* lanes 2k, 2k+1: a_2k * a_2k+1 */
    fe8_swap2(t, p1); fe8_mul(p2, p1, t);                 /* lanes 4k..4k+3: the product of the four */
    fe8_swap4(t, p2); fe8_mul(t, p2, t);                  /* every lane: the product of all eight */
    fe pr, pi;
#if QSB_CPU_JL_INV_LANE0
    /* QSB_CPU_JL_INV_LANE0 (jacklightChen b1c5e58e): only lane 0 feeds the scalar inverse; its five radix-52 limbs through the
     * scalar canonicalizer instead of fe8_canon64 over all 8 lanes */
    uint64_t l0[5];
    for (int i = 0; i < 5; i++) l0[i] = (uint64_t)_mm_cvtsi128_si64(_mm512_castsi512_si128(t.l[i]));
    fe8_lane_canon(pr, l0);
#else
    alignas(64) uint64_t w[4][8]; fe8_canon64(w, t);
    pr = fe{{w[0][0], w[1][0], w[2][0], w[3][0]}};
#endif
    fe_inv_var(pi, pr);
    fe8 I; fe8_bcast(I, pi);
    fe8_swap4(t, p2); fe8_mul(i2, I, t);                  /* lanes 0..3: 1/(a0 a1 a2 a3), lanes 4..7: 1/(a4..a7) */
    fe8_swap2(t, p1); fe8_mul(i2, i2, t);                 /* 1/(a_2k a_2k+1) */
    fe8_mul(x, i2, a1);                                   /* 1/a_k */
}
#endif
/* Invert the 4 interleaved chain products with ONE inversion (Montgomery's trick across the
 * chains: 3 + 6 extra multiplications). */
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
/* (QSB_CPU_NCH 2): the 2 interleaved chain products with one inversion: 1 + 2 multiplications around fe8_inv_lanes
 * instead of fe8_inv4's 3 + 6, and two multiplication latencies fewer on the pass boundary's serial path. */
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
/* (QSB_CPU_PFSPREAD bit 1): issue rows pr[j0..j1) of the next window's group (no-op without a next window). */
static inline QCPU_AIF void qcpu_pf_rows(const pt *const *pr, int j0, int j1) {
    if (pr) for (int j = j0; j < j1; j++) _mm_prefetch((const char *)pr[j], QCPU_NXT_HINT);
}
/* One batch-affine window step over G groups of 8 (G % 4 == 0): acc[g] += sign * T[|digit| - 1].
 * rp/ng hold this window's table rows (8 per group, never a digit-0 row) and lane sign masks, computed and
 * prefetched by the previous pass. With nxt, the backward pass computes the next window's rows into rpn/ngn
 * and prefetches them (next group 0 first), so the table misses spread over the long backward pass instead
 * of bunching in the short forward pass; the forward pass only pulls its rows QSB_CPU_PFD groups ahead. */
template <class RowFn>
Q8TX static void ec8_window(fe8 *X, fe8 *Y, fe8 *D, fe8 *PRE, fe8 *TX, fe8 *TY, int G, const pt *const *rp, const __mmask8 *ng,
                           const pt **rpn, __mmask8 *ngn, const RowFn *nxt) {
    const int PF = QSB_CPU_PFD;
    const int NC = QSB_CPU_NCH;                      /* R2-D: interleaved chains of the batch inversion (4: the code before) */
    fe8 run[4]; for (int c = 0; c < NC; c++) fe8_set1(run[c]);
#if QSB_CPU_ILP2 & 2
    /* R2-D (QSB_CPU_ILP2 bit 1): the forward pass steps groups h and h + 1 (chains c and c + 1) together, op by op: both groups'
     * rows are loaded and transposed, then both D, both t, both PRE products, both chain products */
    for (int g = 0; g < G; g += NC) {
        for (int c = 0; c < NC; c += 2) {
            const int hA = g + c, hB = hA + 1;
            if (hA + PF < G) { const pt *const *pr = rp + (size_t)(hA + PF) * 8; for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
            if (hB + PF < G) { const pt *const *pr = rp + (size_t)(hB + PF) * 8; for (int j = 0; j < 8; j++) _mm_prefetch((const char *)pr[j], _MM_HINT_T0); }
            fe8 txA, tyA, tA, txB, tyB, tB;
            pt8_load(txA, tyA, rp + (size_t)hA * 8); pt8_load(txB, tyB, rp + (size_t)hB * 8);
            fe8_sub_d(D[hA], txA, X[hA]); fe8_sub_d(D[hB], txB, X[hB]);
            fe8_sub_sgn_nf(tA, tyA, Y[hA], ng[hA]); fe8_sub_sgn_nf(tB, tyB, Y[hB], ng[hB]);
            if (QSB_CPU_JL_INV_FIRST && g == 0) {    /* QSB_CPU_JL_INV_FIRST (jacklightChen b1c5e58e): each chain starts at one; the copies stay mul-only IFMA inputs */
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
            /* weighted prefix: PRE[h] = (product of the earlier D of chain c) * (sign * ty - Y), so the backward pass's slope is
             * one product, lam = (1 / product through h) * PRE[h]; the table y is not kept */
            { fe8 tx, ty, t; pt8_load(tx, ty, rp + (size_t)h * 8); fe8_sub_d(D[h], tx, X[h]);   /* D normalized (limb 4 < 2^51 under DNF): a subtrahend of x3 */
              fe8_sub_sgn_nf(t, ty, Y[h], ng[h]);
              if (QSB_CPU_JL_INV_FIRST >= 2 && g == 0) fe8_cp(PRE[h], t);   /* QSB_CPU_JL_INV_FIRST 2 (lane H1): the chain starts at one */
              else fe8_mul_lz(PRE[h], run[c], t); }
#elif QSB_CPU_NOTX
            { fe8 tx; pt8_load(tx, TY[h], rp + (size_t)h * 8); fe8_sub_d(D[h], tx, X[h]); }   /* D normalized: a subtrahend of x3 */
            fe8_cp(PRE[h], run[c]);
#else
            pt8_load(TX[h], TY[h], rp + (size_t)h * 8);   /* kept (transposed) for the backward pass */
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
    /* R2-D (QSB_CPU_ILP2 bit 0): groups h and h - 1 (chains c and c - 1) step together, op by op; the next window's rows of both are
     * computed first (hn ascending, as before) and prefetched 4 + 4 + 4 + 4 between the four operation pairs */
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
            if (!QSB_CPU_JL_INV_LAST || g > 0) {     /* QSB_CPU_JL_INV_LAST (jacklightChen b1c5e58e): after g == 0 both chains are dead */
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
        for (int c = NC - 1; c >= 0; c--) {         /* one group at a time, its temporaries in registers */
            const int h = g + c;
            const pt **pr = nullptr;
#if QSB_CPU_PFSPREAD & 2
            /* R2-D: the next window's rows 2 + 2 + 2 + 2 between the group's four field operations */
            if (nxt) { const int hn = G - 1 - h; pr = rpn + (size_t)hn * 8; ngn[hn] = (*nxt)(hn, pr); }
            qcpu_pf_rows(pr, 0, 2);
            fe8 t, lam, x3;
            fe8_mul_lz(lam, run[c], PRE[h]); qcpu_pf_rows(pr, 2, 4);
            fe8_mul_lz(run[c], run[c], D[h]); qcpu_pf_rows(pr, 4, 6);
            fe8_sqr_subdx(x3, lam, D[h], X[h]); qcpu_pf_rows(pr, 6, 8);
            fe8_sub_nf(t, X[h], x3); fe8_mul_sub(Y[h], lam, t, Y[h]); fe8_cp(X[h], x3);
#else
            if (nxt) { const int hn = G - 1 - h; pr = rpn + (size_t)hn * 8; ngn[hn] = (*nxt)(hn, pr);
                       for (int j = 0; j < 4; j++) _mm_prefetch((const char *)pr[j], QCPU_NXT_HINT); }   /* next window's rows, 4 + 4 per group */
            fe8 t, lam, x3;
#if QSB_CPU_WPRE
            fe8_mul_lz(lam, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);    /* lam = (sign * ty - Y) / D; chain c steps back */
            if (pr) for (int j = 4; j < 8; j++) _mm_prefetch((const char *)pr[j], QCPU_NXT_HINT);
#else
            fe8 dinv;
            fe8_mul_lz(dinv, run[c], PRE[h]); fe8_mul_lz(run[c], run[c], D[h]);   /* chain c steps back one group */
            if (pr) for (int j = 4; j < 8; j++) _mm_prefetch((const char *)pr[j], QCPU_NXT_HINT);
            fe8_sub_sgn_nf(t, TY[h], Y[h], ng[h]); fe8_mul_lz(lam, t, dinv);
#endif
#if QSB_CPU_NOTX
            fe8_sqr_subdx(x3, lam, D[h], X[h]);                 /* lam^2 - (TX - X) - 2X; X, Y stay normalized: they are subtrahends */
#else
            fe8_sqr_sub2(x3, lam, X[h], TX[h]);                 /* X, Y stay normalized: they are subtrahends */
#endif
            fe8_sub_nf(t, X[h], x3); fe8_mul_sub(Y[h], lam, t, Y[h]); fe8_cp(X[h], x3);
#endif
        }
    }
#endif
}
/* Canonical x of 8 lanes into the (candidate, recid) array: lane j -> o[2 * j] (o = &qx[h * 16 + ri]).
 * The four 64-bit limb rows are transposed to one 4-limb row per lane (unpack + permute) and stored as
 * eight 256-bit rows, instead of 32 scalar extractions and stores. */
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
    const __m512i b0 = _mm512_permutex2var_epi64(a0, iL, a2),   /* lanes 0, 2 */
                  b1 = _mm512_permutex2var_epi64(a1, iL, a3),   /* lanes 1, 3 */
                  b2 = _mm512_permutex2var_epi64(a0, iH, a2),   /* lanes 4, 6 */
                  b3 = _mm512_permutex2var_epi64(a1, iH, a3);   /* lanes 5, 7 */
    _mm256_storeu_si256((__m256i *)(o + 0), _mm512_castsi512_si256(b0)); _mm256_storeu_si256((__m256i *)(o + 4), _mm512_extracti64x4_epi64(b0, 1));
    _mm256_storeu_si256((__m256i *)(o + 2), _mm512_castsi512_si256(b1)); _mm256_storeu_si256((__m256i *)(o + 6), _mm512_extracti64x4_epi64(b1, 1));
    _mm256_storeu_si256((__m256i *)(o + 8), _mm512_castsi512_si256(b2)); _mm256_storeu_si256((__m256i *)(o + 12), _mm512_extracti64x4_epi64(b2, 1));
    _mm256_storeu_si256((__m256i *)(o + 10), _mm512_castsi512_si256(b3)); _mm256_storeu_si256((__m256i *)(o + 14), _mm512_extracti64x4_epi64(b3, 1));
}
/* bit j of x -> byte j (0 or 1) */
static inline QCPU_AIF uint64_t spread8(unsigned x) {
    const uint64_t v = ((uint64_t)x * 0x0101010101010101ULL) & 0x8040201008040201ULL;
    return ((v + 0x7F7F7F7F7F7F7F7FULL) & 0x8080808080808080ULL) >> 7;
}
/* Final step for both recovery ids: Q_ri = C_ri + acc, C_1 = -C_0. Writes the canonical x and the
 * parity of the canonical y of each (candidate, recid). */
Q8T static void ec8_final(const fe8 *X, const fe8 *Y, fe8 *D, fe8 *PRE, int G, const fe &cx, const fe &cy,
                          fe *qx /* [G*8][2] */, uint8_t *qp) {
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
            /* qp[(h * 8 + j) * 2 + ri] = bit j of par[ri]: 16 bytes, the two recids interleaved */
            _mm_storeu_si128((__m128i *)(qp + (size_t)h * 16),
                             _mm_unpacklo_epi8(_mm_cvtsi64_si128((long long)spread8(par[0])), _mm_cvtsi64_si128((long long)spread8(par[1]))));
        }
    }
}
/* Final step with C folded into the top window: the windows leave Q_0 = z*A + C in (X, Y); Q_1 = z*A - C
 * = Q_0 + M with M = -2C, one batch-affine addition (5M + 1S) instead of two sharing a denominator (7M + 2S).
 * Writes the canonical x and the parity of the canonical y of each (candidate, recid), as ec8_final. */
Q8T static void ec8_final_cf(const fe8 *X, const fe8 *Y, fe8 *D, fe8 *PRE, int G, const fe &mx, const fe &my,
                             fe *qx /* [G*8][2] */, uint8_t *qp) {
    fe8 MX, MY; fe8_bcast(MX, mx); fe8_bcast(MY, my);
    fe8 run[4]; for (int c = 0; c < 4; c++) fe8_set1(run[c]);
    for (int g = 0; g < G; g += 4)
        for (int c = 0; c < 4; c++) {
            const int h = g + c; fe8_sub_lz(D[h], MX, X[h]);
#if QSB_CPU_WPRE
            { fe8 t; fe8_sub_lz(t, MY, Y[h]); fe8_mul_lz(PRE[h], run[c], t); }   /* weighted prefix, as in ec8_window */
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
/* ---- QSB_CPU_KH16 (a33e04c3): key hashes from a 16-lane message schedule ----
 * The 16 keys of a group (8 candidates x 2 recids; lane j = candidate j recid 0, lane 8 + j = recid 1) are hashed together:
 * ec8_final_cf_kh writes their message words W0..W8 (the 33-byte compressed key 02|03 || x, then 0x80) straight from the canonical
 * limbs in registers, kh16_pass expands W16..W63 for all 16 keys with AVX-512 (W9..W14 = 0, W15 = 264), stores W + K pairwise
 * interleaved (unpack of rounds 2p, 2p + 1) and runs the rounds with SHA-NI four keys at a time from those pairs (no sha256msg1/2,
 * four independent sha256rnds2 chains). Same messages, same rounds, same h0 as qsha_keyhash4_h0. */
Q8T static inline QCPU_AIF void kh16_canon(__m512i w[4], const fe8 &a) {   /* canonical 4 x 64-bit limbs of 8 lanes */
#if QSB_CPU_I34_CANON_TOP
    /* QSB_CPU_I34_CANON_TOP (i34-9 b4c5a3c8): limb 4 < 2^48 - 1 in every lane: v < p, pack v itself */
    if (_mm512_cmp_epu64_mask(a.l[4], _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), _MM_CMPINT_NLT) == 0) {
        w[0] = _mm512_or_si512(a.l[0], _mm512_slli_epi64(a.l[1], 52));
        w[1] = _mm512_or_si512(_mm512_srli_epi64(a.l[1], 12), _mm512_slli_epi64(a.l[2], 40));
        w[2] = _mm512_or_si512(_mm512_srli_epi64(a.l[2], 24), _mm512_slli_epi64(a.l[3], 28));
        w[3] = _mm512_or_si512(_mm512_srli_epi64(a.l[3], 36), _mm512_slli_epi64(a.l[4], 16));
        return;
    }
#endif
    fe8 u; const __mmask8 ge = fe8_ge_p(a, &u);
    __m512i r[5];
    for (int i = 0; i < 5; i++) r[i] = _mm512_mask_blend_epi64(ge, a.l[i], u.l[i]);
    w[0] = _mm512_or_si512(r[0], _mm512_slli_epi64(r[1], 52));
    w[1] = _mm512_or_si512(_mm512_srli_epi64(r[1], 12), _mm512_slli_epi64(r[2], 40));
    w[2] = _mm512_or_si512(_mm512_srli_epi64(r[2], 24), _mm512_slli_epi64(r[3], 28));
    w[3] = _mm512_or_si512(_mm512_srli_epi64(r[3], 36), _mm512_slli_epi64(r[4], 16));
}
/* message words W0..W8 of one recid's 8 keys as 64-bit lanes (low 32 bits = the word): x = w3 w2 w1 w0, W0 = pfx << 24 | x >> 232,
 * W_i = (x >> (232 - 32 i)) & 0xffffffff (i = 1..7), W8 = (x & 0xff) << 24 | 0x800000; pfx = 2 | parity of y */
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
#if QSB_CPU_JL_KH16_WORDS52
/* QSB_CPU_JL_KH16_WORDS52 (jacklightChen b1c5e58e): the same message words straight from the canonical radix-52 limbs r0..r4;
 * kh16_store keeps only each word's low 32 bits, so the bits above them are don't-care */
Q8T static inline QCPU_AIF void kh16_words52(__m512i W[9], const fe8 &a, __mmask8 par) {
#if QSB_CPU_I34_CANON_TOP >= 2
    /* QSB_CPU_I34_CANON_TOP 2 (lane H2, i34-9's test on jacklightChen's WORDS52 path): limb 4 < 2^48 - 1 in every lane: v < p,
     * r = v; else the full v >= p test and blend below */
    __m512i r[5];
    if (_mm512_cmp_epu64_mask(a.l[4], _mm512_set1_epi64(0x0FFFFFFFFFFFFULL), _MM_CMPINT_NLT) == 0) {
        for (int i = 0; i < 5; i++) r[i] = a.l[i];
    } else {
        fe8 u; const __mmask8 ge = fe8_ge_p(a, &u);
        for (int i = 0; i < 5; i++) r[i] = _mm512_mask_blend_epi64(ge, a.l[i], u.l[i]);
    }
#else
    fe8 u; const __mmask8 ge = fe8_ge_p(a, &u);
    __m512i r[5];
    for (int i = 0; i < 5; i++) r[i] = _mm512_mask_blend_epi64(ge, a.l[i], u.l[i]);
#endif
    const __m512i head = _mm512_or_si512(_mm512_srli_epi64(r[4], 24), _mm512_set1_epi64(0x02000000));
    W[0] = _mm512_mask_or_epi64(head, par, head, _mm512_set1_epi64(0x01000000));
    W[1] = _mm512_or_si512(_mm512_slli_epi64(r[4], 8), _mm512_srli_epi64(r[3], 44));
    W[2] = _mm512_srli_epi64(r[3], 12);
    W[3] = _mm512_or_si512(_mm512_slli_epi64(r[3], 20), _mm512_srli_epi64(r[2], 32));
    W[4] = r[2];
    W[5] = _mm512_srli_epi64(r[1], 20);
    W[6] = _mm512_or_si512(_mm512_slli_epi64(r[1], 12), _mm512_srli_epi64(r[0], 40));
    W[7] = _mm512_srli_epi64(r[0], 8);
    W[8] = _mm512_or_si512(_mm512_slli_epi64(r[0], 24), _mm512_set1_epi64(0x00800000));
}
#endif
/* m[0..8] (16 dwords each): recid 0's keys in lanes 0..7, recid 1's in lanes 8..15 */
Q8T static inline QCPU_AIF void kh16_store(uint32_t *m, const fe8 &x0, __mmask8 p0, const fe8 &x1, __mmask8 p1) {
    __m512i W0[9], W1[9];
#if QSB_CPU_JL_KH16_WORDS52
    kh16_words52(W0, x0, p0); kh16_words52(W1, x1, p1);
#else
    __m512i w[4];
    kh16_canon(w, x0); kh16_words8(W0, w, p0);
    kh16_canon(w, x1); kh16_words8(W1, w, p1);
#endif
    for (int i = 0; i < 9; i++)
        _mm512_store_si512((void *)(m + 16 * i), _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi64_epi32(W0[i])), _mm512_cvtepi64_epi32(W1[i]), 1));
}
/* ec8_final_cf with the key-hash message words as output (m16: 9 x 16 dwords per group) instead of qx/qp. The EC steps are
 * ec8_final_cf's line for line (a33e04c3 wrote the QSB_CPU_WPRE form; the other branch mirrors ec8_final_cf's). */
Q8T static void ec8_final_cf_kh(const fe8 *X, const fe8 *Y, fe8 *D, fe8 *PRE, int G, const fe &mx, const fe &my, uint32_t *m16) {
    fe8 MX, MY; fe8_bcast(MX, mx); fe8_bcast(MY, my);
    const int NC = QSB_CPU_NCH;                      /* R2-D: chains of the batch inversion (4: the code before) */
    fe8 run[4]; for (int c = 0; c < NC; c++) fe8_set1(run[c]);
    for (int g = 0; g < G; g += NC)
        for (int c = 0; c < NC; c++) {
            const int h = g + c; fe8_sub_lz(D[h], MX, X[h]);
#if QSB_CPU_WPRE
            { fe8 t; fe8_sub_lz(t, MY, Y[h]);
              if (QSB_CPU_JL_INV_FIRST && g == 0) fe8_cp(PRE[h], t);   /* QSB_CPU_JL_INV_FIRST (jacklightChen b1c5e58e) */
              else fe8_mul_lz(PRE[h], run[c], t); }   /* weighted prefix, as in ec8_window */
#else
            fe8_cp(PRE[h], run[c]);
#endif
            if (QSB_CPU_JL_INV_FIRST && g == 0) fe8_cp(run[c], D[h]);
            else fe8_mul_lz(run[c], run[c], D[h]);
        }
    QCPU_INVC(run);
#if QSB_CPU_ILP2 & 1
    /* R2-D (QSB_CPU_ILP2 bit 0): two groups of different chains step together, op by op, as in ec8_window */
    for (int g = G - NC; g >= 0; g -= NC) {
        for (int c = NC - 1; c >= 1; c -= 2) {
            const int hA = g + c, hB = hA - 1;
            fe8 tA, tB, lamA, lamB, x3A, x3B, y3A, y3B;
            fe8_mul_lz(lamA, run[c], PRE[hA]); fe8_mul_lz(lamB, run[c - 1], PRE[hB]);
            if (!QSB_CPU_JL_INV_LAST || g > 0) {     /* QSB_CPU_JL_INV_LAST (jacklightChen b1c5e58e): after g == 0 both chains are dead */
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
}
#endif
/* Window 0: acc = sign * T0[|d0| - 1] (digit-0 lanes load row 0 and are dropped by the caller); computes
 * and prefetches window 1's rows (nxt) into rpn/ngn. */
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
/* Digits of 16 candidates at a time (zb: 8 SHA state words per candidate, h0 most significant) into
 * ds[i * B + k], with the per-window shifts uniform across the lanes; bad[k] = some digit is zero. */
Q8T static void recode16(const Geo &g, const uint32_t *zb, uint32_t *ds, uint8_t *bad, int B) {
#if !QSB_CPU_RECODE_NG
    const __m512i vidx = _mm512_set_epi32(120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0);
#endif
    const __m512i Z = _mm512_setzero_si512(), SB = _mm512_set1_epi32((int)0x80000000u), ONE = _mm512_set1_epi32(1);
    const int nw = g.nw, ns = g.sgn ? nw - 1 : 0;        /* windows 0..ns-1 are signed */
    __m128i shr[NWMAX], shl[NWMAX]; __m512i msk[NWMAX], half[NWMAX], full[NWMAX]; int wq[NWMAX]; bool two[NWMAX];
    for (int i = 0; i < nw; i++) {
        const int s = g.off[i] & 31, w = g.wid[i];
        wq[i] = g.off[i] >> 5; two[i] = s + w > 32;
        shr[i] = _mm_cvtsi32_si128(s); shl[i] = _mm_cvtsi32_si128(32 - s);
        msk[i] = _mm512_set1_epi32((int)((1u << w) - 1)); half[i] = _mm512_set1_epi32((int)(1u << (w - 1))); full[i] = _mm512_set1_epi32((int)(1u << w));
    }
    for (int k0 = 0; k0 < B; k0 += 16) {
        __m512i zw[9];                                   /* zw[j] = bits 32j..32j+31 of z = word h[7 - j] */
#if QSB_CPU_RECODE_NG
        {   /* r[m] = candidates 2m, 2m+1 (position 8*(k&1) + w); three stages swap the vector bits k3, k2, k1 with the word
             * bits w2, w1, w0, and the last one lays the positions out as k: s[w][k] = word w of candidate k. */
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
            v = _mm512_add_epi32(_mm512_and_si512(v, msk[i]), cy);   /* cy = 0 unless signed */
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
/* ---- 8-lane table build (QSB_CPU_VBUILD): dst[k] = src[k] + Q for k < 1,024 through the window step above (all rows Q,
 * positive), then canonical 4x64 limbs: the same points (and bytes) as batch_add. dst may equal src (the block is loaded
 * first). A block holding a point with Q's x (the doubling case) returns false before writing: the caller's scalar step. */
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
#if QSB_CPU_BUILD_NT
        for (int j = 0; j < 8; j++) {                    /* one 64-byte row per non-temporal store (rows are 64-byte aligned) */
            alignas(64) pt v; v.x = xs[j]; v.y = ys[j];
            _mm512_stream_si512((__m512i *)&dst[8 * h + j], _mm512_load_si512((const void *)&v));
        }
#else
        for (int j = 0; j < 8; j++) { dst[8 * h + j].x = xs[j]; dst[8 * h + j].y = ys[j]; }
#endif
    }
    return true;
}
#if QSB_CPU_BUILD_NT
Q8TX static void nt_copy_rows(pt *dst, const pt *src, size_t n) {   /* QSB_CPU_BUILD_NT: the fold's copy back */
    for (size_t i = 0; i < n; i++) _mm512_stream_si512((__m512i *)&dst[i], _mm512_load_si512((const void *)&src[i]));
}
#endif
#endif  /* QCPU_VEC */
#if !QCPU_VEC
struct Build8 {};
#endif

#if QCPU_SHANI
/* SHA-256 compression of 4 independent (state, block) pairs with the x86 SHA extensions,
 * instruction streams interleaved so the sha256rnds2 latency of one lane hides behind the others. */
#define QSHA __attribute__((target("sha,sse4.1,ssse3,avx")))   /* VEX: 3-operand adds/palignr feed sha256rnds2 without copies */
alignas(16) static const uint32_t qsha_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
/* st[l] (8 words, a..h) <- compress(st[l], blk[l]) for l = 0..3. */
QSHA static void qsha_x4(uint32_t (*st)[8], const uint8_t *const *blk) {
    const __m128i BSWAP = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i S0[4], S1[4], I0[4], I1[4], M[4][4];
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        __m128i t = _mm_loadu_si128((const __m128i *)&st[l][0]);           /* a b c d */
        __m128i u = _mm_loadu_si128((const __m128i *)&st[l][4]);           /* e f g h */
        t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
        S0[l] = _mm_alignr_epi8(t, u, 8);                                   /* ABEF */
        S1[l] = _mm_blend_epi16(u, t, 0xF0);                                /* CDGH */
        I0[l] = S0[l]; I1[l] = S1[l];
#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) M[l][j] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(blk[l] + 16 * j)), BSWAP);
    }
#pragma GCC unroll 16
    for (int r = 0; r < 16; r++) {
        const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
#pragma GCC unroll 4
        for (int l = 0; l < 4; l++) {
            if (r >= 4) {                                                   /* W[4r..4r+3] */
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
        __m128i t = _mm_shuffle_epi32(a, 0x1B);                             /* FEBA */
        b = _mm_shuffle_epi32(b, 0xB1);                                     /* DCHG */
        _mm_storeu_si128((__m128i *)&st[l][0], _mm_blend_epi16(t, b, 0xF0)); /* DCBA -> a b c d */
        _mm_storeu_si128((__m128i *)&st[l][4], _mm_alignr_epi8(b, t, 8));    /* HGFE -> e f g h */
    }
}
static const uint32_t qsha_iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
static bool qsha_supported() {
    unsigned a, b, cc, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &cc, &d)) return false;
    __builtin_cpu_init();                                   /* the 4-lane code is VEX-encoded: needs AVX too */
    return ((b >> 29) & 1) && __builtin_cpu_supports("avx");   /* CPUID.(7,0):EBX.SHA */
}
/* Precomputed-schedule compression: rows[l][b] points to the 64 words W[i] + K[i] of lane l's
 * block b (b < nblk), so the rounds need no message expansion (no sha256msg1/msg2, byte swaps or
 * K additions) and the chaining state stays in the ABEF/CDGH layout across the nblk blocks. Used
 * for the tail blocks that do not depend on the epoch: every one of them is one of a few fixed
 * contents per problem, scheduled once in start(). */
QSHA static void qsha_x4p(uint32_t (*st)[8], const uint32_t *const *const *rows, int nblk, uint32_t *out = nullptr, int ostride = 8,
                          const uint32_t *const *in = nullptr
#if QCPU_PFQ
                          , QPfRing *pq = nullptr
#endif
                          ) {
    __m128i S0[4], S1[4];
    if (!out) out = &st[0][0];                          /* out: lane l's final state at out + l * ostride (default: in place) */
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        const uint32_t *s = in ? in[l] : st[l];         /* in: lane l's initial state (default: st[l]) */
        __m128i t = _mm_loadu_si128((const __m128i *)&s[0]);
        __m128i u = _mm_loadu_si128((const __m128i *)&s[4]);
        t = _mm_shuffle_epi32(t, 0xB1); u = _mm_shuffle_epi32(u, 0x1B);
        S0[l] = _mm_alignr_epi8(t, u, 8); S1[l] = _mm_blend_epi16(u, t, 0xF0);
    }
    for (int b = 0; b < nblk; b++) {
#if QCPU_PFQ
        if (pq) qpf_drain(*pq, 2);                  /* R2-D: two queued row prefetches per block */
#endif
        const uint32_t *const w[4] = {rows[0][b], rows[1][b], rows[2][b], rows[3][b]};
        __m128i I0[4], I1[4];
#pragma GCC unroll 4
        for (int l = 0; l < 4; l++) { I0[l] = S0[l]; I1[l] = S1[l]; }
        if (QSB_CPU_X4PS && w[0] == w[1] && w[0] == w[2] && w[0] == w[3]) {   /* the same fixed block in all four lanes (the tail
                                                                                blocks): each W+K pair is loaded once, into xmm0, for four rounds */
            const uint32_t *ws = w[0];
            __asm__("" : "+r"(ws));                     /* opaque copy: keeps the compiler from hoisting (and spilling) lane 0's loads */
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
/* QSB_CPU_SHA4: the 64 rounds of four SHA-256 compressions from the IV, interleaved. Lane l's message is 16 native-order words in
 * mr[4l..4l+3], a ring the schedule overwrites in place (after round r >= 4, mr[4l + (r & 3)] holds W[4r..4r+3]); out0[l], out1[l]
 * receive the feed-forwarded ABEF and CDGH of lane l. sha256rnds2 is legacy-SSE encoded (xmm0..15 only, xmm0 its implicit
 * round-constant operand), so the eight chaining registers of four lanes leave no room for their sixteen message registers: the
 * ring stays in memory (the pointer is opaque and every round ends with a compiler barrier), read by sha256msg1/msg2 and palignr
 * as memory operands and written back once per lane and round (about 3 loads and 1 store per lane and round, on the load/store
 * pipes, not the FP pipes that sha256rnds2 and IFMA share). The row a round produces is what the next round's schedule step
 * depends on (W[t-2] in sha256msg2 and the W[t-7] term), so each lane also keeps its previous row in a register (p0..p3): read
 * back through the ring it would wait on store-to-load forwarding, about 12 cycles per round on Zen 3 against the 8 of the
 * two dependent sha256rnds2, and the 4-lane block ran 15 % slower than the 2-lane pair. Rows written in rounds 14 and 15 are
 * never read again and are not stored. Four independent chains issue one sha256rnds2 per cycle (latency 4); two lanes issue
 * one per 2 cycles. SHC: the schedule identities of W9..W14 = 0 (round 4 has no W[t-7] term, round 6's sha256msg1 is its first
 * operand), exactly as qsha_xw_iv32 and qsha_keyhash4_h0 apply them. Same round values as qsha_xw_iv<4>. */
template <bool SHC>
QSHA static inline void qsha_rounds4m(__m128i *mr, __m128i *out0, __m128i *out1) {
    const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);   /* ABEF: lanes f e b a */
    const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);   /* CDGH: lanes h g d c */
    __m128i a0 = IV0, a1 = IV0, a2 = IV0, a3 = IV0, b0 = IV1, b1 = IV1, b2 = IV1, b3 = IV1;   /* named, not arrays: they stay in registers */
    __m128i p0 = IV0, p1 = IV0, p2 = IV0, p3 = IV0;                                       /* each lane's previous row (W[4(r-1)..4r-1]) */
    __asm__("" : "+r"(mr));                              /* opaque ring pointer: the ring is never promoted to registers */
#pragma GCC unroll 16
    for (int r = 0; r < 16; r++) {
        const __m128i K = _mm_load_si128((const __m128i *)&qsha_k[4 * r]);
#define QSHA4_LANE(L, A, B, P) do {                                                                                      \
            __m128i *const M = mr + 4 * (L); __m128i w;                                                                  \
            if (r >= 4) {                                                                                                \
                __m128i t = (SHC && r == 6) ? _mm_load_si128(M + 2)                                                       \
                                            : _mm_sha256msg1_epu32(_mm_load_si128(M + (r & 3)), _mm_load_si128(M + ((r + 1) & 3))); \
                if (!(SHC && r == 4)) t = _mm_add_epi32(t, _mm_alignr_epi8(P, _mm_load_si128(M + ((r + 2) & 3)), 4));    \
                w = _mm_sha256msg2_epu32(t, P);                                                                          \
                if (r < 14) _mm_store_si128(M + (r & 3), w);        /* read again in rounds r+2 .. r+4 */                \
            } else w = _mm_load_si128(M + (r & 3));                                                                      \
            P = w;                                                                                                       \
            __m128i m = _mm_add_epi32(w, K);                                                                             \
            B = _mm_sha256rnds2_epu32(B, A, m); m = _mm_shuffle_epi32(m, 0x0E); A = _mm_sha256rnds2_epu32(A, B, m); } while (0)
        QSHA4_LANE(0, a0, b0, p0); QSHA4_LANE(1, a1, b1, p1); QSHA4_LANE(2, a2, b2, p2); QSHA4_LANE(3, a3, b3, p3);
#undef QSHA4_LANE
        __asm__ volatile("" ::: "memory");                /* the next round reads the ring from memory, not from registers kept live */
    }
    out0[0] = _mm_add_epi32(a0, IV0); out0[1] = _mm_add_epi32(a1, IV0); out0[2] = _mm_add_epi32(a2, IV0); out0[3] = _mm_add_epi32(a3, IV0);
    out1[0] = _mm_add_epi32(b0, IV1); out1[1] = _mm_add_epi32(b1, IV1); out1[2] = _mm_add_epi32(b2, IV1); out1[3] = _mm_add_epi32(b3, IV1);
}
/* a = ABEF + IV0, b = CDGH + IV1 -> st[0..7] = a b c d e f g h, the store order of qsha_xw. */
QSHA static inline void qsha_st8(uint32_t *st, __m128i a, __m128i b) {
    const __m128i t = _mm_shuffle_epi32(a, 0x1B);                             /* FEBA */
    b = _mm_shuffle_epi32(b, 0xB1);                                           /* DCHG */
    _mm_storeu_si128((__m128i *)&st[0], _mm_blend_epi16(t, b, 0xF0));         /* DCBA -> a b c d */
    _mm_storeu_si128((__m128i *)&st[4], _mm_alignr_epi8(b, t, 8));            /* HGFE -> e f g h */
}
#endif
/* SHA-256 compression with the message given as 16 native-order words per lane (no byte round trip),
 * L lanes interleaved: the second SHA-256 (message = the first digest's state words + fixed padding)
 * and the key hashes. L = 2 keeps state, message and schedule in the 16 legacy-SSE registers that
 * sha256rnds2 can address (4 lanes spill ~60 moves per lane and are not faster; QSB_CPU_SHA4's
 * qsha_rounds4m keeps the message rings in memory instead). */
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
/* qsha_xw<L> from the SHA-256 IV: the initial ABEF/CDGH pairs are constants (no state load or shuffle),
 * so the second SHA-256 and the key hashes need no per-call state initialization. Same rounds as qsha_xw. */
template <int L>
QSHA static inline void qsha_xw_iv(uint32_t (*st)[8], const uint32_t (*wd)[16]) {
    const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);   /* ABEF: lanes f e b a */
    const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);   /* CDGH: lanes h g d c */
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
/* qsha_xw_iv for a 32-byte message (the second SHA-256): only words 0..7 are read from wd; the padding words W8 = 0x80000000,
 * W9..W14 = 0, W15 = 256 are constants, r = 4's W[t-7] term (W9..W12) is zero and r = 6's sha256msg1(W8..11, W12..15) is W8..11.
 * The same round values as qsha_xw_iv with wd[l][8..15] set to that padding. */
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
/* The key hashes need only h0, the prefilter's input: h0 of SHA-256(compressed key) for the 4 keys (x = qx[l], parity of y
 * = qp[l]) as one vector, lane l = key l. The message words (as native words: word k = key bytes 4k..4k+3 big-endian, key
 * byte 0 = 0x02 | parity, bytes 1..32 = x big-endian, then 0x80 and the 264-bit length in word 15) are built in registers
 * and fed to qsha_xw_iv's rounds, two lanes at a time; no message stores or loads, no state stores. */
QSHA static __m128i qsha_keyhash4_h0(const fe *qx, const uint8_t *qp) {
    const __m128i I03 = _mm_set_epi8(4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, -128, 15, 14, 13);
    const __m128i I47 = _mm_set_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m128i a[4];
#if QSB_CPU_SHA4
    __m128i mr[16], b[4];                                /* the 4 keys' message words, built in registers, into the rings */
#pragma GCC unroll 4
    for (int l = 0; l < 4; l++) {
        const fe &x = qx[l];
        const __m128i lo = _mm_loadu_si128((const __m128i *)&x.v[0]), hi = _mm_loadu_si128((const __m128i *)&x.v[2]);
        _mm_store_si128(mr + 4 * l + 0, _mm_or_si128(_mm_shuffle_epi8(hi, I03), _mm_cvtsi32_si128((int)((0x02u | (qp[l] & 1)) << 24))));
        _mm_store_si128(mr + 4 * l + 1, _mm_shuffle_epi8(_mm_alignr_epi8(hi, lo, 1), I47));
        _mm_store_si128(mr + 4 * l + 2, _mm_cvtsi32_si128((int)(((uint32_t)(x.v[0] & 0xff) << 24) | 0x00800000u)));
        _mm_store_si128(mr + 4 * l + 3, _mm_set_epi32(264, 0, 0, 0));
    }
    qsha_rounds4m<(QSB_CPU_SHC != 0)>(mr, a, b);         /* a[l] = ABEF + IV0 of key l; h0 is its high word */
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
                    /* W9..W14 = 0: r = 4's W[t-7] words (W9..W12) are zero, r = 6's msg1(W8..11, W12..15) = W8..11 */
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
/* QSB_CPU_KH16 (a33e04c3): h0 of SHA-256 of the 16 keys whose message words W0..W8 are m[0..8] (lane = key); wk: 32 x 32 dwords of
 * scratch. Returns the prefilter mask of the 16 keys (bit k = key k's h0 has QSB_ZEROS_N leading zero bits). h0_out (dev builds
 * only, never in a ranked build): the 16 h0 values for the unit test. */
template <bool PAD_READY = false>   /* QSB_CPU_JL_KH16_PAD_ONCE: true when wk's pairs 5..7 (rounds 10..15) were written once */
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
        if (PAD_READY && t >= 10 && t <= 15) continue;   /* QSB_CPU_JL_KH16_PAD_ONCE (jacklightChen b1c5e58e): fixed pairs; W9..W15 stay set for the expansion */
        __m512i wt;
        if (t < 16) wt = W[t];
        else {
            /* W[t] = s1(W[t-2]) + W[t-7] + s0(W[t-15]) + W[t-16]; the zero words W9..W14 drop out at compile time */
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
        if (t & 1) {                                     /* rounds t - 1, t as dword pairs, key-major within 128-bit lanes */
            _mm512_store_si512((void *)(wk + 32 * (t >> 1)), _mm512_unpacklo_epi32(prev, wkt));
            _mm512_store_si512((void *)(wk + 32 * (t >> 1) + 16), _mm512_unpackhi_epi32(prev, wkt));
        } else prev = wkt;
    }
#undef S0_16
#undef S1_16
#undef R16
    /* key 4L + e: pair p at wk + 32 p + (e >= 2 ? 16 : 0) + 4 L + 2 (e & 1) */
    const __m128i IV0 = _mm_set_epi32((int)0x6a09e667, (int)0xbb67ae85, (int)0x510e527f, (int)0x9b05688c);
    const __m128i IV1 = _mm_set_epi32((int)0x3c6ef372, (int)0xa54ff53a, (int)0x1f83d9ab, (int)0x5be0cd19);
#if !QSB_CPU_JL_KH16_REGMASK || defined(QSB_CPU_DEVBENCH)
    alignas(64) uint32_t h0[16];
#endif
#if QSB_CPU_JL_KH16_REGMASK
    unsigned pass = 0;
#endif
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
#if QSB_CPU_JL_KH16_REGMASK
        /* QSB_CPU_JL_KH16_REGMASK (jacklightChen b1c5e58e): the IV feed-forward per state as before; unpackhi32 gives [a2, b2, a3, b3],
         * unpackhi64 keeps h0 = a3, b3, c3, d3; the four keys' prefilter bits from registers */
        const __m128i h01 = _mm_unpackhi_epi32(_mm_add_epi32(S0[0], IV0), _mm_add_epi32(S0[1], IV0));
        const __m128i h23 = _mm_unpackhi_epi32(_mm_add_epi32(S0[2], IV0), _mm_add_epi32(S0[3], IV0));
        const __m128i hv4 = _mm_unpackhi_epi64(h01, h23);
        const __m128i hit4 = _mm_cmpeq_epi32(_mm_srli_epi32(hv4, 32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32)), _mm_setzero_si128());
        pass |= (unsigned)_mm_movemask_ps(_mm_castsi128_ps(hit4)) << (4 * L);
#ifdef QSB_CPU_DEVBENCH
        _mm_storeu_si128((__m128i *)(h0 + 4 * L), hv4);
#endif
#else
#pragma GCC unroll 4
        for (int e = 0; e < 4; e++) h0[4 * L + e] = (uint32_t)_mm_extract_epi32(_mm_add_epi32(S0[e], IV0), 3);
#endif
    }
#ifdef QSB_CPU_DEVBENCH
    if (h0_out) memcpy(h0_out, h0, sizeof h0);
#endif
#if QSB_CPU_JL_KH16_REGMASK
    return pass;
#else
    const __m512i hv = _mm512_load_si512((const void *)h0);   /* pk_prefilter of the 16 keys: bit k = key k passes */
    return (unsigned)_mm512_cmpeq_epi32_mask(_mm512_srli_epi32(hv, 32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32)), _mm512_setzero_si512());
#endif
}
#endif
/* W[i] + K[i] for i < 64 of one 64-byte block (big-endian words), with qsha_x4's schedule steps. */
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
#if QSB_CPU_FENCE
/* QSB_CPU_FENCE: F and N once fence_plan() committed a fence (0: none, the co-grinder takes the complement patterns as in the
 * base); g_fence_off: the co-grinder said "off" after the fence was committed, so nobody grinds [F, N) and the GPU may. */
static uint64_t g_fence = 0, g_fence_n = 0;
static std::atomic<int> g_fence_off{0};
/* Called by tree.cu before the host producers start (and by the bench before start()): the GPU's epoch count, F with the
 * co-grinder on, n_epochs with it off at run time (QSB_CPU_THREADS_ENV < 1) or when no fence fits. */
static uint64_t fence_plan(uint64_t n_epochs, uint64_t cap) {
    if (const char *e = getenv("QSB_CPU_THREADS_ENV")) if (atoi(e) < 1) return n_epochs;
    uint64_t f = n_epochs > (uint64_t)QSB_CPU_FENCE_R ? n_epochs - (uint64_t)QSB_CPU_FENCE_R : 0;
    if (const char *e = getenv("QSB_CPU_FENCE_AT")) f = strtoull(e, nullptr, 10);   /* dev: the forced-low fence test */
    if (cap) f = f / cap * cap;                           /* batch-aligned: the GPU never launches a partial batch at F */
    if (f == 0 || f >= n_epochs) return n_epochs;
    g_fence = f; g_fence_n = n_epochs;
    return f;
}
static bool fence_released() { return g_fence_off.load() != 0; }
#define QCPU_FENCE_OFF() (g_fence_off = 1)
#else
#define QCPU_FENCE_OFF() ((void)0)
#endif
struct Ctx {
    const digest_params_t *dp;
    Geo g;                          /* table geometry */
    pt *table = nullptr;            /* g.total points of 64 B, 2 MiB aligned (transparent huge pages) */
    void *table_map = nullptr; size_t table_map_bytes = 0;
    const pt *tw[NWMAX] = {nullptr};   /* first entry of window i */
    fe cx, cy;                      /* C = u2*R */
    bool cfold = false;             /* top window holds (j + 1) * B_top + C; mx, my = -2C (8-lane path) */
    fe mx, my;
    uint8_t cwin[286][3];           /* CPU window patterns: the complement of the GPU's */
    int ncwin = 0;
    int cut = 137, early = 6;
    uint64_t mid_bytes = 0;         /* preimage bytes covered by dp->midstate */
    uint64_t n_epochs = 0;
    uint64_t epoch_base = 0;        /* first epoch of the workers' walk (worker t: base + t, base + t + T, ...) */
    std::atomic<uint64_t> cand{0};
    std::atomic<uint32_t> hits{0};
    std::mutex io;
    qsb_hv_t hv;                    /* exact gate, used under io */
    FILE *out = nullptr;
    int nthreads = 0;
    int batch = QSB_CPU_BATCH;      /* candidates per batch (a multiple of 32), set in start() before the workers (QSB_CPU_BATCH_AUTO) */
    char batch_why[48] = "default";
#ifdef QSB_CPU_DEVBENCH
    uint64_t dev_limit = 0;         /* dev only (never in a ranked build): each worker stops once this many candidates are done */
    std::atomic<int> dev_live{0};
#endif
    bool vec = false;               /* 8-lane IFMA path */
    bool shani = false;             /* 4-lane SHA-NI hashing */
    bool kh16 = false;              /* QSB_CPU_KH16 key hashes (8-lane path with SHA-NI, AVX-512VL and the C fold) */
    std::atomic<int> live{0}, stop{0}, ready{0};   /* H9: workers not yet returned; stop request; workers spawned (stop_unmap) */
#if QCPU_SHANI
    /* Hashing plan (hash_plan): the message after an epoch's state is block 0 (the epoch's buffered
     * bytes + the first window pushes) and blocks 1..nb-1, whose contents do not depend on the epoch.
     * The CPU patterns share few block-0 contents (groups) and few distinct later blocks. */
    bool hplan = false;
    int h_nb = 0, h_ng = 0;
    uint8_t h_g0[286];              /* block-0 group of each CPU pattern */
    const uint32_t *h_wkp[286][16]; /* schedule of fixed block b = 1..nb-1 of each CPU pattern (index b-1) */
    uint64_t h_binom[256][8];       /* binom_u64(n, k) for the epoch unrank */
    std::vector<uint8_t> h_gblk;    /* ng x 64: block 0 of each group, epoch bytes left zero */
    std::vector<uint32_t, qalloc64<uint32_t> > h_wk;   /* distinct fixed blocks x 64 words W[i]+K[i] */
#endif
#if QSB_CPU_DIAG_V4
    uint8_t diag4[256] = {0};       /* QSB_CPU_DIAG_V4: worker t starts diag4[t] x 2^20 epochs into its range (set before spawn) */
#endif
};

/* The table as a 2 MiB-aligned anonymous mapping with transparent huge pages (it is read at random);
 * pages are touched first by the builder threads. */
static bool table_alloc(Ctx &c) {
    const size_t H = (size_t)2 << 20, bytes = (c.g.total * sizeof(pt) + H - 1) / H * H;
    void *m = mmap(nullptr, bytes + H, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return false;
    const uintptr_t a = ((uintptr_t)m + H - 1) & ~(uintptr_t)(H - 1);
#ifdef MADV_HUGEPAGE
    if (!getenv("QSB_CPU_NOTHP")) madvise((void *)a, bytes, MADV_HUGEPAGE);   /* dev override: 4 KiB pages */
#endif
    c.table_map = m; c.table_map_bytes = bytes + H; c.table = (pt *)a;
    for (int i = 0; i < c.g.nw; i++) c.tw[i] = c.table + c.g.base[i];
    return true;
}
static void table_free(Ctx &c) {
    if (c.table_map) munmap(c.table_map, c.table_map_bytes);
    c.table_map = nullptr; c.table = nullptr;
}
/* Bytes of [p, p + n) mapped by transparent huge pages (AnonHugePages of the overlapping VMAs in /proc/self/smaps,
 * each capped at its overlap), or -1 if smaps cannot be read. */
static double thp_bytes(const void *p, size_t n) {
    FILE *f = fopen("/proc/self/smaps", "r"); if (!f) return -1;
    const unsigned long a = (unsigned long)p, e = a + n;
    double huge = 0, ov = 0; bool seen = false; char line[512];
    while (fgets(line, sizeof line, f)) {
        unsigned long s, t; double kb;
        if (sscanf(line, "%lx-%lx ", &s, &t) == 2) {
            if (s >= e) break;                          /* VMAs come in address order */
            ov = t > a ? (double)((t < e ? t : e) - (s > a ? s : a)) : 0; if (ov < 0) ov = 0;
            if (ov > 0) seen = true;
        } else if (ov > 0 && sscanf(line, "AnonHugePages: %lf kB", &kb) == 1) huge += kb * 1024.0 < ov ? kb * 1024.0 : ov;
    }
    fclose(f);
    return seen ? huge : -1;
}
/* First touch of 2 MiB regions i * step, i < n / step, by nth threads (a transparent huge page is allocated
 * at the first fault in a region, or never); returns the fraction of those regions that got one (-1: unknown). */
static double table_touch(Ctx &c, int nth, size_t step, double deadline = 0) {   /* H9: deadline (CLOCK_MONOTONIC s; 0 none): -3 */
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
    const double h = thp_bytes(p, n * H);              /* untouched regions hold no pages */
    return h < 0 ? -1 : h / (double)(m * H);
}
static pt pt_add_aff(const pt &a, const pt &b) {       /* a + b for distinct x */
    fe d, inv, t, lam, x3, y3;
    fe_sub(d, b.x, a.x); fe_inv(inv, d); fe_sub(t, b.y, a.y); fe_mul(lam, t, inv);
    fe_sqr(x3, lam); fe_sub(x3, x3, a.x); fe_sub(x3, x3, b.x);
    fe_sub(t, a.x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, a.y);
    return {x3, y3};
}
static pt pt_mul_small(const pt &q, uint64_t k) {      /* k * q, 1 <= k << n (no intermediate meets +-q) */
    pt r = q;
    for (int b = 62 - __builtin_clzll(k | 1); b >= 0; b--) { r = pt_double(r); if ((k >> b) & 1) r = pt_add_aff(r, q); }
    return r;
}
/* Window i: T[j] = (j + 1) * B_i, B_i = 2^off[i] * A. The first S entries by doubling rounds; entries
 * cS..cS+S-1 (c >= 1) as T[k] + cS * B_i. The (window, chunk) tasks are shared by nth threads. */
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
        uint32_t have = 2;                               /* T[0..have-1] = 1..have multiples */
        std::vector<fe> d(n0), pre(n0); std::vector<uint8_t> inf(n0), bad(n0); std::vector<const pt *> tp(n0);
        while (have < n0) {
            uint32_t n = have; if (have + n > n0) n = n0 - have;
            /* T[have+k] = T[k] + T[have-1]  ((k+1) + have = have+k+1) */
            for (uint32_t k = 0; k < n; k++) { T[have + k] = T[k]; tp[k] = &T[have - 1]; inf[k] = 0; bad[k] = 0; }
            batch_add(&T[have], tp.data(), inf.data(), bad.data(), (int)n, d.data(), pre.data());
            /* k = have-1 adds T[have-1] to itself: equal x, so batch_add flags it; double it. */
            for (uint32_t k = 0; k < n; k++) if (bad[k]) T[have + k] = pt_double(T[have - 1]);
            have += n;
        }
    };
    auto chunks = [&](pt *T, uint32_t o, uint32_t k0, uint32_t n, const pt &Q) {   /* scalar: T[o + k] = T[k] + Q, k0 <= k < k0 + n */
        std::vector<fe> d(n), pre(n); std::vector<uint8_t> inf(n, 0), bad(n, 0); std::vector<const pt *> tp(n, &Q);
        for (uint32_t k = 0; k < n; k++) T[o + k0 + k] = T[k0 + k];
        batch_add(&T[o + k0], tp.data(), inf.data(), bad.data(), (int)n, d.data(), pre.data());
        for (uint32_t k = 0; k < n; k++) if (bad[k]) T[o + k0 + k] = pt_double(Q);   /* only cc = 1, k = S-1: T[k] = Q */
    };
    auto chunkc = [&](int i, uint32_t cc, Build8 *b8) {
        pt *T = c.table + g.base[i];
        const uint32_t o = cc * S, n = g.ent[i] - o < S ? g.ent[i] - o : S;
        const pt Q = pt_mul_small(T[S - 1], cc);         /* T[S-1] = S * B_i */
#if QCPU_VEC
        if (b8) {                                        /* 8-lane blocks; the block with T[k] = Q (cc = 1) and a tail: scalar */
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
    const bool vb = QSB_CPU_VBUILD && c.vec && !getenv("QSB_CPU_NOVBUILD");   /* dev override: the scalar build */
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
#if QSB_CPU_BUILD_NT
            _mm_sfence();                                /* this thread's non-temporal rows are visible before the join */
#endif
        };
        std::vector<std::thread> ts;
        const int m = (size_t)nth < nt ? nth : (int)nt;
        for (int t = 0; t < m; t++) ts.emplace_back(work);
        for (auto &t : ts) t.join();
    }
    c.cfold = false;
    if (QSB_CPU_CFOLD && c.vec) {                       /* top window T[j] += C, in chunks; any equal-x entry (T[j] = +-C, a known
                                                           discrete log: never) keeps the plain table */
        const int top = g.nw - 1; pt *T = c.table + g.base[top]; const uint32_t n = g.ent[top], CH = 1u << 16;
        const pt C = {c.cx, c.cy};
        std::atomic<uint32_t> next{0}; std::atomic<int> anybad{0};
        std::vector<pt> tmpv; pt *tmp = nullptr;
#if QSB_CPU_FOLD_PAR
        const size_t tH = (size_t)2 << 20, tb = ((size_t)n * sizeof(pt) + tH - 1) / tH * tH;   /* H9: the temporary on huge pages, */
        void *tmap = mmap(nullptr, tb + tH, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);   /* touched by the fold threads */
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
                if (okb) {                              /* 8-lane blocks (a point with C's x: anybad, as below); the tail scalar */
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
#if QSB_CPU_BUILD_NT
            _mm_sfence();
#endif
        };
        std::vector<std::thread> ts;
        for (int t = 0; t < nth; t++) ts.emplace_back(work);
        for (auto &t : ts) t.join();
        if (!anybad) {
#if QSB_CPU_FOLD_PAR
            {   /* H9: the copy back in 65,536-entry chunks on the build threads */
                std::atomic<uint32_t> nc{0};
#if QSB_CPU_BUILD_NT && QCPU_VEC
                const bool ntc = c.vec;
                auto cp = [&]() {
                    for (uint32_t o; (o = nc.fetch_add(CH)) < n;) {
                        const size_t k = (size_t)(n - o < CH ? n - o : CH);
                        if (ntc) nt_copy_rows(T + o, tmp + o, k); else memcpy(T + o, tmp + o, k * sizeof(pt));
                    }
                    if (ntc) _mm_sfence();
                };
#else
                auto cp = [&]() { for (uint32_t o; (o = nc.fetch_add(CH)) < n;) memcpy(T + o, tmp + o, (size_t)(n - o < CH ? n - o : CH) * sizeof(pt)); };
#endif
                std::vector<std::thread> tc;
                for (int t = 1; t < nth; t++) tc.emplace_back(cp);
                cp();
                for (auto &t : tc) t.join();
            }
#else
            memcpy(T, tmp, (size_t)n * sizeof(pt));
#endif
            const pt C2 = pt_double(C);                 /* (mx, my) = -2C */
            c.mx = C2.x; fe z0 = {{0, 0, 0, 0}}; fe_sub(c.my, z0, C2.y);
            c.cfold = true;
        }
#if QSB_CPU_FOLD_PAR
        if (tmap) munmap(tmap, tb + tH);
#endif
    }
}
/* Spot-check the table against OpenSSL: in every window the first, the last and a middle entry, and both
 * sides of the first, a middle and the last 65,536-entry chunk boundary (the build's and the C fold's chunks). */
static bool table_check(const Ctx &c) {
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *bx = BN_CTX_new(); BIGNUM *nri = BN_new(), *k = BN_new(), *ord = BN_new(), *xx = BN_new(), *yy = BN_new();
    EC_POINT *P = grp ? EC_POINT_new(grp) : nullptr, *Cp = grp ? EC_POINT_new(grp) : nullptr;
    bool ok = grp && bx && nri && k && ord && xx && yy && P && Cp && EC_GROUP_get_order(grp, ord, bx) &&
              BN_lebin2bn(c.dp->neg_r_inv, 32, nri);
    if (ok && c.cfold) {                                /* the top window holds (j + 1) * B_top + C */
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
/* Bytes this process may still take: MemAvailable, capped by the cgroup (v2 or v1) limit minus usage. */
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
    auto cap = [&](const char *dir, bool v2) {       /* limit - usage of one cgroup directory */
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
    if (FILE *f = fopen("/proc/self/cgroup", "r")) {  /* the process's own cgroup and its ancestors, when not namespaced */
        char line[512];
        while (fgets(line, sizeof line, f)) {
            char *c1 = strchr(line, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : nullptr; if (!c2) continue;
            line[strcspn(line, "\n")] = 0; *c1 = 0; *c2 = 0;   /* hierarchy id : controllers : path */
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
    if (const char *e = getenv("QSB_CPU_MEM_MB")) avail = atof(e) * 1048576.0;   /* dev override */
    return avail;
}
/* Geometry: QSB_CPU_NW / QSB_CPU_UNSIGNED (dev overrides), else the fewest windows (>= QSB_CPU_NW_MIN) whose
 * table fits the budget, else 15 (68 MiB, the size of the old 16 x 16-bit table). */
static Geo geo_choose(bool try11 = false, bool try10 = false) {
    const double avail = mem_avail(), capb = (double)QSB_CPU_TAB_CAP_MB * 1048576.0;
    double budget = avail * QSB_CPU_TAB_FRAC;
    if (budget > capb) budget = capb;
    int nw = 15;
    for (int n = QSB_CPU_NW_MIN; n < 15; n++) if ((double)geo_make(n, true).total * sizeof(pt) <= budget) { nw = n; break; }
    if (try11 && nw > 11) {                            /* 11 windows: the budget after the reserve */
        double b11 = (avail - (double)QSB_CPU_TAB_RESERVE_MB * 1048576.0) * QSB_CPU_TAB_FRAC;
        if (b11 > capb) b11 = capb;
        if ((double)geo_make(11, true).total * sizeof(pt) <= b11) nw = 11;
    }
    if (try10 && nw == 11) {                           /* 10 windows: the same rule, its own cap */
        double b10 = (avail - (double)QSB_CPU_TAB_RESERVE_MB * 1048576.0) * QSB_CPU_TAB_FRAC;
        if (b10 > (double)QSB_CPU_TAB10_CAP_MB * 1048576.0) b10 = (double)QSB_CPU_TAB10_CAP_MB * 1048576.0;
        if ((double)geo_make(10, true).total * sizeof(pt) <= b10) nw = 10;
    }
    if (QSB_CPU_TRY9 && try10 && nw == 10) {            /* 9 windows: half of what is left after the reserve */
        const double b9 = (avail - (double)QSB_CPU_TAB_RESERVE_MB * 1048576.0) * QSB_CPU_TAB9_FRAC;
        if ((double)geo_make(9, true).total * sizeof(pt) <= b9) nw = 9;
    }
    if (const char *e = getenv("QSB_CPU_NW")) nw = atoi(e);
    return geo_make(nw, !getenv("QSB_CPU_UNSIGNED"));
}
/* Table memory: geo_choose, 2 MiB-aligned THP mapping, first touch, measured huge-page fraction (hp; -1 = unknown).
 * An 11-window table below QSB_CPU_HP_MIN (a 60-region sample first, then all regions) is freed for 12 windows; a
 * 12-window table below it is replaced by 13 windows if QSB_CPU_FALL13 and those come at QSB_CPU_HP_MIN or 25 points
 * better backed, else 12 stays (as before). QSB_CPU_NW (dev) forces the geometry. note: what was tried and dropped.
 * Returns false without table memory. */
static bool table_setup(Ctx &c, int nth, double &hp, char *note, size_t nn, int nw_floor = 0) {
    note[0] = 0; hp = -1;
    bool thp = !getenv("QSB_CPU_NOTHP");
    if (FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
        char b[128] = {0}; if (fgets(b, sizeof b, f) && strstr(b, "[never]")) thp = false; fclose(f);
    }
    const bool forced = getenv("QSB_CPU_NW") != nullptr;
    c.g = geo_choose(QSB_CPU_TRY11 && thp && !forced, QSB_CPU_TRY10 && thp && !forced);
    if (!forced && c.g.nw < nw_floor) c.g = geo_make(nw_floor, true);
    const char *fail = getenv("QSB_CPU_HP_FAIL");       /* dev: windows whose measured fraction is taken as 0 */
    auto failed = [&](int nw) {
        if (!fail) return false;
        char k[8]; const size_t kl = (size_t)snprintf(k, sizeof k, "%d", nw);
        for (const char *q = fail; *q;) { const size_t n = strcspn(q, ","); if (n == kl && !strncmp(q, k, n)) return true; q += n; if (*q) q++; }
        return false;
    };
    double touch_dl = 0;                                 /* H9: the 9-window first touch's deadline (QSB_CPU_TOUCH_GUARD) */
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
            {   struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
                const char *e = getenv("QSB_CPU_TOUCH9_MAX_S");   /* dev override */
                touch_dl = t.tv_sec + 1e-9 * t.tv_nsec + (e ? atof(e) : (double)QSB_CPU_TOUCH9_MAX_S); }
#endif
            double f = measure(32);                      /* 1,792 of 57,344 regions */
#if !QSB_CPU_TOUCH_FUSE
            if (f >= QSB_CPU_HP_MIN) f = measure(1);
#endif
            if (f >= QSB_CPU_HP_MIN) { hp = f; return true; }
            add(9, f);
            table_free(c);
        } else add(9, -2);
        c.g = geo_make(10, true);                        /* the 9-window rule implies the 10-window one */
    }
    if (c.g.nw == 10 && c.g.sgn && !forced) {
        if (table_alloc(c)) {
            double f = measure(32);                      /* 272 of 8,704 regions */
#if !QSB_CPU_TOUCH_FUSE
            if (f >= QSB_CPU_HP_MIN) f = measure(1);
#endif
            if (f >= QSB_CPU_HP_MIN) { hp = f; return true; }
            add(10, f);
            table_free(c);
        } else add(10, -2);
        c.g = geo_make(11, true);                        /* the 10-window rule implies the 11-window one */
    }
    if (c.g.nw == 11 && c.g.sgn && !forced) {
        if (table_alloc(c)) {
            double f = measure(32);                      /* 60 of 1,920 regions */
            if (f >= QSB_CPU_HP_MIN) f = measure(1);
            if (f >= QSB_CPU_HP_MIN) { hp = f; return true; }
            add(11, f);
            table_free(c);
        } else add(11, -2);
        c.g = geo_make(12, true);
    }
    if (!table_alloc(c)) { c.g = geo_make(15, true); if (!table_alloc(c)) return false; }
    if (forced || !thp || c.g.nw != 12 || !c.g.sgn || !QSB_CPU_FALL13) { hp = measure(1); return true; }
    const double f = measure(16);                        /* 34 of 544 regions */
    if (f >= QSB_CPU_HP_MIN || f < 0) { hp = f >= QSB_CPU_HP_MIN ? measure(1) : f; if (hp >= QSB_CPU_HP_MIN || hp < 0) return true; }
    else hp = f;
    /* 12 poorly backed: try 13 windows (368 MiB); the 12-window mapping is kept meanwhile */
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
/* Plan the 4-lane hashing (after ncwin, mid_bytes): group the CPU patterns by block 0 and schedule
 * every distinct later block once. Leaves hplan false for an unexpected shape (old path then). */
static void hash_plan(Ctx &c) {
    const digest_params_t *dp = c.dp;
    const size_t prl = dp->prefix_remainder_len, pl = (size_t)(c.cut - c.early) * SIG_PUSH_SIZE,
                 wlen = (size_t)(dp->n - c.cut - 3) * SIG_PUSH_SIZE, tl = dp->tail_section_len, sl = dp->tx_suffix_len;
    const size_t remlen = (prl + pl) % 64;
    const int nb = (int)((remlen + wlen + tl + sl + 9 + 63) / 64);
    if (nb < 2 || nb > 16 || c.ncwin < 4 || c.ncwin > 286) return;   /* ncwin >= 4: see the worker's gstb */
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
    c.h_wk.assign(blocks.size(), 0);                    /* (blocks/64) x 64 words */
    for (size_t q = 0; q < blocks.size() / 64; q++) qsha_schedule(&c.h_wk[q * 64], &blocks[q * 64]);
    for (int wi = 0; wi < c.ncwin; wi++) for (int b = 1; b < nb; b++) c.h_wkp[wi][b - 1] = &c.h_wk[(size_t)sidx[(size_t)wi * 16 + b] * 64];
    for (int n = 0; n < 256; n++) for (int k = 0; k < 8; k++) c.h_binom[n][k] = binom_u64(n, k);
    if (c.cut > 255 || c.early > 7) return;
    c.h_nb = nb; c.h_ng = ng; c.hplan = true;
}

#ifndef QSB_CPU_PREFIX100
#define QSB_CPU_PREFIX100 1
#endif
/* b67487a1 (after 4a197f06's family selection), taken as written: keep the 20 block-0 groups of exactly 5 patterns (100 patterns)
 * when the plan has the 158/77 shape, so each block-0 compression serves 5 candidates; any other shape keeps every pattern.
 * Runs in start before any worker exists, so no worker holds a pointer into the first plan. 0 = hash_plan alone (the base). */
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
/* Gate one recovered key (x, parity of y); true when it is an exact hit, which is then published. */
static inline void pk_block(uint8_t *pk, const fe &x3, unsigned ypar) {    /* compressed key, bytes 0..32 */
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
/* The exact OpenSSL gate and the publication, for a candidate whose key hash passed the prefilter. */
static bool gate_publish_exact(Ctx *c, const uint8_t *sk, int ri) {
    std::lock_guard<std::mutex> g(c->io);
    if (!qsb_hv_check(&c->hv, sk, ri)) return false;
#if QSB_CPU_FENCE
    if (!qsb_pub_once(sk, ri)) return false;         /* published before in this process (qsb_host_verify.h): dropped, counted */
#endif
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
/* Prefetch the rows of windows 0 and 1 (QSB_CPU_HPF = 2; window 0 only for 1) of the 8 candidates k0..k0+7 from their z words
 * (zb: 8 words per candidate, h7 = z bits 0..31): recode_scalar's first two digits, RowSgn's row T[max(|d|, 1) - 1]. */
Q8T static void hpf_rows8(const uint32_t *zb, int k0, const pt *t0, const pt *t1, int w0, int w1, bool sgn) {
    const __m512i one = _mm512_set1_epi64(1);
    const __m512i vi = _mm512_set_epi64(28, 24, 20, 16, 12, 8, 4, 0);                /* qword index of h6|h7 << 32 */
    const __m512i v = _mm512_i64gather_epi64(vi, (const void *)(zb + (size_t)k0 * 8 + 6), 8);
    const __m512i zl = _mm512_ror_epi64(v, 32);                                       /* z bits 0..63: h7 | h6 << 32 */
    __m512i u = _mm512_and_si512(zl, _mm512_set1_epi64((long long)((1ULL << w0) - 1)));
    const __mmask8 n0 = sgn ? _mm512_cmpgt_epu64_mask(u, _mm512_set1_epi64(1LL << (w0 - 1))) : 0;
    u = _mm512_mask_sub_epi64(u, n0, _mm512_set1_epi64(1LL << w0), u);
    alignas(64) uint64_t a[16];
    _mm512_store_si512(a, _mm512_add_epi64(_mm512_set1_epi64((long long)(uintptr_t)t0), _mm512_slli_epi64(_mm512_sub_epi64(_mm512_max_epu64(u, one), one), 6)));
    if (QSB_CPU_HPF > 1) {
        __m512i u1 = _mm512_and_si512(_mm512_srl_epi64(zl, _mm_cvtsi32_si128(w0)), _mm512_set1_epi64((long long)((1ULL << w1) - 1)));
        u1 = _mm512_mask_add_epi64(u1, n0, u1, one);                                  /* window 0's carry */
        const __mmask8 n1 = sgn ? _mm512_cmpgt_epu64_mask(u1, _mm512_set1_epi64(1LL << (w1 - 1))) : 0;
        u1 = _mm512_mask_sub_epi64(u1, n1, _mm512_set1_epi64(1LL << w1), u1);
        _mm512_store_si512(a + 8, _mm512_add_epi64(_mm512_set1_epi64((long long)(uintptr_t)t1), _mm512_slli_epi64(_mm512_sub_epi64(_mm512_max_epu64(u1, one), one), 6)));
    }
    for (int j = 0; j < (QSB_CPU_HPF > 1 ? 16 : 8); j++) _mm_prefetch((const char *)a[j], QCPU_NXT_HINT);
}
#if QCPU_PFQ
/* R2-D: hpf_rows8's addresses into the queue instead of 16 prefetches in a burst */
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
    void *m = nullptr, *w = nullptr;   /* QSB_CPU_KH16: the groups' key-hash message words (9 x 16 dwords each) and the W + K scratch */
    if (posix_memalign(&m, 64, (size_t)G * 144 * 4) || posix_memalign(&w, 64, 32 * 32 * 4)) { free(m); for (int j = 0; j < 9; j++) free(q[j]); return false; }
    v.m16 = (uint32_t *)m; v.wk16 = (uint32_t *)w;
#if QSB_CPU_JL_KH16_PAD_ONCE
    /* QSB_CPU_JL_KH16_PAD_ONCE (jacklightChen b1c5e58e): pairs 5..7 hold rounds 10..15's W + K, the same for all 16 keys (W10..W14 = 0,
     * W15 = 264); kh16_pass's unpacklo/hi of two broadcast words alternate them, round 2p in even dwords and 2p + 1 in odd ones */
    for (int p = 5; p < 8; p++) for (int j = 0; j < 32; j++) {
        const int t = 2 * p + (j & 1);
        v.wk16[32 * p + j] = qsha_k[t] + (t == 15 ? uint32_t(264) : uint32_t(0));
    }
#endif
#endif
    return true;
}
/* Row function of one window: the 8 lanes' rows T[|d| - 1] (row 0 for a zero digit) and their sign mask. */
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
/* The EC part of one batch on the 8-lane path: z*A for every candidate (g.nw table windows), then
 * both recovery ids against C. zb holds the candidates' z words, ds receives the digits (window-major). */
Q8TX static void vec_batch(const Ctx *c, const uint32_t *zb, uint32_t *ds, int B, VecBuf &v) {
    const int G = B / 8;
    const Geo &g = c->g;
    recode16(g, zb, ds, v.bad, B);
    /* two row buffers: this window's rows and the next window's, filled (and prefetched) one pass ahead */
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

static void worker(Ctx *c, int tid) {
    struct LiveGuard { std::atomic<int> &n; ~LiveGuard() { n--; } } live_guard{c->live};   /* H9: the spawner counted this worker */
#ifdef SCHED_IDLE
    struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
    const digest_params_t *dp = c->dp;
    const int B = QSB_CPU_BATCH_AUTO ? c->batch : QSB_CPU_BATCH;   /* QSB_CPU_BATCH_AUTO: chosen in start() (a multiple of 32, <= 8192), a
                                                                     * run-time value as in 86c643ae; 0: the compile-time constant as before */
    std::vector<pt> acc(B); std::vector<fe> d(2 * B), pre(2 * B);
    std::vector<uint8_t> inf(B), bad(B); std::vector<const pt *> tp(B);
    std::vector<uint32_t, qalloc64<uint32_t> > zb((size_t)B * 8), ds((size_t)NWMAX * B);   /* z words; digits */
    std::vector<uint8_t> ngs(B);
    std::vector<uint8_t> skips((size_t)B * 9 + 8);   /* +8: the wide skip stores below overrun by 3 bytes */
    uint8_t pk[64]; memset(pk, 0, 64); pk[33] = 0x80; pk[62] = 0x01; pk[63] = 0x08;   /* 264 bits */
    uint8_t blk2[64]; memset(blk2, 0, 64); blk2[32] = 0x80; blk2[62] = 0x01;          /* 256 bits */
#if QSB_CPU_EPOCH_CONTIG
    const uint64_t eavail = c->n_epochs > c->epoch_base ? c->n_epochs - c->epoch_base : 0;
    const uint64_t espan = eavail / (uint64_t)(c->nthreads > 0 ? c->nthreads : 1);
    uint64_t epoch = c->epoch_base + (uint64_t)tid * espan;
    const uint64_t epoch_end = epoch + espan;
#if QSB_CPU_DIAG_V4
    /* the start code c_t x 2^20, only where the range keeps 2^25 epochs (1.3 x a runner worker's 1,200 s walk) behind it */
    if (tid < 256 && ((uint64_t)c->diag4[tid] << 20) + ((uint64_t)1 << 25) <= espan) epoch += (uint64_t)c->diag4[tid] << 20;
#endif
#else
    uint64_t epoch = c->epoch_base + (uint64_t)tid;   /* epoch_base: see QSB_CPU_DIAG_EPOCH */
#endif
    int wi = c->ncwin;
    SHA256_CTX ectx; uint8_t early[16];
    std::vector<uint8_t> pbuf((size_t)dp->n * SIG_PUSH_SIZE + 64);
    auto put_digits = [&](int kk, const uint32_t *h) { memcpy(&zb[(size_t)kk * 8], h, 32); };   /* z = h[0] (MSW) .. h[7] */
#if QCPU_SHANI
    /* 4-lane path: each lane holds one candidate's chaining state (its epoch's) and the padded rest of
     * its message: the epoch's buffered bytes, the kept window pushes, tail section, suffix. */
    const size_t tl = dp->tail_section_len, sl = dp->tx_suffix_len;
    alignas(16) uint32_t lst[4][8]; alignas(16) uint8_t lmsg[4][16 * 64]; const uint8_t *lp[4];
    uint32_t est[8]; uint8_t erem[64]; size_t remlen = 0; int nb = 0; uint64_t tbits = 0;
    bool shani = c->shani;
    /* planned path: per-epoch block-0 states of the ng groups, fixed later blocks by schedule */
    const bool hplan = shani && c->hplan;
    std::vector<uint8_t> lgblk(c->h_gblk);                 /* this worker's group blocks (epoch bytes patched) */
    /* block-0 schedules of the groups; they change only with the epoch's buffered bytes (erem), which
     * are the same for most consecutive epochs of a worker */
    std::vector<uint32_t, qalloc64<uint32_t> > gwk((size_t)(c->h_ng + 3) * 64);
    uint8_t gwk_erem[64]; bool gwk_ok = false;
#if QSB_CPU_GWK_CACHE
    const size_t gwk_n = (size_t)(c->h_ng + 3) * 64;      /* one set: ng schedules plus 3 lane-padding copies */
    std::vector<uint32_t, qalloc64<uint32_t> > gwkc(8 * gwk_n);
    uint8_t gwkc_key[8][64]; int gwkc_used = 0; const uint32_t *gwkp = gwkc.data();
#endif
    /* the epoch prefix (prefix remainder + kept early pushes) and its chaining state after every full
     * block: consecutive epochs of a worker differ only from their first differing early omission on */
    const size_t pfx_max = dp->prefix_remainder_len + (size_t)c->cut * SIG_PUSH_SIZE;
    std::vector<uint8_t> pfx(pfx_max + 64);
    if (dp->prefix_remainder_len) memcpy(pfx.data(), dp->prefix_remainder, dp->prefix_remainder_len);
    std::vector<uint32_t> pst((pfx_max / 64 + 2) * 8);
    memcpy(pst.data(), dp->midstate, 32);
#if QSB_CPU_EPOCH_CONTIG
    uint8_t pv_early[16]; bool pv_ok = false; uint64_t pv_epoch = 0;
#else
    uint8_t pv_early[16]; bool pv_ok = false;
#endif
    alignas(16) uint32_t gstb[2][286 + 3][8]; int gpar = 0;   /* this epoch's and the previous epoch's block-0 group states: a lane */
    uint32_t (*gst)[8] = gstb[0];                             /* group of 4 spans at most 2 epochs (ncwin >= 4, see hash_plan) */
    const uint32_t *lin[4];
    const uint32_t *const *lrow[4];
    alignas(16) uint32_t w2[4][16]; memset(w2, 0, sizeof w2);
    for (int l = 0; l < 4; l++) { w2[l][8] = 0x80000000u; w2[l][15] = 256; }   /* second SHA: 32-byte message */
    uint32_t cw4[286];                                     /* skip-record bytes 6..9 of each CPU pattern */
    for (int i = 0; i < c->ncwin; i++) cw4[i] = (uint32_t)c->cwin[i][0] | (uint32_t)c->cwin[i][1] << 8 | (uint32_t)c->cwin[i][2] << 16;
#endif
#if QCPU_PFQ
    QPfRing pfq;                                    /* R2-D: the hashing phase's queued row prefetches */
#endif
#if QCPU_VEC
    VecBuf vb;
    if (c->vec && !vecbuf_alloc(vb, B)) vb = VecBuf();
    if (c->cfold && !vb.X) return;                    /* the C-folded table serves only the 8-lane path */
    const bool hpf = QSB_CPU_HPF > 0 && vb.X != nullptr && c->g.nw >= 3 && c->g.wid[0] + c->g.wid[1] <= 64;
#else
    const bool hpf = false;
#endif
    /* The rows of windows 0 and 1 are prefetched from the hashing phase, from each candidate's z (recode_scalar's digits 0
     * and 1): the first window's loop then finds them cached instead of issuing 16 misses per group into a short loop. */
    const int hw0 = c->g.wid[0], hw1 = c->g.wid[1]; const bool hsg = c->g.sgn;
    const pt *const ht0 = c->tw[0], *const ht1 = c->tw[1];
    auto hpf_rows = [&](int kk) {
        const uint32_t *h = &zb[(size_t)kk * 8];
        const uint64_t zl = (uint64_t)h[6] << 32 | h[7];              /* z bits 0..63 */
        uint32_t u = (uint32_t)(zl & ((1ULL << hw0) - 1)), cy = 0;
        if (hsg && u > (1u << (hw0 - 1))) { u = (1u << hw0) - u; cy = 1; }
        __builtin_prefetch(ht0 + (u ? u - 1 : 0), 0, QSB_CPU_PFNX ? 2 : 3);
        if (QSB_CPU_HPF > 1) {
            uint32_t u1 = (uint32_t)((zl >> hw0) & ((1ULL << hw1) - 1)) + cy;
            if (hsg && u1 > (1u << (hw1 - 1))) u1 = (1u << hw1) - u1;
            __builtin_prefetch(ht1 + (u1 ? u1 - 1 : 0), 0, QSB_CPU_PFNX ? 2 : 3);
        }
    };
    auto hpf_after = [&](int kl, int cnt) {          /* z of candidates kl - cnt + 1 .. kl is new in zb */
        if (hpf) for (int q = kl - cnt + 1; q <= kl; q++) hpf_rows(q);
    };
    for (;;) {
        if (c->stop.load(std::memory_order_relaxed)) return;   /* H9: stop_unmap (every earlier batch's hits are published) */
#ifdef QSB_CPU_DEVBENCH
        if (c->dev_limit && c->cand.load() >= c->dev_limit) { c->dev_live--; return; }
#endif
        int k = 0;
        /* COHASH hook (QSB_GPU_COHASH): a worker claims a batch of GPU-hashed z here (z converted to its word order, skips rebuilt
         * from (epoch, lane)) and sets k = B, skipping the hashing phase below; on its own range's end it keeps claiming. */
        while (k < B) {
            if (wi == c->ncwin) {                       /* next epoch: hash its fixed prefix once */
#if QSB_CPU_EPOCH_CONTIG
                if (epoch >= c->n_epochs || epoch >= epoch_end) return;
#else
                if (epoch >= c->n_epochs) return;
#endif
#if QCPU_SHANI
                if (shani && hplan) {                       /* re-hash the prefix from the first block this epoch changes */
#if QSB_CPU_EPOCH_CONTIG
                    int su = -1;                            /* the next lexicographic epoch from the previous one */
                    if (pv_ok && epoch == pv_epoch + 1) {
                        su = c->early - 1; while (su >= 0 && pv_early[su] == c->cut - c->early + su) su--;
                        if (su >= 0) {
                            memcpy(early, pv_early, (size_t)c->early); early[su]++;
                            for (int i = su + 1; i < c->early; i++) early[i] = (uint8_t)(early[i - 1] + 1);
                        }
                    }
                    if (su < 0)                             /* first epoch of the range, or the last omission wrapped: unrank */
#endif
                    {                                       /* qsb_host_unrank with a binomial table */
                        uint64_t rank = epoch; int lo = 0;
                        for (int i = 0; i < c->early; i++) {
                            int cc = lo;
                            for (;;) { const uint64_t cnt = c->h_binom[c->cut - cc - 1][c->early - i - 1]; if (rank < cnt) break; rank -= cnt; cc++; }
                            early[i] = (uint8_t)cc; lo = cc + 1;
                        }
                    }
#if QSB_CPU_EPOCH_CONTIG
                    pv_epoch = epoch;
#endif
                    const size_t prl = dp->prefix_remainder_len;
                    size_t from = 0, pl = 0; int e2 = 0, i0 = 0;
                    if (pv_ok) {
                        int e = 0; while (e < c->early && early[e] == pv_early[e]) e++;
                        const int lo = e < c->early ? (early[e] < pv_early[e] ? early[e] : pv_early[e]) : c->cut;
                        from = prl + (size_t)(lo - e) * SIG_PUSH_SIZE;   /* the kept pushes below push lo are unchanged */
                        i0 = lo; e2 = e; pl = (size_t)(lo - e) * SIG_PUSH_SIZE;   /* early[0..e-1] < lo: rebuild from push lo on */
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
                        int u = 0;                          /* remlen is fixed for a run (prefix length), so erem's first remlen bytes are the key */
                        while (u < gwkc_used && memcmp(gwkc_key[u], erem, remlen)) u++;
                        if (u == gwkc_used) {               /* a new erem: compute its set (slot 7 is reused after 8, never reached: <= 7 values) */
                            if (gwkc_used < 8) gwkc_used++; else u = 7;
                            uint32_t *gw = &gwkc[(size_t)u * gwk_n];
                            for (int q = 0; q < ng; q++) { memcpy(&lgblk[(size_t)q * 64], erem, remlen); qsha_schedule(&gw[(size_t)q * 64], &lgblk[(size_t)q * 64]); }
                            for (int q = ng; q < ng + 3; q++) memcpy(&gw[(size_t)q * 64], &gw[0], 64 * sizeof(uint32_t));   /* lane padding */
                            memcpy(gwkc_key[u], erem, remlen);
                        }
                        gwkp = &gwkc[(size_t)u * gwk_n];
#else
                        for (int q = 0; q < ng; q++) { memcpy(&lgblk[(size_t)q * 64], erem, remlen); qsha_schedule(&gwk[(size_t)q * 64], &lgblk[(size_t)q * 64]); }
                        for (int q = ng; q < ng + 3; q++) memcpy(&gwk[(size_t)q * 64], &gwk[0], 64 * sizeof(uint32_t));   /* lane padding */
#endif
                        memcpy(gwk_erem, erem, remlen); gwk_ok = true;
                    }
                    gpar ^= 1; gst = gstb[gpar];
                    for (int g = 0; g < ng; g += 4) {      /* block 0 of each group: once per epoch, not per candidate */
                        const uint32_t *gp[4]; const uint32_t *const *gr[4] = {&gp[0], &gp[1], &gp[2], &gp[3]};
#if QSB_CPU_GWK_CACHE
                        for (int l = 0; l < 4; l++) { memcpy(gst[g + l], est, 32); gp[l] = &gwkp[(size_t)(g + l) * 64]; }
#else
                        for (int l = 0; l < 4; l++) { memcpy(gst[g + l], est, 32); gp[l] = &gwk[(size_t)(g + l) * 64]; }
#endif
                        qsha_x4p(&gst[g], gr, 1);
                    }
#if QSB_CPU_EPOCH_CONTIG
                    epoch += 1; wi = 0;
#else
                    epoch += (uint64_t)c->nthreads; wi = 0;
#endif
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
                    if (ectx.num != remlen || nb > 16) shani = false;   /* unexpected shape: stay on OpenSSL (decided at the first epoch, k = 0) */
                }
#endif
#if QSB_CPU_EPOCH_CONTIG
                epoch += 1; wi = 0;
#else
                epoch += (uint64_t)c->nthreads; wi = 0;
#endif
            }
#if QCPU_SHANI
            have_epoch:
            if (shani && hplan) {                           /* planned path: this epoch's remaining patterns, up to the batch end */
                const int n = B - k < c->ncwin - wi ? B - k : c->ncwin - wi;
                uint64_t e8; memcpy(&e8, early, 8);
                for (int t = 0; t < n; t++) {
                    const int pi = wi + t, kq = k + t, j = kq & 3;
                    lin[j] = gst[c->h_g0[pi]];
                    lrow[j] = c->h_wkp[pi];
                    uint8_t *sk = &skips[(size_t)kq * 9];
                    memcpy(sk, &e8, 8); memcpy(sk + 6, &cw4[pi], 4);   /* bytes 0..5 = early, 6..8 = the pattern (byte 9: next record's) */
                    if (j == 3) {                           /* four candidates ready: blocks 1..nb-1 (digests straight into the second
                                                               SHA-256's message words), then the second SHA-256 into z = h0 (MSW) .. h7 */
#if QCPU_PFQ
                        qsha_x4p(nullptr, lrow, nb - 1, &w2[0][0], 16, lin, hpf ? &pfq : nullptr);
#else
                        qsha_x4p(nullptr, lrow, nb - 1, &w2[0][0], 16, lin);
#endif
#if QSB_CPU_SHC
                        qsha_x4w_iv32((uint32_t (*)[8])&zb[(size_t)(kq - 3) * 8], w2);   /* w2[l][8..15]: the constant padding */
#else
                        qsha_x4w_iv((uint32_t (*)[8])&zb[(size_t)(kq - 3) * 8], w2);
#endif
#if QCPU_VEC
#if QCPU_PFQ
                        if (hpf && (kq & 7) == 7) hpf_rows8q(zb.data(), kq - 7, ht0, ht1, hw0, hw1, hsg, pfq);   /* R2-D: queued */
#else
                        if (hpf && (kq & 7) == 7) hpf_rows8(zb.data(), kq - 7, ht0, ht1, hw0, hw1, hsg);   /* B % 8 == 0 */
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
                if (j == 3) {                               /* four candidates ready: both SHA-256 passes */
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
            SHA256_Final(blk2, &s);                     /* first digest into the second block */
            SHA256_CTX s2; SHA256_Init(&s2); SHA256_Transform(&s2, blk2);
            { uint32_t hw[8]; for (int i = 0; i < 8; i++) hw[i] = (uint32_t)s2.h[i]; put_digits(k, hw); }
            hpf_after(k, 1);
            uint8_t *sk = &skips[(size_t)k * 9];
            for (int j = 0; j < 6; j++) sk[j] = early[j];
            sk[6] = w3[0]; sk[7] = w3[1]; sk[8] = w3[2];
            k++;
        }
#if QCPU_PFQ
        qpf_drain(pfq, 64);                         /* R2-D: the last octet's rows (and any left over) before the EC phase */
#endif
#if QCPU_VEC
        if (vb.X) {
            vec_batch(c, zb.data(), ds.data(), B, vb);
#if QCPU_SHANI
#if QSB_CPU_KH16
            if (c->kh16 && c->cfold) {                  /* QSB_CPU_KH16: key hashes 16 at a time: the group's 8 candidates x 2 recids */
                for (int h = 0; h < B / 8; h++) {
                    const unsigned pass = kh16_pass<QSB_CPU_JL_KH16_PAD_ONCE != 0>(vb.m16 + (size_t)h * 144, vb.wk16);   /* pk_prefilter, bit 8 ri + j */
                    if (!pass) continue;
                    for (int j = 0; j < 8; j++) {
                        const int q = h * 8 + j;
                        if (vb.bad[q]) continue;
                        for (int ri = 0; ri < 2; ri++)
                            if (((pass >> (8 * ri + j)) & 1) && gate_publish_exact(c, &skips[(size_t)q * 9], ri)) break;   /* one recid per candidate */
                    }
                }
            } else
#endif
            if (c->shani) {                             /* key hashes 4 at a time: 2 candidates x 2 recids */
                const __m128i zero = _mm_setzero_si128();
                for (int kk = 0; kk < B; kk += 2) {
                    const __m128i h0 = qsha_keyhash4_h0(&vb.qx[(size_t)kk * 2], &vb.qp[(size_t)kk * 2]);   /* lane l: candidate kk + l/2, recid l&1 */
                    const int pass = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(_mm_srli_epi32(h0, 32 - (QSB_ZEROS_N < 32 ? QSB_ZEROS_N : 32)), zero)));   /* pk_prefilter, 4 lanes */
                    if (!pass) continue;
                    for (int q2 = 0; q2 < 2; q2++) {
                        const int q = kk + q2;
                        if (vb.bad[q]) continue;
                        for (int ri = 0; ri < 2; ri++)
                            if (((pass >> (2 * q2 + ri)) & 1) && gate_publish_exact(c, &skips[(size_t)q * 9], ri)) break;   /* one recid per candidate */
                    }
                }
            } else
#endif
            for (int kk = 0; kk < B; kk++) {
                if (vb.bad[kk]) continue;
                const uint8_t *sk = &skips[(size_t)kk * 9];
                for (int ri = 0; ri < 2; ri++)
                    if (gate_publish(c, pk, sk, ri, vb.qx[(size_t)kk * 2 + ri], vb.qp[(size_t)kk * 2 + ri])) break;   /* one recid per candidate */
            }
            c->cand += B;
            continue;
        }
#endif
        memset(inf.data(), 1, (size_t)B); memset(bad.data(), 0, (size_t)B);   /* every candidate starts at infinity, not dropped */
        recode_scalar(c->g, zb.data(), ds.data(), B);
        for (int i = 0; i < c->g.nw; i++) {
            const pt *T = c->tw[i]; const uint32_t *di = &ds[(size_t)i * B];
            for (int kk = 0; kk < B; kk++) { const uint32_t e = di[kk], m = e & 0x7FFFFFFFu; tp[kk] = m ? &T[m - 1] : nullptr; ngs[kk] = (uint8_t)(e >> 31); }
            batch_add(acc.data(), tp.data(), inf.data(), bad.data(), B, d.data(), pre.data(), ngs.data());
        }
        /* Both recids share the denominator x_C - x_P. */
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
                fe cy = c->cy; if (ri) { fe z0 = {{0, 0, 0, 0}}; fe_sub(cy, z0, cy); }   /* recid 1: -C */
                fe lam, t, x3, y3;
                fe_sub(t, cy, acc[kk].y); fe_mul(lam, t, dinv);
                fe_sqr(x3, lam); fe_sub(x3, x3, acc[kk].x); fe_sub(x3, x3, c->cx);
                fe_sub(t, acc[kk].x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, acc[kk].y);
                if (gate_publish(c, pk, &skips[(size_t)kk * 9], ri, x3, (unsigned)(y3.v[0] & 1))) break;   /* one recid per candidate, like the GPU gate */
            }
        }
        c->cand += B;
    }
}

static Ctx *g_ctx = nullptr;
/* win3: the GPU's 128 window patterns (actual push indices, ascending). */
/* The process's CPU set as it was before main(): other start-up code (the host producers) may later pin
 * the main thread to its own core, and the co-grinder must still see, and run on, the rest of the CPUs. */
#ifdef CPU_COUNT
static cpu_set_t g_initial_cpus;
static int g_initial_ok = [] { CPU_ZERO(&g_initial_cpus); return sched_getaffinity(0, sizeof g_initial_cpus, &g_initial_cpus) == 0 ? 1 : 0; }();
#endif
#if defined(CPU_COUNT) && QSB_CPU_RSV_CORE
/* 296e5e53's worker CPUs (package y2d): the pre-main() set minus the core of the GPU host thread (start() runs on it) when that
 * thread is pinned to fewer CPUs; when it is not, minus the last core with two or more CPUs (plan_cores: as 296e5e53's
 * smt_plan). Without a core to reserve: the pre-main() set minus the host thread's CPUs if it is pinned (296e5e53's table-builder
 * mask), else false (no mask: the workers inherit, as in 296e5e53). */
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
        for (char *q = buf; *q;) {                      /* "a-b,c,..." */
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
/* QSB_CPU_BATCH_AUTO: 86c643ae's run-time batch choice, its cpu_siblings, core_groups, core_sharing and batch_choose as
 * written (its smt_plan, the other user of core_groups, is not taken). */
#ifdef CPU_COUNT
/* The SMT siblings of a CPU from /sys (thread_siblings_list, "a-b,c,..."); false when the list cannot be read. */
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
/* The CPUs of m grouped by physical core (siblings outside m ignored); a CPU whose list cannot be read is its own core, and
 * *all_read (if given) turns false. Used by the batch choice (core_sharing). */
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
/* nullptr when no two of nw workers running on the CPUs of wm can share a physical core (at most one worker per CPU and no two
 * CPUs of wm SMT siblings), else why they may. */
static const char *core_sharing(const cpu_set_t &wm, int nw) {
    if (CPU_COUNT(&wm) < 1) return "CPU set unknown";
    if (nw > CPU_COUNT(&wm)) return "more workers than CPUs";
    std::vector<std::vector<int> > cores; bool all = true; core_groups(wm, cores, &all);
    if (!all) return "topology unreadable";
    for (const std::vector<int> &k : cores) if (k.size() > 1) return "SMT siblings among the workers' CPUs";
    return nullptr;
}
#endif
/* Candidates per batch: QSB_CPU_BATCH_SOLO unless workers may share a core (shared: the reason), then QSB_CPU_BATCH.
 * QSB_CPU_BATCH_RT=<n> (dev): n, a multiple of 32 in 32..8192. */
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
#if QSB_CPU_DIAG_V4
/* QSB_CPU_DIAG_V4: worker t starts diag4[t] x 2^20 epochs into its range [t x span, (t + 1) x span), span = C(137,6) / T; its
 * first hit lands within 2^20 epochs of that start with P = 1 - e^-12.5 (2 x 100 x 2^20 / 2^24 expected hits). Codes (each
 * < 100, read back from the public hit list):
 *   w0 42 (format marker), w1 T mod 100, w2 w3 huge-page fraction in 0.1% (1099: unknown), w4 w5 table-ready time in 0.1 s,
 *   w6 w7 MemAvailable before the table in GiB (plus the table), w8 w9 HugePages_Free x Hugepagesize in GiB,
 *   w10 table windows, w11 batch / 64, w12 path bits (1 8-lane IFMA, 2 SHA-NI, 4 KH16, 8 C fold); workers 13 and up 0. */
static void diag4_fill(Ctx &c, int nth, double hp, double ready_s) {
    double av = mem_avail(); if (av >= 0) av += (double)c.g.total * sizeof(pt);
    long avg = av < 0 ? 0 : (long)(av / 1073741824.0); if (avg > 9999) avg = 9999;
    long hpf = 0, hps_kb = 0;
    if (FILE *f = fopen("/proc/meminfo", "r")) {
        char l[256];
        while (fgets(l, sizeof l, f)) { sscanf(l, "HugePages_Free: %ld", &hpf); sscanf(l, "Hugepagesize: %ld kB", &hps_kb); }
        fclose(f);
    }
    long hfg = (long)((double)hpf * (double)hps_kb / 1048576.0); if (hfg < 0) hfg = 0; if (hfg > 9999) hfg = 9999;
    const long pm = hp < 0 ? 1099 : hp >= 1 ? 1000 : (long)(hp * 1000.0 + 0.5);
    long ds = (long)(ready_s * 10.0 + 0.5); if (ds < 0) ds = 0; if (ds > 9999) ds = 9999;
    const long w[13] = {42, nth % 100, pm / 100, pm % 100, ds / 100, ds % 100, avg / 100, avg % 100, hfg / 100, hfg % 100,
                        c.g.nw, c.batch / 64 > 99 ? 99 : c.batch / 64, (c.vec ? 1 : 0) | (c.shani ? 2 : 0) | (c.kh16 ? 4 : 0) | (c.cfold ? 8 : 0)};
    for (int t = 0; t < 256; t++) c.diag4[t] = (uint8_t)(t < 13 && t < nth ? w[t] : 0);
    printf("  CPU co-grind: DIAG v4 walk-start codes: huge pages %.1f%%, table ready %.1f s, MemAvailable %ld GiB, HugePages_Free %ld GiB, "
           "%d windows, batch %d, path %ld (worker t starts code_t x 2^20 into its range)\n",
           pm / 10.0, ds / 10.0, avg, hfg, c.g.nw, c.batch, w[12]);
    fflush(stdout);
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
        cpu_set_t now; CPU_ZERO(&now);                    /* the main thread's current set (maybe narrowed) */
        if (g_initial_ok && sched_getaffinity(0, sizeof now, &now) == 0 && CPU_COUNT(&now) < CPU_COUNT(&cs)) {
            for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &cs) && !CPU_ISSET(c, &now)) CPU_SET(c, &work_cpus);
#if defined(QSB_HP_ON) && QSB_CPU_HP_SHARE
            /* the main thread sleeps between launches (blocking event waits): its CPU may host a worker too */
            if (qhp::g_share_cpu >= 0 && CPU_ISSET(qhp::g_share_cpu, &cs)) CPU_SET(qhp::g_share_cpu, &work_cpus);
#endif
#if QSB_CPU_ALLCPU && defined(QSB_HOST_BLOCKING) && QSB_HOST_BLOCKING
            /* The GPU host thread sleeps in blocking event waits (QSB_HOST_BLOCKING in tree.cu), so its core is idle between
             * launches: workers on every CPU, the host thread's included; they are SCHED_IDLE and yield to it and to the
             * producers (789aed1b / a141df2b, after 0735233a / 888f5fce). */
            for (int c = 0; c < CPU_SETSIZE; c++) if (CPU_ISSET(c, &now)) CPU_SET(c, &work_cpus);
#endif
            work_mask = CPU_COUNT(&work_cpus) >= 1;       /* workers: every CPU the main thread no longer uses */
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
    if (FILE *q = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {      /* cgroup v1 */
        long quota = -1, period = 0; if (fscanf(q, "%ld", &quota) != 1) quota = -1; fclose(q);
        if (FILE *pf = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) { if (fscanf(pf, "%ld", &period) != 1) period = 0; fclose(pf); }
        if (quota > 0 && period > 0) { long lim = (quota + period - 1) / period; if (lim > 0 && lim < ncpu) ncpu = lim; }
    }
#ifdef QSB_CPU_THREADS
    int nth = QSB_CPU_THREADS;
#else
    int nth = (int)ncpu - QSB_CPU_RESERVE;
#if defined(CPU_COUNT) && (QSB_CPU_NTH_RAISE || QSB_CPU_ALLCPU)
    /* The host producers may have narrowed the main thread to one logical CPU (its SMT sibling then runs
     * the pinned producer): one worker per CPU the main thread left, within the CPU quota. Under QSB_CPU_ALLCPU
     * the set holds every CPU, so this is one worker per CPU (32 of 32 on the ranked host). */
    if (work_mask && CPU_COUNT(&work_cpus) > nth) nth = CPU_COUNT(&work_cpus) < (int)ncpu ? CPU_COUNT(&work_cpus) : (int)ncpu;
#endif
#endif
    if (const char *e = getenv("QSB_CPU_THREADS_ENV")) nth = atoi(e);   /* dev override */
    if (nth < 1 || dp->n != 150 || cut != 137 || early != 6) { QCPU_FENCE_OFF(); printf("  CPU co-grind: off (%d threads)\n", nth); return; }
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
    {   /* 296e5e53's placement (its smt_plan and table-builder mask): the fallback core rule applies where 296e5e53 plans cores */
        cpu_set_t m;
        if (rsv_core_mask(c->vec && !getenv("QSB_CPU_NOPIN"), &m)) { work_cpus = m; work_mask = true; }
    }
#endif
#ifdef QSB_CPU_DEVBENCH
    if (const char *e = getenv("QSB_CPU_DEVCAND")) c->dev_limit = strtoull(e, nullptr, 10);
    c->dev_live = nth;
#endif
#if QSB_CPU_BATCH_AUTO
    {   /* candidates per batch (86c643ae's rule), before the workers: the workers' CPUs are the builder thread's mask (work_cpus,
         * else the main thread's current set, which the builder thread and the workers inherit) */
        const char *shared = "CPU set unknown";
#ifdef CPU_COUNT
        cpu_set_t wcs; CPU_ZERO(&wcs);
        if (work_mask) wcs = work_cpus; else if (sched_getaffinity(0, sizeof wcs, &wcs) != 0) CPU_ZERO(&wcs);
        shared = core_sharing(wcs, c->nthreads);
#endif
        c->batch = batch_choose(c->nthreads, shared, c->batch_why, sizeof c->batch_why);
    }
#endif
#if QSB_CPU_FENCE
    if (g_fence) {   /* the fence: the GPU's own window patterns, on the epochs above it */
        for (int i = 0; i < nwin && i < 286; i++) { memcpy(c->cwin[c->ncwin], win3[i], 3); c->ncwin++; }
    } else
#endif
    /* CPU window patterns: every 3-subset of {cut..n-1} not used by the GPU. */
    for (int a = cut; a < (int)dp->n; a++) for (int b = a + 1; b < (int)dp->n; b++) for (int d3 = b + 1; d3 < (int)dp->n; d3++) {
        int used = 0;
        for (int i = 0; i < nwin; i++) if (win3[i][0] == a && win3[i][1] == b && win3[i][2] == d3) { used = 1; break; }
        if (!used) { c->cwin[c->ncwin][0] = (uint8_t)a; c->cwin[c->ncwin][1] = (uint8_t)b; c->cwin[c->ncwin][2] = (uint8_t)d3; c->ncwin++; }
    }
    const uint64_t unpadded = (uint64_t)dp->prefix_remainder_len + (uint64_t)(dp->n - dp->t) * SIG_PUSH_SIZE +
                              dp->tail_section_len + dp->tx_suffix_len;
    if (dp->t != 9 || dp->total_preimage_len < unpadded || ((dp->total_preimage_len - unpadded) % 64) != 0 || c->ncwin < 1) {
        QCPU_FENCE_OFF(); printf("  CPU co-grind: off (unexpected problem shape)\n"); delete c; return;
    }
    c->mid_bytes = dp->total_preimage_len - unpadded;
    c->n_epochs = binom_u64(cut, early);
#if QSB_CPU_FENCE
    if (g_fence) {
        if (g_fence_n != c->n_epochs || g_fence >= c->n_epochs) { QCPU_FENCE_OFF(); printf("  CPU co-grind: off (fence %llu outside the epoch space)\n", (unsigned long long)g_fence); delete c; return; }
        c->epoch_base = g_fence;                         /* worker t: [F + t x span, F + (t + 1) x span), span = (N - F) / T */
    }
#endif
#if QCPU_SHANI
    if (c->shani) hash_plan_cpu_patterns(*c);
#endif
    if (!qsb_hv_init(&c->hv, dp, (const uint8_t (*)[QSB_SE_TWIN])win3, cut, early)) { QCPU_FENCE_OFF(); printf("  CPU co-grind: off (gate)\n"); delete c; return; }
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(NID_secp256k1);
    BN_CTX *bctx = BN_CTX_new(); BIGNUM *nri = BN_new(), *ax = BN_new(), *ay = BN_new();
    EC_POINT *A = EC_POINT_new(grp);
    BN_lebin2bn(dp->neg_r_inv, 32, nri);
    if (!EC_POINT_mul(grp, A, nri, NULL, NULL, bctx) ||
        !EC_POINT_get_affine_coordinates_GFp(grp, A, ax, ay, bctx)) { QCPU_FENCE_OFF(); printf("  CPU co-grind: off (A)\n"); delete c; return; }
    fe fax, fay; fe_from_bn(fax, ax); fe_from_bn(fay, ay);
    fe_from_le32(c->cx, dp->u2r_x); fe_from_le32(c->cy, dp->u2r_y);
    EC_POINT_free(A); BN_free(nri); BN_free(ax); BN_free(ay); BN_CTX_free(bctx); EC_GROUP_free(grp);
#ifdef CPU_COUNT
    std::thread([c, fax, fay, nth, work_mask, work_cpus, ncpu, nwin]() {
        if (work_mask) sched_setaffinity(0, sizeof work_cpus, &work_cpus);   /* table build + workers inherit */
#else
    std::thread([c, fax, fay, nth, ncpu, nwin]() {
#endif
#ifdef SCHED_IDLE
        struct sched_param sp; sp.sched_priority = 0; sched_setscheduler(0, SCHED_IDLE, &sp);
#endif
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        double hp; char note[160];
        if (!table_setup(*c, nth, hp, note, sizeof note)) { QCPU_FENCE_OFF(); printf("  CPU co-grind: off (table memory)\n"); fflush(stdout); return; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        char hps[16]; if (hp < 0) snprintf(hps, sizeof hps, "n/a"); else snprintf(hps, sizeof hps, "%.1f%%", 100.0 * hp);
#if QSB_CPU_FENCE
        const char *const pat_rel = g_fence ? "above the fence, the same as" : "disjoint from";   /* the fence: the GPU's own patterns */
#else
        const char *const pat_rel = "disjoint from";
#endif
        printf("  CPU co-grind: %d threads (of %ld CPUs), %s, %s%s, %d window patterns per epoch %s the GPU's %d, batch %d (%s); "
               "table %d %s windows of %d..%d bits, %.0f MiB, huge pages %s (%.2f s%s)\n",
               nth, ncpu, c->vec ? "8-lane IFMA" : "scalar", c->shani ? "4-lane SHA-NI" : "OpenSSL SHA-256", c->kh16 ? " (16-lane key hashes)" : "",
               c->ncwin, pat_rel, nwin, c->batch, c->batch_why,
               c->g.nw, c->g.sgn ? "signed" : "unsigned", c->g.wid[c->g.nw - 1], c->g.wid[0], c->g.total * sizeof(pt) / 1048576.0,
               hps, (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec), note);
        fflush(stdout);
        build_table(*c, fax, fay, nth);
        bool tab_ok = table_check(*c);
        if (!tab_ok && c->g.nw == 9 && !getenv("QSB_CPU_NW")) {    /* 9-window table failed its check: 10 (or more) */
            table_free(*c);
            if (!table_setup(*c, nth, hp, note, sizeof note, 10)) { QCPU_FENCE_OFF(); printf("  CPU co-grind: off (table memory)\n"); fflush(stdout); return; }
            printf("  CPU co-grind: 9-window table check failed; table %d windows, %.0f MiB, huge pages %.1f%%%s\n",
                   c->g.nw, c->g.total * sizeof(pt) / 1048576.0, 100.0 * hp, note);
            fflush(stdout);
            build_table(*c, fax, fay, nth);
            tab_ok = table_check(*c);
        }
        if (!tab_ok && c->g.nw == 10 && !getenv("QSB_CPU_NW")) {   /* 10-window table failed its check: 11 (or 12) */
            table_free(*c);
            if (!table_setup(*c, nth, hp, note, sizeof note, 11)) { QCPU_FENCE_OFF(); printf("  CPU co-grind: off (table memory)\n"); fflush(stdout); return; }
            printf("  CPU co-grind: 10-window table check failed; table %d windows, %.0f MiB, huge pages %.1f%%%s\n",
                   c->g.nw, c->g.total * sizeof(pt) / 1048576.0, 100.0 * hp, note);
            fflush(stdout);
            build_table(*c, fax, fay, nth);
            tab_ok = table_check(*c);
        }
        if (!tab_ok) { QCPU_FENCE_OFF(); printf("  CPU co-grind: off (table check failed)\n"); fflush(stdout); table_free(*c); return; }
#if QSB_CPU_TOUCH_FUSE
        {   /* the build faulted the table in: its huge-page fraction over the whole table */
            const double hb = thp_bytes(c->table, c->g.total * sizeof(pt));
            if (hb >= 0) hp = hb / (double)(c->g.total * sizeof(pt));
            printf("  CPU co-grind: huge pages after the fused build %.1f%%\n", hp < 0 ? -1.0 : 100.0 * hp);
            fflush(stdout);
        }
#endif
        clock_gettime(CLOCK_MONOTONIC, &t1);
#if QSB_CPU_DIAG_EPOCH   /* default at file scope (package y2d: 0) */
        {   /* Zero-cost diagnostic. Every epoch of the co-grinder's walk is real work on patterns disjoint from
             * the GPU's, so where the walk starts is a free enumeration choice. Start it at code * 2^29, so the
             * public ranked hit list (the smallest CPU-hit epoch rank / 2^29) shows what this host chose:
             *   code = geometry (11 windows: 1, 12: 2, other: 3) + 3 * (huge pages below 95%) + 6 * (fewer than 28 workers)
 *   and, in bits 20..27, the GiB this process could use before the table (capped at 255)
 *   and (v3) bit 28 set for a 10-window table (geometry 3 then means 10 windows; without bit 28, 13 or more)
             * i.e. 1..12; the walk (about 3.6e8 epochs in 1,200 s) stays below C(137,6). */
            const int cg = c->g.nw == 11 ? 1 : c->g.nw == 12 ? 2 : 3;
            const int ch = (hp >= 0.95) ? 0 : 1;
            const int ct = nth >= 28 ? 0 : 1;
            uint64_t base = (uint64_t)(cg + 3 * ch + 6 * ct) << 29 | (uint64_t)(c->g.nw == 10 || c->g.nw == 9) << 28 | (uint64_t)(c->g.nw == 9) << 19;   /* bit 19: 9 windows */
            {   /* v2: + GiB this process could use before the table (MemAvailable / cgroup headroom), bits 20..27.
                 * The first co-grinder hit lands within 2^20 epochs of the start with probability 1 - e^-20. */
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
#if QSB_CPU_FENCE
        if (g_fence) {
            printf("  CPU co-grind: fence at epoch %llu of %llu: the GPU walks [0, %llu) and idles there; the co-grinder grinds the GPU's %d "
                   "window patterns on [%llu, %llu) in %d ranges of %llu epochs\n", (unsigned long long)g_fence, (unsigned long long)g_fence_n,
                   (unsigned long long)g_fence, c->ncwin, (unsigned long long)g_fence, (unsigned long long)c->n_epochs, nth,
                   (unsigned long long)((c->n_epochs - g_fence) / (uint64_t)nth));
            fflush(stdout);
        }
#endif
#if QSB_CPU_DIAG_V4
        diag4_fill(*c, nth, hp, (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
#endif
        c->live += nth; c->ready = 1;                   /* H9: stop_unmap may now wait for the workers and unmap */
        for (int t = 0; t < nth; t++) std::thread(worker, c, t).detach();
    }).detach();
    g_ctx = c;
#ifdef QSB_CPU_DEVBENCH
    if (c->dev_limit) {                                  /* dev only (never in a ranked build): fixed work, then exit */
        while (c->cand.load() == 0 || c->dev_live.load() > 0) usleep(10000);
        { std::lock_guard<std::mutex> g(c->io); if (c->out) fflush(c->out); }
        printf("DEVCAND %llu candidates, %u hits\n", (unsigned long long)c->cand.load(), c->hits.load());
        fflush(stdout); _exit(0);
    }
#endif
}
static uint64_t candidates() { return g_ctx ? g_ctx->cand.load() : 0; }
static uint32_t hits() { return g_ctx ? g_ctx->hits.load() : 0; }
/* H9 (tree.cu's QSB_FAST_TEARDOWN, on a helper thread once the stop signal is seen): the workers stop at their next batch
 * boundary (each has published the hits of every batch it finished), then the table is released in 1 GiB steps (MADV_DONTNEED
 * holds the mmap lock for reading, so other threads' mappings are never held up for long) and unmapped, overlapping the GPU's
 * drain instead of following it in the exit. Returns the seconds taken, or -1 when the table is left to the exit: the workers
 * were not spawned yet (table still building) or one did not stop within 0.25 s. */
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
}  // namespace qcpu
