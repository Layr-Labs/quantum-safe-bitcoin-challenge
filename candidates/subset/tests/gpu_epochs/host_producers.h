/* host_producers.h -- QSB_HOST_PRODUCERS: exact host twins of the per-batch producer kernels .
 *
 * Host-only code: the device image (and the carrier cubin) is unchanged, and none of the switches below is in
 * QSB_CARRIER_KNOBS. This file only picks the producers implementation; the two files it includes are unchanged
 * copies of their sources and export the same qhp:: interface (start, acquire, upload, release, stats, shutdown).
 *
 * QSB_HP_V3 (default 1): host_producers_v3.h, the producers of public subset submissions 4da17ebc / 3ff68d21 /
 *   57065b7d / 2f690f1a (blob c59fd0b4, after 2d1631b0 / 789aed1b / a141df2b): every SHA-NI compression uses a
 *   precomputed message schedule (W[i] + K[i] rows of the omission-free stream, first-block class rows cached per
 *   thread), and the placement switches QSB_HP_PLACE (default 0: the main thread on both SMT siblings of its core,
 *   QSB_HP_THREADS floating producers, as before) / QSB_HP_HELPER / QSB_HP_BLOCKSYNC. At PLACE 0 the helper and
 *   qhp::g_share_cpu are inert; qhp::stats() gains a defaulted sixth argument (helper chunks).
 * QSB_HP_V3 0: host_producers_v1.h, this tree's previous producers (blob 382acc1d: 4-lane SHA-NI or the 16-lane
 *   AVX-512 path chosen by a batch-0 calibration), byte for byte. */
#pragma once
#ifndef QSB_HP_V3
#define QSB_HP_V3 1
#endif
#if QSB_FIRST_PLANES && !QSB_HP_V3
#error "QSB_FIRST_PLANES requires the updated v3 host producer"
#endif
#if QSB_HP_V3
#include "host_producers_v3.h"
#else
#include "host_producers_v1.h"
#endif
