// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#ifndef QSB_SUBSET_SHA_PRODUCER
#define QSB_SUBSET_SHA_PRODUCER 1
#endif
#ifndef QSB_SUBSET_SHA_BLOCKS
#define QSB_SUBSET_SHA_BLOCKS 4
#endif
#ifndef QSB_SUBSET_SCALAR_STREAM
#define QSB_SUBSET_SCALAR_STREAM 1
#endif
#ifndef QSB_SUBSET_SINGLE_BLOCKS
#define QSB_SUBSET_SINGLE_BLOCKS 12
#endif
#ifndef QSB_SUBSET_SHA_SINGLE
#define QSB_SUBSET_SHA_SINGLE 1
#endif
#ifndef QSB_SUBSET_SHA_AUTOTUNE
#define QSB_SUBSET_SHA_AUTOTUNE QSB_SUBSET_SHA_PRODUCER
#endif
#if QSB_SUBSET_SHA_AUTOTUNE && !QSB_SUBSET_SHA_PRODUCER
#error "SHA autotuning requires the scalar producer"
#endif

#ifndef QSB_SUBSET_TILED
#define QSB_SUBSET_TILED 1
#endif
#ifndef QSB_SUBSET_TILE_BLOCKS
#define QSB_SUBSET_TILE_BLOCKS 512
#endif
#if QSB_SUBSET_TILE_BLOCKS < 1
#error "A scalar tile must contain at least one consumer block"
#endif

#ifndef QSB_SUBSET_TILE_GRAPHS
#define QSB_SUBSET_TILE_GRAPHS 1
#endif
