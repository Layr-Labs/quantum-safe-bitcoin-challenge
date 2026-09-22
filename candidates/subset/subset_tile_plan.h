// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "subset_sha_flags.h"
struct qsb_subset_tile {
    unsigned first_block, blocks, first_epoch, epochs;
};
inline unsigned qsb_subset_tile_capacity(unsigned launch_blocks) {
#if QSB_SUBSET_TILED
    return launch_blocks<QSB_SUBSET_TILE_BLOCKS?launch_blocks:QSB_SUBSET_TILE_BLOCKS;
#else
    return launch_blocks;
#endif
}
inline qsb_subset_tile qsb_subset_next_tile(unsigned first,unsigned blocks,
                                           unsigned epochs,unsigned pair_mul) {
    const unsigned count=qsb_subset_tile_capacity(blocks-first);
    const unsigned ep=first*pair_mul,cap=count*pair_mul;
    return {first,count,ep,epochs-ep<cap?epochs-ep:cap};
}
