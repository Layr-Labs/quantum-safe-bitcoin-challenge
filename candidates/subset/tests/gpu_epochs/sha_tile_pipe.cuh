// SPDX-License-Identifier: GPL-3.0-only
// Bounded scratch ownership for overlapping exact SHA and curve stages.
#pragma once
#ifndef QSB_SHA_TILE_PIPE
#define QSB_SHA_TILE_PIPE 1
#endif
#ifndef QSB_SHA_TILE_EPOCHS
#define QSB_SHA_TILE_EPOCHS 2048
#endif

#if QSB_SHA_STAGE && QSB_SHA_TILE_PIPE
static_assert(QSB_SHA_TILE_EPOCHS > 0 && QSB_SHA_TILE_EPOCHS % QSB_PAIR_MUL == 0,
              "full tiles must preserve the paired consumer geometry");
#define QSB_TILE_ARG(x) ,x
#define QSB_TILE_EPOCH(x) ((x)+tile_epoch_offset)

static bool qsb_tile_check(cudaError_t error) {
    if(error==cudaSuccess)return true;
    fprintf(stderr,"SHA tile pipeline: %s\n",cudaGetErrorString(error));
    return false;
}

struct QsbShaTilePipe {
    cudaStream_t producer=0;
    cudaEvent_t first_ready=0, scalar_ready[2]={0,0}, consumed[2]={0,0};
    uint64_t *scalars[2]={nullptr,nullptr};

    bool init() {
        if(!qsb_tile_check(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking)) ||
           !qsb_tile_check(cudaEventCreateWithFlags(&first_ready,cudaEventDisableTiming)))return false;
        const size_t bytes=(size_t)QSB_SHA_TILE_EPOCHS*QSB_SE_WINDOWS*4*sizeof(uint64_t);
        for(int s=0;s<2;s++) {
            if(!qsb_tile_check(cudaMalloc(&scalars[s],bytes)) ||
               !qsb_tile_check(cudaEventCreateWithFlags(&scalar_ready[s],cudaEventDisableTiming)) ||
               !qsb_tile_check(cudaEventCreateWithFlags(&consumed[s],cudaEventDisableTiming)))return false;
        }
        return true;
    }
    ~QsbShaTilePipe() {
        // Normal completion has already joined both streams through the hit
        // readback. Synchronize on early-error cleanup before freeing scratch.
        if(producer)cudaStreamSynchronize(producer);
        for(int s=0;s<2;s++) {
            if(scalars[s])cudaFree(scalars[s]);
            if(scalar_ready[s])cudaEventDestroy(scalar_ready[s]);
            if(consumed[s])cudaEventDestroy(consumed[s]);
        }
        if(first_ready)cudaEventDestroy(first_ready);
        if(producer)cudaStreamDestroy(producer);
    }
};
#else
#define QSB_TILE_ARG(x)
#define QSB_TILE_EPOCH(x) (x)
#endif
