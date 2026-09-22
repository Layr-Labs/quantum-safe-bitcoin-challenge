// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "subset_sha_flags.h"
// Graphs contain only scalar/curve tiles. Epoch production and exact hit
// verification remain on the legacy stream before and after graph execution.
struct qsb_subset_tile_graphs {
    cudaStream_t capture_stream=nullptr;
    cudaGraphExec_t exec[3]={nullptr,nullptr,nullptr};
    bool disabled[3]={false,false,false};
    qsb_subset_tile_graphs()=default;
    qsb_subset_tile_graphs(const qsb_subset_tile_graphs&)=delete;
    ~qsb_subset_tile_graphs(){
        for(unsigned i=1;i<3;i++)if(exec[i])cudaGraphExecDestroy(exec[i]);
        if(capture_stream)cudaStreamDestroy(capture_stream);
    }
    template<class Launch>
    cudaError_t run(unsigned route,bool full_batch,const Launch &launch) {
        if(route<1||route>2)return cudaErrorInvalidValue;
        auto previous=cudaGetLastError();if(previous!=cudaSuccess)return previous;
#if QSB_SUBSET_TILE_GRAPHS && QSB_SUBSET_TILED
        if(full_batch&&!disabled[route]) {
            if(!exec[route]) {
                cudaError_t err=cudaSuccess;
                if(!capture_stream)err=cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking);
                if(err==cudaSuccess)err=cudaStreamBeginCapture(capture_stream,cudaStreamCaptureModeThreadLocal);
                cudaGraph_t graph=nullptr;
                if(err==cudaSuccess) {
                    launch(capture_stream);
                    const cudaError_t launch_error=cudaGetLastError();
                    // Capture enqueues no work. Even an invalidated capture must be ended.
                    err=cudaStreamEndCapture(capture_stream,&graph);
                    if(launch_error!=cudaSuccess)err=launch_error;
                    if(err==cudaSuccess)err=cudaGraphInstantiate(&exec[route],graph,nullptr,nullptr,0);
                    if(graph)cudaGraphDestroy(graph);
                }
                if(err!=cudaSuccess) {
                    disabled[route]=true;cudaGetLastError();
                    if(exec[route]){cudaGraphExecDestroy(exec[route]);exec[route]=nullptr;}
                }
            }
            // A graph launch failure must not retry: execution may have begun.
            if(exec[route])return cudaGraphLaunch(exec[route],nullptr);
        }
#endif
        launch(nullptr);return cudaGetLastError();
    }
};
