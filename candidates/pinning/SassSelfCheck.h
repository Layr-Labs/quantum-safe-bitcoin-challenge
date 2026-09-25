#pragma once
// Compare full stage-0 state against the unmodified native image before grinding.
// This is differential sampling, not a proof of correctness for every input.
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void qsb_sass_check_cuda(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "SASS self-check CUDA error: %s\n", cudaGetErrorString(error));
        exit(2);
    }
}

template<class Launch>
static void qsb_sass_selfcheck(Launch launch) {
    static_assert(QSB_TREE_OFFLOAD == 0, "self-check requires in-kernel tree");
    const int count = 262144;
    const size_t state_bytes = (size_t)count * QSB_STATE_PLANES * sizeof(ulonglong2);
    const size_t root_bytes = (size_t)((count+QSB_TREE_N-1)/QSB_TREE_N) * 8 * sizeof(uint64_t);
    ulonglong2 *state = nullptr;
    uint64_t *roots = nullptr;
    qsb_sass_check_cuda(cudaDeviceSynchronize());
    qsb_sass_check_cuda(cudaMalloc(&state, state_bytes));
    qsb_sass_check_cuda(cudaMalloc(&roots, root_bytes));
    unsigned char *reference = (unsigned char*)malloc(state_bytes+root_bytes);
    unsigned char *actual = (unsigned char*)malloc(state_bytes+root_bytes);
    if (!reference || !actual) { fprintf(stderr,"SASS self-check allocation failed\n"); exit(2); }
    for (int arm=0; arm<2; ++arm) {
        qsb_sass_check_cuda(cudaMemset(state,0,state_bytes));
        qsb_sass_check_cuda(cudaMemset(roots,0,root_bytes));
        launch(arm == 0 ? QK_N : QK_S0, state, roots, count);
        qsb_sass_check_cuda(cudaDeviceSynchronize());
        unsigned char *out = arm == 0 ? reference : actual;
        qsb_sass_check_cuda(cudaMemcpy(out,state,state_bytes,cudaMemcpyDeviceToHost));
        qsb_sass_check_cuda(cudaMemcpy(out+state_bytes,roots,root_bytes,cudaMemcpyDeviceToHost));
    }
    if (memcmp(reference,actual,state_bytes+root_bytes) != 0) {
        fprintf(stderr,"SASS self-check FAILED: native control and candidate disagree\n");
        exit(2);
    }
    free(reference); free(actual);
    qsb_sass_check_cuda(cudaFree(state)); qsb_sass_check_cuda(cudaFree(roots));
    printf("SASS self-check PASS: %d candidates, %zu state/root bytes identical\n",
           count,state_bytes+root_bytes);
    fflush(stdout);
}
