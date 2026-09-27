#pragma once
// SPDX-License-Identifier: GPL-3.0-only
/*
 * Reuse the prepare -> root inverse -> finish dependency chain as one graph
 * per state-buffer ring entry. No device arithmetic or candidate enumeration
 * changes. Graphs execute on independent ring streams; stream ordering prevents
 * reuse of a ring's buffers before its preceding finish has completed.
 *
 * A graph launch stream does not select kernel SM resources. Capture preserves
 * the original green contexts, priorities, and access-policy windows. Update
 * through the context-aware driver API, using a library kernel + the converted
 * green CUcontext. The older kernel-node updater can drop the green context.
 * The fixed harness link line needs no -lcuda: resolve through the runtime.
 */
#include <cuda.h>
#include <cudaTypedefs.h>

#ifndef QSB_SUBGRAPH
#define QSB_SUBGRAPH 1
#endif

namespace qsb_sg {
struct Ring {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t prepare = nullptr, root = nullptr, finish = nullptr;
    CUDA_KERNEL_NODE_PARAMS_v2 pp = {}, rp = {}, fp = {};
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    int root_count = 0;
};
static Ring rings[QSB_SUBRING];
static bool enabled = false;
static CUcontext contexts[3] = {};
static PFN_cuGraphKernelNodeGetParams_v12000 get_params;
static PFN_cuGraphExecNodeSetParams_v12020 set_params;
static PFN_cuStreamGetCtx_v12050 get_context;
static PFN_cuCtxFromGreenCtx_v12040 from_green;

static void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "Subgraph %s failed: %s\n", what, cudaGetErrorString(e));
        exit(2);
    }
}
static void driver_check(CUresult e, const char *what) {
    if (e != CUDA_SUCCESS) {
        fprintf(stderr, "Subgraph %s failed (CUDA driver %d)\n", what, (int)e);
        exit(2);
    }
}
static bool entry(const char *name, void **ptr, unsigned version) {
    cudaDriverEntryPointQueryResult status;
    return cudaGetDriverEntryPointByVersion(name, ptr, version, cudaEnableDefault, &status)
           == cudaSuccess && status == cudaDriverEntryPointSuccess && *ptr;
}
static bool init(cudaStream_t prepare, cudaStream_t root, cudaStream_t finish) {
    if (!QSB_SUBGRAPH || getenv("QSB_SUBGRAPH_OFF") ||
        !qsb_carrier_has(QK_S0) || !qsb_carrier_has(QK_RF) || !qsb_carrier_has(QK_S2))
        return false;
    if (!entry("cuGraphKernelNodeGetParams", (void**)&get_params, 12000) ||
        !entry("cuGraphExecNodeSetParams", (void**)&set_params, 12020) ||
        !entry("cuStreamGetCtx", (void**)&get_context, 12050) ||
        !entry("cuCtxFromGreenCtx", (void**)&from_green, 12040)) {
        (void)cudaGetLastError();
        printf("  Subgraph: driver APIs unavailable; using stream pipeline\n");
        return false;
    }
    const cudaStream_t streams[3] = {prepare, root, finish};
    for (int i = 0; i < 3; i++) {
        CUgreenCtx green = nullptr;
        CUresult e = get_context((CUstream)streams[i], &contexts[i], &green);
        if (e == CUDA_SUCCESS && green) e = from_green(&contexts[i], green);
        if (e != CUDA_SUCCESS) {
            (void)cudaGetLastError();
            printf("  Subgraph: context query unavailable; using stream pipeline\n");
            return false;
        }
    }
    for (int r = 0; r < QSB_SUBRING; r++) {
        check(cudaStreamCreateWithFlags(&rings[r].stream, cudaStreamNonBlocking), "ring stream");
        check(cudaEventCreateWithFlags(&rings[r].done, cudaEventDisableTiming), "ring event");
    }
    printf("  Subgraph: enabled (captured contexts and priorities, one graph per state ring)\n");
    return true;
}
static void instantiate(Ring &g, int count) {
    size_t n = 0;
    check(cudaGraphGetNodes(g.graph, nullptr, &n), "node count");
    if (n != 3) { fprintf(stderr, "Subgraph expected three kernel nodes, got %zu\n", n); exit(2); }
    cudaGraphNode_t nodes[3];
    check(cudaGraphGetNodes(g.graph, nodes, &n), "nodes");
    for (int i = 0; i < 3; i++) {
        cudaGraphNodeType type;
        check(cudaGraphNodeGetType(nodes[i], &type), "node type");
        if (type != cudaGraphNodeTypeKernel) { fprintf(stderr, "Subgraph unexpected node type\n"); exit(2); }
        size_t before = 0, after = 0;
#if CUDART_VERSION >= 13000
        check(cudaGraphNodeGetDependencies(nodes[i], nullptr, nullptr, &before), "dependencies");
        check(cudaGraphNodeGetDependentNodes(nodes[i], nullptr, nullptr, &after), "dependents");
#else
        check(cudaGraphNodeGetDependencies(nodes[i], nullptr, &before), "dependencies");
        check(cudaGraphNodeGetDependentNodes(nodes[i], nullptr, &after), "dependents");
#endif
        if (!before && after == 1) g.prepare = nodes[i];
        else if (before == 1 && !after) g.finish = nodes[i];
        else if (before == 1 && after == 1) g.root = nodes[i];
        else { fprintf(stderr, "Subgraph unexpected topology\n"); exit(2); }
    }
    if (!g.prepare || !g.root || !g.finish) { fprintf(stderr, "Subgraph incomplete chain\n"); exit(2); }
    driver_check(get_params((CUgraphNode)g.prepare, &g.pp), "prepare parameters");
    driver_check(get_params((CUgraphNode)g.root, &g.rp), "root parameters");
    driver_check(get_params((CUgraphNode)g.finish, &g.fp), "finish parameters");
    CUDA_KERNEL_NODE_PARAMS_v2 *p[3] = {&g.pp, &g.rp, &g.fp};
    for (int i = 0; i < 3; i++) {
        if (!p[i]->kern || p[i]->ctx != contexts[i]) {
            fprintf(stderr, "Subgraph capture changed kernel execution context\n"); exit(2);
        }
    }
    check(cudaGraphInstantiateWithFlags(&g.exec, g.graph, cudaGraphInstantiateFlagUseNodePriority), "instantiate");
    g.root_count = count;
}
template<typename... P, typename... A, size_t... I>
static void update_impl(cudaGraphExec_t exec, cudaGraphNode_t node,
                        const CUDA_KERNEL_NODE_PARAMS_v2 &saved, dim3 grid,
                        std::index_sequence<I...>, A&&... a) {
    std::tuple<typename std::decay<P>::type...> values(std::forward<A>(a)...);
    void *args[sizeof...(P)] = {(void*)&std::get<I>(values)...};
    CUgraphNodeParams params = {};
    params.type = CU_GRAPH_NODE_TYPE_KERNEL;
    auto &k = params.kernel;
    k.func = nullptr; k.kern = saved.kern; k.ctx = saved.ctx;
    k.gridDimX = grid.x; k.gridDimY = grid.y; k.gridDimZ = grid.z;
    k.blockDimX = saved.blockDimX; k.blockDimY = saved.blockDimY; k.blockDimZ = saved.blockDimZ;
    k.sharedMemBytes = saved.sharedMemBytes;
    k.kernelParams = args;
    // CUDA copies these argument values. Updates affect only future launches,
    // including when earlier instances remain queued on this ring's stream.
    driver_check(set_params((CUgraphExec)exec, (CUgraphNode)node, &params), "argument update");
}
template<typename... P, typename... A>
static void update(void (*)(P...), cudaGraphExec_t exec, cudaGraphNode_t node,
                   const CUDA_KERNEL_NODE_PARAMS_v2 &saved, dim3 grid, A&&... a) {
    static_assert(sizeof...(P) == sizeof...(A), "subgraph argument count");
    update_impl<P...>(exec, node, saved, grid, std::index_sequence_for<P...>{}, std::forward<A>(a)...);
}
} // namespace qsb_sg
