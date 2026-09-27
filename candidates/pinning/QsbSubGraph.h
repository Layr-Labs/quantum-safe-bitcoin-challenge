#pragma once
/* Independently implemented from f4ef0994's public host-graph description.
 * Three explicit dependent nodes, one executable per ring, original native image.
 * API references: CUDA 12.8 Driver Graph/Green Context documentation. */
#if QSB_GREEN && QSB_ROOT_FUSED
static decltype(&cuGraphCreate) qsg_create;
static decltype(&cuGraphAddKernelNode) qsg_add;
static decltype(&cuGraphInstantiate) qsg_instantiate;
static decltype(&cuGraphExecNodeSetParams) qsg_update;
static decltype(&cuGraphKernelNodeSetAttribute) qsg_attr;
static decltype(&cuGraphLaunch) qsg_launch;
static decltype(&cuGraphDestroy) qsg_destroy;
static decltype(&cuGraphExecDestroy) qsg_exec_destroy;
struct QsbRingGraph { CUgraph graph; CUgraphExec exec; CUgraphNode node[3]; int blocks; };
static QsbRingGraph qsg_ring[QSB_SUBRING] = {};
static int qsg_enabled = -1;
static void qsg_cleanup() {
    for (auto &G : qsg_ring) {
        if (G.exec) qsg_exec_destroy(G.exec);
        if (G.graph) qsg_destroy(G.graph);
        G = {};
    }
}
static bool qsb_subgraph_available() {
    if (qsg_enabled >= 0) return qsg_enabled != 0;
    qsg_enabled = 0;
    if (getenv("QSB_SUBGRAPH_OFF") || !qsb_graph_ctxA || !qsb_graph_ctxB ||
        !qsb_carrier_has(QK_S0) || !qsb_carrier_has(QK_RF) || !qsb_carrier_has(QK_S2)) return false;
    struct { const char *n; void **p; } api[] = {
        {"cuGraphCreate",(void**)&qsg_create}, {"cuGraphAddKernelNode",(void**)&qsg_add},
        {"cuGraphInstantiate",(void**)&qsg_instantiate}, {"cuGraphExecNodeSetParams",(void**)&qsg_update},
        {"cuGraphKernelNodeSetAttribute",(void**)&qsg_attr}, {"cuGraphLaunch",(void**)&qsg_launch},
        {"cuGraphDestroy",(void**)&qsg_destroy}, {"cuGraphExecDestroy",(void**)&qsg_exec_destroy}};
    for (auto &a : api) {
        cudaDriverEntryPointQueryResult qr;
        if (cudaGetDriverEntryPoint(a.n,a.p,cudaEnableDefault,&qr) != cudaSuccess || !*a.p) {
            cudaGetLastError(); return false;
        }
    }
    qsg_enabled = 1; atexit(qsg_cleanup);
    printf("  Sub-batch CUDA Graph: enabled (unchanged native kernels)\n");
    return true;
}
template<class T,size_t... I>
static void qsg_argv(T &v,void **args,std::index_sequence<I...>) {
    int unused[] = {0, (args[I]=(void*)&std::get<I>(v),0)...}; (void)unused;
}
static bool qsg_policy(CUgraphNode node,cudaStream_t stream) {
    int priority=0;
    if (cudaStreamGetPriority(stream,&priority)!=cudaSuccess) return false;
    CUkernelNodeAttrValue val = {}; val.priority=priority;
    if (qsg_attr(node,CU_KERNEL_NODE_ATTRIBUTE_PRIORITY,&val)!=CUDA_SUCCESS) return false;
    cudaStreamAttrValue av = {};
    if (cudaStreamGetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&av)!=cudaSuccess) return false;
    val = {};
    val.accessPolicyWindow.base_ptr=av.accessPolicyWindow.base_ptr;
    val.accessPolicyWindow.num_bytes=av.accessPolicyWindow.num_bytes;
    val.accessPolicyWindow.hitRatio=av.accessPolicyWindow.hitRatio;
    val.accessPolicyWindow.hitProp=(CUaccessProperty)av.accessPolicyWindow.hitProp;
    val.accessPolicyWindow.missProp=(CUaccessProperty)av.accessPolicyWindow.missProp;
    return qsg_attr(node,CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW,&val)==CUDA_SUCCESS;
}
static CUDA_KERNEL_NODE_PARAMS qsg_params(int kid,CUcontext ctx,unsigned grid,unsigned block,void **args) {
    CUDA_KERNEL_NODE_PARAMS p = {}; p.kern=(CUkernel)g_qsb_carrier.k[kid]; p.ctx=ctx;
    p.gridDimX=grid; p.gridDimY=p.gridDimZ=1;
    p.blockDimX=block; p.blockDimY=p.blockDimZ=1; p.kernelParams=args; return p;
}
static CUresult qsg_set(QsbRingGraph &G,int i,const CUDA_KERNEL_NODE_PARAMS &p) {
    /* Generic API retains explicit converted green context, unlike the legacy update. */
    CUgraphNodeParams np = {}; np.type=CU_GRAPH_NODE_TYPE_KERNEL;
    np.kernel.kern=p.kern; np.kernel.ctx=p.ctx;
    np.kernel.gridDimX=p.gridDimX; np.kernel.gridDimY=np.kernel.gridDimZ=1;
    np.kernel.blockDimX=p.blockDimX; np.kernel.blockDimY=np.kernel.blockDimZ=1;
    np.kernel.kernelParams=p.kernelParams;
    return qsg_update(G.exec,G.node[i],&np);
}
template<class A,class R,class F>
static bool qsb_subgraph_run(int r,int blocks,cudaStream_t st,A &a,R &root,F &finish) {
    QsbRingGraph &G=qsg_ring[r];
    void *aa[std::tuple_size<A>::value], *rr[std::tuple_size<R>::value], *ff[std::tuple_size<F>::value];
    qsg_argv(a,aa,std::make_index_sequence<std::tuple_size<A>::value>{});
    qsg_argv(root,rr,std::make_index_sequence<std::tuple_size<R>::value>{});
    qsg_argv(finish,ff,std::make_index_sequence<std::tuple_size<F>::value>{});
    CUDA_KERNEL_NODE_PARAMS p[3] = {
        qsg_params(QK_S0,qsb_graph_ctxA,blocks,QSB_S0_THREADS,aa),
        qsg_params(QK_RF,QSB_GREEN_RT_B ? qsb_graph_ctxB : qsb_graph_ctxA,1,QSB_RF_LANES,rr),
        qsg_params(QK_S2,qsb_graph_ctxB,blocks,QSB_S2_THREADS,ff)};
    if (!G.exec) {
        if (qsg_create(&G.graph,0)!=CUDA_SUCCESS) goto fallback;
        for (int i=0;i<3;i++) {
            if (qsg_add(&G.node[i],G.graph,i?&G.node[i-1]:nullptr,i?1:0,&p[i])!=CUDA_SUCCESS) goto fallback;
            cudaStream_t s=i==0?st:i==1?g_qsb_sub.rt:g_qsb_sub.s2b[g_qsb_sub.g&1ull];
            if (!qsg_policy(G.node[i],s)) goto fallback;
        }
        if (qsg_instantiate(&G.exec,G.graph,CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY)!=CUDA_SUCCESS) goto fallback;
        G.blocks=blocks;
    } else {
        if (qsg_set(G,0,p[0])!=CUDA_SUCCESS || qsg_set(G,2,p[2])!=CUDA_SUCCESS) goto fallback;
        if (G.blocks!=blocks) {
            if (qsg_set(G,1,p[1])!=CUDA_SUCCESS) goto fallback;
            G.blocks=blocks;
        }
    }
    /* The input reset and previous ring completion must precede the entire graph. */
    if (cudaStreamWaitEvent(st,g_qsb_sub.ev_in,0)!=cudaSuccess) {
        fprintf(stderr,"Subgraph input ordering failed\n"); exit(2);
    }
    if (qsg_launch(G.exec,(CUstream)st)!=CUDA_SUCCESS) {
        /* Ambiguous launch errors must not retry enumeration through fallback. */
        fprintf(stderr,"Subgraph launch failed; refusing duplicate fallback\n"); exit(2);
    }
    return true;
fallback:
    /* No current graph launch occurred. Prior enqueued launches remain valid. */
    qsg_enabled=0; cudaGetLastError();
    printf("  Sub-batch CUDA Graph setup/update unavailable; original stream path\n");
    return false;
}
#endif
