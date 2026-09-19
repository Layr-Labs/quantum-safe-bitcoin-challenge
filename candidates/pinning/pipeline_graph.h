#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Host-only prototype: retain the definition while its parameter storage is
// used for executable updates. Each graph belongs to one pipeline slot.
struct QsbPipelineGraph {
    cudaGraph_t definition;
    cudaGraphExec_t executable;
    cudaGraphNode_t search_nodes[2];
    cudaKernelNodeParams search_params[2];
};

static void qsb_graph_check(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "Pipeline graph %s failed: %s\n", operation, cudaGetErrorString(status));
    exit(2);
}

static void qsb_graph_instantiate(QsbPipelineGraph *graph, cudaStream_t stream,
                                  void *prepare, void *finish) {
    cudaGraphNode_t nodes[5];
    size_t count = 0;
    qsb_graph_check(cudaGraphGetNodes(graph->definition, nullptr, &count), "node count");
    if (count != 5) {
        fprintf(stderr, "Pipeline graph expected five kernels, got %zu nodes\n", count);
        exit(2);
    }
    qsb_graph_check(cudaGraphGetNodes(graph->definition, nodes, &count), "nodes");
    cudaStreamAttrValue stream_attribute = {};
    qsb_graph_check(cudaStreamGetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                                           &stream_attribute), "read cache policy");
    cudaKernelNodeAttrValue node_attribute = {};
    node_attribute.accessPolicyWindow = stream_attribute.accessPolicyWindow;
    for (size_t i = 0; i < count; i++) {
        cudaGraphNodeType type;
        qsb_graph_check(cudaGraphNodeGetType(nodes[i], &type), "node type");
        if (type != cudaGraphNodeTypeKernel) {
            fputs("Pipeline graph contains a non-kernel node\n", stderr);
            exit(2);
        }
        qsb_graph_check(cudaGraphKernelNodeSetAttribute(nodes[i],
            cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute), "cache policy");
        cudaKernelNodeParams parameters = {};
        qsb_graph_check(cudaGraphKernelNodeGetParams(nodes[i], &parameters), "parameters");
        int index = parameters.func == prepare ? 0 : parameters.func == finish ? 1 : -1;
        if (index >= 0) {
            if (graph->search_nodes[index] || !parameters.kernelParams || parameters.extra) {
                fputs("Pipeline graph has unexpected search parameters\n", stderr);
                exit(2);
            }
            graph->search_nodes[index] = nodes[i];
            graph->search_params[index] = parameters;
        }
    }
    if (!graph->search_nodes[0] || !graph->search_nodes[1]) {
        fputs("Pipeline graph is missing a search kernel\n", stderr);
        exit(2);
    }
    qsb_graph_check(cudaGraphInstantiate(&graph->executable, graph->definition,
                                        nullptr, nullptr, 0), "instantiate");
}

static void qsb_graph_launch(QsbPipelineGraph *graph, cudaStream_t stream,
                             uint32_t sequence, uint32_t locktime) {
    // Both stage signatures have 22 arguments. Only these two scalar values
    // vary for a full-size fast batch; fixed pointers include slot-local buffers
    // and shared read-only buffers, all retained throughout graph replay.
    // SetParams copies their values before the stack arrays go out of scope.
    for (int i = 0; i < 2; i++) {
        cudaKernelNodeParams parameters = graph->search_params[i];
        void *arguments[22];
        memcpy(arguments, parameters.kernelParams, sizeof(arguments));
        arguments[6] = &sequence;
        arguments[7] = &locktime;
        parameters.kernelParams = arguments;
        qsb_graph_check(cudaGraphExecKernelNodeSetParams(graph->executable,
            graph->search_nodes[i], &parameters), "update search arguments");
    }
    qsb_graph_check(cudaGraphLaunch(graph->executable, stream), "launch");
}
