#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/system/cuda/execution_policy.h>

#include "defs.h"

// uncomment to turn on built-in profiling
//#define PROFILER

#ifdef PROFILER
#include "profiler.hpp"

static Profiler profiler;

#define PROFILER_TOC(n) do { cudaDeviceSynchronize(); profiler.toc(n); } while(0)
#define PROFILER_TIC() do { profiler.tic(); } while(0)
#define PROFILER_REPORT(n) do { std::cout << profiler; } while(0)
#else
#define PROFILER_TOC(n) do {} while(0)
#define PROFILER_TIC() do {} while(0)
#define PROFILER_REPORT(n) do {} while(0)
#endif

// Block size for CUDA calls
#define BLOCK_SIZE 128

// We use shared memory to speed up atomic operations during best edge search
// amount of shared memory used is BEST_BLOCKS * BLOCK_SIZE ints
// Depending on graph size and type this may improve results or not
// Reasonable values are 1-4 or 0 (do not use shared memory)
#define BEST_BLOCKS 3

template<class T> inline T* D(thrust::device_vector<T> *v) {
    return v->data().get();
}

using namespace std;

typedef vertex_id_t my_edge_id_t; // 32 bits for edge ID are enough for graph sizes up to 26


// We use texture memory for graph data. The following functions and macros encapsulate access to graph data.

#define TX_WIDTH_BITS 15
#define TX_WIDTH (1 << TX_WIDTH_BITS)
#define TX_MASK (TX_WIDTH - 1)
#define TX_HEIGHT(s) (((s) + TX_WIDTH - 1)/TX_WIDTH)
#define TX_ROUND(s) (TX_HEIGHT(s) * TX_WIDTH)
#define TX_PITCH(t) (TX_WIDTH * sizeof(t))

__device__ inline double tex2Dweight(texture<uint2, 2, cudaReadModeElementType> tx, int i, int j) {
    uint2 v = tex2D(tx, i, j);
    return *((double *) &v);
}

texture<vertex_id_t, 2, cudaReadModeElementType> t_dst;
texture<my_edge_id_t, 2, cudaReadModeElementType> t_rowEnds;
texture<uint2, 2, cudaReadModeElementType> t_weight;
#define GET_DST(y) (tex2D(t_dst, (y) & TX_MASK, (y) >> TX_WIDTH_BITS))
#define GET_WEIGHT(y) (tex2Dweight(t_weight, (y) & TX_MASK, (y) >> TX_WIDTH_BITS))
#define GET_ROW_END(x) (tex2D(t_rowEnds, (x) & TX_MASK, (x) >> TX_WIDTH_BITS))

// Problem data kept on device
static thrust::device_vector<my_edge_id_t> *d_rowStarts, *d_rowEnds, *d_currentEdge;
static thrust::device_vector<vertex_id_t> *d_parent;
static thrust::device_vector<uint8_t> *d_mst;
static thrust::device_vector<uint2> *d_weight;
static thrust::device_vector<vertex_id_t> *d_dst;
static thrust::device_vector<my_edge_id_t> *d_bestEdge;
static thrust::device_vector<vertex_id_t> *d_fragmentCount;

// Problem data not copied to device
static std::vector<edge_id_t> id;
static std::vector<vertex_id_t> vertexPermutation;
static my_edge_id_t edgeCount;

// Lock-free implementation of Union-Find.
// We do not use ranks and do not use path compression.
//
// No ranks => component roots are likely to be all small numbers, improving memory locality.
// No path compression => less divergence.

// Try to merge two components.
// Returns True if merge has been actually performed.
__device__ inline bool merge(vertex_id_t *parent, vertex_id_t x, vertex_id_t y) {
    // here we should have called find(x), find(y), but we take a different approach to reduce divergence
    // anyway, if threads diverge during find, they will create significant congestion during atomicCAS
    x = parent[x];
    y = parent[y];
    while (x != y) {
        if (y < x) {
            vertex_id_t t = x;
            x = y;
            y = t;
        }
        vertex_id_t z = atomicCAS(&parent[y], y, x);
        if (z == y) {
            return true;
        }
        x = parent[parent[x]];
        y = parent[parent[z]]; // reuse value returned by atomicCAS
    }
    return false;
}

// Flatten Union-Find structure by replacing parent pointers with pointers to the root
// Note that since parent[x] < x for all non-root x, threads started earlier help those started later
// (this works like path compresssion, but there is no actual compression - there is not even a find method for
// our Union-Find implementation).
__device__ inline void flatten(vertex_id_t *parent, vertex_id_t x) {
    if (x != parent[x]) {
        vertex_id_t y = parent[x];
        if (y != parent[y]) {
            do {
                y = parent[y];
            } while (y != parent[y]);
            parent[x] = y;
        }
    }
}

__global__ void flatten_vertices(vertex_id_t *parent) {
    vertex_id_t x = blockIdx.x * BLOCK_SIZE | threadIdx.x;
    flatten(parent, x);
}

// Merge fragments using best edges found so far
// Calculate number of active components using atomic increments
__global__ void merge_fragments(my_edge_id_t *bestEdge, vertex_id_t *parent, uint8_t *mst, vertex_id_t *fragmentCount) {
    vertex_id_t x = blockIdx.x * BLOCK_SIZE | threadIdx.x;
    my_edge_id_t y = bestEdge[x];
    if (y) {
        // special value 0 is used to indicate that component is empty (does not have inter-component
        // edges and so does not have a best edge). Note that there is an edge with zero ID, but
        // it has been already processed during initialization, so this is not a problem.
        bestEdge[x] = 0;
        if (merge(parent, x, GET_DST(y))) {  // try to merge using best edge
            mst[y] = 1; // add edge to MST if merge was successful
        } else {
            atomicInc(fragmentCount, (vertex_id_t) -1); // unsuccessful merge => increase active fragment count
        }
    }
}

// Merge fragments during initialization
// Here we don't count remaining fragments, but instead we automatically skip first edge of each vertex, whether
// it was successfully merged or not.
__global__ void merge_fragments0(my_edge_id_t *currentEdge, vertex_id_t *parent, uint8_t *mst) {
    vertex_id_t x = blockIdx.x * BLOCK_SIZE | threadIdx.x;
    my_edge_id_t y = currentEdge[x];
    if (y != GET_ROW_END(x)) {
        if (merge(parent, x, GET_DST(y))) {
            mst[y] = 1;
        }
        currentEdge[x]++;
    }
}

// Skip intra-component edges and update best edges for each component, possibly using shared memory
__global__ void skip_edges(my_edge_id_t *currentEdge, const vertex_id_t *parent, my_edge_id_t *bestEdge) {
    vertex_id_t x = blockIdx.x * BLOCK_SIZE | threadIdx.x;
    // this block of shared memory will store best edges of vertices with small indices
    // recall that on later iterations, component roots are likely to have small indices
    __shared__ my_edge_id_t bestEdge0[BLOCK_SIZE * BEST_BLOCKS + 1];
    my_edge_id_t y = currentEdge[x], y1 = GET_ROW_END(x);
    if (y != y1) {
        // skip all intra-component edges
        vertex_id_t root = parent[x];
        for (; y != y1; y++) {
            if (parent[GET_DST(y)] != root) {
                break;
            }
        }
        currentEdge[x] = y; // update vertex's current edge
        // if we have an inter-component edge, update root's best edge
        if (y != y1) {
            my_edge_id_t best = bestEdge[root], *rootBest = root < BLOCK_SIZE * BEST_BLOCKS ? &bestEdge0[root] : &bestEdge[root];
            while (!best || GET_WEIGHT(best) > GET_WEIGHT(y)) {
                my_edge_id_t oldBest = atomicCAS(rootBest, best, y);
                if (oldBest == best) {
                    break;
                }
                best = oldBest;
            }
        }
    }
    if (BEST_BLOCKS) {
        // if BEST_BLOCKS is turned on, do actual update of best edges for vertices with small indices
        __syncthreads();
        for (int i = 0; i < BEST_BLOCKS; i++) {
            y = bestEdge0[BLOCK_SIZE * i | threadIdx.x];
            if (y) {
                my_edge_id_t best = bestEdge[BLOCK_SIZE * i | threadIdx.x];
                while (!best || GET_WEIGHT(best) > GET_WEIGHT(y)) {
                    my_edge_id_t oldBest = atomicCAS(&bestEdge[BLOCK_SIZE * i | threadIdx.x], best, y);
                    if (oldBest == best) {
                        break;
                    }
                    best = oldBest;
                }
            }
            bestEdge0[BLOCK_SIZE * i | threadIdx.x] = 0;
        }
    }
}

// The same as previous, but for small graphs where number of vertices is smaller than the amount of shared memory used
__global__ void skip_edges_small(my_edge_id_t *currentEdge, const vertex_id_t *parent, my_edge_id_t *bestEdge) {
    vertex_id_t x = blockIdx.x * BLOCK_SIZE | threadIdx.x;
    my_edge_id_t y = currentEdge[x], y1 = GET_ROW_END(x);
    if (y != y1) {
        vertex_id_t root = parent[x];
        for (; y != y1; y++) {
            if (parent[GET_DST(y)] != root) {
                my_edge_id_t best = bestEdge[root];
                while (!best || GET_WEIGHT(best) > GET_WEIGHT(y)) {
                    my_edge_id_t oldBest = atomicCAS(&bestEdge[root], best, y);
                    if (oldBest == best) {
                        break;
                    }
                    best = oldBest;
                }
                break;
            }
        }
        currentEdge[x] = y;
    }
}

struct compare_vertices_by_edge_count {
    graph_t *g;

    compare_vertices_by_edge_count(graph_t *g) : g(g) {}

    inline bool operator()(edge_id_t a, edge_id_t b) const {
        return (g->rowsIndices[a + 1] - g->rowsIndices[a]) > (g->rowsIndices[b + 1] - g->rowsIndices[b]);
    }
};

struct compare_edge_id_by_weight {
    graph_t *g;

    compare_edge_id_by_weight(graph_t *g) : g(g) {}

    inline bool operator()(edge_id_t a, edge_id_t b) const {
        return g->weights[a] < g->weights[b];
    }
};

struct compare_edge_id_by_dst_and_weight {
    graph_t *g;

    compare_edge_id_by_dst_and_weight(graph_t *g) : g(g) {}

    inline bool operator()(edge_id_t a, edge_id_t b) const {
        return g->endV[a] < g->endV[b] || (g->endV[a] == g->endV[b] && g->weights[a] < g->weights[b]);
    }
};

struct equals_edge_id_by_dst {
    graph_t *g;

    equals_edge_id_by_dst(graph_t *g) : g(g) {}

    inline bool operator()(edge_id_t a, edge_id_t b) const {
        return g->endV[a] == g->endV[b];
    }
};

static cudaStream_t stream1, stream2, stream3;

extern "C" void init_mst(graph_t *G) {
    // sort vertices by number of edges, heavy vertices first
    // IMPORTANT: this greatly decreases divergence, as threads of the warp are likely to process approximately
    // the same number of edges during each iteration
    std::vector<vertex_id_t> vertexIDs(G->n);
    thrust::sequence(vertexIDs.begin(), vertexIDs.end());
    std::sort(vertexIDs.begin(), vertexIDs.end(), compare_vertices_by_edge_count(G));
    vertexPermutation.resize(G->n);
    for (vertex_id_t i = 0; i < G->n; i++) {
        vertexPermutation[vertexIDs[i]] = i;
    }

    thrust::host_vector<my_edge_id_t> rowStarts(G->n);
    thrust::host_vector<my_edge_id_t> rowEnds(G->n);
    id.resize(G->m);
    my_edge_id_t currentEdge = 0;
    for(vertex_id_t i = 0; i < G->n; i++) {
        vertex_id_t v = vertexIDs[i];

        // filter out loop edges
        rowStarts[i] = currentEdge;
        for(edge_id_t e = G->rowsIndices[v]; e < G->rowsIndices[v + 1]; e++) {
            if (G->endV[e] != v) {
                id[currentEdge++] = (my_edge_id_t) e;
            }
        }

        // filter out duplicate edges for this vertex
        std::sort(&id[rowStarts[i]], &id[currentEdge], compare_edge_id_by_dst_and_weight(G));
        rowEnds[i] = currentEdge = (my_edge_id_t) std::distance(&id[0], unique(&id[rowStarts[i]], &id[currentEdge], equals_edge_id_by_dst(G)));
        // sort remaining edges by weight
        std::sort(&id[rowStarts[i]], &id[currentEdge], compare_edge_id_by_weight(G));
    }
    edgeCount = currentEdge;
    id.resize(edgeCount);

    thrust::host_vector<uint2> weight(TX_ROUND(edgeCount));
    thrust::host_vector<vertex_id_t> dst(TX_ROUND(edgeCount));
    for(my_edge_id_t e = 0; e < edgeCount; e++) {
        weight[e] = *((uint2 *) &G->weights[id[e]]);
        dst[e] = vertexPermutation[G->endV[id[e]]];
    }

    d_rowStarts = new thrust::device_vector<my_edge_id_t>(rowStarts);
    d_rowEnds = new thrust::device_vector<my_edge_id_t>(rowEnds);
    d_currentEdge = new thrust::device_vector<my_edge_id_t>(G->n);
    d_weight = new thrust::device_vector<uint2>(weight);
    d_dst = new thrust::device_vector<vertex_id_t>(dst);
    d_bestEdge = new thrust::device_vector<my_edge_id_t>(G->n);
    d_fragmentCount = new thrust::device_vector<vertex_id_t>(1);
    d_parent = new thrust::device_vector<vertex_id_t>(G->n);
    d_mst = new thrust::device_vector<uint8_t>(edgeCount);

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    cudaFuncSetCacheConfig(&merge_fragments, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(&merge_fragments0, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(&flatten_vertices, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(&skip_edges, cudaFuncCachePreferL1);

    // Create textures for graph data
    cudaChannelFormatDesc descriptor1 = cudaCreateChannelDesc<vertex_id_t>();
    cudaBindTexture2D(0, &t_dst, D(d_dst), &descriptor1, TX_WIDTH, TX_HEIGHT(edgeCount), TX_PITCH(vertex_id_t));
    cudaChannelFormatDesc descriptor2 = cudaCreateChannelDesc<uint2>();
    cudaBindTexture2D(0, &t_weight, D(d_weight), &descriptor2, TX_WIDTH, TX_HEIGHT(edgeCount), TX_PITCH(uint2));
    cudaChannelFormatDesc descriptor3 = cudaCreateChannelDesc<my_edge_id_t>();
    cudaBindTexture2D(0, &t_rowEnds, D(d_rowEnds), &descriptor3, TX_WIDTH, TX_HEIGHT(edgeCount), TX_PITCH(my_edge_id_t));
}

void* MST(graph_t *G) {
    PROFILER_TIC();
    dim3 dimBlock(std::min(G->n, (vertex_id_t) BLOCK_SIZE), 1);
    dim3 dimGrid(G->n / dimBlock.x, 1);

    // asynchronous initialization of data structures, to improve memory bandwidth
    cudaMemsetAsync(D(d_mst), 0, edgeCount, stream1);
    cudaMemcpyAsync(D(d_currentEdge), D(d_rowStarts), sizeof(my_edge_id_t) * G->n, cudaMemcpyDeviceToDevice, stream2);
    cudaMemsetAsync(D(d_bestEdge), 0, sizeof(my_edge_id_t) * G->n, stream3);
    thrust::sequence(d_parent->begin(), d_parent->end());
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    PROFILER_TOC(10);

    // initialization + initial merge
    merge_fragments0<<<dimGrid, dimBlock>>>(D(d_currentEdge), D(d_parent), D(d_mst));
    PROFILER_TOC(20);

    int iteration = 0;
    vertex_id_t fragmentCount = (vertex_id_t) -1;
    while (true) {
        // flatten Union-Find structure
        flatten_vertices<<<dimGrid, dimBlock>>>(D(d_parent));
        PROFILER_TOC(30);

        // stop if there remains only one active component
        // for RMAT graphs, this allows to completely skip a good share of edges
        // note that even we stop here, we still need the previous flatten operation
        // because we need flattened parent pointers during post-processing
        cudaStreamSynchronize(stream3);
        if (fragmentCount <= 1) {
            break;
        }
        cudaMemsetAsync(D(d_fragmentCount), 0, sizeof(vertex_id_t));

        // skip intra-component edges and update best edge of each component
        if (G->n < BEST_BLOCKS * BLOCK_SIZE) {
            skip_edges_small<<<dimGrid, dimBlock>>>(D(d_currentEdge), D(d_parent), D(d_bestEdge));
        } else {
            skip_edges<<<dimGrid, dimBlock>>>(D(d_currentEdge), D(d_parent), D(d_bestEdge));
        }
        PROFILER_TOC(40 + (++iteration));

        // put least-weight edge of each component into MST
        merge_fragments<<<dimGrid, dimBlock>>>(D(d_bestEdge), D(d_parent), D(d_mst), D(d_fragmentCount));
        cudaMemcpyAsync(&fragmentCount, D(d_fragmentCount), sizeof(vertex_id_t), cudaMemcpyDeviceToHost, stream3);
        PROFILER_TOC(50);
    }
    // make sure all ongoing operations are finished
    cudaStreamSynchronize(0);
    return &d_mst;
}

extern "C" void finalize_mst(graph_t *G) {
    PROFILER_REPORT();
    delete d_rowStarts;
    delete d_rowEnds;
    delete d_currentEdge;
    delete d_bestEdge;
    delete d_weight;
    delete d_dst;
    delete d_fragmentCount;
    delete d_parent;
    delete d_mst;
}

extern "C" void convert_to_output(graph_t *G, void* result, forest_t *trees_output)
{
    // Result:
    // bit mask of edges
    thrust::host_vector<uint8_t> mst(*d_mst);
    // ids of connected components assigned to each vertex
    thrust::host_vector<vertex_id_t> parent(*d_parent);
    // and also:
    // id - mapping of edge ids (ours to original)
    // vertexPermutation - mapping of vertex ids (original to ours)

    convert_mask_and_parent_to_output(G, &id[0], mst.size(), &mst[0], &vertexPermutation[0], &parent[0], sizeof(vertex_id_t), trees_output);
}