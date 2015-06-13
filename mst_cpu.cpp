#include <stdlib.h>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <float.h>
#include <cmath>
#include <cstring>
#include <omp.h>

#include "defs.h"

using namespace std;

typedef uint32_t my_edge_id_t; // 32 bits will be enough for graph sizes up to 26
typedef uint64_t my_weight_t; // double weights are replaced with 64-bit integers

// Implementation of wait-free union-find based on paper
// Anderson, Richard J, and Heather Woll. “Wait-Free Parallel Algorithms for the Union-Find Problem,”
// 370–80, New York, New York, USA: ACM Press, 1991. doi:10.1145/103418.103458.
//
// Path compression is turned off for performance reasons, and also we do not make
// additional update_root call upon a successful merge, so this implementation is actually lock-free,
// not wait-free - but it performs better on our data.
//
// Instead of path compression, we flatten the Union-Find structure during each round of Boruvka algorithm.

template<class T>
struct union_find_record_t {
    T parent;
    T rank;
};

template<class T>
struct union_find_t {
    vector<union_find_record_t<T>> a;

    union_find_t(T size) : a(size) {
    }

    inline void clear(T p) {
        a[p] = {p, 0};
    }

    inline T getParent(T x) {
        return a[x].parent;
    }

    // Find operation without path compression.
    inline T find_fast(T x) {
        while (a[x].parent != x) {
            x = a[x].parent;
        }
        return x;
    }

    inline void compress(T x) {
        T y = a[x].parent;
        if (y != x) {
            T z = a[y].parent;
            if (a[y].parent != y) {
                do {
                    y = a[y].parent;
                } while (y != a[y].parent);
                a[x].parent = y;
            }
        }
    }

    // Flatten the union-find structure, such that each node's parent points to the root of the component.
    inline void flatten(T p) {
        if (a[p].parent != p) {
            compress(p);
        }
    }

    // See cited paper for explanation of update_root and merge.
    inline bool update_root(T &x, T oldRank, T y, T newRank) {
        union_find_record_t<T> old_record = {x, oldRank};
        if (atomic_compare_exchange_weak((atomic<union_find_record_t<T>> *)&a[x], &old_record, {y, newRank})) {
            return true;
        } else {
            // here we re-use actual parent value returned by the atomic operation
            x = old_record.parent;
            return false;
        }
    }

    // Retruns true if merge was successful (i.e. two components were actually merged)
    inline bool merge(T x, T y) {
        x = find_fast(x);
        y = find_fast(y);
        if (x != y) {
            do {
                T xRank = a[x].rank, yRank = a[y].rank;
                if (xRank > yRank) {
                    if (update_root(y, yRank, x, yRank)) {
                        return true;
                    }
                } else if (xRank == yRank) {
                    if (x < y) {
                        if (update_root(y, yRank, x, yRank)) {
                            update_root(x, xRank, x, xRank + 1);
                            return true;
                        }
                    } else {
                        if (update_root(x, xRank, y, xRank)) {
                            update_root(y, yRank, y, yRank + 1);
                            return true;
                        }
                    }
                } else {
                    if (update_root(x, xRank, y, xRank)) {
                        return true;
                    }
                }
            } while (x != y);
        }
        return false;
    }
};

typedef union_find_t<vertex_id_t> equivalence_t;

struct edge_weight_comparator {
    graph_t *g;

    inline edge_weight_comparator(graph_t *g) : g(g) {}

    inline bool operator()(my_edge_id_t a, my_edge_id_t b) const {
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

struct compare_vertices_by_edge_count {
    graph_t *g;
    int k;

    compare_vertices_by_edge_count(graph_t *g, int k) : g(g), k(k) {}

    inline bool operator()(edge_id_t a, edge_id_t b) const {
        return (a % k < b % k) || ((a % k == b % k) && (g->rowsIndices[a + 1] - g->rowsIndices[a]) > (g->rowsIndices[b + 1] - g->rowsIndices[b]));
    }
};

static equivalence_t equivalence(0);
static std::vector<my_edge_id_t> edgeIDs, bestEdges;
static std::vector<vertex_id_t> edgeEnds;
static std::vector<my_weight_t> globalEdgeWeights;
static std::vector<std::vector<my_edge_id_t> *> mstEdges; // MST edges found so far, one vector per thread
static std::vector<std::vector<my_edge_id_t> *> rowIndices; // row indices, one vector per thread
static std::vector<vertex_id_t> mstEdgeCounts, vertexPermutation;

extern "C" void init_mst(graph_t *G) {
    // sort vertices by number of edges, heavy vertices first (or heavy vertices on thread first)
    std::vector<vertex_id_t> vertexIDs(G->n);
    for (vertex_id_t i = 0; i < G->n; i++) {
        vertexIDs[i] = i;
    }
    // There are three variants of sort order, each giving best results for certain combination of graph size and type.
    // Variant A: each thread will have approximately the same number of edges, and vertices will be sorted
    // by the number of edges within each thread
    std::sort(vertexIDs.begin(), vertexIDs.end(), compare_vertices_by_edge_count(G, omp_get_max_threads()));
    // Variant B: each CPU (there are 2 CPUs on target system) will have approximately the same number of edges,
    // and vertices will be sorted by the number of edges within each CPU
//    std::sort(vertexIDs.begin(), vertexIDs.end(), compare_vertices_by_edge_count(G, 2));
    // Variant C: vertices will be sorted by the number of edges globally
//    std::sort(vertexIDs.begin(), vertexIDs.end(), compare_vertices_by_edge_count(G, 1));
    vertexPermutation.resize(G->n);
    for (vertex_id_t i = 0; i < G->n; i++) {
        vertexPermutation[vertexIDs[i]] = i;
    }

    std::vector<my_edge_id_t> globalRowIndices(G->n + 1);
    bestEdges.resize(G->n);
    equivalence = equivalence_t(G->n);
    edgeIDs.resize(G->m);

    my_edge_id_t currentEdge = 0;
    for(vertex_id_t i = 0; i < G->n; i++) {
        vertex_id_t v = vertexIDs[i];

        globalRowIndices[i] = currentEdge;
        for (edge_id_t e = G->rowsIndices[v]; e < G->rowsIndices[v + 1]; e++) {
            if (v != G->endV[e]) {
                edgeIDs[currentEdge++] = (my_edge_id_t) e;
            }
        }

        // filter out duplicate edges for this vertex
        sort(&edgeIDs[globalRowIndices[i]], &edgeIDs[currentEdge], compare_edge_id_by_dst_and_weight(G));
        globalRowIndices[i + 1] = currentEdge = (my_edge_id_t) distance(&edgeIDs[0], unique(&edgeIDs[globalRowIndices[i]], &edgeIDs[currentEdge], equals_edge_id_by_dst(G)));
        // sort remaining edges by weight
        sort(&edgeIDs[globalRowIndices[i]], &edgeIDs[globalRowIndices[i + 1]], edge_weight_comparator(G));
    }

    edgeIDs.resize(currentEdge);
    globalEdgeWeights.resize(currentEdge);
    edgeEnds.resize(currentEdge);

    // Sort edges of each vertex by weight
#pragma omp parallel for
    for (vertex_id_t v = 0; v < G->n; v++) {
        for (my_edge_id_t e = globalRowIndices[v]; e < globalRowIndices[v + 1]; e++) {
            my_edge_id_t id = edgeIDs[e];
            globalEdgeWeights[e] = (my_weight_t) ((1ul << 62) * G->weights[id]); // convert double weights to 64-bit ints
            edgeEnds[e] = vertexPermutation[G->endV[id]];
        }
    }

    mstEdges.resize(omp_get_max_threads());
    mstEdgeCounts.resize(omp_get_max_threads());
    rowIndices.resize(omp_get_max_threads());
#pragma omp parallel
    {
        int rank = omp_get_thread_num(), size = omp_get_num_threads();
        double k = ((double) G->n) / size;
        vertex_id_t v1 = (vertex_id_t) std::round(rank * k), v2 = (vertex_id_t) std::round((rank + 1) * k),
                chunkSize = v2 - v1;

        // allocate thread-local data by this particular thread, to improve memory locality
        mstEdges[rank] = new std::vector<my_edge_id_t>(G->n - 1);
        rowIndices[rank] = new std::vector<my_edge_id_t>(&globalRowIndices[v1], &globalRowIndices[v2 + 1]);
    }
}

// Update root's best edge atomically
inline void update_best(my_edge_id_t *best_edge, my_edge_id_t edge) {
    my_edge_id_t current_best = *best_edge;
    while (!current_best || globalEdgeWeights[edge] < globalEdgeWeights[current_best]) {
        if (atomic_compare_exchange_weak((atomic<my_edge_id_t> *) best_edge, &current_best, edge)) {
            return;
        }
    }
}

// Concurrent Boruvka
void* MST(graph_t *G) {
    vertex_id_t totalFragmentCount = (vertex_id_t) -1; // keep track of the number of remaining fragments
#pragma omp parallel
    {
        // Determine our share of work
        int rank = omp_get_thread_num(), size = omp_get_num_threads();
        double k = ((double) G->n) / size;
        vertex_id_t v1 = (vertex_id_t) std::round(rank * k), v2 = (vertex_id_t) std::round((rank + 1) * k),
                chunkSize = v2 - v1;
        std::vector<my_edge_id_t> &mstEdgesLocal = *mstEdges[rank], &rowIndicesLocal = *rowIndices[rank];
        vertex_id_t edgeCount = 0;

        // Clear equivalence relation from previous runs
        for (vertex_id_t v = v1; v < v2; v++) {
            equivalence.clear(v);
        }
        // We need this barrier because equivalence equivalence relation will be used during init
#pragma omp barrier

        // Combined initialization and initial merge
        vertex_id_t vertexCount = 0, fragmentCount = 0, fragmentRoots[chunkSize], vertexIDs[chunkSize];
        my_edge_id_t currentEdge[chunkSize + 1], *lastEdge = &rowIndicesLocal[1];
        // Reset best edge of each vertex
        memset(&bestEdges[v1], 0, sizeof(my_edge_id_t) * chunkSize);
        // Reset current edge of each vertex to its first edge (in weight order)
        memcpy(currentEdge, &rowIndicesLocal[0], sizeof(my_edge_id_t) * (chunkSize + 1));
        for (vertex_id_t v = v1; v < v2; v++) {
            vertex_id_t w = v - v1;
            if (currentEdge[w] != currentEdge[w + 1]) { // if vertex is not isolated
                fragmentRoots[fragmentCount++] = v; // keep track of vertices that are fragment roots
                if (equivalence.merge(v, edgeEnds[currentEdge[w]])) { // try to merge using first vertex
                    mstEdgesLocal[edgeCount++] = currentEdge[w]; // add edge to MST if merge was successful
                }
                if (++currentEdge[w] != currentEdge[w + 1]) { // do we have more vertices?
                    vertexIDs[vertexCount++] = w; // keep track of active vertices (those with more edges remaining)
                }
            }
        }
        // We need this barrier because equivalence relation will be used during flatten
#pragma omp barrier

        while (true) {
            // update roots for each vertex, such that parent always points to the root
            // this way we do not need to call "find" to get component ID
            for (vertex_id_t v = v1; v < v2; v++) {
                equivalence.flatten(v);
            }

            // stop if there remains only one active component
            // for RMAT graphs, this allows to completely skip a good share of edges
            // note that even we stop here, we still need the previous flatten operation
            // because we need flattened parent pointers during post-processing
            if (totalFragmentCount <= 1) {
                break;
            }
            // We need this barrier because parent pointers in equivalence relation will be used during next stage
#pragma omp barrier

            // skip intra-component edges, update best edges of components
            vertex_id_t newVertexCount = 0;
            for (vertex_id_t i = 0; i < vertexCount; i++) {
                vertex_id_t w = vertexIDs[i], root = equivalence.getParent(w + v1);
                do {
                    if (equivalence.getParent(edgeEnds[currentEdge[w]]) != root) {
                        // we have found an inter-component edge
                        update_best(&bestEdges[root], currentEdge[w]); // try to update root's best edge
                        vertexIDs[newVertexCount++] = w; // return to this vertex during the next iteration
                        break;
                    }
                } while (++currentEdge[w] != lastEdge[w]);
            }
            vertexCount = newVertexCount;

            totalFragmentCount = 0;
            // We need this barrier because best edges will be used during subsequent merge phase
#pragma omp barrier

            // put least-weight edge of each component into tree
            int newFragmentCount = 0, remainingFragmentCount = 0;
            for (vertex_id_t i = 0; i < fragmentCount; i++) {
                vertex_id_t v = fragmentRoots[i];
                my_edge_id_t edge = bestEdges[v];
                if (edge) {
                    // special value 0 is used to indicate that component is empty (does not have inter-component
                    // edges and so does not have a best edge). Note that there is an edge with zero ID, but
                    // it has been already processed during initialization, so this is not a problem.
                    bestEdges[v] = 0; // reset best edge for subsequent iteration
                    fragmentRoots[newFragmentCount++] = v; // remember to return to this component during the next iteration
                    if (equivalence.merge(v, edgeEnds[edge])) { // try to merge using best edge
                        mstEdgesLocal[edgeCount++] = edge; // add edge to MST if merge was successful
                    } else {
                        remainingFragmentCount++; // unsuccessful merge => increase active fragment count
                    }
                }
            }
            // reduce fragment count
            fragmentCount = (vertex_id_t) newFragmentCount;
#pragma omp atomic update
            totalFragmentCount += remainingFragmentCount;
            // We need this barrier because equivalence relation will be used during flatten on next iteration
#pragma omp barrier
        }

        // store this thread's number of MST edges
        mstEdgeCounts[rank] = edgeCount;
    }
    return nullptr;
}

extern "C" void finalize_mst(graph_t *G) {
}

extern "C" void convert_to_output(graph_t *G, void* result, forest_t *trees_output) {
#ifdef __APPLE__
    std::map<edge_id_t, vertex_id_t> componentId;
#else
    std::unordered_map<edge_id_t, vertex_id_t> componentId;
#endif

    // Postprocessing

    // Output format:
    // - mst edges found by each thread (mstEdges), number of such edges (mstEdgeCounts)
    //   (in our numbering)
    // - IDs of edges (edgeIDs, translates our numbering into original IDs)
    // - permutation of vertices (vertexPermutation)
    // - component ID for each vertex (equivalence.a[v].parent)

    // renumber non-empty connected components - O(n)
    vertex_id_t index = 0;
    for(edge_id_t i = 0; i < G->n; i++) {
        vertex_id_t component = equivalence.a[i].parent;
        if (componentId.find(component) == componentId.end()) {
            componentId[component] = index++;
        }
    }

    // Copy edges to vector of vectors (trees_output) - O(n)
    std::vector<std::vector<edge_id_t> > trees_mst(componentId.size());
    for (int i = 0; i < mstEdges.size(); i++) {
        for (edge_id_t j = 0; j < mstEdgeCounts[i]; j++) {
            edge_id_t id = edgeIDs[(*mstEdges[i])[j]];
            trees_mst[componentId[equivalence.a[vertexPermutation[G->endV[id]]].parent]].push_back(id);
        }
    }

    convert_vectors_to_output(G, trees_mst, trees_output);
}