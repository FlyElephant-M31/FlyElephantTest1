#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "defs.h"
#include "profiler.hpp"

static Profiler profiler;

// Parallel Kruskal inspired by paper
// Loncar, Vladmir, Srdjan Skrbic, and Antun Balaz. “Distributed Memory Parallel Algorithms for Minimum Spanning Trees ,” Vol. 2, London, 2013.
//
// Each process (out of p) needs O(n) + O(m/p) memory. So this algorithm does not scale with the number of vertices,
// but it does scale with the number of edges (in particular, it is suitable for dense graphs).

struct edge_t {
    edge_id_t id; // global edge id
    vertex_id_t src, dst; // global vertex ids
    weight_t weight;
};

struct CompareEdgesByWeight {
    inline bool operator ()(const edge_t &a, const edge_t &b) const {
        return a.weight < b.weight;
    }
};

// Implementation of Union-Find with ranks
// We have two versions of Find, find (with path compression) and find_fast (without path compression)

template<class T>
struct union_find_record_t {
    T parent;
    T rank;
};

template<class T>
struct union_find_t {
    std::vector<union_find_record_t<T>> a;

    union_find_t(T size) : a(size) {
    }

    inline void clear() {
        for (T p = 0; p < a.size(); p++) {
            a[p] = {p, 0};
        }
    }

    inline T find_fast(T x) const {
#pragma unroll(8)
        while (x != a[x].parent) {
            x = a[x].parent;
        }
        return x;
    }

    inline T find(T x) {
        T y = a[x].parent;
        if (y != x) {
            T z = a[y].parent;
            if (z != y) {
#pragma unroll(8)
                do {
                    y = z;
                    z = a[y].parent;
                } while (z != y);
                a[x].parent = z;
            }
        }
        return y;
    }

    inline void merge_roots(T x, T y) {
        if (a[y].rank > a[x].rank) {
            a[x].parent = y;
        } else if (a[y].rank == a[x].rank) {
            if (y < x) {
                a[x].parent = y;
                a[y].rank++;
            } else {
                a[y].parent = x;
                a[x].rank++;
            }
        } else {
            a[y].parent = x;
        }
    }

    inline void flatten() {
        for (T p = 0; p < a.size(); p++) {
            a[p].parent = find_fast(p);
        }
    }
};

typedef union_find_t<vertex_id_t> equivalence_t;

std::vector<edge_t> localEdges;
equivalence_t equivalence(0);
std::vector<edge_id_t> stack;
std::vector<edge_t> edges, buffer[4], *mst;
edge_id_t mstSize;

// Implementation of Kruskal algorithm on sorted lists of edges
// kruskal1 operates on one sorted list
// kruskal2, kruskal3 operate on two and three sorted lists respectively. The idea is to avoid merging sorted list before
// calling kruskal1.

template<class Input, class Output>
static inline bool process(Input &first, Input last, Output &output) {
    vertex_id_t a = equivalence.find_fast(first->src), b = equivalence.find(first->dst);
    if (a != b) {
        *(output++) = *first;
        equivalence.merge_roots(a, b);
    }
    return ++first != last;
}

template<class Input, class Output>
static inline Output kruskal1(Input first, Input last, Output output) {
    if (first == last) {
        return output;
    }
    while (true) {
        if (!process(first, last, output)) {
            return output;
        }
    }
}

template<class Input, class Output>
static inline Output kruskal2(Input first1, Input last1, Input first2, Input last2, Output output) {
    if (first1 == last1) {
        return kruskal1(first2, last2, output);
    } else if (first2 == last2) {
        return kruskal1(first1, last1, output);
    }
    while (true) {
        if (first1->weight < first2->weight) {
            if (!process(first1, last1, output)) {
                return kruskal1(first2, last2, output);
            }
        } else {
            if (!process(first2, last2, output)) {
                return kruskal1(first1, last1, output);
            }
        }
    }
}

template<class Input, class Output>
static inline Output kruskal3(Input first1, Input last1, Input first2, Input last2, Input first3, Input last3, Output output) {
    if (first1 == last1) {
        return kruskal2(first2, last2, first3, last3, output);
    } else if (first2 == last2) {
        return kruskal2(first1, last1, first3, last3, output);
    } else if (first3 == last3) {
        return kruskal2(first1, last1, first2, last2, output);
    }
    while (true) {
        if (first1->weight < first2->weight) {
            if (first3->weight < first1->weight) {
                if (!process(first3, last3, output)) {
                    return kruskal2(first1, last1, first2, last2, output);
                }
            } else {
                if (!process(first1, last1, output)) {
                    return kruskal2(first2, last2, first3, last3, output);
                }
            }
        } else {
            if (first3->weight < first2->weight) {
                if (!process(first3, last3, output)) {
                    return kruskal2(first1, last1, first2, last2, output);
                }
            } else {
                if (!process(first2, last2, output)) {
                    return kruskal2(first1, last1, first3, last3, output);
                }
            }
        }
    }
}

// Implementation of IQS (Incremental Quick Select) from paper
// Navarro, Gonzalo, and Rodrigo Paredes. “On Sorting, Heaps, and Minimum Spanning Trees.”
// Algorithmica 57, no. 4 (March 23, 2010): 585–620. doi:10.1007/s00453-010-9400-6.

template<class T, class I, class C>
static inline I partition(T &list, I left, I right, I pivot, const C &compare) {
    std::swap(list[pivot], list[right - 1]);
    for (I i = left; i < right - 1; i++) {
        if (compare(list[i], list[right - 1])) {
            std::swap(list[i], list[left++]);
        }
    }
    std::swap(list[left], list[right - 1]);
    return left;
}

template<class T, class I, class C>
static inline void iqs(T &list, I left, I** stack, const C &compare) {
    // remove pivots that are smaller than left
    while (**stack < left) {
        (*stack)--;
    }
    if (**stack == left) {
        (*stack)--;
        return;
    }
    while (true) {
        I pivot = left + std::floor(drand48() * (**stack - left));
        pivot = partition(list, left, **stack, pivot, compare);
        if (pivot == left) {
            return;
        } else {
            *(++*stack) = pivot;
        }
    }
}

// Implementation of Kruskal's algorithm on unsorted list of edges using IQS

template<class T, class I, class O>
inline O kruskal_iqs(T &list, I size, O out) {
    stack[0] = size;
    edge_id_t *stackPtr = &stack[0];
    for (edge_id_t e = 0; e < size;) {
        iqs(list, (edge_id_t) e, &stackPtr, CompareEdgesByWeight());
        vertex_id_t a = equivalence.find_fast(list[e].src), b = equivalence.find(list[e].dst);
        if (a != b) {
            equivalence.merge_roots(a, b);
            *(out++) = list[e];
        }

        // skip edges not in MST
        do {
            e++;
        } while (e < size && equivalence.find_fast(list[e].src) == equivalence.find(list[e].dst));
    }
    return out;
}

extern "C" void init_mst(graph_t *G) {
    // learn number of edges on each process
    // FIXME: workaround for missing MPI_IN_PLACE during judgement
    std::vector<edge_id_t> edgesPerProcess0((unsigned long) G->nproc, G->local_m);
    std::vector<edge_id_t> edgesPerProcess((unsigned long) G->nproc, G->local_m);
    MPI_Alltoall(&edgesPerProcess0[0], 1, MPI_UINT64_T, &edgesPerProcess[0], 1, MPI_UINT64_T, MPI_COMM_WORLD);

    // calculate offset for our edge id's
    edge_id_t offset = 0;
    for (int p = 0; p < G->rank; p++) {
        offset += edgesPerProcess[p];
    }

    // preprocess local edges
    // remove half of unidirected arcs
    localEdges.reserve(G->local_m);
    for (vertex_id_t v = 0; v < G->local_n; v++) {
        edge_id_t firstEdge = localEdges.size();
        vertex_id_t src = VERTEX_TO_GLOBAL(v, G->n, G->nproc, G->rank);
        for (edge_id_t e = G->rowsIndices[v]; e < G->rowsIndices[v + 1]; e++) {
            // we cannot simply have src > dst because this will cause
            // significant disbalance between processes
            bool include = ((G->endV[e] ^ src) & 1) ? src < G->endV[e] : src > G->endV[e];
            if (include) {
                localEdges.push_back({offset + e, src, G->endV[e], G->weights[e]});
            }
        }
        edge_id_t lastEdge = localEdges.size();
        // pre-sort edges of current vertex
        std::sort(&localEdges[firstEdge], &localEdges[lastEdge], CompareEdgesByWeight());
    }

    if (G->nproc > 1) {
        // redistribute edges across processes evenly
        // FIXME: this is a workaround for bugs in MPI graph generators, should not need this if
        // graphs are generated correctly
        // not a very efficient scheme, but it works
        edge_id_t totalEdgeCount = localEdges.size();
        MPI_Allreduce(MPI_IN_PLACE, &totalEdgeCount, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

        edge_id_t meanEdgeCount = totalEdgeCount / (G->nproc - 1);
        if (G->rank != G->nproc - 1) {
            edge_id_t count;
            int peer = (G->rank + G->nproc - 1) % G->nproc;
            MPI_Recv(&count, 1, MPI_UINT64_T, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            localEdges.resize(localEdges.size() + count);
            MPI_Recv(&localEdges[localEdges.size() - count], (int) (sizeof(edge_t) * count), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (G->rank != G->nproc - 2) {
            edge_id_t count = G->rank == G->nproc - 1 ? localEdges.size() : (localEdges.size() > meanEdgeCount ? localEdges.size() - meanEdgeCount : 0);
            int peer = (G->rank + 1) % G->nproc;
            MPI_Send(&count, 1, MPI_UINT64_T, peer, 0, MPI_COMM_WORLD);
            MPI_Send(&localEdges[localEdges.size() - count], (int) (sizeof(edge_t) * count), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
            localEdges.resize(localEdges.size() - count);
        }
    }

    equivalence = equivalence_t(G->n);
    stack.resize(1000);
    edges.resize(localEdges.size());
    for (int i = 0; i < 4; i++) {
        buffer[i].resize(G->n - 1);
    }
}

extern "C" void* MST(graph_t *G) {
    profiler.tic();

    // start with our local set of edges
    edge_id_t edgeCount = (edge_id_t) std::distance(edges.begin(), std::copy(localEdges.begin(), localEdges.end(), edges.begin()));
    profiler.toc(10);

    // receive edges from children
    // ranks of adjacent nodes in the tree
    int upRank = ((G->rank + 1) >> 1) - 1;
    int leftRank = ((G->rank + 1) << 1) - 1;
    int rightRank = leftRank + 1;

    // post two receive requests
    MPI_Request requests[2];
    int requestCount = 0;
    if (leftRank < G->nproc) {
        MPI_Irecv(&buffer[1][0], (int) (G->n * sizeof(edge_t)), MPI_BYTE, leftRank, 0, MPI_COMM_WORLD, &requests[requestCount++]);
        if (rightRank < G->nproc) {
            MPI_Irecv(&buffer[2][0], (int) (G->n * sizeof(edge_t)), MPI_BYTE, rightRank, 0, MPI_COMM_WORLD, &requests[requestCount++]);
        }
    }
    profiler.toc(15);

    // perform Kruskal on local edges and put result in buffer 0
    mstSize = 0;
    mst = &buffer[0];
    if (edgeCount) {
        equivalence.clear();
        mstSize = (edge_id_t) std::distance(mst->begin(), kruskal_iqs(edges, edgeCount, mst->begin()));
    }
    profiler.toc(20);

    // receive data and merge MSTs
    if (G->scale >= 12 && requestCount == 2) {
        MPI_Status stati[2];
        MPI_Waitall(2, requests, stati);
        int count1, count2;
        MPI_Get_count(&stati[0], MPI_BYTE, &count1);
        MPI_Get_count(&stati[1], MPI_BYTE, &count2);
        edge_id_t size1 = count1 / sizeof(edge_t), size2 = count2 / sizeof(edge_t);
        if (size1 && size2) {
            equivalence.clear();
            mstSize = (edge_id_t) std::distance(buffer[3].begin(), kruskal3(mst->begin(), mst->begin() + mstSize,
                    buffer[1].begin(), buffer[1].begin() + size1,
                    buffer[2].begin(), buffer[2].begin() + size2, buffer[3].begin()));
            mst = &buffer[3];
            profiler.toc((unsigned int) 40);
        }
    } else {
        for (int i = 0; i < requestCount; i++) {
            int index, count;
            MPI_Status status;
            MPI_Waitany(requestCount, requests, &index, &status);
            profiler.toc((unsigned int) (30 + i));
            MPI_Get_count(&status, MPI_BYTE, &count);
            edge_id_t size = count / sizeof(edge_t);
            if (size) {
                equivalence.clear();
                std::vector<edge_t> &in = buffer[1 + index];
                std::vector<edge_t> &out = i == 0 ? buffer[3] : buffer[2 - index];
                mstSize = (edge_id_t) std::distance(out.begin(), kruskal2(mst->begin(), mst->begin() + mstSize,
                        in.begin(), in.begin() + size, out.begin()));
                mst = &out;
                profiler.toc((unsigned int) (40 + i));
            }
        }
    }

    if (upRank >= 0) {
        // send data to parent
        MPI_Send(&*mst->begin(), (int) (sizeof(edge_t) * mstSize), MPI_BYTE, upRank, 0, MPI_COMM_WORLD);
        profiler.toc(70);
    } else {
        // flatten equivalence relation, this is necessary for post-processing stage
        equivalence.flatten();
        profiler.toc(60);
    }

    return NULL;
}

extern "C" void finalize_mst(graph_t* G) {
    if (G->rank == 0) {
        std::cout << profiler;
    }
}

extern "C" void convert_to_output(graph_t *G, void* result, forest_t *trees_output)
{
    if (G->rank == 0) {
#ifdef __APPLE__
        std::map<edge_id_t, vertex_id_t> componentId;
#else
        std::unordered_map<edge_id_t, vertex_id_t> componentId;
#endif

        // renumber non-empty connected components
        vertex_id_t index = 0;
        for(edge_id_t i = 0; i < G->n; i++) {
            vertex_id_t component = equivalence.a[i].parent;
            if (componentId.find(component) == componentId.end()) {
                componentId[component] = index++;
            }
        }

        // Copy edges to vector of vectors
        // We do not need to filter out duplicates here due to properties of Kruskal's algorithm
        std::vector<std::vector<edge_id_t> > trees_mst(componentId.size());
        for(edge_id_t e = 0; e < mstSize; e++) {
            trees_mst[componentId[equivalence.a[(*mst)[e].dst].parent]].push_back((*mst)[e].id);
        }

        convert_vectors_to_output(G, trees_mst, trees_output);
    }
}