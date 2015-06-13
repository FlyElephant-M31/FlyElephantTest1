#ifndef __GRAPH_HPC_DEFS_H
#define __GRAPH_HPC_DEFS_H
#define __STDC_FORMAT_MACROS

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <vector>
#ifdef __APPLE__
#include <map>
#else
#include <unordered_map>
#include <unordered_set>
#endif
#include <set>

#define DEFAULT_ARITY 16
#define SMALL_COMPONENT_EDGES_THRESHOLD   2
#define FNAME_LEN   256
#define WEIGHT_ERROR 0.0001

typedef uint32_t vertex_id_t;
typedef uint64_t edge_id_t;
typedef double weight_t;

/* The graph data structure*/
typedef struct
{
    /***
     The minimal graph repesentation consists of:
     n        -- the number of vertices
     m        -- the number of edges
     endV     -- an array of size m that stores the 
                 destination ID of an edge <src->dest>.
     rowsIndices -- an array of size n+1 that stores the degree 
                 (out-degree in case of directed graphs) and pointers to
                 the endV array. The degree of vertex i is given by 
                 rowsIndices[i+1]-rowsIndices[i], and the edges out of i are
                 stored in the contiguous block endV[rowsIndices[i] .. rowsIndices[i+1]-1].
     Vertices are ordered from 0 in our internal representation
     ***/
    vertex_id_t n;
    edge_id_t m;
    edge_id_t* rowsIndices;
    vertex_id_t* endV;

    /* Edge weights */
    weight_t* weights;
    weight_t min_weight, max_weight;

    /* other graph parameters */
    int scale; /* log2 of vertices number */
    int avg_vertex_degree; /* relation m / n */
    bool directed; 

    /* RMAT graph parameters */
    double a, b, c;     
    bool permute_vertices;
    
    /* Distributed version variables */
    int nproc, rank;
    vertex_id_t local_n; /* local vertices number */
    edge_id_t local_m; /* local edges number */
    edge_id_t* num_edges_of_any_process;

    char filename[FNAME_LEN]; /* filename for output graph */
} graph_t;

typedef struct
{
    vertex_id_t numTrees;
    edge_id_t numEdges;
    edge_id_t* p_edge_list;
    edge_id_t* edge_id;

} forest_t;

/* write graph to file */
void writeGraph(graph_t *G, char *filename);
void writeBinaryGraph(graph_t *G, char *filename);
void writeTextGraph(graph_t *G);
void writeTextGraph_MPI(graph_t *G);
void writeBinaryGraph_MPI(graph_t *G);

/* read graph from file */
void readGraph(graph_t *G, char *filename);
void readGraph_rankFiles_MPI(graph_t *G, char *filename);
void readGraph_singleFile_MPI(graph_t *G, char *filename);

/* free graph memory */
void freeGraph(graph_t *G);

#ifdef __cplusplus
    #define EXTERN_DECL extern "C"
#else
    #define EXTERN_DECL 
#endif

/* Minimum spanning tree */
EXTERN_DECL void* MST (graph_t *G);
EXTERN_DECL void convert_to_output(graph_t *G, void *result, forest_t* output); 

/* initialize algorithm memory */
EXTERN_DECL void init_mst(graph_t *G);
EXTERN_DECL void finalize_mst(graph_t *G);
EXTERN_DECL void gen_SSCA2_graph_MPI(graph_t *G);
EXTERN_DECL void gen_RMAT_graph_MPI(graph_t *G);

/* returns global number of the edge from local number */
edge_id_t edge_to_global(edge_id_t, graph_t*);

#define MOD_SIZE(v) ((v) % size)
#define DIV_SIZE(v) ((v) / size)
#define MUL_SIZE(x) ((x) * size)

/* returns number of vertex owner, v - the global vertex number, TotVertices - the global number of vertices, size - the number of processes*/
inline int VERTEX_OWNER(const vertex_id_t v, const vertex_id_t TotVertices, const int size)
{
    vertex_id_t mod_size = MOD_SIZE(TotVertices);
    vertex_id_t div_size = DIV_SIZE(TotVertices);
    if (!mod_size)
        return v / div_size;
    else
    {
        if (v / (div_size + 1) < mod_size)
            return v / (div_size + 1);
        else
            return (v - mod_size * (div_size + 1)) / div_size + mod_size;
    }
}

/* returns local vertex number, v - the global vertex number, TotVertices - the global number of vertices, size - the number of processes, rank - the process number*/
inline vertex_id_t VERTEX_LOCAL(const vertex_id_t v, const vertex_id_t TotVertices, const int size, const int rank) 
{
    if (MOD_SIZE(TotVertices) <= (unsigned int) rank)
        return ((v - MOD_SIZE(TotVertices) * (DIV_SIZE(TotVertices) + 1))%DIV_SIZE(TotVertices));
    else
        return (v%(DIV_SIZE(TotVertices) + 1));
}

/* returns global vertex number, v_local - the local vertex number, TotVertices - the global number of vertices, size - the number of processes, rank - the process number*/
inline vertex_id_t VERTEX_TO_GLOBAL(const vertex_id_t v_local, const vertex_id_t TotVertices, const int size, const int rank)
{
    if(MOD_SIZE(TotVertices) > (unsigned int) rank )
        return ((DIV_SIZE(TotVertices) + 1)*rank + (vertex_id_t) v_local);
    else
        return (MOD_SIZE(TotVertices)*(DIV_SIZE(TotVertices) + 1) + DIV_SIZE(TotVertices)*(rank - MOD_SIZE(TotVertices)) + v_local);
}

#define LOC(v) (VERTEX_LOCAL((v), G->n, G->nproc, G->rank))
#define OWN(v) (VERTEX_OWNER((v), G->n, G->nproc))
#define GLO(lv) (VERTEX_TO_GLOBAL((lv), G->n, G->nproc, G->rank))
#define REM(v) (VERTEX_LOCAL((v), G->n, G->nproc, OWN(v)))

#ifdef __APPLE__

#include <mach/mach_time.h>
#define ORWL_NANO (+1.0E-9)
#define ORWL_GIGA UINT64_C(1000000000)

static double orwl_timebase = 0.0;
static uint64_t orwl_timestart = 0;

static inline struct timespec gettime(void) {
    // be more careful in a multithreaded environement
    if (!orwl_timestart) {
        mach_timebase_info_data_t tb = { 0 };
        mach_timebase_info(&tb);
        orwl_timebase = tb.numer;
        orwl_timebase /= tb.denom;
        orwl_timestart = mach_absolute_time();
    }
    struct timespec t;
    double diff = (mach_absolute_time() - orwl_timestart) * orwl_timebase;
    t.tv_sec = diff * ORWL_NANO;
    t.tv_nsec = diff - (t.tv_sec * ORWL_GIGA);
    return t;
}

#else
#include <time.h>
#if defined(CLOCK_MONOTONIC)
#define CLOCK CLOCK_MONOTONIC
#elif defined(CLOCK_REALTIME)
#define CLOCK CLOCK_REALTIME
#else
#error "Failed to find a timing clock."
#endif

static inline struct timespec gettime() {
    struct timespec t;
    clock_gettime(CLOCK, &t);
    return t;
}
#endif

static inline void convert_vectors_to_output(graph_t *G, std::vector<std::vector<edge_id_t> > &trees_mst, forest_t *trees_output)
{
    trees_output->p_edge_list = (edge_id_t *)malloc(trees_mst.size()*2 * sizeof(edge_id_t));
    edge_id_t number_of_edges = 0;
    for (vertex_id_t i = 0; i < trees_mst.size(); i++) number_of_edges += trees_mst[i].size();
    trees_output->edge_id = (edge_id_t *)malloc(number_of_edges * sizeof(edge_id_t));
    trees_output->p_edge_list[0] = 0;
    trees_output->p_edge_list[1] = trees_mst[0].size();
    for (vertex_id_t i = 1; i < trees_mst.size(); i++) {
        trees_output->p_edge_list[2*i] = trees_output->p_edge_list[2*i-1];
        trees_output->p_edge_list[2*i +1] = trees_output->p_edge_list[2*i-1] + trees_mst[i].size();
    }
    int k = 0;
    for (vertex_id_t i = 0; i < trees_mst.size(); i++) {
        for (edge_id_t j = 0; j < trees_mst[i].size(); j++) {
            trees_output->edge_id[k] = trees_mst[i][j];
            k++;
        }
    }

    trees_output->numTrees = trees_mst.size();
    trees_output->numEdges = number_of_edges;
}

static inline void convert_mask_and_parent_to_output(graph_t *G, edge_id_t *edgeIDs, edge_id_t maskSize, uint8_t *mask,
        vertex_id_t *vertexPermutation, void *parent, size_t parentStride, forest_t *trees_output) {
#define __GET_PARENT(i) (*((vertex_id_t*) (((char*) parent) + (i) * parentStride)))
#ifdef __APPLE__
    std::map<edge_id_t, vertex_id_t> componentId;
    std::set<weight_t> usedEdges;
#else
    std::unordered_map<edge_id_t, vertex_id_t> componentId;
    std::unordered_set<weight_t> usedEdges;
#endif

    // renumber non-empty connected components
    vertex_id_t index = 0;
    for(edge_id_t i = 0; i < G->n; i++) {
        vertex_id_t component = __GET_PARENT(i);
        if (componentId.find(component) == componentId.end()) {
            componentId[component] = index++;
        }
    }

    // Copy edges to vector of vectors. Filter out duplicate half-edges belonging to the same full edge.
    std::vector<std::vector<edge_id_t> > trees_mst(componentId.size());
    for(edge_id_t i = 0; i < maskSize; i++) {
        if (mask[i]) {
            edge_id_t id = edgeIDs ? edgeIDs[i] : i;
            vertex_id_t edgeEnd = vertexPermutation ? vertexPermutation[G->endV[id]] : G->endV[id];
            vertex_id_t component = componentId[__GET_PARENT(edgeEnd)];
            weight_t edge = G->weights[id];
            if (usedEdges.find(edge) == usedEdges.end()) {
                trees_mst[component].push_back(id);
                usedEdges.insert(edge);
            }
        }
    }

    convert_vectors_to_output(G, trees_mst, trees_output);
#undef __GET_PARENT
}

#ifdef OUTPUT_FORMAT_VECTOR_VECTOR

typedef std::vector<std::vector<edge_id_t > > result_t;

/* Convert MST implementation output to GraphHPC-2015 forest_t data type. forest_t will be used in validation
 * NOTE: isolated vertex is also tree, such tree must be represented as separate element of trees_mst vector with zero-length edges list
 * FIXME: If you change MST output data structure, you must change this function */
extern "C" void convert_to_output(graph_t *G, void* result, forest_t *trees_output)
{
    convert_vectors_to_output(G, *reinterpret_cast<result_t*>(result), trees_output);
}

#endif

#ifdef OUTPUT_FORMAT_MASK_AND_PARENT

struct result_t {
    uint8_t *mask;
    void *parent;
    size_t stride;
};

extern "C" void convert_to_output(graph_t *G, void* result, forest_t *trees_output)
{
    result_t &mst = *reinterpret_cast<result_t*>(result);
    convert_mask_and_parent_to_output(G, mst.mask, mst.parent, mst.stride, trees_output);
}

#endif

#ifdef OUTPUT_FORMAT_MASK

typedef std::vector<uint8_t> result_t;

namespace result {

    struct edge_t {
        edge_id_t id;
        vertex_id_t src, dst;
        weight_t weight;
        vertex_id_t component;
    };

    template<class T>
    struct union_find_t {
        std::vector<T> parents;

        union_find_t(T size) : parents(size, 0) {
        }

        inline T find(T x) {
            T y = parents[x];
            if (!y) return x;
            do {
                T z = parents[y];
                if (!z) {
                    return y;
                }
                y = parents[x] = z;
            } while (true);
        }

        inline void merge(T x, T y) {
            x = find(x);
            y = find(y);
            if (y > x) {
                parents[x] = y;
            } else if (y < x) {
                parents[y] = x;
            }
        }
    };

#define COMPARE(x, y) do { if( (x) < (y) ) return true; if ( (x) > (y) ) return false; } while(0)

    inline bool operator < (const edge_t &a, const edge_t &b) {
        COMPARE(a.component, b.component);
        COMPARE(a.src, b.src);
        COMPARE(a.dst, b.dst);
        COMPARE(a.weight, b.weight);
        COMPARE(a.id, b.id);
        return false;
    }

#undef COMPARE
}

/* Convert MST implementation output to GraphHPC-2015 forest_t data type. forest_t will be used in validation
 * NOTE: isolated vertex is also tree, such tree must be represented as separate element of trees_mst vector with zero-length edges list
 * FIXME: If you change MST output data structure, you must change this function */
extern "C" void convert_to_output(graph_t *G, void* result, forest_t *trees_output)
{
    result_t &mst = *reinterpret_cast<result_t*>(result);

    // we need edges startV, vector of edge starts, analog of G.endV
    std::vector<vertex_id_t> startV(G->m);
    for(vertex_id_t v = 0; v < G->n; v++) {
        for(edge_id_t e = G->rowsIndices[v]; e < G->rowsIndices[v + 1]; e++) {
            startV[e] = v;
        }
    }

    // determine connected components according to MST
    std::vector<result::edge_t> edges;
    result::union_find_t<vertex_id_t> equivalence(G->n);
    for(edge_id_t e = 0; e < G->m; e++) {
        if( mst[e] ) {
            result::edge_t edge = {e, std::min(startV[e], G->endV[e]), std::max(startV[e], G->endV[e]), G->weights[e], 0};
            edges.push_back(edge);
            equivalence.merge(startV[e], G->endV[e]);
        }
    }
    for(edge_id_t e = 0; e < edges.size(); e++) {
        edges[e].component = equivalence.find(edges[e].src);
    }
    // sort edges by component and then by vertices and weights
    std::sort(edges.begin(), edges.end());

    // filter out duplicate (half)edges
    edge_id_t e0 = 1;
    for(edge_id_t e = 1; e < edges.size(); e++) {
        if( edges[e].src != edges[e-1].src || edges[e].dst != edges[e-1].dst || edges[e].weight != edges[e-1].weight) {
            edges[e0++] = edges[e];
        }
    }
    edges.resize(e0);

    // count connected components, including isolated vertices
    vertex_id_t componentCount = 0;
    for(vertex_id_t v = 0; v < G->n; v++) {
        if (equivalence.find(v) == v) {
            componentCount++;
        }
    }

    trees_output->p_edge_list = (edge_id_t *)malloc(componentCount * 2 * sizeof(edge_id_t));
    trees_output->edge_id = (edge_id_t *)malloc(edges.size() * sizeof(edge_id_t));

    vertex_id_t component = 0;
    trees_output->p_edge_list[2*component] = 0;
    for(edge_id_t e = 0; e < edges.size(); e++) {
        trees_output->edge_id[e] = edges[e].id;
        if( e > 0 && edges[e].component != edges[e-1].component ) {
            trees_output->p_edge_list[2*component + 1] = e;
            component++;
            trees_output->p_edge_list[2*component] = e;
        }
    }
    trees_output->p_edge_list[2*component + 1] = edges.size();
    for(vertex_id_t c = component + 1; c < componentCount; c++) {
        trees_output->p_edge_list[2*c] = edges.size();
        trees_output->p_edge_list[2*c + 1] = edges.size();
    }

    trees_output->numTrees = componentCount;
    trees_output->numEdges = edges.size();
}

#endif

#endif
