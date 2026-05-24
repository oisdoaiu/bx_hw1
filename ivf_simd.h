#ifndef IVF_SIMD_H
#define IVF_SIMD_H

#include "flat_simd.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <queue>
#include <utility>

// ===== KMeans clustering =====
inline void kmeans_cluster(const float* base, size_t base_number, size_t vecdim,
                            float* centroids, size_t nlist, int max_iters)
{
    // random init: pick first nlist vectors as initial centroids
    for(size_t c = 0; c < nlist; c++){
        size_t idx = (c * (base_number / nlist)) % base_number;
        std::memcpy(centroids + c * vecdim, base + idx * vecdim, vecdim * sizeof(float));
    }

    std::vector<int> assignments(base_number);

    for(int iter = 0; iter < max_iters; iter++){
        // assign
        for(size_t i = 0; i < base_number; i++){
            const float* v = base + i * vecdim;
            float best_dist = FLT_MAX;
            int best_c = 0;
            for(size_t c = 0; c < nlist; c++){
                float d = InnerProductSIMD(v, centroids + c * vecdim, vecdim);
                if(d < best_dist){
                    best_dist = d;
                    best_c = (int)c;
                }
            }
            assignments[i] = best_c;
        }

        // update
        std::vector<float> sum(nlist * vecdim, 0.0f);
        std::vector<int> counts(nlist, 0);
        for(size_t i = 0; i < base_number; i++){
            int c = assignments[i];
            const float* v = base + i * vecdim;
            float* dst = sum.data() + c * vecdim;
            for(size_t d = 0; d < vecdim; d++) dst[d] += v[d];
            counts[c]++;
        }
        for(size_t c = 0; c < nlist; c++){
            if(counts[c] > 0){
                float inv = 1.0f / (float)counts[c];
                float* cent = centroids + c * vecdim;
                for(size_t d = 0; d < vecdim; d++)
                    cent[d] = sum[c * vecdim + d] * inv;
            }
        }
    }
}

// ===== IVF index =====
static std::vector<float> g_ivf_centroids;
static std::vector<std::vector<int>> g_ivf_lists;
static size_t g_ivf_nlist;
static const float* g_ivf_base;

inline void build_ivf(const float* base, size_t base_number, size_t vecdim)
{
    g_ivf_nlist = 256;
    const char* env_nl = std::getenv("IVF_NLIST");
    if(env_nl) g_ivf_nlist = (size_t)std::atoi(env_nl);

    g_ivf_base = base;
    g_ivf_centroids.resize(g_ivf_nlist * vecdim);
    kmeans_cluster(base, base_number, vecdim, g_ivf_centroids.data(), g_ivf_nlist, 15);

    // assign all vectors to clusters
    g_ivf_lists.resize(g_ivf_nlist);
    for(size_t c = 0; c < g_ivf_nlist; c++) g_ivf_lists[c].clear();

    for(size_t i = 0; i < base_number; i++){
        const float* v = base + i * vecdim;
        float best_dist = FLT_MAX;
        int best_c = 0;
        for(size_t c = 0; c < g_ivf_nlist; c++){
            float d = InnerProductSIMD(v, g_ivf_centroids.data() + c * vecdim, vecdim);
            if(d < best_dist){ best_dist = d; best_c = (int)c; }
        }
        g_ivf_lists[best_c].push_back((int)i);
    }

    size_t total = 0;
    for(size_t c = 0; c < g_ivf_nlist; c++) total += g_ivf_lists[c].size();
    std::cerr << "IVF built: nlist=" << g_ivf_nlist << " total_vectors=" << total << "\n";
}

// ===== IVF search =====
inline std::priority_queue<std::pair<float, int>> ivf_search(
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t nprobe)
{
    // coarse: find nearest nprobe clusters (negate dist for min-heap)
    std::priority_queue<std::pair<float, int>> cluster_heap;
    for(size_t c = 0; c < g_ivf_nlist; c++){
        float d = InnerProductSIMD(query, g_ivf_centroids.data() + c * vecdim, vecdim);
        cluster_heap.push(std::make_pair(-d, (int)c));
    }

    std::vector<int> probes;
    probes.reserve(nprobe);
    for(size_t i = 0; i < nprobe && !cluster_heap.empty(); i++){
        probes.push_back(cluster_heap.top().second);
        cluster_heap.pop();
    }

    // fine: scan selected clusters
    std::priority_queue<std::pair<float, int>> result;
    for(int c : probes){
        for(int idx : g_ivf_lists[c]){
            float dist = InnerProductSIMD(g_ivf_base + (size_t)idx * vecdim, query, vecdim);
            if(result.size() < k){
                result.push(std::make_pair(dist, idx));
            }else if(dist < result.top().first){
                result.pop();
                result.push(std::make_pair(dist, idx));
            }
        }
    }

    return result;
}

inline std::priority_queue<std::pair<float, int>> ivf_solve(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k)
{
    size_t nprobe = 8;
    const char* env_np = std::getenv("IVF_NPROBE");
    if(env_np) nprobe = (size_t)std::atoi(env_np);
    return ivf_search(query, base_number, vecdim, k, nprobe);
}

#endif
