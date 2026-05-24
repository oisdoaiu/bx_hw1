#ifndef IVF_PQ_SIMD_H
#define IVF_PQ_SIMD_H

#include "pq_gather.h"
#include "ivf_simd.h"
#include <vector>

// IVF-PQ search: IVF coarse ranking + PQ approximate distance fine ranking
// PQ distance = 1.0 - sum(LUT[code[m]]) for m=0..M-1
// This replaces InnerProductSIMD (96 madds) with M lookups + adds (8 with M=8)
inline std::priority_queue<std::pair<float, int>> ivf_pq_search(
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t nprobe)
{
    // Build PQ LUT for this query
    std::vector<float> lut(g_pq_M * g_pq_Ks);
    build_lut(query, lut.data(), g_pq_centroids.data(), vecdim, g_pq_M, g_pq_Ks);

    // Coarse: find nearest nprobe clusters (same as IVF, exact float distance)
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

    // Fine: scan selected clusters using PQ approximate distance
    std::priority_queue<std::pair<float, int>> result;
    for(int c : probes){
        for(int idx : g_ivf_lists[c]){
            // PQ distance: sum up LUT lookups across M subspaces
            float ip_sum = 0.0f;
            for(size_t m = 0; m < g_pq_M; m++){
                ip_sum += lut[m * g_pq_Ks + g_pq_codes[m * base_number + (size_t)idx]];
            }
            float dist = 1.0f - ip_sum;

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

// Re-rank: after IVF-PQ coarse filter, refine top candidates with exact float distance
// p: number of PQ candidates to re-rank
inline std::priority_queue<std::pair<float, int>> ivf_pq_rerank_search(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t rerank_p)
{
    // Build PQ LUT
    std::vector<float> lut(g_pq_M * g_pq_Ks);
    build_lut(query, lut.data(), g_pq_centroids.data(), vecdim, g_pq_M, g_pq_Ks);

    // Coarse
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

    // Fine: PQ scan to get top rerank_p candidates
    std::priority_queue<std::pair<float, int>> pq_heap;
    for(int c : probes){
        for(int idx : g_ivf_lists[c]){
            float ip_sum = 0.0f;
            for(size_t m = 0; m < g_pq_M; m++){
                ip_sum += lut[m * g_pq_Ks + g_pq_codes[m * base_number + (size_t)idx]];
            }
            float dist = 1.0f - ip_sum;
            if(pq_heap.size() < rerank_p){
                pq_heap.push(std::make_pair(dist, idx));
            }else if(dist < pq_heap.top().first){
                pq_heap.pop();
                pq_heap.push(std::make_pair(dist, idx));
            }
        }
    }

    // Re-rank: exact float distance on top candidates
    std::priority_queue<std::pair<float, int>> result;
    while(!pq_heap.empty()){
        int idx = pq_heap.top().second; pq_heap.pop();
        float dist = InnerProductSIMD(base + (size_t)idx * vecdim, query, vecdim);
        if(result.size() < k){
            result.push(std::make_pair(dist, idx));
        }else if(dist < result.top().first){
            result.pop();
            result.push(std::make_pair(dist, idx));
        }
    }
    return result;
}

inline std::priority_queue<std::pair<float, int>> ivf_pq_solve(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k)
{
    size_t nprobe = 16;
    const char* env_np = std::getenv("IVF_NPROBE");
    if(env_np) nprobe = (size_t)std::atoi(env_np);

    const char* env_rerank = std::getenv("IVF_RERANK");
    if(env_rerank){
        size_t rerank_p = (size_t)std::atoi(env_rerank);
        return ivf_pq_rerank_search(base, query, base_number, vecdim, k, nprobe, rerank_p);
    }
    return ivf_pq_search(query, base_number, vecdim, k, nprobe);
}

#endif
