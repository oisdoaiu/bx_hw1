#ifndef ALGO_IVF_H
#define ALGO_IVF_H

#define IVF_NLIST_VAL 256
#define IVF_NPROBE_VAL 8

#include "simd_wrapper.h"
#include "flat_simd.h"
#include <queue>
#include <utility>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <iostream>

static std::vector<float> g_ivf_centroids;
static std::vector<std::vector<int>> g_ivf_lists;
static size_t g_ivf_nlist;
static const float* g_ivf_base;

inline void kmeans_cluster(const float* base, size_t bn, size_t vd,
    float* centroids, size_t nlist, int max_iters)
{
    for(size_t c = 0; c < nlist; c++){
        size_t idx = (c * (bn / nlist)) % bn;
        std::memcpy(centroids + c * vd, base + idx * vd, vd * sizeof(float));
    }
    std::vector<int> assignments(bn);
    for(int iter = 0; iter < max_iters; iter++){
        for(size_t i = 0; i < bn; i++){
            const float* v = base + i * vd;
            float best_d = FLT_MAX; int best_c = 0;
            for(size_t c = 0; c < nlist; c++){
                float d = InnerProductSIMD(v, centroids + c * vd, vd);
                if(d < best_d){ best_d = d; best_c = (int)c; }
            }
            assignments[i] = best_c;
        }
        std::vector<float> sum(nlist * vd, 0.0f);
        std::vector<int> counts(nlist, 0);
        for(size_t i = 0; i < bn; i++){
            int c = assignments[i]; const float* v = base + i * vd;
            float* dst = sum.data() + c * vd;
            for(size_t d = 0; d < vd; d++) dst[d] += v[d];
            counts[c]++;
        }
        for(size_t c = 0; c < nlist; c++){
            if(counts[c] > 0){
                float inv = 1.0f / (float)counts[c];
                for(size_t d = 0; d < vd; d++) centroids[c*vd+d] = sum[c*vd+d] * inv;
            }
        }
    }
}

inline void build_ivf_index(const float* base, size_t bn, size_t vd)
{
    g_ivf_nlist = 256;
    g_ivf_nlist = IVF_NLIST_VAL;
    g_ivf_base = base;
    g_ivf_centroids.resize(g_ivf_nlist * vd);
    kmeans_cluster(base, bn, vd, g_ivf_centroids.data(), g_ivf_nlist, 15);
    g_ivf_lists.resize(g_ivf_nlist);
    for(size_t c = 0; c < g_ivf_nlist; c++) g_ivf_lists[c].clear();
    for(size_t i = 0; i < bn; i++){
        const float* v = base + i * vd;
        float best_d = FLT_MAX; int best_c = 0;
        for(size_t c = 0; c < g_ivf_nlist; c++){
            float d = InnerProductSIMD(v, g_ivf_centroids.data() + c * vd, vd);
            if(d < best_d){ best_d = d; best_c = (int)c; }
        }
        g_ivf_lists[best_c].push_back((int)i);
    }
}

inline std::priority_queue<std::pair<float,int>> ivf_search(
    const float* query, size_t bn, size_t vd, size_t k, size_t nprobe)
{
    std::priority_queue<std::pair<float,int>> ch;
    for(size_t c = 0; c < g_ivf_nlist; c++){
        float d = InnerProductSIMD(query, g_ivf_centroids.data() + c * vd, vd);
        ch.push(std::make_pair(-d, (int)c));
    }
    std::vector<int> probes; probes.reserve(nprobe);
    for(size_t i = 0; i < nprobe && !ch.empty(); i++){
        probes.push_back(ch.top().second); ch.pop();
    }
    std::priority_queue<std::pair<float,int>> result;
    for(int c : probes){
        for(int idx : g_ivf_lists[c]){
            float dist = InnerProductSIMD(g_ivf_base + (size_t)idx * vd, query, vd);
            if(result.size() < k) result.push(std::make_pair(dist, idx));
            else if(dist < result.top().first){ result.pop(); result.push(std::make_pair(dist, idx)); }
        }
    }
    return result;
}

inline void build_ivf_algo(const float* base, size_t bn, size_t vd)
{
    build_ivf_index(base, bn, vd);
}

inline std::priority_queue<std::pair<float, int>> ivf_algo_solve(
    const float* base, const float* query, size_t bn, size_t vd, size_t k)
{
    size_t nprobe = IVF_NPROBE_VAL;
    return ivf_search(query, bn, vd, k, nprobe);
}

#endif
