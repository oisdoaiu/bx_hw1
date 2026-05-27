#ifndef ALGO_IVF_OMP_H
#define ALGO_IVF_OMP_H

#define IVF_NLIST_VAL 256
#define IVF_NPROBE_VAL 12
#define NUM_THREADS_VAL 8

#include "simd_wrapper.h"
#include "flat_simd.h"
#include <queue>
#include <utility>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <iostream>
#include <omp.h>

static std::vector<float> g_ivf_ct;
static std::vector<std::vector<int>> g_ivf_ls;
static size_t g_ivf_nl;
static const float* g_ivf_bb;

inline void kmeans_cluster(const float* base, size_t bn, size_t vd,
    float* centroids, size_t nlist, int max_iters)
{
    for(size_t c = 0; c < nlist; c++){
        size_t idx = (c * (bn / nlist)) % bn;
        std::memcpy(centroids + c * vd, base + idx * vd, vd * sizeof(float));
    }
    std::vector<int> as(bn);
    for(int iter = 0; iter < max_iters; iter++){
        for(size_t i = 0; i < bn; i++){
            const float* v = base + i * vd;
            float bd = FLT_MAX; int bc = 0;
            for(size_t c = 0; c < nlist; c++){
                float d = InnerProductSIMD(v, centroids + c * vd, vd);
                if(d < bd){ bd = d; bc = (int)c; }
            }
            as[i] = bc;
        }
        std::vector<float> sum(nlist * vd, 0.0f); std::vector<int> cnt(nlist, 0);
        for(size_t i = 0; i < bn; i++){
            int c = as[i]; const float* v = base + i * vd;
            float* dst = sum.data() + c * vd;
            for(size_t d = 0; d < vd; d++) dst[d] += v[d];
            cnt[c]++;
        }
        for(size_t c = 0; c < nlist; c++){
            if(cnt[c] > 0){
                float inv = 1.0f / (float)cnt[c];
                for(size_t d = 0; d < vd; d++) centroids[c*vd+d] = sum[c*vd+d] * inv;
            }
        }
    }
}

inline void build_ivf_index(const float* base, size_t bn, size_t vd)
{
    g_ivf_nl = 256;
    g_ivf_nl = IVF_NLIST_VAL;
    g_ivf_bb = base;
    g_ivf_ct.resize(g_ivf_nl * vd);
    kmeans_cluster(base, bn, vd, g_ivf_ct.data(), g_ivf_nl, 15);
    g_ivf_ls.resize(g_ivf_nl);
    for(size_t c = 0; c < g_ivf_nl; c++) g_ivf_ls[c].clear();
    for(size_t i = 0; i < bn; i++){
        const float* v = base + i * vd;
        float bd = FLT_MAX; int bc = 0;
        for(size_t c = 0; c < g_ivf_nl; c++){
            float d = InnerProductSIMD(v, g_ivf_ct.data() + c * vd, vd);
            if(d < bd){ bd = d; bc = (int)c; }
        }
        g_ivf_ls[bc].push_back((int)i);
    }
}

inline void build_ivf_omp(const float* base, size_t bn, size_t vd)
{
    build_ivf_index(base, bn, vd);
}

inline std::priority_queue<std::pair<float, int>> ivf_omp_solve(
    const float* base, const float* query, size_t bn, size_t vd, size_t k)
{
    int nt = NUM_THREADS_VAL;
    size_t nprobe = IVF_NPROBE_VAL;

    std::priority_queue<std::pair<float,int>> ch;
    for(size_t c = 0; c < g_ivf_nl; c++){
        float d = InnerProductSIMD(query, g_ivf_ct.data() + c * vd, vd);
        ch.push(std::make_pair(-d, (int)c));
    }
    std::vector<int> pr; pr.reserve(nprobe);
    for(size_t p = 0; p < nprobe && !ch.empty(); p++){ pr.push_back(ch.top().second); ch.pop(); }

    std::vector<std::pair<float,int>> all_cand;
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num(), nts = omp_get_num_threads();
        size_t ck = (pr.size() + nts - 1) / nts, cs = tid * ck, ce = std::min(cs + ck, pr.size());
        std::priority_queue<std::pair<float,int>> local_pq;
        for(size_t pi = cs; pi < ce; pi++){
            int c = pr[pi];
            for(int idx : g_ivf_ls[c]){
                float dist = InnerProductSIMD(g_ivf_bb + (size_t)idx * vd, query, vd);
                if(local_pq.size() < k) local_pq.push(std::make_pair(dist, idx));
                else if(dist < local_pq.top().first){ local_pq.pop(); local_pq.push(std::make_pair(dist, idx)); }
            }
        }
        #pragma omp critical
        { while(!local_pq.empty()){ all_cand.push_back(local_pq.top()); local_pq.pop(); } }
    }

    std::priority_queue<std::pair<float,int>> result;
    for(auto& c : all_cand){
        if(result.size() < k) result.push(c);
        else if(c.first < result.top().first){ result.pop(); result.push(c); }
    }
    return result;
}

#endif
