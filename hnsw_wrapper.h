#ifndef HNSW_WRAPPER_H
#define HNSW_WRAPPER_H

#include <queue>
#include <utility>
#include <vector>
#include <cstdlib>
#include "hnswlib.h"

static hnswlib::HierarchicalNSW<float>* g_hnsw = nullptr;
static hnswlib::InnerProductSpace* g_hnsw_space = nullptr;
static int g_hnsw_ef = 50;

inline void build_hnsw(const float* base, size_t base_number, size_t vecdim)
{
    g_hnsw_space = new hnswlib::InnerProductSpace(vecdim);
    g_hnsw = new hnswlib::HierarchicalNSW<float>(g_hnsw_space, base_number, 16, 200);

    const char* env_ef = std::getenv("HNSW_EF");
    if(env_ef) g_hnsw_ef = std::atoi(env_ef);
    g_hnsw->ef_ = g_hnsw_ef;

    for(size_t i = 0; i < base_number; i++){
        g_hnsw->addPoint((void*)(base + i * vecdim), (size_t)i);
    }

    std::cerr << "HNSW built: n=" << base_number << " dim=" << vecdim
              << " M=16 ef_construction=200 ef=" << g_hnsw->ef_ << "\n";
}

inline std::priority_queue<std::pair<float, int>> hnsw_solve(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k)
{
    auto pq = g_hnsw->searchKnn((void*)query, k);

    std::vector<std::pair<float, int>> tmp;
    while(!pq.empty()){
        tmp.push_back(std::make_pair(pq.top().first, (int)pq.top().second));
        pq.pop();
    }

    std::priority_queue<std::pair<float, int>> result;
    for(int i = (int)tmp.size()-1; i >= 0; i--){
        result.push(std::make_pair(1.0f - tmp[i].first, tmp[i].second));
    }
    return result;
}

#endif
