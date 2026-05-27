#ifndef ALGO_HNSW_H
#define ALGO_HNSW_H

#define HNSW_M_VAL 16
#define HNSW_EFC_VAL 200
#define HNSW_EF_VAL 50

#include <queue>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <omp.h>

static hnswlib::HierarchicalNSW<float>* g_hnsw_ptr = nullptr;
static hnswlib::InnerProductSpace* g_hnsw_space_p = nullptr;

inline void build_hnsw_algo(const float* base, size_t base_number, size_t vecdim)
{
    int M = HNSW_M_VAL;
    int efc = HNSW_EFC_VAL;
    int ef = HNSW_EF_VAL;

    g_hnsw_space_p = new hnswlib::InnerProductSpace(vecdim);
    g_hnsw_ptr = new hnswlib::HierarchicalNSW<float>(g_hnsw_space_p, base_number, M, efc);
    g_hnsw_ptr->ef_ = ef;

    g_hnsw_ptr->addPoint((void*)base, 0);
    #pragma omp parallel for
    for(size_t i = 1; i < base_number; i++)
        g_hnsw_ptr->addPoint((void*)(base + i * vecdim), (size_t)i);
}

inline std::priority_queue<std::pair<float, int>> hnsw_algo_solve(
    const float* base, const float* query,
    size_t base_number, size_t vecdim, size_t k)
{
    int ef = HNSW_EF_VAL;
    g_hnsw_ptr->ef_ = ef;

    auto pq = g_hnsw_ptr->searchKnn((void*)query, k);

    std::vector<std::pair<float,int>> tmp;
    while(!pq.empty()){
        tmp.push_back(std::make_pair(pq.top().first, (int)pq.top().second));
        pq.pop();
    }
    std::priority_queue<std::pair<float,int>> result;
    for(int i = (int)tmp.size() - 1; i >= 0; i--)
        result.push(std::make_pair(1.0f - tmp[i].first, tmp[i].second));
    return result;
}

#endif
